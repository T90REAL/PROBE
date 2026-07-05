"""
Tools are plain functions registered on a ``ToolRegistry`` with a name, description, and JSON-schema-style parameter declarations. 
The registry converts an allowlist into the OpenAI function-calling payload and dispatches tool calls coming back from the model.

Every tool receives the shared run context as its first positional argument
and the model-supplied arguments as keyword arguments:

    registry = ToolRegistry()

    @registry.tool(
        name="run_pytest",
        description="Execute a pytest test file and report the outcome.",
        params={"code": {"type": "string", "description": "full test file content"}},
    )
    def run_pytest(ctx, *, code: str) -> dict:
        ...
"""

import json
import keyword
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping


_JSON_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


class ToolRegistrationError(Exception):
    """Raised for invalid tool declarations (bad name, duplicate, bad schema)."""


@dataclass(slots=True, frozen=True)
class ToolSpec:
    name: str
    description: str
    params: dict[str, dict[str, Any]]
    required: tuple[str, ...]
    func: Callable[..., Any]

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.params,
                    "required": list(self.required),
                    "additionalProperties": False,
                },
            },
        }


class ToolRegistry:
    def __init__(self, *, max_chars_per_field: int = 16000):
        self._tools: dict[str, ToolSpec] = {}
        self.max_chars_per_field = max_chars_per_field

    def tool(
        self,
        *,
        name: str,
        description: str,
        params: Mapping[str, Mapping[str, Any]] | None = None,
        required: Iterable[str] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator form of :meth:`register`; returns the function unchanged."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.register(name=name, description=description, params=params, required=required, func=func)
            return func

        return decorator

    def register(
        self,
        *,
        name: str,
        description: str,
        func: Callable[..., Any],
        params: Mapping[str, Mapping[str, Any]] | None = None,
        required: Iterable[str] | None = None,
    ) -> None:
        if not name.isidentifier():
            raise ToolRegistrationError(f"tool name {name!r} must be a valid identifier")
        if name in self._tools:
            raise ToolRegistrationError(f"tool {name!r} is already registered")
        if not description.strip():
            raise ToolRegistrationError(f"tool {name!r} needs a non-empty description")

        cleaned: dict[str, dict[str, Any]] = {}
        for param, schema in (params or {}).items():
            if not isinstance(param, str) or not param.isidentifier() or keyword.iskeyword(param):
                raise ToolRegistrationError(
                    f"tool {name!r} parameter name {param!r} must be a valid non-keyword Python identifier"
                )
            if not isinstance(schema, Mapping) or schema.get("type") not in _JSON_TYPE_CHECKS:
                raise ToolRegistrationError(
                    f"tool {name!r} parameter {param!r} needs a schema with a JSON type "
                    f"(one of {sorted(_JSON_TYPE_CHECKS)})"
                )
            cleaned[param] = dict(schema)

        # Parameters are required unless declared otherwise.
        required_tuple = tuple(cleaned) if required is None else tuple(required)
        unknown_required = set(required_tuple) - set(cleaned)
        if unknown_required:
            raise ToolRegistrationError(f"tool {name!r} requires undeclared parameters: {sorted(unknown_required)}")

        self._tools[name] = ToolSpec(
            name=name,
            description=description.strip(),
            params=cleaned,
            required=required_tuple,
            func=func,
        )

    def names(self) -> frozenset[str]:
        return frozenset(self._tools)

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"no tool registered under {name!r}") from None

    def as_openai_tools(self, allowlist: Iterable[str] | None = None) -> list[dict[str, Any]]:
        """Build the ``tools=`` payload for an allowlist (default: all tools).

        Unknown names raise: the loader already validates allowlists, so a miss
        here is a programming error, not a model mistake.
        """
        names = list(self._tools) if allowlist is None else list(allowlist)
        missing = [n for n in names if n not in self._tools]
        if missing:
            raise KeyError(f"allowlist names unregistered tools: {sorted(missing)}")
        return [self._tools[n].to_openai() for n in names]

    def dispatch(
        self,
        ctx: Any,
        name: str,
        args: Mapping[str, Any] | str | None,
        *,
        allowlist: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Execute one model-issued tool call. Never raises.

        ``args`` may be an already-parsed mapping or the raw JSON string of an
        OpenAI tool call (``function.arguments``).
        """
        if name not in self._tools:
            return self._error(f"unknown tool {name!r}; available tools: {sorted(self._tools)}")
        if allowlist is not None:
            allowed = set(allowlist)
            if name not in allowed:
                return self._error(f"tool {name!r} is not allowed in the current task; allowed: {sorted(allowed)}")

        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError as exc:
                return self._error(f"tool {name!r} received invalid JSON arguments: {exc}")
        if args is None:
            args = {}
        if not isinstance(args, Mapping):
            return self._error(f"tool {name!r} arguments must be a JSON object, got {type(args).__name__}")
        args = dict(args)

        spec = self._tools[name]
        unexpected = set(args) - set(spec.params)
        if unexpected:
            return self._error(f"tool {name!r} got unexpected arguments: {sorted(unexpected)}")
        missing = [p for p in spec.required if p not in args]
        if missing:
            return self._error(f"tool {name!r} is missing required arguments: {missing}")
        for param, value in args.items():
            expected = spec.params[param]["type"]
            if not _JSON_TYPE_CHECKS[expected](value):
                return self._error(
                    f"tool {name!r} argument {param!r} must be of JSON type "
                    f"{expected!r}, got {type(value).__name__}"
                )

        try:
            result = spec.func(ctx, **args)
        except Exception as exc:  # deliberate blanket catch: the loop must survive any tool crash
            return self._error(f"tool {name!r} failed: {type(exc).__name__}: {exc}")

        if not isinstance(result, dict):
            result = {"result": result}
        try:
            json.dumps(result)
        except (TypeError, ValueError):
            return self._error(f"tool {name!r} returned a non-JSON-serializable result")
        return _truncate(result, self.max_chars_per_field)

    def _error(self, message: str) -> dict[str, Any]:
        """Build an error result with the same truncation budget as successes."""
        return _truncate({"error": message}, self.max_chars_per_field)


def _truncate(value: Any, limit: int) -> Any:
    """Recursively cap every string leaf at ``limit`` characters."""
    if isinstance(value, str) and len(value) > limit:
        dropped = len(value) - limit
        return value[:limit] + f"...[truncated {dropped} chars]"
    if isinstance(value, dict):
        return {k: _truncate(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [_truncate(v, limit) for v in value]
    return value
