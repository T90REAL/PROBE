"""
Load and validate agent workflow definitions.

A workflow yaml (see ``default.yaml``) declares parameters using shell-style
env substitution ``${ENV_VAR:-default}``, resolved once at load time. Every
other field references resolved parameters with ``{{name}}`` placeholders,
rendered here so downstream consumers (the agent loop) receive finished text.
"""

import os
import re
import yaml
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


_ENV_PATTERN = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")
_PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\}\}")

_TASK_REQUIRED_FIELDS = ("name", "instruction", "allowed_tools", "output_artifact", "output_schema")


class WorkflowLoadError(Exception):
    """Raised when a workflow file is missing, malformed, or fails validation."""


@dataclass(slots=True, frozen=True)
class TaskSpec:
    name: str
    instruction: str
    allowed_tools: tuple[str, ...]
    output_artifact: str
    output_schema: str


@dataclass(slots=True, frozen=True)
class WorkflowSpec:
    name: str
    goal: str
    system_prompt: str
    parameters: dict[str, str]
    tasks: tuple[TaskSpec, ...]

    def task(self, name: str) -> TaskSpec:
        for task in self.tasks:
            if task.name == name:
                return task
        raise KeyError(f"workflow {self.name!r} has no task named {name!r}")


def load_workflow(
    path: str | Path,
    *,
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, str] | None = None,
    known_tools: Iterable[str] | None = None,
) -> WorkflowSpec:
    """Parse, resolve, render, and validate a workflow yaml file.

    ``env`` defaults to ``os.environ``; inject a mapping in tests.
    ``overrides`` wins over both env and yaml defaults (e.g. CLI arguments).
    ``known_tools``, when given, restricts task allowlists to registered tools.
    """
    path = Path(path)
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise WorkflowLoadError(f"cannot read workflow file {path}: {exc}") from exc

    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise WorkflowLoadError(f"invalid yaml in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise WorkflowLoadError(f"workflow file {path} must contain a mapping at the top level")

    name = _require_str(data, "name", context=str(path))
    system_prompt = _require_str(data, "system_prompt", context=str(path))
    goal = data.get("goal", "")

    parameters = _resolve_parameters(
        data.get("parameters") or {},
        env=os.environ if env is None else env,
        overrides=overrides or {},
    )

    goal = _render(goal, parameters, where="goal")
    system_prompt = _render(system_prompt, parameters, where="system_prompt")
    tasks = _parse_tasks(data.get("workflow"), parameters)

    _validate(tasks, parameters, known_tools)
    return WorkflowSpec(name=name, goal=goal, system_prompt=system_prompt, parameters=parameters, tasks=tasks)


def _resolve_parameters(
    raw: Any,
    *,
    env: Mapping[str, str],
    overrides: Mapping[str, str],
) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise WorkflowLoadError("'parameters' must be a mapping of name -> value")

    unknown_overrides = set(overrides) - set(raw)
    if unknown_overrides:
        raise WorkflowLoadError(f"overrides for undeclared parameters: {sorted(unknown_overrides)}")

    resolved: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.isidentifier():
            raise WorkflowLoadError(f"parameter name {key!r} must be a valid identifier")
        if key in overrides:
            resolved[key] = str(overrides[key])
            continue
        if not isinstance(value, str):
            raise WorkflowLoadError(f"parameter {key!r} must be a string, got {type(value).__name__}")
        resolved[key] = _substitute_env(value, env, parameter=key)
    return resolved


def _substitute_env(value: str, env: Mapping[str, str], *, parameter: str) -> str:
    def replace(match: re.Match[str]) -> str:
        var = match.group("name")
        default = match.group("default")
        if var in env:
            return env[var]
        if default is not None:
            return default
        raise WorkflowLoadError(
            f"parameter {parameter!r} references ${{{var}}} which is unset and has no default"
        )

    return _ENV_PATTERN.sub(replace, value)


def _render(text: str, parameters: Mapping[str, str], *, where: str) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group("name")
        if key not in parameters:
            available = ", ".join(sorted(parameters)) or "<none>"
            raise WorkflowLoadError(
                f"unknown placeholder {{{{{key}}}}} in {where}; declared parameters: {available}"
            )
        return parameters[key]

    return _PLACEHOLDER_PATTERN.sub(replace, text)


def _parse_tasks(raw: Any, parameters: Mapping[str, str]) -> tuple[TaskSpec, ...]:
    if not isinstance(raw, list) or not raw:
        raise WorkflowLoadError("'workflow' must be a non-empty list of tasks")

    tasks: list[TaskSpec] = []
    for position, entry in enumerate(raw):
        if not isinstance(entry, dict) or set(entry) != {"task"} or not isinstance(entry["task"], dict):
            raise WorkflowLoadError(f"workflow entry #{position} must be a mapping with a single 'task' key")
        body = entry["task"]
        label = body.get("name") or f"#{position}"
        missing = [f for f in _TASK_REQUIRED_FIELDS if f not in body]
        if missing:
            raise WorkflowLoadError(f"task {label} is missing required fields: {missing}")
        unknown = set(body) - set(_TASK_REQUIRED_FIELDS)
        if unknown:
            raise WorkflowLoadError(f"task {label} has unknown fields: {sorted(unknown)}")

        task_name = _require_str(body, "name", context=f"task {label}")
        tools = body["allowed_tools"]
        if (
            not isinstance(tools, list)
            or not tools
            or not all(isinstance(t, str) and t.strip() for t in tools)
        ):
            raise WorkflowLoadError(f"task {task_name!r}: 'allowed_tools' must be a non-empty list of tool names")

        tasks.append(
            TaskSpec(
                name=task_name,
                instruction=_render(
                    _require_str(body, "instruction", context=f"task {task_name}"),
                    parameters,
                    where=f"task {task_name!r} instruction",
                ),
                allowed_tools=tuple(t.strip() for t in tools),
                output_artifact=_render(
                    _require_str(body, "output_artifact", context=f"task {task_name}"),
                    parameters,
                    where=f"task {task_name!r} output_artifact",
                ),
                output_schema=_require_str(body, "output_schema", context=f"task {task_name}"),
            )
        )
    return tuple(tasks)


def _validate(
    tasks: tuple[TaskSpec, ...],
    parameters: Mapping[str, str],
    known_tools: Iterable[str] | None,
) -> None:
    seen: set[str] = set()
    for task in tasks:
        if task.name in seen:
            raise WorkflowLoadError(f"duplicate task name {task.name!r}")
        seen.add(task.name)

        duplicate_tools = [t for t in task.allowed_tools if task.allowed_tools.count(t) > 1]
        if duplicate_tools:
            raise WorkflowLoadError(f"task {task.name!r} lists duplicate tools: {sorted(set(duplicate_tools))}")

        artifact = PurePosixPath(task.output_artifact)
        if artifact.is_absolute() or ".." in artifact.parts:
            raise WorkflowLoadError(f"task {task.name!r}: output_artifact {task.output_artifact!r} must be a relative path without '..'")
        output_dir = parameters.get("output_dir")
        if output_dir and not _is_within(artifact, PurePosixPath(output_dir)):
            raise WorkflowLoadError(f"task {task.name!r}: output_artifact {task.output_artifact!r} must live under output_dir {output_dir!r}")

    if known_tools is not None:
        registry = set(known_tools)
        for task in tasks:
            unknown = [t for t in task.allowed_tools if t not in registry]
            if unknown:
                raise WorkflowLoadError(f"task {task.name!r} allows unregistered tools: {sorted(unknown)}")


def _is_within(path: PurePosixPath, root: PurePosixPath) -> bool:
    return path.parts[: len(root.parts)] == root.parts


def _require_str(data: Mapping[str, Any], key: str, *, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise WorkflowLoadError(f"{context}: {key!r} must be a non-empty string")
    return value
