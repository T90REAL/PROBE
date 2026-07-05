import json
import litellm
from typing import Any
from dataclasses import dataclass
from collections.abc import Mapping

from probe.utils.toolcall import ToolCall


class LLMError(RuntimeError):
    """Raised when the LLM backend cannot complete or parse a request."""


@dataclass(slots=True)
class LLMReply:
    content: str
    tool_calls: list[ToolCall]
    finish_reason: str | None = None
    # usage: dict[str, Any] | None = None
    raw: Any | None = None


class LitellmModel:
    _abort_exceptions = (
        litellm.exceptions.UnsupportedParamsError,
        litellm.exceptions.NotFoundError,
        litellm.exceptions.PermissionDeniedError,
        litellm.exceptions.ContextWindowExceededError,
        litellm.exceptions.AuthenticationError,
    )

    def __init__(
        self,
        model_name: str,
        *,
        api_key: str,
        base_url: str,
        temperature: float = 0.8,
        timeout: float = 120.0,
        num_retries: int = 1,
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if not model_name:
            raise ValueError("model_name is required!")

        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout
        self.num_retries = num_retries
        self.model_kwargs = dict(model_kwargs or {})


    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {k: v for k, v in msg.items() if k != "extra"}
            for msg in messages
        ]


    def _query(
        self,
        messages: list[dict[str, Any]],
        **kwargs
    ) -> Any:
        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "api_key" : self.api_key,
            "base_url" : self.base_url,
            "messages": messages,
            "temperature": self.temperature,
            "timeout": self.timeout,
        }

        attempts = max(1, self.num_retries + 1)
        for attempt in range(attempts):
            try:
                return litellm.completion(**(request_kwargs | self.model_kwargs | kwargs))
            except self._abort_exceptions as exc:
                raise LLMError(f"LiteLLM request failed: {exc}") from exc
            except Exception as exc:
                if attempt == attempts - 1:
                    raise LLMError(f"LiteLLM request failed after {attempts} attempt(s): {exc}") from exc

        raise LLMError("LiteLLM request failed")


    def _parse_reply(self, response: Any) -> LLMReply:
        try:
            choice = response.choices[0]
            message = choice.message
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise LLMError(f"LiteLLM response has unexpected shape: {exc}") from exc

        content = getattr(message, "content", None) or ""
        finish_reason = getattr(choice, "finish_reason", None)

        return LLMReply(
            content=content,
            tool_calls=self._parse_tool_calls(getattr(message, "tool_calls", None) or []),
            finish_reason=finish_reason,
            raw=response,
        )


    def _parse_tool_calls(self, raw_tool_calls: Any) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []
        for raw in raw_tool_calls:
            try:
                function = raw.function
                raw_arguments = function.arguments or "{}"
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise LLMError(f"Invalid tool call arguments JSON: {exc}") from exc
            except AttributeError as exc:
                raise LLMError(f"Tool call has unexpected shape: {exc}") from exc
            if not isinstance(arguments, dict):
                raise LLMError(f"Tool call arguments must be a JSON object, got {type(arguments).__name__}")

            tool_calls.append(
                ToolCall(
                    id=raw.id,
                    name=function.name,
                    arguments=arguments,
                    raw_arguments=raw_arguments,
                )
            )
        return tool_calls


    def query(
        self,
        messages: list[dict[str, Any]],
        **kwargs
    ) -> LLMReply:
        response = self._query(messages=self._prepare_messages(messages), **kwargs)
        return self._parse_reply(response)
