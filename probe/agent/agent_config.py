from typing import Any
from dataclasses import dataclass


@dataclass(slots=True)
class AgentConfig:
    system_prompt: str
    max_steps: int = 100
    tool_choice: str | dict[str, Any] | None = "auto" # the model has to support tool calling
