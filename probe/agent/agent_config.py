from typing import Any, Literal
from pathlib import Path
from dataclasses import dataclass

ActiveModel = Literal["normal", "adv"]


@dataclass(slots=True)
class AgentConfig:
    system_prompt: str
    model_name: str = ""
    api_key: str = ""
    base_url: str = ""
    adv_model_name: str = ""
    adv_api_key: str = ""
    adv_base_url: str = ""
    active_model: ActiveModel = "normal"
    temperature: float = 0.8
    timeout_per_step: float = 120.0
    max_steps: int = 100
    context_window: int = 0
    traj_path: str | Path | None = None
    tool_choice: str | dict[str, Any] | None = "auto" # the model has to support tool calling

    def __post_init__(self) -> None:
        self.validate_active_model()

    def validate_active_model(self) -> None:
        if self.active_model not in ("normal", "adv"):
            raise ValueError("active_model must be 'normal' or 'adv'")

    @property
    def active_model_name(self) -> str:
        return self.adv_model_name if self.active_model == "adv" else self.model_name

    @property
    def active_api_key(self) -> str:
        return self.adv_api_key if self.active_model == "adv" else self.api_key

    @property
    def active_base_url(self) -> str:
        return self.adv_base_url if self.active_model == "adv" else self.base_url
