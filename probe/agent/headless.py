import os
import json
import tomllib
from typing import Any
from pathlib import Path

from probe.tools.registry import ToolRegistry
from probe.agent.agent_config import ActiveModel, AgentConfig
from probe.utils.litellm_backend import LitellmModel, LLMReply, LLMError


CONFIG_PATH = Path("probe.toml")


class HeadlessAgent:
    def __init__(
        self,
        *,
        llm: LitellmModel,
        tools: ToolRegistry,
        ctx: Any,
        config: AgentConfig,
        allowlist: list[str] | None = None,
        histories: list[dict[str, Any]] | None = None,
    ) -> None:
        self.llm = llm
        self.ctx = ctx
        self.tools = tools
        self.config = config
        self.allowlist = allowlist

        self.step_cnt = 0
        self.current_task = ""
        self.messages: list[dict[str, Any]] = list(histories or [])

    @property
    def traj_path(self) -> Path | None:
        if self.config.traj_path is None:
            return None
        return Path(self.config.traj_path)

    @property
    def active_model(self) -> ActiveModel:
        return self.config.active_model

    @active_model.setter
    def active_model(self, value: ActiveModel) -> None:
        self.set_active_model(value)


    @classmethod
    def from_toml(
        cls,
        *,
        tools: ToolRegistry,
        ctx: Any,
        system_prompt: str,
        allowlist: list[str] | None = None,
        histories: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        active_model: ActiveModel = "normal",
    ) -> "HeadlessAgent":
        data = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        config = AgentConfig(
            system_prompt=system_prompt,
            model_name=_require_toml_str(data, "model_name"),
            api_key=_resolve_api_key(_require_toml_str(data, "api_key")),
            base_url=_require_toml_str(data, "base_url"),
            adv_model_name=_require_toml_str(data, "adv_model_name"),
            adv_api_key=_resolve_api_key(_require_toml_str(data, "adv_api_key")),
            adv_base_url=_require_toml_str(data, "adv_base_url"),
            active_model=active_model,
            temperature=float(data.get("temperature", 0.8)),
            timeout_per_step=float(data.get("timeout_per_step", 120)),
            max_steps=int(data.get("max_steps", 100)),
            context_window=int(data.get("context_window", 0)),
            traj_path=_toml_str(data.get("traj_path")),
            tool_choice=tool_choice,
        )

        return cls(
            llm=_build_litellm_model(config),
            tools=tools,
            ctx=ctx,
            config=config,
            allowlist=allowlist,
            histories=histories,
        )

    def set_active_model(self, active_model: ActiveModel) -> None:
        self.config.active_model = active_model
        self.config.validate_active_model()
        self.llm = _build_litellm_model(self.config)


    def run(self, task: str) -> LLMReply:
        """
        -> initialize history / message
        -> while not done:
             step()
         """
        self.messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": task},
        ]

        self.current_task = task
        self.step_cnt = 0
        try:
            while True:
                reply = self.step()
                if not reply.tool_calls:
                    self.save_trajectory(task=task, status="ok", final=reply.content)
                    return reply
        except Exception as exc:
            self.save_trajectory(task=task, status="error", error=str(exc))
            raise


    def save_trajectory(
        self,
        *,
        task: str,
        status: str,
        final: str = "",
        error: str = "",
    ) -> None:
        traj_path = self.traj_path
        if traj_path is None:
            return

        traj_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "status": status,
            "task": task,
            "final": final,
            "error": error,
            "steps": self.step_cnt,
            "messages": self.messages,
        }
        with traj_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


    def _to_assistant_message(self, reply: LLMReply) -> dict[str, Any]:
        message = {
            "role": "assistant",
            "content": reply.content,
        }
        if reply.tool_calls:
            message["tool_calls"] = [call.to_openai() for call in reply.tool_calls]
        return message


    def _to_tool_message(self, tool_call_id: str, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(result, ensure_ascii=False),
        }


    def step(self) -> LLMReply:
        """ 
        -> query model 
        -> parse actions / tool_calls 
        -> execute tools 
        -> append observations
        """
        if self.step_cnt >= self.config.max_steps:
            raise LLMError(f"max steps {self.config.max_steps} exceeded!")

        self.step_cnt += 1
        message = self.llm.query(
            messages=self.messages,
            tools=self.tools.as_openai_tools(self.allowlist),
            tool_choice=self.config.tool_choice,
        )

        self.messages.append(self._to_assistant_message(message))
        
        for call in message.tool_calls:
            result = self.tools.dispatch(
                self.ctx,
                call.name,
                call.arguments,
                allowlist=self.allowlist,
            )
            self.messages.append(self._to_tool_message(call.id, result))

        self.save_trajectory(task=self.current_task, status="step")
        return message


def _toml_str(value: Any) -> str | None:
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _require_toml_str(data: dict[str, Any], key: str) -> str:
    value = _toml_str(data.get(key))
    if value is None:
        raise LLMError(f"missing required config field: {key}")
    return value


def _resolve_api_key(value: str | None) -> str:
    if not value:
        raise LLMError("missing api key: set api_key in probe.toml")
    return os.environ.get(value) or value


def _build_litellm_model(config: AgentConfig) -> LitellmModel:
    return LitellmModel(
        model_name=config.active_model_name,
        api_key=config.active_api_key,
        base_url=config.active_base_url,
        temperature=config.temperature,
        timeout=config.timeout_per_step,
    )
