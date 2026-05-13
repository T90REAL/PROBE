import os
import json
import tomllib
from typing import Any
from pathlib import Path
from dataclasses import dataclass
from openai import AsyncOpenAI, OpenAI


@dataclass(slots=True)
class RuntimeConfig:
    model_name: str
    base_url: str | None
    api_key: str
    python_executable: str
    llm_concurrency: int
    request_timeout_seconds: float
    pytest_timeout_seconds: int
    temperature: float
    enforce_strength_checks: bool
    reject_weak_suites: bool

    @classmethod
    def load(cls, config_path: str | None = None, *, api_key_override: str | None = None, base_url_override: str | None = None):
        data: dict[str, str] = {}
        if config_path:
            raw = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
            data = {k: str(v) for k, v in raw.get("runtime", {}).items()}

        model = data.get("model_name") or os.environ.get("PROBE_MODEL") or "gpt-4.1"
        base_url = (
            base_url_override
            if base_url_override is not None
            else (data.get("base_url") or os.environ.get("OPENAI_BASE_URL"))
        )
        api_key = (
            api_key_override
            if api_key_override is not None
            else (data.get("api_key") or os.environ.get("OPENAI_API_KEY") or os.environ.get("PROBE_API_KEY", ""))
        )
        python_executable = (
            data.get("python_executable")
            or os.environ.get("PROBE_PYTHON")
            or os.environ.get("PYTHON", "python3")
        )
        llm_concurrency = int(data.get('llm_concurrency') or os.environ.get('PROBE_LLM_CONCURRENCY') or '1')
        request_timeout_seconds = float(data.get('request_timeout_seconds') or os.environ.get('PROBE_LLM_TIMEOUT_SECONDS') or '120')
        pytest_timeout_seconds = int(data.get('pytest_timeout_seconds') or os.environ.get('PROBE_PYTEST_TIMEOUT_SECONDS') or '120')
        temperature = float(data.get('temperature') or os.environ.get('PROBE_LLM_TEMPERATURE') or '0.2')
        enforce_strength_checks_raw = (
            data.get("enforce_strength_checks")
            or os.environ.get("PROBE_ENFORCE_STRENGTH_CHECKS")
            or "true"
        )
        reject_weak_suites_raw = (
            data.get("reject_weak_suites")
            or os.environ.get("PROBE_REJECT_WEAK_SUITES")
            or "true"
        )
        enforce_strength_checks = enforce_strength_checks_raw.lower() in {"1", "true", "yes", "on"}
        reject_weak_suites = reject_weak_suites_raw.lower() in {"1", "true", "yes", "on"}
        if not api_key and base_url and "localhost" in base_url:
            api_key = "None"
        if not api_key:
            raise ValueError("Missing API key. Set OPENAI_API_KEY/PROBE_API_KEY or provide config.")
        return cls(model_name=model, base_url=base_url, api_key=api_key, python_executable=python_executable, llm_concurrency=llm_concurrency, request_timeout_seconds=request_timeout_seconds, pytest_timeout_seconds=pytest_timeout_seconds, temperature=temperature, enforce_strength_checks=enforce_strength_checks, reject_weak_suites=reject_weak_suites)


class LLMClient:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=config.request_timeout_seconds, max_retries=0)
        self.async_client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url, timeout=config.request_timeout_seconds, max_retries=0)

    def _build_kwargs(self, system_prompt: str, user_prompt: str, require_json: bool):
        kwargs: dict[str, Any] = {
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self._uses_minimax_model():
            kwargs["extra_body"] = {"reasoning_split": True}
        if require_json:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def _uses_minimax_model(self):
        model = (self.config.model_name or "").lower()
        base_url = (self.config.base_url or "").lower()
        return "minimax" in model or "minimaxi.com" in base_url

    def chat(self, system_prompt: str, user_prompt: str, require_json: bool = False):
        kwargs = self._build_kwargs(system_prompt, user_prompt, require_json)
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception:
            if not require_json:
                raise
            kwargs.pop("response_format", None)
            response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        return content or ""

    async def achat(self, system_prompt: str, user_prompt: str, require_json: bool = False):
        kwargs = self._build_kwargs(system_prompt, user_prompt, require_json)
        try:
            response = await self.async_client.chat.completions.create(**kwargs)
        except Exception:
            if not require_json:
                raise
            kwargs.pop("response_format", None)
            response = await self.async_client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        return content or ""


def load_json_payload(raw: str):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("\n", 1)
        if len(parts) == 2:
            cleaned = parts[1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for start_token in ("{", "["):
        start = cleaned.find(start_token)
        if start == -1:
            continue
        try:
            payload, _ = decoder.raw_decode(cleaned[start:])
            return payload
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("Could not parse JSON payload", cleaned, 0)
