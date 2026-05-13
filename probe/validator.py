import ast
import json
import textwrap

from .llm import LLMClient, load_json_payload
from .models import CounterImplementation, TestingTask, ValidationDecision
from .prompts import COUNTER_IMPLEMENTATION_SYSTEM, VALIDATION_SYSTEM


class Validator:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def diagnose(self, task: TestingTask, pbt_code: str, error_message: str, combined_pbt_code: str | None = None):
        payload = {
            "task": task.to_prompt_dict(),
            "pbt_code": pbt_code,
            "error_message": error_message,
        }
        if combined_pbt_code is not None:
            payload["combined_pbt_code"] = combined_pbt_code
        raw = self.llm.chat(VALIDATION_SYSTEM, json.dumps(payload, ensure_ascii=False, indent=2), require_json=True)
        try:
            data = load_json_payload(raw)
        except json.JSONDecodeError:
            return self._non_json_decision(raw)
        return ValidationDecision(error_type=data.get('error_type', 'code_defect'), reasoning=data.get('reasoning', ''), fix_suggestion=data.get('fix_suggestion', ''))

    # yes, this just works
    async def adiagnose(self, task: TestingTask, pbt_code: str, error_message: str, combined_pbt_code: str | None = None):
        payload = {
            "task": task.to_prompt_dict(),
            "pbt_code": pbt_code,
            "error_message": error_message,
        }
        if combined_pbt_code is not None:
            payload["combined_pbt_code"] = combined_pbt_code
        raw = await self.llm.achat(VALIDATION_SYSTEM, json.dumps(payload, ensure_ascii=False, indent=2), require_json=True)
        try:
            data = load_json_payload(raw)
        except json.JSONDecodeError:
            return self._non_json_decision(raw)
        return ValidationDecision(error_type=data.get('error_type', 'code_defect'), reasoning=data.get('reasoning', ''), fix_suggestion=data.get('fix_suggestion', ''))

    def _non_json_decision(self, raw: str):
        snippet = raw.strip()
        if len(snippet) > 1000:
            snippet = snippet[:997].rstrip() + "..."
        return ValidationDecision(error_type='code_defect', reasoning='Validator returned a non-JSON diagnosis; continue repairing the generated test harness from the execution error.', fix_suggestion=snippet)

    def generate_counter_implementation(self, task: TestingTask, combined_pbt_code: str):
        impls = self.generate_counter_implementations(task, combined_pbt_code, max_candidates=1)
        return impls[0] if impls else None

    def generate_counter_implementations(self, task: TestingTask, combined_pbt_code: str, max_candidates: int = 3):
        payload = {
            "task": task.to_prompt_dict(),
            "combined_pbt_code": self._compact_code_for_prompt(combined_pbt_code),
            "max_candidates": max_candidates,
        }
        # TODO: 这里有可能和LLM的参数有很大关系，do some search
        raw = self.llm.chat(COUNTER_IMPLEMENTATION_SYSTEM, json.dumps(payload, ensure_ascii=False, indent=2), require_json=True)
        data = load_json_payload(raw)
        return self._parse_counter_implementations(task, data, max_candidates=max_candidates)

    async def agenerate_counter_implementation(self, task: TestingTask, combined_pbt_code: str):
        impls = await self.agenerate_counter_implementations(task, combined_pbt_code, max_candidates=1)
        return impls[0] if impls else None

    async def agenerate_counter_implementations(self, task: TestingTask, combined_pbt_code: str, max_candidates: int = 3):
        payload = {
            "task": task.to_prompt_dict(),
            "combined_pbt_code": self._compact_code_for_prompt(combined_pbt_code),
            "max_candidates": max_candidates,
        }
        raw = await self.llm.achat(COUNTER_IMPLEMENTATION_SYSTEM, json.dumps(payload, ensure_ascii=False, indent=2), require_json=True)
        data = load_json_payload(raw)
        return self._parse_counter_implementations(task, data, max_candidates=max_candidates)

    def _parse_counter_implementations(self, task: TestingTask, data: dict, max_candidates: int):
        raw_items = list(data.get("counter_implementations") or [])
        if not raw_items and data.get("has_counter_implementation"):
            item = data.get("counter_implementation")
            if item:
                raw_items = [item]

        results: list[CounterImplementation] = []
        seen: set[str] = set()
        for item in raw_items:
            code = self._normalize_counter_code(task, (item.get("code") or "").strip())
            if not code or code in seen: continue
            seen.add(code)
            results.append(CounterImplementation(description=item.get('description', ''), code=code, what_it_violates=item.get('what_it_violates', '')))
            if len(results) >= max_candidates: break
        return results

    def _normalize_counter_code(self, task: TestingTask, code: str):
        if code.startswith("```"): code = code.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        if not code: return None
        if self._defines_target_function(code, task.target.short_name()): return code

        body = textwrap.dedent(code).strip()
        if not body: return None
        signature = task.target.signature.strip() or "()"
        if not signature.startswith("("):
            signature = f"({signature})"
        wrapped = f"def {task.target.short_name()}{signature}:\n{textwrap.indent(body, '    ')}\n"


        if self._defines_target_function(wrapped, task.target.short_name()): return wrapped

        return None

    def _defines_target_function(self, code: str, function_name: str):
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        return any((isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name for node in tree.body))

    def _compact_code_for_prompt(self, code: str, max_chars: int = 12000):
        norm = code.strip()
        if len(norm) <= max_chars: return norm
        # remove blank
        head = max_chars * 2 // 3
        tail = max_chars - head
        return (
            norm[:head].rstrip()
            + "\n\n# ... combined test suite truncated for prompt budget ...\n\n"
            + norm[-tail:].lstrip()
        )
