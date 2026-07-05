import re
import json
from .llm import LLMClient, load_json_payload
from .prompts import CONTEXT_RETRIEVAL_SYSTEM, PROPERTY_PLANNING_SYSTEM, PROPERTY_REPLAN_SYSTEM, STRENGTHEN_PROPERTY_SYSTEM
from .models import CounterExecutionResult, CounterImplementation, FunctionInfo, GuardConstraint, PropertyCandidate, SemanticEdge

# TODO: 删掉所有comments和print

class Planner:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def select_context_functions(self, target: FunctionInfo, constraints: list[GuardConstraint], candidate_functions: list[FunctionInfo], semantic_edges: list[SemanticEdge], max_functions: int = 4):
        if not candidate_functions:
            return []
        prompt = {
            "target": target.to_prompt_dict(compact=True),
            "constraints": [c.to_prompt_dict() for c in constraints],
            "candidate_functions": [item.to_prompt_dict(compact=True) for item in candidate_functions],
            "semantic_edges": [edge.to_prompt_dict() for edge in semantic_edges],
            "max_functions": max_functions,
        }
        raw = self.llm.chat(CONTEXT_RETRIEVAL_SYSTEM, json.dumps(prompt, ensure_ascii=False, indent=2), require_json=True)
        data = load_json_payload(raw)



        return self._parse_selected_functions(data, candidate_functions, max_functions)

    async def aselect_context_functions(self, target: FunctionInfo, constraints: list[GuardConstraint], candidate_functions: list[FunctionInfo], semantic_edges: list[SemanticEdge], max_functions: int = 4):
        if not candidate_functions:
            return []
        prompt = {
            "target": target.to_prompt_dict(compact=True),
            "constraints": [c.to_prompt_dict() for c in constraints],
            "candidate_functions": [item.to_prompt_dict(compact=True) for item in candidate_functions],
            "semantic_edges": [edge.to_prompt_dict() for edge in semantic_edges],
            "max_functions": max_functions,
        }
        raw = await self.llm.achat(CONTEXT_RETRIEVAL_SYSTEM, json.dumps(prompt, ensure_ascii=False, indent=2), require_json=True)
        data = load_json_payload(raw)
        return self._parse_selected_functions(data, candidate_functions, max_functions)

    def plan_properties(self, target: FunctionInfo, constraints: list[GuardConstraint], context_functions: list[FunctionInfo], semantic_edges: list[SemanticEdge], existing_properties: list[str], max_properties: int = 3, planning_mode: str = "target_only"):
        prompt = {
            "planning_mode": planning_mode,
            "target": target.to_prompt_dict(compact=True),
            "constraints": [c.to_prompt_dict() for c in constraints],
            "context_functions": [f.to_prompt_dict(compact=True) for f in context_functions],
            "semantic_edges": [e.to_prompt_dict() for e in semantic_edges],
            "existing_properties": existing_properties,
            "max_properties": max_properties,
        }
        heur_cands = self._merge_candidates(self._heuristic_candidates(target, context_functions))
        if self._should_use_heuristic_fast_path(target, heur_cands, existing_properties, max_properties):
            return heur_cands[:max_properties]
        raw = self.llm.chat(PROPERTY_PLANNING_SYSTEM, json.dumps(prompt, ensure_ascii=False, indent=2), require_json=True)
        
        data = load_json_payload(raw)


        llm_cands = self._parse_candidates(data)
        return self._merge_candidates(heur_cands + llm_cands)[:max_properties]

    async def aplan_properties(self, target: FunctionInfo, constraints: list[GuardConstraint], context_functions: list[FunctionInfo], semantic_edges: list[SemanticEdge], existing_properties: list[str], max_properties: int = 3, planning_mode: str = "target_only"):
        prompt = {
            "planning_mode": planning_mode,
            "target": target.to_prompt_dict(compact=True),
            "constraints": [c.to_prompt_dict() for c in constraints],
            "context_functions": [f.to_prompt_dict(compact=True) for f in context_functions],
            "semantic_edges": [e.to_prompt_dict() for e in semantic_edges],
            "existing_properties": existing_properties,
            "max_properties": max_properties,
        }
        heur_cands = self._merge_candidates(self._heuristic_candidates(target, context_functions))
        if self._should_use_heuristic_fast_path(target, heur_cands, existing_properties, max_properties):
            return heur_cands[:max_properties]
        raw = await self.llm.achat(PROPERTY_PLANNING_SYSTEM, json.dumps(prompt, ensure_ascii=False, indent=2), require_json=True)
        data = load_json_payload(raw)
        llm_cands = self._parse_candidates(data)
        return self._merge_candidates(heur_cands + llm_cands)[:max_properties]

    def replan_property(self, target: FunctionInfo, constraints: list[GuardConstraint], context_functions: list[FunctionInfo], semantic_edges: list[SemanticEdge], existing_properties: list[str], original_property: str, error_reason: str):
        prompt = {
            "target": target.to_prompt_dict(compact=True),
            "constraints": [c.to_prompt_dict() for c in constraints],
            "context_functions": [f.to_prompt_dict(compact=True) for f in context_functions],
            "semantic_edges": [e.to_prompt_dict() for e in semantic_edges],
            "existing_properties": existing_properties,
            "original_property": original_property,
            "error_reason": error_reason,
        }
        raw = self.llm.chat(PROPERTY_REPLAN_SYSTEM, json.dumps(prompt, ensure_ascii=False, indent=2), require_json=True)
        data = load_json_payload(raw)
        return self._parse_candidate(data)

    async def areplan_property(self, target: FunctionInfo, constraints: list[GuardConstraint], context_functions: list[FunctionInfo], semantic_edges: list[SemanticEdge], existing_properties: list[str], original_property: str, error_reason: str):
        prompt = {
            "target": target.to_prompt_dict(compact=True),
            "constraints": [c.to_prompt_dict() for c in constraints],
            "context_functions": [f.to_prompt_dict(compact=True) for f in context_functions],
            "semantic_edges": [e.to_prompt_dict() for e in semantic_edges],
            "existing_properties": existing_properties,
            "original_property": original_property,
            "error_reason": error_reason,
        }
        raw = await self.llm.achat(PROPERTY_REPLAN_SYSTEM, json.dumps(prompt, ensure_ascii=False, indent=2), require_json=True)
        data = load_json_payload(raw)
        return self._parse_candidate(data)

    def strengthen_property(self, target: FunctionInfo, constraints: list[GuardConstraint], context_functions: list[FunctionInfo], semantic_edges: list[SemanticEdge], existing_properties: list[str], counter_implementation: CounterImplementation, counter_execution: CounterExecutionResult):
        prompt = {
            "target": target.to_prompt_dict(compact=True),
            "constraints": [c.to_prompt_dict() for c in constraints],
            "context_functions": [f.to_prompt_dict(compact=True) for f in context_functions],
            "semantic_edges": [e.to_prompt_dict() for e in semantic_edges],
            "existing_properties": existing_properties,
            "counter_implementation": counter_implementation.to_prompt_dict(),
            "counter_execution": counter_execution.to_prompt_dict(),
        }
        raw = self.llm.chat(STRENGTHEN_PROPERTY_SYSTEM, json.dumps(prompt, ensure_ascii=False, indent=2), require_json=True)
        data = load_json_payload(raw)
        return self._parse_candidate(data, default_evidence_type="code", default_confidence="high")

    async def astrengthen_property(self, target: FunctionInfo, constraints: list[GuardConstraint], context_functions: list[FunctionInfo], semantic_edges: list[SemanticEdge], existing_properties: list[str], counter_implementation: CounterImplementation, counter_execution: CounterExecutionResult):
        prompt = {
            "target": target.to_prompt_dict(compact=True),
            "constraints": [c.to_prompt_dict() for c in constraints],
            "context_functions": [f.to_prompt_dict(compact=True) for f in context_functions],
            "semantic_edges": [e.to_prompt_dict() for e in semantic_edges],
            "existing_properties": existing_properties,
            "counter_implementation": counter_implementation.to_prompt_dict(),
            "counter_execution": counter_execution.to_prompt_dict(),
        }
        raw = await self.llm.achat(STRENGTHEN_PROPERTY_SYSTEM, json.dumps(prompt, ensure_ascii=False, indent=2), require_json=True)
        data = load_json_payload(raw)
        return self._parse_candidate(data, default_evidence_type="code", default_confidence="high")

    def _parse_candidates(self, data: dict | list):
        results: list[PropertyCandidate] = []
        if isinstance(data, list):
            raw_items = data
        elif isinstance(data, dict):
            raw_items = data.get("properties", [])
        else:
            raw_items = []
        for item in raw_items:
            cand = self._parse_candidate(item)
            if cand is not None:
                results.append(cand)
        return self._merge_candidates(results)

    def _merge_candidates(self, candidates: list[PropertyCandidate]):
        ordered = sorted(candidates, key=self._candidate_priority, reverse=True)
        unique: list[PropertyCandidate] = []
        seen_text: set[str] = set()
        seen_modes: set[str] = set()
        for cand in ordered:
            text_key = cand.property_text.strip().lower()
            if text_key in seen_text:
                continue
            mode_key = self._candidate_mode_key(cand)
            if mode_key and mode_key in seen_modes:
                continue
            unique.append(cand)
            seen_text.add(text_key)
            if mode_key:
                seen_modes.add(mode_key)
        return unique

    def _heuristic_candidates(self, target: FunctionInfo, context_functions: list[FunctionInfo]):
        _ = target, context_functions
        return []

    def _should_use_heuristic_fast_path(self, target: FunctionInfo, heuristic_candidates: list[PropertyCandidate], existing_properties: list[str], max_properties: int):
        _ = target, heuristic_candidates, existing_properties, max_properties
        return False

    def _parse_candidate(self, data: dict | list, default_evidence_type: str = "docstring", default_confidence: str = "medium"):
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                cand = self._parse_candidate(item, default_evidence_type, default_confidence)
                if cand is not None:
                    return cand
            return None
        if not isinstance(data, dict):
            return None
        if data.get("status") in {"NO_VALID_PROPERTY", "NO_STRONGER_PROPERTY"}:
            return None
        if not data.get("property"):
            return None
        return PropertyCandidate(property_text=data['property'], evidence=data.get('evidence', ''), evidence_type=data.get('evidence_type', default_evidence_type), confidence=data.get('confidence', default_confidence), mode_label=data.get('mode', ''), oracle_hint=data.get('oracle_hint', ''), relevant_functions=list(data.get('relevant_functions', [])))

    def _parse_selected_functions(self, data: dict | list, candidate_functions: list[FunctionInfo], max_functions: int):
        known = {item.full_name for item in candidate_functions}
        selected: list[str] = []
        if isinstance(data, list):
            raw_names = data
        else:
            raw_names = data.get("selected_functions", [])
        for name in raw_names:
            if name in known and name not in selected:
                selected.append(name)
        if selected:
            return selected[:max_functions]
        return [item.full_name for item in candidate_functions[:max_functions]]

    def _candidate_priority(self, candidate: PropertyCandidate):
        text = candidate.property_text.lower()
        mode = candidate.mode_label.lower()
        oracle_hint = candidate.oracle_hint.lower()
        score = 0.0

        if candidate.evidence_type in {"cross_function_contract", "mathematical_convention"}:
            score += 2.5
        elif candidate.evidence_type == "docstring":
            score += 0.9
        if candidate.confidence == "high":
            score += 1.5
        elif candidate.confidence == "medium":
            score += 0.5
        if candidate.oracle_hint:
            score += 0.8
        if candidate.mode_label:
            score += 0.4

        strong = (
            "roundtrip",
            "round-trip",
            "inverse",
            "equivalent",
            "same result",
            "same sequence",
            "reconstruct",
            "reconstruction",
            "concatenation",
            "preserve",
            "preserves",
            "decompose",
            "denominator",
            "numerator",
            "slice",
            "remaining",
            "advance",
            "partition",
            "order",
            "oracle",
            "reference",
            "prefix",
            "non-empty",
            "exact",
            "trace",
            "simple pole",
            "coefficient",
        )
        weak = (
            "returns tuple",
            "single list",
            "is exhausted",
            "raises stopiteration",
            "does not raise",
            "does not crash",
            "is not none",
            "type",
            "shape",
        )
        cases = (
            r"\bmaxsplit\s*=\s*0\b",
            r"\bn\s*(?:=|is)\s*none\b",
            r"\bkeep_separator\s*=\s*true\b",
            r"\ball inputs are scalars\b",
            r"\bwindow_size\s*(?:=|is)\s*none\b",
            r"\bstrict\s*=\s*false\b",
            r"\bif\s+[^.]{0,50}\b0\b",
            r"\bwhen\s+[^.]{0,50}\b0\b",
            r"\bif\s+[^.]{0,50}\bnone\b",
            r"\bwhen\s+[^.]{0,50}\bnone\b",
        )

        # 这里用LLM-as-judge会不会更好？
        score += sum(1.2 for hint in strong if hint in text)
        score -= sum(1.8 for hint in weak if hint in text)
        score -= sum(2.5 for pattern in cases if re.search(pattern, text))
        score += sum(0.8 for hint in strong if hint in oracle_hint)
        score -= sum(1.5 for pattern in cases if re.search(pattern, mode))

        if any(token in mode for token in ("scripted", "mixed denominator", "simple pole", "fresh iterator", "literal corpus", "component invariants")):
            score += 1.1

        if text.startswith("if ") or text.startswith("when "):
            score -= 0.75

        return (score, len(candidate.relevant_functions))

    def _candidate_mode_key(self, candidate: PropertyCandidate):
        mode = candidate.mode_label.strip().lower()
        if mode:
            return re.sub(r"\s+", " ", mode)
        return ""
