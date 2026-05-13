from typing import Any
from dataclasses import dataclass, field


def _truncate(text: str, max_chars: int):
    norm = text.strip()
    if len(norm) <= max_chars:
        return norm
    return norm[: max_chars - 3].rstrip() + "..."


def _extract_docstring_examples(docstring: str, max_examples: int = 6):
    examples: list[str] = []
    current: list[str] = []
    collect = False

    for raw_line in docstring.splitlines():
        stripped = raw_line.rstrip()
        norm = stripped.lstrip()
        if norm.startswith(">>>"):
            if current:
                examples.append("\n".join(current).strip())
                if len(examples) >= max_examples:
                    return examples
                current = []
            current.append(norm)
            collect = True
            continue
        if collect and (norm.startswith("...") or norm):
            current.append(norm)
            continue
        if current:
            examples.append("\n".join(current).strip())
            if len(examples) >= max_examples:
                return examples
            current = []
        collect = False

    if current and len(examples) < max_examples:
        examples.append("\n".join(current).strip())
    return examples[:max_examples]


@dataclass(slots=True)
class FunctionInfo:
    name: str
    full_name: str
    module_location: str
    signature: str
    docstring: str
    code: str
    function_type: str
    file_path: str
    package: str
    class_name: str | None = None
    class_docstring: str = ""
    base_classes: tuple[str, ...] = ()
    local_imports: tuple[str, ...] = ()
    calls: tuple[str, ...] = ()

    def short_name(self):
        return self.name.split(".")[-1]

    def docstring_examples(self, max_examples: int = 6):
        return _extract_docstring_examples(self.docstring, max_examples=max_examples)

    def to_prompt_dict(self, compact: bool = False):
        docstring = self.docstring
        cls_doc = self.class_docstring
        code = self.code
        examples = self.docstring_examples()
        cls_examples = _extract_docstring_examples(cls_doc)
        if compact:
            docstring = _truncate(docstring, 600)
            cls_doc = _truncate(cls_doc, 1000)
            code = _truncate(code, 1200)
            examples = examples[:6]
            cls_examples = cls_examples[:6]
        return {
            "name": self.name,
            "full_name": self.full_name,
            "module_location": self.module_location,
            "signature": self.signature,
            "docstring": docstring,
            "docstring_examples": examples,
            "class_docstring": cls_doc,
            "class_docstring_examples": cls_examples,
            "code": code,
            "function_type": self.function_type,
            "class_name": self.class_name,
            "base_classes": list(self.base_classes),
            "local_imports": list(self.local_imports),
            "calls": list(self.calls),
            "file_path": self.file_path,
        }


@dataclass(slots=True)
class GuardConstraint:
    guard_expression: str
    valid_constraint: str
    rationale: str

    def to_prompt_dict(self):
        return {
            "guard_expression": self.guard_expression,
            "valid_constraint": self.valid_constraint,
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class SemanticEdge:
    source: str
    target: str
    edge_type: str
    evidence: str
    weight: int

    def to_prompt_dict(self):
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type,
            "evidence": self.evidence,
            "weight": self.weight,
        }


@dataclass(slots=True)
class PropertyCandidate:
    property_text: str
    evidence: str
    evidence_type: str
    confidence: str
    mode_label: str = ""
    oracle_hint: str = ""
    relevant_functions: list[str] = field(default_factory=list)

    def to_prompt_dict(self):
        return {
            "property": self.property_text,
            "evidence": self.evidence,
            "evidence_type": self.evidence_type,
            "confidence": self.confidence,
            "mode": self.mode_label,
            "oracle_hint": self.oracle_hint,
            "relevant_functions": list(self.relevant_functions),
        }


@dataclass(slots=True)
class TestingTask:
    target: FunctionInfo
    constraints: list[GuardConstraint]
    property_candidate: PropertyCandidate
    context_functions: list[FunctionInfo]
    semantic_edges: list[SemanticEdge]

    def to_prompt_dict(self):
        return {
            "target": self.target.to_prompt_dict(compact=True),
            "constraints": [c.to_prompt_dict() for c in self.constraints],
            "property": self.property_candidate.to_prompt_dict(),
            "context_functions": [f.to_prompt_dict(compact=True) for f in self.context_functions],
            "semantic_edges": [e.to_prompt_dict() for e in self.semantic_edges],
        }


@dataclass(slots=True)
class CounterImplementation:
    description: str
    code: str
    what_it_violates: str

    def to_prompt_dict(self):
        return {
            "description": self.description,
            "code": self.code,
            "what_it_violates": self.what_it_violates,
        }


@dataclass(slots=True)
class CounterExecutionResult:
    survived: bool
    execution_status: str
    return_code: int | None = None
    output: str = ""
    error_message: str = ""

    def to_prompt_dict(self):
        observed = self.error_message or self.output
        return {
            "survived": self.survived,
            "execution_status": self.execution_status,
            "return_code": self.return_code,
            "observed_output": _truncate(observed, 800) if observed else "",
        }

    def to_dict(self):
        return {
            "survived": self.survived,
            "execution_status": self.execution_status,
            "return_code": self.return_code,
            "output": self.output,
            "error_message": self.error_message,
        }


@dataclass(slots=True)
class ValidationDecision:
    error_type: str
    reasoning: str
    fix_suggestion: str = ""


@dataclass(slots=True)
class GenerationResult:
    status: str
    code: str | None
    error_message: str | None = None
    reasoning: str | None = None
    fix_suggestion: str | None = None


@dataclass(slots=True)
class PropertyAttempt:
    property_text: str
    status: str
    evidence: str
    evidence_type: str
    confidence: str
    mode_label: str = ""
    oracle_hint: str = ""
    code: str | None = None
    error_message: str | None = None
    reasoning: str | None = None

    def to_dict(self):
        return {
            "property": self.property_text,
            "status": self.status,
            "evidence": self.evidence,
            "evidence_type": self.evidence_type,
            "confidence": self.confidence,
            "mode": self.mode_label,
            "oracle_hint": self.oracle_hint,
            "code": self.code,
            "error_message": self.error_message,
            "reasoning": self.reasoning,
        }


@dataclass(slots=True)
class ProbeRunResult:
    status: str
    target_function: str
    final_pbt_code: str | None
    properties: list[PropertyAttempt] = field(default_factory=list)
    strengthening_rounds: int = 0
    bug_report: dict[str, Any] | None = None
    adversarial_checks: list[dict[str, Any]] = field(default_factory=list)
