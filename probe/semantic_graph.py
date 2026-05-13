from collections import defaultdict

from .models import FunctionInfo, SemanticEdge
from .repository import RepositoryIndex

# Some adhoc hints
INVERSE_PAIRS = [
    ("encode", "decode"),
    ("dump", "load"),
    ("dumps", "loads"),
    ("serialize", "deserialize"),
    ("pack", "unpack"),
    ("compress", "decompress"),
    ("format", "parse"),
]

STATE_PAIRS = [
    ("add", "remove"),
    ("push", "pop"),
    ("append", "pop"),
    ("insert", "delete"),
    ("open", "close"),
]

IDENTITY_GROUPS = [
    {"variance", "stdev", "stddev", "std"},
    {"covariance", "correlation"},
    {"sin", "asin"},
    {"cos", "acos"},
    {"tan", "atan"},
]

# 这里应该是adhoc pattern + LLM search (this is useful)
# But did not do ablation on the adhoc pattern, not sure if is this works or not, but at least no side-effects I think.
class SemanticGraphBuilder:
    def __init__(self, repository_index: RepositoryIndex):
        self.repository_index = repository_index

    def build_for(self, target: FunctionInfo, max_nodes: int = 8):
        cands, edge_map = self.collect_candidates(target, max_candidates=max_nodes)
        edges: list[SemanticEdge] = []
        for cand in cands:
            edges.extend(edge_map.get(cand.full_name, []))
        return cands, edges

    def collect_candidates(self, target: FunctionInfo, max_candidates: int = 12):
        cand_names = self._collect_structural_neighborhood(target)
        cands = [
            self.repository_index.functions_by_full_name[name]
            for name in cand_names
            if name in self.repository_index.functions_by_full_name and name != target.full_name
        ]
        scored: list[tuple[int, FunctionInfo, list[SemanticEdge]]] = []
        for cand in cands:
            edges = self._infer_edges(target, cand)
            if not edges:
                continue
            score = max(edge.weight for edge in edges)
            scored.append((score, cand, edges))
        scored.sort(key=lambda item: (-item[0], item[1].full_name))

        ctx_funcs: list[FunctionInfo] = []
        edge_map: dict[str, list[SemanticEdge]] = {}
        for _, cand, cand_edges in scored[:max_candidates]:
            ctx_funcs.append(cand)
            edge_map[cand.full_name] = cand_edges
        return ctx_funcs, edge_map

    def _collect_structural_neighborhood(self, target: FunctionInfo):
        names: set[str] = set()
        module = target.module_location
        names.update((info.full_name for info in self.repository_index.functions_by_full_name.values() if info.module_location == module))

        for dep in self.repository_index.module_imports.get(module, set()):
            names.update((info.full_name for info in self.repository_index.functions_by_full_name.values() if info.module_location == dep))
        for dep in self.repository_index.reverse_module_imports.get(module, set()):
            names.update((info.full_name for info in self.repository_index.functions_by_full_name.values() if info.module_location == dep))

        if target.class_name:
            class_key = f"{module}.{target.class_name}"
            names.update(self.repository_index.class_methods.get(class_key, []))
            for base in self.repository_index.class_bases.get(class_key, ()):
                names.update(self._methods_for_base_name(base))
            for child in self.repository_index.class_children.get(class_key, set()):
                names.update(self.repository_index.class_methods.get(child, []))
        return names

    def _methods_for_base_name(self, base_name: str):
        matches: set[str] = set()
        suffix = f".{base_name}"
        for class_key, methods in self.repository_index.class_methods.items():
            if class_key.endswith(suffix) or class_key == base_name:
                matches.update(methods)
        return matches

    def _infer_edges(self, target: FunctionInfo, candidate: FunctionInfo):
        edges: list[SemanticEdge] = []
        if candidate.module_location == target.module_location:
            edges.append(SemanticEdge(source=target.full_name, target=candidate.full_name, edge_type='same_module', evidence='Located in the same module neighborhood.', weight=20))
        if candidate.short_name() in target.calls:
            edges.append(SemanticEdge(source=target.full_name, target=candidate.full_name, edge_type='call_dependency', evidence=f'{target.short_name()} directly calls {candidate.short_name()}.', weight=35))
        if target.short_name() in candidate.calls:
            edges.append(SemanticEdge(source=candidate.full_name, target=target.full_name, edge_type='reverse_call_dependency', evidence=f'{candidate.short_name()} directly calls {target.short_name()}.', weight=30))
        if target.class_name and candidate.class_name == target.class_name and candidate.module_location == target.module_location:
            edges.append(SemanticEdge(source=target.full_name, target=candidate.full_name, edge_type='same_class', evidence='Method appears in the same class hierarchy scope.', weight=40))

        relation = _name_relation(target.short_name().lower(), candidate.short_name().lower())
        if relation:
            edge_type, evidence, weight = relation
            edges.append(SemanticEdge(source=target.full_name, target=candidate.full_name, edge_type=edge_type, evidence=evidence, weight=weight))
        return edges


def _name_relation(target: str, candidate: str):
    for left, right in INVERSE_PAIRS:
        if _contains_token(target, left) and _contains_token(candidate, right):
            return ("inverse_operation", f"Name pair suggests inverse relation: {left}/{right}.", 90)
        if _contains_token(target, right) and _contains_token(candidate, left):
            return ("inverse_operation", f"Name pair suggests inverse relation: {right}/{left}.", 90)
    for left, right in STATE_PAIRS:
        if _contains_token(target, left) and _contains_token(candidate, right):
            return ("state_invariant", f"Verb pair suggests state-preserving operation pair: {left}/{right}.", 75)
        if _contains_token(target, right) and _contains_token(candidate, left):
            return ("state_invariant", f"Verb pair suggests state-preserving operation pair: {right}/{left}.", 75)
    for group in IDENTITY_GROUPS:
        if any(_contains_token(target, token) for token in group) and any(_contains_token(candidate, token) for token in group):
            label = ", ".join(sorted(group))
            return ("relational_identity", f"Names fall in the same relational identity family: {label}.", 70)
    tgt_parts = [part for part in target.replace("__", "_").split("_") if part]
    cand_parts = [part for part in candidate.replace("__", "_").split("_") if part]
    if (
        len(tgt_parts) >= 2
        and len(cand_parts) >= 2
        and tgt_parts[0] == cand_parts[0]
        and tgt_parts != cand_parts
    ):
        return (
            "same_prefix_family",
            f"Names share the operation-family prefix '{tgt_parts[0]}' but differ in mode suffixes.",
            55,
        )
    return None


def _contains_token(name: str, token: str):
    parts = [part for part in name.replace("__", "_").split("_") if part]
    return token in parts or name == token or name.endswith(token)
