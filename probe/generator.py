import os
import re
import ast
import json
import builtins
import tempfile
import subprocess
from typing import Any

from .llm import LLMClient
from .prompts import PBT_FIX_SYSTEM, PBT_GENERATION_SYSTEM
from .models import FunctionInfo, GenerationResult, TestingTask


class Generator:
    def __init__(self, llm: LLMClient, python_executable: str, repo_root: str | None = None, max_fix_retries: int = 3, enforce_strength_checks: bool = True, pytest_timeout_seconds: int = 120):
        self.llm = llm
        self.python_executable = python_executable
        self.repo_root = repo_root
        self.max_fix_retries = max_fix_retries
        self.enforce_strength_checks = enforce_strength_checks
        self.pytest_timeout_seconds = pytest_timeout_seconds
        self.current_code: str | None = None

    def generate_initial_code(self, task: TestingTask):
        self.current_code = self._generate_initial(task)
        return self.current_code

    async def agenerate_initial_code(self, task: TestingTask):
        self.current_code = await self._agenerate_initial(task)
        return self.current_code

    def repair_code(self, task: TestingTask, failing_code: str, error_message: str, validator_feedback: dict[str, str] | None = None, combined_pbt_code: str | None = None):
        payload = {
            "task": task.to_prompt_dict(),
            "failing_code": failing_code,
            "error_message": error_message,
            "validator_feedback": validator_feedback or {},
        }
        if combined_pbt_code is not None:
            payload["combined_pbt_code"] = combined_pbt_code
        raw = self.llm.chat(PBT_FIX_SYSTEM, json.dumps(payload, ensure_ascii=False, indent=2))
        return self._extract_code(raw)

    async def arepair_code(self, task: TestingTask, failing_code: str, error_message: str, validator_feedback: dict[str, str] | None = None, combined_pbt_code: str | None = None):
        payload = {
            "task": task.to_prompt_dict(),
            "failing_code": failing_code,
            "error_message": error_message,
            "validator_feedback": validator_feedback or {},
        }
        if combined_pbt_code is not None:
            payload["combined_pbt_code"] = combined_pbt_code
        raw = await self.llm.achat(PBT_FIX_SYSTEM, json.dumps(payload, ensure_ascii=False, indent=2))
        return self._extract_code(raw)

    def _generate_initial(self, task: TestingTask):
        raw = self.llm.chat(PBT_GENERATION_SYSTEM, json.dumps(task.to_prompt_dict(), ensure_ascii=False, indent=2))
        return self._extract_code(raw)

    # 断点测试
    async def _agenerate_initial(self, task: TestingTask):
        raw = await self.llm.achat(PBT_GENERATION_SYSTEM, json.dumps(task.to_prompt_dict(), ensure_ascii=False, indent=2))
        return self._extract_code(raw)

    def _extract_code(self, raw: str):
        match = re.search(r"```python\n(.*?)```", raw, re.DOTALL)
        if not match:
            return None
        code = match.group(1).strip()
        return self._ensure_settings(code)

    def _ensure_settings(self, code: str):
        code = self._ensure_hypothesis_import(code, "settings")
        if "@example(" in code:
            code = self._ensure_hypothesis_import(code, "example")
        try:
            return self._ensure_settings_decorators(code)
        except SyntaxError:
            code = re.sub(r"^([ \t]*)@settings\(.*\)\s*$", r"\1@settings(max_examples=1000, deadline=None)", code, flags=re.MULTILINE)
            if "@settings" not in code:
                code = re.sub(r"^([ \t]*)@given\(", r"\1@settings(max_examples=1000, deadline=None)\n\1@given(", code, flags=re.MULTILINE)
            return code

    def _ensure_settings_decorators(self, code: str):
        tree = ast.parse(code)
        lines = code.splitlines()
        repls: list[tuple[int, int, list[str]]] = []

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            has_given = False
            has_cfg = False
            given_line: int | None = None
            g_indent = ""
            for decorator in node.decorator_list:
                name = self._decorator_name(decorator)
                if name == "given":
                    has_given = True
                    if given_line is None:
                        given_line = decorator.lineno
                        g_indent = lines[decorator.lineno - 1][: decorator.col_offset]
                elif name == "settings":
                    has_cfg = True
                    start = decorator.lineno - 1
                    end = decorator.end_lineno or decorator.lineno
                    indent = lines[start][: decorator.col_offset]
                    repls.append((start, end, [f"{indent}@settings(max_examples=1000, deadline=None)"]))

            if has_given and not has_cfg and given_line is not None:
                insert_at = given_line - 1
                repls.append((insert_at, insert_at, [f"{g_indent}@settings(max_examples=1000, deadline=None)"]))

        if not repls:
            return code
        for start, end, repl in sorted(repls, key=lambda item: item[0], reverse=True):
            lines[start:end] = repl + lines[start:end] if start == end else repl
        newline = "\n" if code.endswith("\n") else ""
        return "\n".join(lines) + newline

    def _decorator_name(self, decorator: ast.AST):
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Name):
            return target.id
        if isinstance(target, ast.Attribute):
            return target.attr
        return ""

    def _ensure_hypothesis_import(self, code: str, name: str):
        if "from hypothesis import" in code:
            match = re.search(r"from hypothesis import ([^\n]+)", code)
            if match:
                items = [item.strip() for item in match.group(1).split(",")]
                if name not in items:
                    items.append(name)
                    repl = f"from hypothesis import {', '.join(sorted(set(items)))}"
                    code = code[:match.start()] + repl + code[match.end():]
            return code
        return f"from hypothesis import {name}\n" + code

    def execute_code(self, code: str, timeout: int | None = None):
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
                handle.write(code)
                temp_file = handle.name
            env = os.environ.copy()
            if self.repo_root:
                env["PYTHONPATH"] = f"{self.repo_root}:{env.get('PYTHONPATH', '')}".rstrip(":")
            timeout_s = timeout or self.pytest_timeout_seconds
            result = subprocess.run([self.python_executable, '-m', 'pytest', temp_file, '-q', '--tb=short', '-p', 'no:cacheprovider'], capture_output=True, text=True, timeout=timeout_s, env=env)
            output = (result.stdout or "") + "\n" + (result.stderr or "")
            if result.returncode == 0 and self._all_tests_skipped(output):
                return {"success": False, "error_message": "Vacuous success: all generated tests were skipped."}
            if result.returncode == 0:
                return {"success": True, "error_message": None}
            return {"success": False, "error_message": output}
        except subprocess.TimeoutExpired:
            return {"success": False, "error_message": "Timeout"}
        finally:
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)

    def detect_strength_issues(self, task: TestingTask, code: str): # 直接匹配这里可能会有问题
        issues: list[str] = []
        norm = code.lower()
        prop_text = task.property_candidate.property_text.lower()
        mode_label = task.property_candidate.mode_label.lower()
        hint = task.property_candidate.oracle_hint.lower()
        docs = task.target.docstring_examples()


        if "pytest.skip(" in norm:
            issues.append("The test uses pytest.skip, which can make the property vacuous.")
        if "st.functions(" in norm:
            issues.append("The test uses st.functions for arbitrary callables instead of a concrete deterministic oracle/predicate.")
        if self._has_broad_exception_swallowing(code):
            issues.append("The test contains broad try/except logic that may swallow behavioral failures.")
        if norm.count("assert ") <= 1 and any((token in prop_text for token in ('equivalent', 'reconstruct', 'same', 'preserve', 'concatenation', 'remaining', 'count', 'order'))):
            issues.append("The assertion structure is too shallow for a relational property.")
        if (
            (re.search(r"\bmaxsplit\s*=\s*0\b", norm) and "maxsplit=0" not in prop_text)
            or (
                re.search(r"\bn\s*=\s*none\b", norm)
                and "n is none" not in prop_text
                and "n=none" not in prop_text
            )
        ):
            issues.append("The test hard-codes a narrow special-case parameter value instead of exercising the general property.")
        if norm.count("assume(") >= 3:
            issues.append("The test relies on too many assume() calls and may be vacuous.")
        if self._has_gating_control_flow(code):
            issues.append("The test uses gating control flow (skip/return/one-sided conditionals) instead of an exhaustive oracle or strategy-encoded antecedent.")
        if docs and "@example(" not in code:
            issues.append("The test does not anchor any documented docstring examples with @example.")
        if any(token in prop_text or token in mode_label for token in ("n is none", "maxsplit=0", "all inputs are scalars", "window_size is none", "strict=false", "slice access", "exact length")):
            if len(docs) >= 2 and not any(token in prop_text for token in ("reconstruct", "reference", "equivalent", "remaining", "raises", "slice")):
                issues.append("The property focuses on a single documented mode without a strong oracle even though the API exposes multiple documented modes.")
        if "raises" in prop_text and "pytest.raises" not in code:
            issues.append("The property claims exception behavior but the test does not use pytest.raises.")
        if "reference" in hint and not self._has_reference_shape(code):
            issues.append("The oracle hint suggests a reference model, but the current test only performs partial assertions.")
        if self._is_exception_only_mutator_suite(task, code):
            issues.append('For remove/discard/pop-like mutators, an exception-only absent-input check is too weak; add a successful state-change invariant such as count/length delta or sibling-mutator/reference equivalence.')

        return issues

    def detect_preflight_issues(self, code: str):
        issues: list[str] = []
        norm = code.lower()
        if ".example()" in norm:
            issues.append("Do not call Strategy.example() inside the test body; derive concrete values from generated inputs or fixed helper predicates.")
        if "pytest.mark.example(" in norm:
            issues.append("Use Hypothesis @example decorators, not pytest.mark.example.")
        if "@pytest.mark.example" in norm:
            issues.append("Use Hypothesis @example decorators or plain example tests, not pytest.mark.example markers.")
        if "st.iterables(" in norm:
            issues.append("Avoid st.iterables for properties that may iterate inputs more than once; prefer reusable concrete containers such as lists or tuples.")
        if "s.order(" in norm:
            issues.append("Use Order(...) rather than S.Order(...) when constructing SymPy order terms.")
        if "@example(" in norm and "@given(" not in norm:
            issues.append("@example decorators require a matching @given-decorated Hypothesis test; otherwise pytest will treat the function arguments as fixtures.")
        issues.extend(self._detect_invalid_hypothesis_api_usage(code))
        issues.extend(self._detect_undefined_example_names(code))
        return issues

    def _detect_invalid_hypothesis_api_usage(self, code: str):
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
        issues: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func_name = self._safe_unparse(node.func)
            kw_names = {kw.arg for kw in node.keywords if kw.arg}
            if func_name == "st.tuples" and kw_names.intersection({"min_size", "max_size"}):
                issues.append("st.tuples has fixed arity; do not pass min_size/max_size to it. Use st.lists/st.tuples with explicit element strategies instead.")
            if func_name == "pytest.raises" and "match" in kw_names:
                issues.append("Avoid brittle pytest.raises(..., match=...) checks on exact exception messages unless the property explicitly requires the message text.")
        return issues

    def _detect_undefined_example_names(self, code: str):
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
        scope_names = set(dir(builtins))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                scope_names.add(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    scope_names.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    scope_names.add(alias.asname or alias.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    scope_names.update(self._collect_assigned_names(target))
            elif isinstance(node, ast.AnnAssign):
                scope_names.update(self._collect_assigned_names(node.target))

        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call) or self._safe_unparse(decorator.func) != "example":
                    continue
                free = {
                    child.id
                    for child in ast.walk(decorator)
                    if isinstance(child, ast.Name) and child.id not in scope_names
                }
                if free:
                    missing = ", ".join(sorted(free))
                    return [f"@example arguments use undefined module-scope names: {missing}. Use literals or imported/defined constants only."]
        return []

    def _collect_assigned_names(self, target: ast.AST):
        names: set[str] = set()
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                names.update(self._collect_assigned_names(element))
        return names

    def _safe_unparse(self, node: ast.AST):
        try:
            return ast.unparse(node)
        except Exception:
            return ""

    def _has_gating_control_flow(self, code: str):
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.name.startswith("test_"):
                continue
            unguarded = self._count_unguarded_assertive_behaviors(node)
            for child in ast.walk(node):
                if not isinstance(child, ast.If):
                    continue
                if self._contains_gating_exit(child.body) or self._contains_gating_exit(child.orelse):
                    return True
                body_ok = self._contains_assertive_behavior(child.body)
                else_ok = self._contains_assertive_behavior(child.orelse)
                if (body_ok or else_ok) and unguarded == 0:
                    return True
        return False

    def _count_unguarded_assertive_behaviors(self, func_node: ast.AST):
        parents: dict[int, ast.AST] = {}
        for parent in ast.walk(func_node):
            for child in ast.iter_child_nodes(parent):
                parents[id(child)] = parent
        cnt = 0
        for node in ast.walk(func_node):
            if not self._is_assertive_node(node):
                continue
            current = parents.get(id(node))
            in_if = False
            while current is not None and current is not func_node:
                if isinstance(current, ast.If):
                    in_if = True
                    break
                current = parents.get(id(current))
            if not in_if:
                cnt += 1
        return cnt

    def _is_assertive_node(self, node: ast.AST):
        if isinstance(node, ast.Assert):
            return True
        if isinstance(node, ast.With):
            for item in node.items:
                if isinstance(item.context_expr, ast.Call) and ast.unparse(item.context_expr.func) == "pytest.raises":
                    return True
        return False

    def _contains_assertive_behavior(self, nodes: list[ast.stmt]):
        for node in nodes:
            for child in ast.walk(node):
                if isinstance(child, ast.Assert):
                    return True
                if isinstance(child, ast.With):
                    for item in child.items:
                        if isinstance(item.context_expr, ast.Call) and ast.unparse(item.context_expr.func) == "pytest.raises":
                            return True
        return False

    def _contains_gating_exit(self, nodes: list[ast.stmt]):
        for node in nodes:
            for child in ast.walk(node):
                if isinstance(child, ast.Return):
                    return True
                if isinstance(child, ast.Call) and ast.unparse(child.func) == "pytest.skip":
                    return True
                if isinstance(child, ast.Call) and ast.unparse(child.func) == "assume":
                    return True
        return False

    def _has_broad_exception_swallowing(self, code: str):
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            for handler in node.handlers:
                if handler.type is None:
                    return True
                if isinstance(handler.type, ast.Name) and handler.type.id in {"Exception", "BaseException"}:
                    return True
                if isinstance(handler.type, ast.Tuple):
                    names = {elt.id for elt in handler.type.elts if isinstance(elt, ast.Name)}
                    if "Exception" in names or "BaseException" in names:
                        return True
        return False

    def _has_reference_shape(self, code: str):
        norm = code.lower()
        ref_sigs = (
            "expected =",
            "reference",
            "sorted(",
            "list(",
            "tuple(",
            "dict(",
            "for i,",
            "enumerate(",
        )
        return any(signal in norm for signal in ref_sigs)

    def _is_exception_only_mutator_suite(self, task: TestingTask, code: str):
        name = task.target.short_name()
        if name not in {"remove", "discard", "pop"}:
            return False
        norm = code.lower()
        prop_text = f"{task.property_candidate.property_text} {task.property_candidate.mode_label}".lower()
        exc_sigs = ("pytest.raises", "must raise", "raises", "valueerror", "error behavior", "not in list")
        if not any(signal in norm or signal in prop_text for signal in exc_sigs):
            return False
        state_sigs = (
            "count(",
            "len(",
            ".discard(",
            ".index(",
            "before =",
            "after =",
            "expected",
            "reference",
            "remaining",
        )
        return not any(signal in norm for signal in state_sigs)

    def _all_tests_skipped(self, output: str):
        norm = output.lower()
        return "skipped" in norm and "passed" not in norm

    # TODO: do code simplfy
    def combine_pbt_codes(self, pbt_codes: list[str], task_or_target: TestingTask | FunctionInfo | None = None):
        if not pbt_codes:
            return ""
        imports: dict[str, set[str]] = {}
        direct: set[str] = set()
        bodies: list[str] = []
        target = task_or_target.target if isinstance(task_or_target, TestingTask) else task_or_target
        local_tgts: set[str] = set()
        target_mod = ""
        if target is not None and target.function_type == "local_function":
            local_tgts = {target.name, target.short_name()}
            target_mod = target.module_location
        idx = 0 # cnt
        for code in pbt_codes:
            lines = code.strip().splitlines()
            body_lines: list[str] = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("from "):
                    match = re.match(r"from\s+([\w.]+)\s+import\s+(.+)", stripped)
                    if match:
                        mod_name = match.group(1)
                        imp_items = [part.strip() for part in match.group(2).split(",")]
                        if local_tgts and mod_name == target_mod:
                            imp_items = [
                                item for item in imp_items
                                if item.split(" as ")[0].strip() not in local_tgts
                            ]
                            if not imp_items: continue
                        imports.setdefault(mod_name, set()).update(imp_items)
                    else: direct.add(stripped)
                elif stripped.startswith("import "): direct.add(stripped)
                else: body_lines.append(line)
            body = "\n".join(body_lines).strip()
            if "def test_" in body:
                idx += 1
                local_idx = 0

                def rename_test(match: re.Match[str]):
                    nonlocal local_idx
                    local_idx += 1
                    suffix = str(idx) if local_idx == 1 else f"{idx}_{local_idx}"
                    return f"def test_property_{suffix}("

                body = re.sub(r"^def test_[A-Za-z0-9_]+\(", rename_test, body, flags=re.MULTILINE)

            if body:
                bodies.append(body)

                
        imp_lines = [f"from {module} import {', '.join(sorted(items))}" for module, items in sorted(imports.items())]
        imp_lines.extend(sorted(direct))
        combined = "\n".join(imp_lines) + "\n\n\n" + "\n\n\n".join(bodies)
        combined = self._prepend_local_target_support(target, combined.strip())
        return self._ensure_settings(combined.strip())

    def _prepend_local_target_support(self, target: FunctionInfo | None, code: str):
        if target is None or target.function_type != "local_function":
            return code
        local_def = textwrap.dedent(target.code).strip()
        if not local_def: return code
        prelude = f"from {target.module_location} import *\n\n{local_def}"

        if not code: return prelude

        return f"{prelude}\n\n\n{code}"
