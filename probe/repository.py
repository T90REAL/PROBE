import ast
import inspect
import textwrap
import importlib.util
from typing import Any
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

from .models import FunctionInfo

# TODO: change this
SKIP_PATTERNS = ("tests", "testing", "conftest", "__pycache__")


def _should_skip(path: Path):
    text = str(path)
    return any(part in text for part in SKIP_PATTERNS)


def _module_name(package_root: Path, file_path: Path):
    rel = file_path.relative_to(package_root.parent)
    parts = list(rel.parts)
    parts[-1] = parts[-1][:-3]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _format_argument(arg: ast.arg, default: ast.expr | None = None):
    text = arg.arg
    if arg.annotation:
        text += f": {ast.unparse(arg.annotation)}"
    if default is not None:
        text += f" = {ast.unparse(default)}"
    return text


def _extract_signature(node: ast.FunctionDef | ast.AsyncFunctionDef):
    parts: list[str] = []
    args = node.args
    pos_args = list(args.posonlyargs) + list(args.args)
    defaults: list[ast.expr | None] = [None] * (len(pos_args) - len(args.defaults)) + list(args.defaults)

    for idx, (arg, default) in enumerate(zip(pos_args, defaults)):
        parts.append(_format_argument(arg, default))
        if args.posonlyargs and idx == len(args.posonlyargs) - 1:
            parts.append("/")

    if args.vararg:
        text = f"*{args.vararg.arg}"
        if args.vararg.annotation:
            text += f": {ast.unparse(args.vararg.annotation)}"
        parts.append(text)
    elif args.kwonlyargs:
        parts.append("*")

    for idx, arg in enumerate(args.kwonlyargs):
        text = arg.arg
        if arg.annotation:
            text += f": {ast.unparse(arg.annotation)}"
        if idx < len(args.kw_defaults) and args.kw_defaults[idx]:
            text += f" = {ast.unparse(args.kw_defaults[idx])}"
        parts.append(text)

    if args.kwarg:
        text = f"**{args.kwarg.arg}"
        if args.kwarg.annotation:
            text += f": {ast.unparse(args.kwarg.annotation)}"
        parts.append(text)

    suffix = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"({', '.join(parts)}){suffix}"


class _CallCollector(ast.NodeVisitor):
    def __init__(self):
        self.calls: set[str] = set()

    def visit_Call(self, node: ast.Call):
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name:
            self.calls.add(name)
        self.generic_visit(node)


class _ImportCollector(ast.NodeVisitor):
    def __init__(self, package_name: str, known_modules: set[str], current_module: str, *, is_package_module: bool):
        self.package_name = package_name
        self.known_modules = known_modules
        self.current_module = current_module
        self.is_package_module = is_package_module
        self.imports: set[str] = set()

    def _record(self, module_name: str | None):
        if not module_name:
            return
        if module_name == self.package_name or module_name.startswith(f"{self.package_name}."):
            if module_name in self.known_modules:
                self.imports.add(module_name)
                return
            for known in self.known_modules:
                if known.startswith(f"{module_name}."):
                    self.imports.add(known)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self._record(alias.name)

    def _resolve_relative_module(self, module_name: str | None, level: int):
        if level <= 0:
            return module_name

        pkg_parts = self.current_module.split(".")
        if not self.is_package_module:
            pkg_parts = pkg_parts[:-1]
        keep = len(pkg_parts) - level + 1
        if keep <= 0:
            return None

        resolved = pkg_parts[:keep]
        if module_name:
            resolved.extend(part for part in module_name.split(".") if part)
        return ".".join(resolved)

    def _record_import_from_target(self, module_name: str, imported_name: str | None = None):
        if imported_name:
            cand = f"{module_name}.{imported_name}"
            before = len(self.imports)
            self._record(cand)
            if len(self.imports) > before:
                return
        self._record(module_name)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        mod_name = node.module
        if node.level:
            mod_name = self._resolve_relative_module(mod_name, node.level)
        if not mod_name:
            return

        if node.module is None:
            for alias in node.names:
                self._record_import_from_target(mod_name, alias.name)
            return

        matched_mod = False
        for alias in node.names:
            before = len(self.imports)
            self._record(f"{mod_name}.{alias.name}")
            if len(self.imports) > before:
                matched_mod = True
        if not matched_mod:
            self._record(mod_name)

    def collect_top_level(self, tree: ast.Module):
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.visit_Import(node)
            elif isinstance(node, ast.ImportFrom):
                self.visit_ImportFrom(node)
        return self.imports


# TODO: fix this
@dataclass(slots=True)
class RepositoryIndex:
    package_name: str
    package_root: Path
    functions_by_full_name: dict[str, FunctionInfo]
    functions_by_name: dict[str, list[FunctionInfo]]
    module_imports: dict[str, set[str]]
    reverse_module_imports: dict[str, set[str]]
    class_methods: dict[str, list[str]]
    class_bases: dict[str, tuple[str, ...]]
    class_children: dict[str, set[str]]

    @classmethod
    def from_package(cls, package_root: Path, package_name: str):
        py_files = [
            path for path in package_root.rglob("*.py")
            if not _should_skip(path)
        ]
        known = {_module_name(package_root, path) for path in py_files}

        funcs_full: dict[str, FunctionInfo] = {}
        funcs_name: dict[str, list[FunctionInfo]] = defaultdict(list)
        mod_imports: dict[str, set[str]] = defaultdict(set)
        cls_methods: dict[str, list[str]] = defaultdict(list)
        cls_bases: dict[str, tuple[str, ...]] = {}
        kids: dict[str, set[str]] = defaultdict(set)

        for path in py_files:
            source = path.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue

            mod_name = _module_name(package_root, path)
            imports = _ImportCollector(package_name, known, mod_name, is_package_module=path.name == '__init__.py')
            mod_imports[mod_name].update(imports.collect_top_level(tree))

            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    info = _build_function_info(node=node, source=source, module_name=mod_name, package_name=package_name, file_path=path)
                    funcs_full[info.full_name] = info
                    funcs_name[info.name].append(info)
                    for child in node.body:
                        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            continue
                        nested_full = f"{mod_name}.{child.name}"
                        if nested_full in funcs_full:
                            nested_full = f"{mod_name}.{node.name}.{child.name}"
                        info = _build_function_info(node=child, source=source, module_name=mod_name, package_name=package_name, file_path=path, function_type='local_function', full_name_override=nested_full)
                        funcs_full[info.full_name] = info
                        funcs_name[info.name].append(info)
                elif isinstance(node, ast.ClassDef):
                    bases = tuple(_expr_to_name(base) for base in node.bases if _expr_to_name(base))
                    cls_doc = ast.get_docstring(node) or ""
                    class_key = f"{mod_name}.{node.name}"
                    cls_bases[class_key] = bases
                    for base in bases:
                        kids[base].add(class_key)
                    for child in node.body:
                        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            continue
                        info = _build_function_info(node=child, source=source, module_name=mod_name, package_name=package_name, file_path=path, class_name=node.name, class_docstring=cls_doc, base_classes=bases)
                        funcs_full[info.full_name] = info
                        funcs_name[info.name].append(info)
                        cls_methods[class_key].append(info.full_name)

        rev_imports: dict[str, set[str]] = defaultdict(set)
        for mod_name, deps in mod_imports.items():
            for dep in deps:
                rev_imports[dep].add(mod_name)

        return cls(package_name=package_name, package_root=package_root, functions_by_full_name=funcs_full, functions_by_name=dict(funcs_name), module_imports=dict(mod_imports), reverse_module_imports=dict(rev_imports), class_methods=dict(cls_methods), class_bases=cls_bases, class_children=dict(kids))

    def find_function(self, api_info: dict[str, Any]):
        cands = [
            api_info.get("full_name", ""),
            f"{api_info.get('module', '')}.{api_info.get('name', '')}".strip("."),
            api_info.get("name", ""),
        ]
        for cand in cands:
            if cand in self.functions_by_full_name:
                return self.functions_by_full_name[cand]
        name = api_info.get("name", "")
        matches = self.functions_by_name.get(name, [])
        if len(matches) == 1:
            return matches[0]
        raise KeyError(f"Could not resolve target function from candidates: {cands}")


def _expr_to_name(node: ast.AST):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _expr_to_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _build_function_info(node: ast.FunctionDef | ast.AsyncFunctionDef, source: str, module_name: str, package_name: str, file_path: Path, class_name: str | None = None, class_docstring: str = "", base_classes: tuple[str, ...] = (), function_type: str | None = None, full_name_override: str | None = None):
    code = textwrap.dedent(ast.get_source_segment(source, node) or "").strip()
    docstring = ast.get_docstring(node) or ""
    calls = _CallCollector()
    calls.visit(node)
    name = f"{class_name}.{node.name}" if class_name else node.name
    full_name = full_name_override or f"{module_name}.{name}"
    return FunctionInfo(name=name, full_name=full_name, module_location=module_name, signature=_extract_signature(node), docstring=docstring, code=code, function_type=function_type or ('method' if class_name else 'function'), file_path=str(file_path), package=package_name, class_name=class_name, class_docstring=class_docstring, base_classes=base_classes, calls=tuple(sorted(calls.calls)))


def infer_package_root(file_path: str, package_name: str):
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(file_path)
    indices = [i for i, part in enumerate(path.parts) if part == package_name or part == package_name.replace("-", "_")]
    if not indices:
        raise ValueError(f"Could not infer package root from {file_path}")
    pkg_idx = indices[-1]
    return Path(*path.parts[:pkg_idx + 1])


def find_import_root(package_name: str):
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        raise ModuleNotFoundError(package_name)
    if spec.submodule_search_locations:
        pkg_root = Path(list(spec.submodule_search_locations)[0])
    else:
        pkg_root = Path(spec.origin).parent
    return pkg_root, pkg_root.parent
