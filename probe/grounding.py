import ast
import textwrap
from .models import FunctionInfo, GuardConstraint


class ConstraintExtractor:
    def extract(self, function_info: FunctionInfo):
        if not function_info.code.strip():
            return []

        tree = ast.parse(textwrap.dedent(function_info.code))
        func_node = next((node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
        if func_node is None: return []

        cons: list[GuardConstraint] = []
        self._extract_from_statements(func_node.body, {}, [], cons)

        dedup: list[GuardConstraint] = []
        seen: set[str] = set()
        for item in cons:
            key = item.valid_constraint.strip()
            if not key or key in seen: continue
            seen.add(key)
            dedup.append(item)
        return dedup

    def _record_assignment(self, node: ast.stmt, assignments: dict[str, ast.AST]):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            assignments[node.targets[0].id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            assignments[node.target.id] = node.value


    def _extract_from_statements(self, statements: list[ast.stmt], assignments: dict[str, ast.AST], path_conditions: list[tuple[str, str]], constraints: list[GuardConstraint]):
        for node in statements:
            if isinstance(node, ast.If):
                condition = self._condition_pair(node.test, assignments)
                self._extract_from_statements(node.body, assignments.copy(), path_conditions + [condition], constraints)
                if node.orelse:
                    self._extract_from_statements(node.orelse, assignments.copy(), path_conditions + [self._invert_condition(condition)], constraints)
            elif isinstance(node, ast.Assert):
                test = self._resolve(node.test, assignments)
                if path_conditions:
                    bad_path = path_conditions + [(f"not ({test})", test)]
                    constraints.append(GuardConstraint(guard_expression=self._join_path(bad_path), valid_constraint=self._negate_path(bad_path), rationale='path condition required for assertion to hold'))
                else:
                    constraints.append(GuardConstraint(guard_expression=test, valid_constraint=test, rationale='assertion required for valid execution'))
            elif isinstance(node, ast.Raise) and path_conditions:
                constraints.append(GuardConstraint(guard_expression=self._join_path(path_conditions), valid_constraint=self._negate_path(path_conditions), rationale='negated guard path that raises on invalid inputs'))
            self._record_assignment(node, assignments)

    # clean
    def _condition_pair(self, expr: ast.AST, assignments: dict[str, ast.AST]):
        return self._resolve(expr, assignments), self._negate(expr, assignments)



    def _invert_condition(self, condition: tuple[str, str]):
        cond, negated = condition
        return negated, cond

    def _join_path(self, path_conditions: list[tuple[str, str]]):
        if len(path_conditions) == 1: return path_conditions[0][0]
        return " and ".join(f"({condition})" for condition, _ in path_conditions)

    def _negate_path(self, path_conditions: list[tuple[str, str]]):
        if len(path_conditions) == 1: return path_conditions[0][1]
        # res = " or ".join(f"({negated})" for _, negated in path_conditions)
        # print(res)

        return " or ".join(f"({negated})" for _, negated in path_conditions)

    def _resolve(self, expr: ast.AST, assignments: dict[str, ast.AST], depth: int = 0): # TODO: make threshold configurable
        if depth > 4:
            return ast.unparse(expr)
        if isinstance(expr, ast.Name) and expr.id in assignments:
            return self._resolve(assignments[expr.id], assignments, depth + 1)
        if isinstance(expr, ast.Compare):
            left = self._resolve(expr.left, assignments, depth + 1)
            chunks = [left]
            for op, cmp_op in zip(expr.ops, expr.comparators):
                chunks.append(self._op_text(op))
                chunks.append(self._resolve(cmp_op, assignments, depth + 1))
            return " ".join(chunks)
        if isinstance(expr, ast.BoolOp):
            joiner = " and " if isinstance(expr.op, ast.And) else " or "
            return joiner.join(f"({self._resolve(v, assignments, depth + 1)})" for v in expr.values)
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not): # TODO: 这里好像不会进来
            return f"not ({self._resolve(expr.operand, assignments, depth + 1)})"
        if isinstance(expr, ast.Call):
            func = self._resolve(expr.func, assignments, depth + 1)
            args = ", ".join(self._resolve(arg, assignments, depth + 1) for arg in expr.args)
            return f"{func}({args})"
        if isinstance(expr, ast.Attribute):
            return f"{self._resolve(expr.value, assignments, depth + 1)}.{expr.attr}"
        return ast.unparse(expr)

    def _negate(self, expr: ast.AST, assignments: dict[str, ast.AST]):
        if isinstance(expr, ast.Compare) and len(expr.ops) == 1 and len(expr.comparators) == 1:
            left = self._resolve(expr.left, assignments)
            right = self._resolve(expr.comparators[0], assignments)
            op = expr.ops[0]
            inv_map = {
                ast.Lt: ">=",
                ast.LtE: ">",
                ast.Gt: "<=",
                ast.GtE: "<",
                ast.Eq: "!=",
                ast.NotEq: "==",
                ast.Is: "is not",
                ast.IsNot: "is",
                ast.In: "not in",
                ast.NotIn: "in",
            }
            for key, symbol in inv_map.items():
                if isinstance(op, key):
                    return f"{left} {symbol} {right}"
        if isinstance(expr, ast.BoolOp):
            joiner = " or " if isinstance(expr.op, ast.And) else " and "
            return joiner.join(f"({self._negate(v, assignments)})" for v in expr.values)
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
            return self._resolve(expr.operand, assignments)
        return f"not ({self._resolve(expr, assignments)})"

    def _op_text(self, op: ast.AST): # TODO: replace with LLM (test)
        mapping = {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
            ast.Is: "is",
            ast.IsNot: "is not",
            ast.In: "in",
            ast.NotIn: "not in",
        }
        for kind, text in mapping.items():
            # print(text)
            if isinstance(op, kind): return text


        return ast.unparse(op)
