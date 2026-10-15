import ast
import math
import operator
import re
from app.tools.base import tool

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_ALLOWED_FUNCTIONS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}


def _safe_eval_expr(node):
    if isinstance(node, ast.Expression):
        return _safe_eval_expr(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        left = _safe_eval_expr(node.left)
        right = _safe_eval_expr(node.right)
        return _ALLOWED_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval_expr(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func_name = node.func.id
        if func_name in _ALLOWED_FUNCTIONS:
            args = [_safe_eval_expr(arg) for arg in node.args]
            return _ALLOWED_FUNCTIONS[func_name](*args)
    raise ValueError(f"Biểu thức không được hỗ trợ: {ast.dump(node)}")


def make_calculator_tool():
    @tool
    def calculate_metrics(expression: str) -> str:
        """
        Công cụ máy tính toán học chính xác dùng để tính toán số liệu, tỷ lệ phần trăm, độ chênh lệch
        (ví dụ: '74.03 - 72.16', '(36000 / 50000) * 100', 'round((74.03 - 72.16) / 72.16 * 100, 2)').
        Hỗ trợ các phép toán +, -, *, /, **, % và các hàm sqrt, log, log10, exp, abs, round, min, max.
        """
        print(f"\n[Tool: Calculator] Đang tính biểu thức: '{expression}'")
        try:
            cleaned = expression.strip().replace("^", "**")
            cleaned = re.sub(r"[^0-9a-zA-Z_\+\-\*\/\%\(\)\.\,\s]", "", cleaned)
            parsed = ast.parse(cleaned, mode="eval")
            result = _safe_eval_expr(parsed)
            if isinstance(result, float):
                result = round(result, 6)
            return f"Kết quả tính toán của `{expression}` = {result}"
        except Exception as e:
            return f"Không thể tính biểu thức '{expression}': {e}"

    return calculate_metrics
