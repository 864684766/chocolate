import ast
import operator as _op
from langchain_core.tools import tool

# 安全计算：仅允许 +, -, *, / 和括号的数学表达式
_ALLOWED_BIN_OPS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
}
_ALLOWED_UNARY_OPS = {ast.UAdd: lambda x: x, ast.USub: _op.neg}


def _eval_ast(node):
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if hasattr(ast, "Num") and isinstance(node, ast.Num):  # 兼容 Python <3.8
        return node.n
    if isinstance(node, ast.Constant):  # Python >=3.8
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("仅支持数字常量")
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _ALLOWED_BIN_OPS:
            raise ValueError("仅支持 + - * / 运算")
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        return _ALLOWED_BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _ALLOWED_UNARY_OPS:
            raise ValueError("仅支持一元正负号，如 -3 或 +5")
        operand = _eval_ast(node.operand)
        return _ALLOWED_UNARY_OPS[type(node.op)](operand)
    raise ValueError("不支持的表达式")


def _safe_eval(expr: str) -> float:
    parsed = ast.parse(expr, mode="eval")
    return float(_eval_ast(parsed))


@tool
def calc(expression: str) -> str:
    """计算器工具：安全计算仅包含 +, -, *, / 和括号的数学表达式。
    例如："1 + 2 * (3 - 0.5)" -> "4.0"
    """
    try:
        result = _safe_eval(expression)
        return str(result)
    except Exception as e:
        return f"计算失败: {e}"