import ast
import operator as _op
from typing import Type, Callable, Union, cast, Any

from langchain_core.tools import tool

# 这里的类型提示可以保留，但警告实际上是关于字典字面量本身
# 它希望你提供的键类型是 Type[ast.operator]，而不是 Union[Type[ast.Add], ...]

# 为二元操作符类型定义一个联合类型
BinaryOpKeyType = Union[Type[ast.Add], Type[ast.Sub], Type[ast.Mult], Type[ast.Div]]
# 为二元操作符函数定义一个类型：接受两个 Any，返回 Any
BinaryOpValueType = Callable[[Any, Any], Any]

_ALLOWED_BIN_OPS: dict[BinaryOpKeyType, BinaryOpValueType] = { # type: ignore [dict-item]
    # 或者直接在该行末尾添加 # type: ignore
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
}

# 为一元操作符类型定义一个联合类型
UnaryOpKeyType = Union[Type[ast.UAdd], Type[ast.USub]]
# 为一元操作符函数定义一个类型：接受一个 Any，返回 Any
UnaryOpValueType = Callable[[Any], Any]

_ALLOWED_UNARY_OPS: dict[UnaryOpKeyType, UnaryOpValueType] = { # type: ignore [dict-item]
    # 或者直接在该行末尾添加 # type: ignore
    ast.UAdd: lambda x: x,
    ast.USub: _op.neg
}

# 修改 _eval_ast 函数
def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_ast(cast(ast.AST, node.body))

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("仅支持数字常量")

    if isinstance(node, ast.BinOp):
        # 这里的 type(node.op) 已经通过前面的类型提示匹配
        if type(node.op) not in _ALLOWED_BIN_OPS:
            raise ValueError("仅支持 + - * / 运算")

        left = _eval_ast(cast(ast.AST, node.left))
        right = _eval_ast(cast(ast.AST, node.right))
        return _ALLOWED_BIN_OPS[type(node.op)](left, right)# type: ignore

    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _ALLOWED_UNARY_OPS:
            raise ValueError("仅支持一元正负号，如 -3 或 +5")
        operand = _eval_ast(cast(ast.AST, node.operand))
        #
        return _ALLOWED_UNARY_OPS[type(node.op)](operand)# type: ignore

    raise ValueError(f"不支持的表达式类型: {type(node).__name__}")


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