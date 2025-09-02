# 文件: app/core/exceptions.py

# 自定义异常类
class DatabaseConnectionError(Exception):
    """数据库连接异常"""
    pass


class DatabaseOperationError(Exception):
    """数据库操作异常"""
    pass


from fastapi import FastAPI, Request
# 导入 Exception 用于类型提示
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse

# --- 修改这里的类型提示 ---
# 将 exc: RequestValidationError 改为 exc: Exception
async def validation_exception_handler(request: Request, exc: Exception):
    """
    捕获并自定义处理请求体参数验证错误
    """
    # 确保我们处理的是正确的异常类型，这样使用 exc.errors() 才安全
    if isinstance(exc, RequestValidationError):
        error_messages = [error["msg"] for error in exc.errors()]
        return JSONResponse(
            status_code=422,
            content={'success': False, 'message': '请求参数验证失败', 'errors': error_messages}
        )
    # 对于其他类型的异常（虽然不太可能进入这里），可以返回一个通用错误
    return JSONResponse(
        status_code=500,
        content={'success': False, 'message': '服务器内部错误'}
    )

def register_exception_handlers(app: FastAPI) -> None:
    """
    将自定义的异常处理器注册到 FastAPI 应用实例上
    """

    # 现在这里的警告应该消失了
    app.add_exception_handler(RequestValidationError, validation_exception_handler)