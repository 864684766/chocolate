"""FastAPI 应用模块初始化。

提供模块化的 API 路由组织结构，按业务领域分离不同路由。
"""
from fastapi import FastAPI,Request
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse
from app.core.exceptions import register_exception_handlers
from .health import router as health_router
from .agent import router as agent_router

def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用。"""
    app = FastAPI(
        title="Chocolate Agent API",
        version="0.1.0",
        description="基于 LangChain 的智能对话 Agent API 服务",
    )
    # 调用注册函数，将异常处理器绑定到 app 实例上
    register_exception_handlers(app)
    # 健康检查路由
    app.include_router(health_router, tags=["system"])
    
    # Agent 相关路由
    app.include_router(agent_router, prefix="/agent", tags=["agent"])
    
    return app