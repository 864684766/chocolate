# from __future__ import annotations
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, Dict
#
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
#
from ..core.agent_service import initialize_agent_chain, get_runnable_config
from ..config import get_config_manager
from .schemas import BaseResponse, ResponseCode, ResponseMessage
from datetime import datetime, timezone

#
router = APIRouter()
#
# # 简易的内存会话存储（生产请替换为 Redis/数据库）
_session_store: Dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
        print('session_id---',_session_store[session_id])
    return _session_store[session_id]
#
class InvokeRequest(BaseModel):
    input: str
    ai_type:Optional[str]= None
    provider:Optional[str]= None
    session_id: Optional[str] = None

class InvokeData(BaseModel):
    answer: str

# 在模块级别获取一次 Agent 链实例，后续请求直接使用


@router.post("/invoke", response_model=BaseResponse)
def agent_invoke(req: InvokeRequest):
    session_id = req.session_id or "guest"
    runnable_config_obj = get_runnable_config(session_id)
    try:
        chain = initialize_agent_chain(req.ai_type, req.provider)
        result = chain.invoke({"input": req.input}, config=runnable_config_obj)
        answer = result.get("output", result) if isinstance(result, dict) else str(result)
        return BaseResponse(
            code=ResponseCode.OK,
            message=ResponseMessage.SUCCESS,
            data=InvokeData(answer=answer).model_dump(),
            request_id=runnable_config_obj.get("configurable", {}).get("session_id", ""),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except (RuntimeError, ValueError) as e:
        cfg = get_config_manager()
        settings = cfg.get_settings(req.ai_type, req.provider)
        model_cfg = cfg.get_model_config(settings.provider, settings.model)
        detail = {
            "error": str(e),
            "provider": settings.provider,
            "model": settings.model,
            "generation_config": model_cfg.get("generation_config"),
            "safety": model_cfg.get("safety"),
        }
        return BaseResponse(
            code=ResponseCode.UPSTREAM_ERROR,
            message=ResponseMessage.UPSTREAM_ERROR,
            data=detail,
            request_id=runnable_config_obj.get("configurable", {}).get("session_id", ""),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )