# from __future__ import annotations
from fastapi import APIRouter, HTTPException
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from typing import Optional, Dict
#
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
#
from ..agent import build_agent
from ..core.agent_service import initialize_agent_chain, get_runnable_config

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
    session_id: Optional[str] = None

class InvokeResponse(BaseModel):
    answer: str

# 在模块级别获取一次 Agent 链实例，后续请求直接使用
_global_agent_chain_instance = initialize_agent_chain()

@router.post("/invoke", response_model=InvokeResponse)
def agent_invoke(req: InvokeRequest):
    try:
        # 配置会话ID（无则使用固定访客ID，仍然实现无状态）x
        session_id = req.session_id or "guest"
        runnable_config_obj = get_runnable_config(session_id)
        result = _global_agent_chain_instance.invoke(
            {"input": req.input},
            config=runnable_config_obj
        )

        # AgentExecutor 默认返回 {"output": str}
        answer = result.get("output", result) if isinstance(result, dict) else str(result)
        return InvokeResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))