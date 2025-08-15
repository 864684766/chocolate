from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from ..agent import build_agent

router = APIRouter()

# 简易的内存会话存储（生产请替换为 Redis/数据库）
_session_store: Dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]

class InvokeRequest(BaseModel):
    input: str
    session_id: Optional[str] = None

class InvokeResponse(BaseModel):
    answer: str

@router.post("/invoke", response_model=InvokeResponse)
def agent_invoke(req: InvokeRequest):
    try:
        executor = build_agent()

        # 包装为带会话历史的可运行对象
        chain_with_history = RunnableWithMessageHistory(
            executor,
            get_session_history,
            input_messages_key="input",  # 请求中用户输入的键
            history_messages_key="history",  # 传入历史记录的键
        )

        # 配置会话ID（无则使用固定访客ID，仍然实现无状态）
        session_id = req.session_id or "guest"

        result = chain_with_history.invoke(
            {"input": req.input},
            config={
                "configurable": {"session_id": session_id}
            },
        )

        # AgentExecutor 默认返回 {"output": str}
        answer = result.get("output", result) if isinstance(result, dict) else str(result)
        return InvokeResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))