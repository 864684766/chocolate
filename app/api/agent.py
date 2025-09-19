# from __future__ import annotations
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, Dict
#
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
#
from ..core.agent_service import initialize_agent_chain, get_runnable_config
from app.rag.retrieval.schemas import RetrievalQuery
from app.rag.retrieval.vector_retriever import VectorRetriever
from app.rag.retrieval.reranker import CrossEncoderReranker
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


@router.post("/retrieval/search", response_model=BaseResponse)
def retrieval_search(req: InvokeRequest):
    """
    与 agent_invoke 同构的检索接口：
    - 输入：InvokeRequest.input 作为 query；其他字段可留空
    - 流程：Vector 召回 →（可选）交叉编码重排 → 返回拼接预览与检索明细
    - 输出：BaseResponse(data=InvokeData(answer=...))，answer 为拼接预览
    """
    session_id = req.session_id or "guest"
    runnable_config_obj = get_runnable_config(session_id)
    try:
        # 读取检索与重排配置
        cfg = get_config_manager().get_config("retrieval") or {}
        rerank_cfg = cfg.get("rerank", {})
        enable_rerank = bool(rerank_cfg.get("enabled", False))
        rerank_model = str(rerank_cfg.get("model_name", "")) or None
        top_n = int(rerank_cfg.get("top_n", 10))

        # 执行向量召回
        retriever = VectorRetriever()
        q = RetrievalQuery(query=req.input, where=None, top_k=top_n, score_threshold=0.0)
        result = retriever.search(q)

        items = result.items
        if enable_rerank and items:
            reranker = CrossEncoderReranker(model_name=rerank_model)
            items = reranker.rerank(items, top_n=top_n, query=req.input)

        # 生成简短预览作为 answer（不调用 LLM）
        preview_texts = [it.text for it in items[: min(3, len(items))]]
        preview = "\n\n".join(preview_texts) if preview_texts else ""

        data = InvokeData(answer=preview).model_dump()
        return BaseResponse(
            code=ResponseCode.OK,
            message=ResponseMessage.SUCCESS,
            data=data,
            request_id=runnable_config_obj.get("configurable", {}).get("session_id", ""),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except (RuntimeError, ValueError) as e:
        return BaseResponse(
            code=ResponseCode.UPSTREAM_ERROR,
            message=ResponseMessage.UPSTREAM_ERROR,
            data={"error": str(e)},
            request_id=runnable_config_obj.get("configurable", {}).get("session_id", ""),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )