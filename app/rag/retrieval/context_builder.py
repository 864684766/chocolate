from __future__ import annotations

"""
上下文构建器：对检索结果做去重、合并、截断，生成可注入 LLM 的上下文文本与引用信息。

注意：
- 真实实现可结合 quality_utils、中文优先合并、媒体类型分组等策略；
- 此处给出最小实现骨架，便于先打通流程。
"""

from typing import List, Dict, Any
from .schemas import RetrievalResult, BuiltContext
from app.core.tokenization.provider import TokenCounter, get_token_counter


class ContextBuilder:
    """最小上下文构建实现。"""

    @staticmethod
    def build(
        result: RetrievalResult,
        max_tokens: int = 1500,
        citation: bool = True,
        token_counter: TokenCounter | None = None,
        ai_type: str | None = None,
        provider: str | None = None,
    ) -> BuiltContext:
        # 若未显式传入 TokenCounter，则按应用层当前的 ai_type/provider 获取
        if token_counter is None:
            # 初始化分词计数器失败时静默降级为字符近似
            try:
                token_counter = get_token_counter(ai_type, provider)
            except (ImportError, OSError, ValueError, KeyError):
                token_counter = None
        lines: List[str] = []
        citations: List[Dict[str, Any]] = []
        used_ids: List[str] = []
        used_budget: int = 0

        for it in result.items:
            if citation:
                citations.append({
                    "id": it.id,
                    "score": it.score,
                    "source": it.metadata.get("source"),
                    "filename": it.metadata.get("filename"),
                    "chunk_index": it.metadata.get("chunk_index"),
                })
            text_piece = it.text or ""
            cost = (token_counter.count_tokens(text_piece) if token_counter else len(text_piece))
            if used_budget + cost > max_tokens:
                break
            lines.append(text_piece)
            used_ids.append(it.id)
            used_budget += cost

        text = "\n\n".join(lines)
        return BuiltContext(text=text, citations=citations, used_tokens=used_budget, from_items=used_ids)


