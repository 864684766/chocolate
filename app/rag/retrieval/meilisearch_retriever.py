from __future__ import annotations

"""
Meilisearch 检索器（占位）。

说明：
- 与 VectorRetriever.search 保持相同方法签名，后续实现 BM25/容错搜索等。
- 当前占位：不提供实际检索实现。
"""

from .schemas import RetrievalQuery, RetrievalResult


class MeilisearchRetriever:
    """关键词检索占位实现。"""

    def __init__(self, index_name: str | None = None):
        self.index_name = index_name

    @staticmethod
    def search(q: RetrievalQuery) -> RetrievalResult:
        # 占位：后续接入 Meilisearch SDK 实现检索
        print(q)
        return RetrievalResult(items=[], latency_ms=0, debug_info={"source": "meili", "note": "placeholder"})


