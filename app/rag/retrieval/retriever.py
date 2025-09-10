from __future__ import annotations

"""
检索器实现（向量检索 / 关键词检索）。

说明：
- 应用层应先进行“智能路由”，选择使用 Vector / Keyword / Hybrid；
- 本文件仅提供执行能力，不负责决策。
"""

from typing import List, Dict, Any
import time

from .schemas import RetrievalQuery, RetrievalResult, RetrievedItem
from app.infra.database.chroma.db_helper import ChromaDBHelper
from app.rag.vectorization.embedder import Embedder
from app.rag.vectorization.config import VectorizationConfig
from app.infra.logging import get_logger


class VectorRetriever:
    """向量检索器：将 query 编码为向量并在 ChromaDB 中召回 TopK。"""

    def __init__(self, collection_name: str | None = None):
        self.logger = get_logger(__name__)
        cfg = VectorizationConfig.from_config_manager()
        # 兼容：严格从 vectorization.database.collection_name 读取，避免未解析属性告警
        self.collection_name = collection_name or getattr(cfg, "collection_name")
        self.db = ChromaDBHelper()
        self.embedder = Embedder(cfg)

    def search(self, q: RetrievalQuery) -> RetrievalResult:
        t0 = time.time()
        embeddings = self.embedder.encode([q.query])
        result = self.db.query(
            collection_name=self.collection_name,
            query_embeddings=embeddings,
            n_results=q.top_k,
            where=q.where,
            include=["documents", "metadatas", "distances", "data"],
        )

        items: List[RetrievedItem] = []
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        ids = result.get("ids", [[]])[0]
        dists = result.get("distances", [[]])[0]

        for doc, meta, _id, dist in zip(docs, metas, ids, dists):
            score = 1.0 - float(dist) if dist is not None else 0.0
            if score < q.score_threshold:
                continue
            # 规范化 metadatas：将 Mapping 转为普通 dict，None → {}
            meta_dict: Dict[str, Any] = dict(meta) if isinstance(meta, dict) else dict(meta or {})
            items.append(RetrievedItem(id=_id, text=doc, score=score, metadata=meta_dict))

        latency_ms = int((time.time() - t0) * 1000)
        return RetrievalResult(items=items, latency_ms=latency_ms, debug_info={"source": "vector"})


class KeywordRetriever:
    """关键词/结构化检索：依赖 where 过滤（Chroma 的 metadata 过滤）。"""

    def __init__(self, collection_name: str | None = None):
        self.logger = get_logger(__name__)
        cfg = VectorizationConfig.from_config_manager()
        self.collection_name = collection_name or getattr(cfg, "collection_name")
        self.db = ChromaDBHelper()

    def search(self, q: RetrievalQuery) -> RetrievalResult:
        t0 = time.time()
        # 关键词检索的最小实现：仅使用 where 过滤 + 返回文档
        # 如果需要倒排/全文关键词匹配，可按后续需求扩展（例如维护 keyword index）
        result = self.db.get(
            collection_name=self.collection_name,
            where=q.where,
            include=["documents", "metadatas", "data"],
        )

        docs: List[str] = result.get("documents", [])
        metas_raw = result.get("metadatas", [])
        metas: List[Dict[str, Any]] = [dict(m) if isinstance(m, dict) else dict(m or {}) for m in metas_raw]
        ids: List[str] = result.get("ids", [])

        items: List[RetrievedItem] = []
        for doc, meta, _id in zip(docs, metas, ids):
            items.append(RetrievedItem(id=_id, text=doc, score=1.0, metadata=meta))

        # 按固定分数、限制 top_k（真正排序由 Hybrid 或应用层决定）
        items = items[: q.top_k]
        latency_ms = int((time.time() - t0) * 1000)
        return RetrievalResult(items=items, latency_ms=latency_ms, debug_info={"source": "keyword"})


