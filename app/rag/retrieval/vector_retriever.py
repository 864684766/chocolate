from __future__ import annotations

"""
向量检索器（独立文件）。

职责：
- 将查询文本规范化并编码为向量
- 在 ChromaDB 中按相似度召回 TopK
- 按配置化度量将距离/相似度映射为 [0,1] 分数

说明：
- 不负责路由/融合/重排，仅执行向量召回本身
- 融合（RRF/加权）与重排（交叉编码器）由上层对接
"""

from typing import List, Dict, Any
import time

from .schemas import RetrievalQuery, RetrievalResult, RetrievedItem
from app.config import get_config_manager
from app.rag.retrieval.utils.scoring import score_from_distance, Metric
from app.infra.database.chroma.db_helper import ChromaDBHelper
from app.rag.vectorization.embedder import Embedder
from app.rag.vectorization.config import VectorizationConfig
from app.infra.logging import get_logger
from app.rag.retrieval.utils.query_cleaner import clean_query_basic


class VectorRetriever:
    """向量检索器：将 query 编码为向量并在 ChromaDB 中召回 TopK。

    方法体不超过 20 行，复杂逻辑拆分到工具函数与依赖类中。
    """

    def __init__(self, collection_name: str | None = None):
        self.logger = get_logger(__name__)
        cfg = VectorizationConfig.from_config_manager()
        # 从 vectorization.database.collection_name 读取集合名
        self.collection_name = collection_name or getattr(cfg, "collection_name")
        self.db = ChromaDBHelper()
        self.embedder = Embedder(cfg)

    def search(self, q: RetrievalQuery) -> RetrievalResult:
        """执行向量检索。

        Args:
            q: 检索请求（含 query/where/top_k/score_threshold）。

        Returns:
            RetrievalResult: 命中项、耗时等信息。
        """
        t0 = time.time()
        query_text = clean_query_basic(q.query)
        embeddings = self.embedder.encode([query_text])
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

        # 读取评分配置
        r_cfg = (get_config_manager().get_config("retrieval") or {}).get("scoring", {})
        metric_cfg = str(r_cfg.get("metric", "cosine_distance"))
        # 将配置字符串安全映射为受限字面量类型 Metric
        allowed: Dict[str, Metric] = {
            "cosine_distance": "cosine_distance",
            "cosine_similarity": "cosine_similarity",
            "l2": "l2",
            "inner_product": "inner_product",
        }
        metric: Metric = allowed.get(metric_cfg, "cosine_distance")
        normalize = bool(r_cfg.get("normalize", True))
        alpha = float((r_cfg.get("params") or {}).get("alpha", 1.0))

        for doc, meta, _id, dist in zip(docs, metas, ids, dists):
            raw = float(dist) if dist is not None else 0.0
            score = score_from_distance(raw, metric=metric, normalize=normalize, alpha=alpha)
            if score < q.score_threshold:
                continue
            meta_dict: Dict[str, Any] = dict(meta) if isinstance(meta, dict) else dict(meta or {})
            items.append(RetrievedItem(id=_id, text=doc, score=score, metadata=meta_dict))

        latency_ms = int((time.time() - t0) * 1000)
        return RetrievalResult(
            items=items,
            latency_ms=latency_ms,
            applied_where=q.where,
            matched_count=len(items),
        )


