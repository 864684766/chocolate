from __future__ import annotations

"""
基于 Neo4j 的邻居扩展检索器。
"""

from typing import List, Dict, Any, Set
import time

from app.infra.database.neo4j.db_helper import Neo4jDBHelper
from app.infra.database.chroma.db_helper import ChromaDBHelper
from app.rag.vectorization.config import VectorizationConfig
from app.rag.retrieval.schemas import RetrievedItem, RetrievalResult, RetrievalQuery
from app.rag.retrieval.vector_retriever import VectorRetriever
from app.infra.logging import get_logger


class GraphRetriever:
    """从图中查询相邻块并拉取原文内容。"""

    def __init__(self) -> None:
        """
        初始化图检索器，准备 Neo4j 与 Chroma 访问。

        用处：
        - 绑定 Neo4j 用于关系查询。
        - 绑定 Chroma 用于根据 chunk_id 拉取文本内容。
        """
        self.db = Neo4jDBHelper()
        self.chroma = ChromaDBHelper()
        cfg = VectorizationConfig.from_config_manager()
        self.collection = cfg.collection_name
        self.logger = get_logger(__name__)

    def is_enabled(self) -> bool:
        """
        判断图检索是否可用。

        Returns:
            bool: 配置存在时返回 True。
        """
        return self.db.has_config()

    def expand_neighbors(
        self,
        base_items: List[RetrievedItem],
        max_hops: int = 1,
        limit: int = 10,
    ) -> List[RetrievedItem]:
        """
        使用图数据库扩展相邻块。

        Args:
            base_items: 原始检索结果列表。
            max_hops: 最多跳数，默认 1。
            limit: 邻居最大返回数量，默认 10。

        Returns:
            List[RetrievedItem]: 扩展并去重后的结果。
        """
        if not self.is_enabled() or not base_items:
            return base_items
        base_ids = [it.id for it in base_items if it.id]
        neighbor_ids = self._query_neighbor_ids(base_ids, max_hops, limit)
        extra_items = self._fetch_chunks(neighbor_ids)
        merged = self._merge_items(base_items, extra_items)
        return merged

    def _query_neighbor_ids(self, ids: List[str], hops: int, limit: int) -> List[str]:
        """
        查询相邻块 ID。

        Args:
            ids: 起始块 ID 列表。
            hops: 最大跳数。
            limit: 返回数量上限。

        Returns:
            List[str]: 去重后的邻居块 ID 列表。

        说明:
        - Neo4j 不允许在关系模式中使用参数化的跳数，需要直接拼接
        - 已添加安全检查，确保 hops 是安全的整数（限制在 1-10 之间）
        """
        # Neo4j 不允许在关系模式中使用参数化的跳数，需要直接拼接
        # 确保 hops 是正整数，防止注入（限制在 1-10 之间）
        hops = max(1, min(int(hops), 10))
        
        query = f"""
        MATCH (c:Chunk) WHERE c.id IN $ids
        MATCH (c)-[:NEXT*1..{hops}]-(n:Chunk)
        RETURN DISTINCT n.id AS id
        LIMIT $limit
        """
        try:
            rows = self.db.run_read(query, {"ids": ids, "limit": limit})
            return [str(r.get("id")) for r in rows if r.get("id")]
        except Exception as exc:
            self.logger.warning(f"查询图邻居失败: {exc}")
            return []

    def _fetch_chunks(self, ids: List[str]) -> List[RetrievedItem]:
        """
        从 Chroma 读取邻居块内容。

        Args:
            ids: 需要拉取的邻居块 ID 列表。

        Returns:
            List[RetrievedItem]: 填充 text/metadata 的邻居结果。
        """
        if not ids:
            return []
        try:
            result = self.chroma.get(
                collection_name=self.collection,
                ids=ids,
                include=["documents", "metadatas"],
            )
            # ChromaDB 的 get 方法返回扁平列表，不是嵌套列表
            # query 方法返回嵌套列表（因为可能有多个查询），但 get 方法返回扁平列表
            docs = result.get("documents", [])
            metas = result.get("metadatas", [])
            res_ids = result.get("ids", [])
            
            items: List[RetrievedItem] = []
            for _id, doc, meta in zip(res_ids, docs, metas):
                if not _id or not doc:
                    continue
                meta_dict: Dict[str, Any] = dict(meta) if isinstance(meta, dict) else {}
                items.append(RetrievedItem(id=_id, text=doc, score=meta_dict.get("score", 0.0), metadata=meta_dict))
            return items
        except Exception as exc:
            self.logger.warning(f"读取邻居块内容失败: {exc}")
            return []

    def search_independent(
        self,
        query: RetrievalQuery,
        max_hops: int = 1,
        limit: int = 10,
    ) -> RetrievalResult:
        """
        独立图检索：基于查询文本，通过向量检索得到初始结果，然后从图中扩展。

        用处：
        - 作为独立的检索源参与 RRF 融合
        - 先通过向量检索找到相关块，然后通过图关系扩展，得到更多相关上下文

        Args:
            query: 检索请求（含 query/where/top_k/score_threshold）
            max_hops: 图扩展的最大跳数，默认 1
            limit: 图扩展的最大邻居数量，默认 10

        Returns:
            RetrievalResult: 图检索结果（包含初始向量检索结果和扩展的邻居结果）

        说明：
        - 先通过向量检索得到初始结果（top_k 可以大一些，如 query.top_k * 2）
        - 然后从这些初始结果中，通过图关系扩展，得到更多的相关块
        - 返回扩展后的结果作为图检索的结果
        """
        if not self.is_enabled():
            return RetrievalResult(items=[], latency_ms=0, matched_count=0)

        t0 = time.time()

        # 先通过向量检索得到初始结果（扩大 top_k 以获取更多候选）
        initial_top_k = max(query.top_k * 2, 20)
        initial_query = RetrievalQuery(
            query=query.query,
            where=query.where,
            top_k=initial_top_k,
            score_threshold=query.score_threshold
        )
        vector_retriever = VectorRetriever()
        initial_result = vector_retriever.search(initial_query)

        if not initial_result.items:
            latency_ms = int((time.time() - t0) * 1000)
            return RetrievalResult(items=[], latency_ms=latency_ms, matched_count=0)

        # 从初始结果中扩展邻居
        base_ids = [it.id for it in initial_result.items if it.id]
        neighbor_ids = self._query_neighbor_ids(base_ids, max_hops, limit)
        extra_items = self._fetch_chunks(neighbor_ids)

        # 合并初始结果和扩展结果
        merged_items = self._merge_items(initial_result.items, extra_items)

        # 限制返回数量
        final_items = merged_items[:query.top_k]

        latency_ms = int((time.time() - t0) * 1000)
        return RetrievalResult(
            items=final_items,
            latency_ms=latency_ms,
            matched_count=len(final_items)
        )

    @staticmethod
    def _merge_items(
            base_items: List[RetrievedItem], extra_items: List[RetrievedItem]
    ) -> List[RetrievedItem]:
        """
        将邻居结果与原结果去重合并，保持原有排序优先。

        Args:
            base_items: 原始检索结果。
            extra_items: 图扩展得到的邻居结果。

        Returns:
            List[RetrievedItem]: 合并去重后的列表。
        """
        seen: Set[str] = set()
        merged: List[RetrievedItem] = []
        for it in base_items + extra_items:
            if it.id and it.id not in seen:
                merged.append(it)
                seen.add(it.id)
        return merged

