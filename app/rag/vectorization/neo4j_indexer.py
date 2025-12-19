from __future__ import annotations

"""
Neo4j 索引器：将分块信息写入图数据库，保留文档与相邻块关系。
"""

from typing import Dict, List, Any, Tuple

from app.infra.database.neo4j.db_helper import Neo4jDBHelper
from app.infra.logging import get_logger


class Neo4jIndexer:
    """Neo4j 索引器：写入 Document / Chunk 节点及相邻关系。"""

    def __init__(self) -> None:
        """
        初始化索引器，读取 Neo4j 配置。

        用处：准备 Neo4jDBHelper，便于后续批量写入图数据。
        """
        self.db = Neo4jDBHelper()
        self.logger = get_logger(__name__)

    def is_enabled(self) -> bool:
        """
        判断 Neo4j 是否可用（配置存在）。

        Returns:
            bool: 当配置中存在 url/user 时返回 True。
        """
        return self.db.has_config()

    def index_chunks(self, ids: List[str], metadatas: List[Dict[str, Any]]) -> int:
        """
        写入分块节点与关系，返回写入条数。

        Args:
            ids: 分块 ID 列表（与 metadatas 一一对应）。
            metadatas: 分块元数据列表，需包含 doc_id、chunk_index 等字段。

        Returns:
            int: 实际写入的分块数（写入失败时返回 0）。
        """
        if not self.is_enabled() or not ids:
            return 0
        chunks = self._build_chunk_records(ids, metadatas)
        if not chunks:
            return 0
        pairs = self._build_next_pairs(chunks)
        self._write_chunks(chunks)
        if pairs:
            self._write_next_relations(pairs)
        return len(chunks)

    @staticmethod
    def _build_chunk_records(
            ids: List[str], metas: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        组装写入图的块记录。

        Args:
            ids: 分块 ID 列表。
            metas: 元数据列表，需包含 doc_id、chunk_index 等。

        Returns:
            List[Dict[str, Any]]: 可直接写入 Neo4j 的记录集合。
        """
        records: List[Dict[str, Any]] = []
        for cid, meta in zip(ids, metas):
            doc_id = str(meta.get("doc_id") or meta.get("source") or "").strip()
            if not doc_id:
                continue
            records.append(
                {
                    "id": cid,
                    "doc_id": doc_id,
                    "chunk_index": int(meta.get("chunk_index", -1)),
                    "total_chunks": int(meta.get("total_chunks", 0)),
                    "section": meta.get("section"),
                    "page_number": meta.get("page_number"),
                }
            )
        return records

    @staticmethod
    def _build_next_pairs(
            chunks: List[Dict[str, Any]]
    ) -> List[Tuple[str, str]]:
        """
        根据 chunk_index 构建相邻关系对。

        Args:
            chunks: 已包含 doc_id 与 chunk_index 的块记录。

        Returns:
            List[Tuple[str, str]]: (前块ID, 后块ID) 对列表。
        """
        pairs: List[Tuple[str, str]] = []
        by_doc: Dict[str, List[Dict[str, Any]]] = {}
        for ch in chunks:
            by_doc.setdefault(ch["doc_id"], []).append(ch)
        for doc_id, items in by_doc.items():
            ordered = sorted(items, key=lambda x: x.get("chunk_index", -1))
            for prev, curr in zip(ordered, ordered[1:]):
                if prev.get("chunk_index", -1) >= 0 and curr.get("chunk_index", -1) >= 0:
                    pairs.append((prev["id"], curr["id"]))
        return pairs

    def _write_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        将块与文档关系写入 Neo4j。

        Args:
            chunks: 结构化的块记录列表。
        """
        query = """
        UNWIND $chunks AS ch
        MERGE (d:Document {doc_id: ch.doc_id})
        SET d.updated_at = timestamp()
        MERGE (c:Chunk {id: ch.id})
        SET c.doc_id = ch.doc_id,
            c.chunk_index = ch.chunk_index,
            c.total_chunks = ch.total_chunks,
            c.section = ch.section,
            c.page_number = ch.page_number
        MERGE (c)-[:PART_OF]->(d)
        """
        self.db.run_write(query, {"chunks": chunks})

    def _write_next_relations(self, pairs: List[Tuple[str, str]]) -> None:
        """
        写入相邻块的 NEXT 关系。

        Args:
            pairs: (前块ID, 后块ID) 对列表。
        """
        query = """
        UNWIND $pairs AS p
        MATCH (a:Chunk {id: p[0]}), (b:Chunk {id: p[1]})
        MERGE (a)-[:NEXT]->(b)
        """
        self.db.run_write(query, {"pairs": pairs})

