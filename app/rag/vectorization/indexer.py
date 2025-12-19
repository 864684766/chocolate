from typing import List, Optional, Dict
from app.rag.processing.interfaces import ProcessedChunk
from app.infra.database.chroma.db_helper import ChromaDBHelper
from .config import VectorizationConfig
from .embedder import Embedder
from .metadata_utils import build_metadata_from_meta
from .utils_writer import normalize_and_build_id, dedup_in_batch, slice_new_records, flatten_metadatas
from .meilisearch_indexer import MeilisearchIndexer
from .neo4j_indexer import Neo4jIndexer
from app.infra.logging import get_logger


class VectorIndexer:
    """向量索引器：将文本块编码为向量并写入向量库。

    职责：
    - 规范化文本与元数据，生成稳定ID（幂等）
    - 批内去重与库内过滤，避免重复写入
    - 对新增/需更新的条目进行向量编码与持久化
    - 同步写入 Meilisearch（如果启用）

    说明：
    - 文本规范化不进行长度截断，确保可检索文本完整
    - 稳定ID格式："{doc_id or filename}:{chunk_index}:{sha1(norm_text)[:16]}"
    - 统计日志字段：raw / batch_dedup / existed / written / meilisearch_synced
    - Meilisearch 同步失败不影响 ChromaDB 写入（错误隔离）
    """

    def __init__(self, config: VectorizationConfig):
        """初始化向量索引器。

        Args:
            config: 向量化配置对象
        """
        self.config = config
        self.db = ChromaDBHelper()
        self.embedder = Embedder(config)
        self.logger = get_logger(__name__)
        
        # 初始化 Meilisearch 索引器（如果启用）
        self._meili_indexer: Optional[MeilisearchIndexer] = None
        try:
            meili_indexer = MeilisearchIndexer()
            # 检查是否启用
            if meili_indexer.is_enabled():
                self._meili_indexer = meili_indexer
        except Exception as e:
            self.logger.warning(f"Meilisearch 索引器初始化失败，将跳过同步: {e}")
            self._meili_indexer = None
        # 初始化 Neo4j 索引器（如果启用）
        self._neo4j_indexer: Optional[Neo4jIndexer] = None
        try:
            neo4j_indexer = Neo4jIndexer()
            if neo4j_indexer.is_enabled():
                self._neo4j_indexer = neo4j_indexer
        except Exception as e:
            self.logger.warning(f"Neo4j 索引器初始化失败，将跳过图写入: {e}")
            self._neo4j_indexer = None

    def index_chunks(self, chunks: List[ProcessedChunk]) -> int:
        """写入（或更新）一批处理后的文本块。

        Args:
            chunks: 由处理流水线产生的块列表，每项含 `text` 与 `meta`。

        Returns:
            int: 实际写入（新插入或更新）的条目数量。

        Notes:
            - 幂等：相同来源/块序/内容的重复写入不会造成重复数据。
            - 性能：先批内去重，再与库对比过滤，仅对新增部分编码与持久化。
        """
        if not chunks:
            return 0

        # 准备数据：规范化、去重、对比
        prep_result = self._prepare_chunks(chunks)
        if not prep_result:
            return 0

        # 写入 ChromaDB
        written_n, updated_n = self._index_to_chromadb(
            prep_result["ids_new"], prep_result["texts_new"], prep_result["metadatas_new"],
            prep_result["ids_update"], prep_result["texts_update"], prep_result["metas_update"]
        )

        # 同步到 Meilisearch
        meili_synced = self._sync_to_meilisearch(
            prep_result["ids_new"], prep_result["texts_new"], prep_result["metadatas_new"],
            prep_result["ids_update"], prep_result["texts_update"], prep_result["metas_update"]
        )
        # 同步到 Neo4j
        neo4j_synced = self._sync_to_neo4j(
            prep_result["ids_new"], prep_result["metadatas_new"],
            prep_result["ids_update"], prep_result["metas_update"]
        )

        # 记录统计日志
        self._log_index_stats(
            prep_result["raw_n"], prep_result["after_batch_n"], prep_result["existed_n"],
            written_n, updated_n, meili_synced, neo4j_synced
        )

        return written_n + updated_n

    def _prepare_chunks(self, chunks: List[ProcessedChunk]) -> Optional[Dict]:
        """准备索引数据：规范化、去重、对比库内记录。

        Returns:
            dict: 包含 ids_new, texts_new, metadatas_new, ids_update, texts_update, metas_update,
                  raw_n, after_batch_n, existed_n, uniq。如果无需写入则返回 None。
        """
        # 文本/元数据统一化 + 生成稳定ID
        normalized = [normalize_and_build_id(c.text, c.meta) for c in chunks if isinstance(c.text, str)]
        uniq = dedup_in_batch(normalized)

        raw_n = len([c for c in chunks if isinstance(c.text, str)])
        after_batch_n = len(uniq)
        ids = list(uniq.keys())
        texts = [uniq[i][0] for i in ids]
        metadatas = flatten_metadatas(uniq, ids)

        if not ids:
            return None

        # 与库对比，过滤已存在的ID
        existed = self.db.get(
            collection_name=self.config.collection_name,
            ids=ids,
            include=[],
        )
        exist_ids = set(existed.get("ids", []))
        ids_new, texts_new, metadatas_new = slice_new_records(ids, exist_ids, texts, metadatas)
        existed_n = len(ids) - len(ids_new)

        # 构造更新集：对已存在ID，若内容或元数据变化则更新
        ids_update: List[str] = []
        texts_update: List[str] = []
        metas_update: List[dict] = []
        if exist_ids:
            existed_full = self.db.get(
                collection_name=self.config.collection_name,
                ids=list(exist_ids),
                include=["documents", "metadatas"],
            )
            old_docs_map = {i: d for i, d in zip(existed_full.get("ids", []), existed_full.get("documents", []))}
            old_metas_map = {i: m for i, m in zip(existed_full.get("ids", []), existed_full.get("metadatas", []))}
            for _id in exist_ids:
                new_text = uniq[_id][0]
                new_meta_flat = build_metadata_from_meta(uniq[_id][1])
                if old_docs_map.get(_id) != new_text or old_metas_map.get(_id) != new_meta_flat:
                    ids_update.append(_id)
                    texts_update.append(new_text)
                    metas_update.append(new_meta_flat)

        if not ids_new and not ids_update:
            self.logger.info(
                "vector index: raw=%s, batch_dedup=%s, existed=%s, written=0, updated=0",
                raw_n, after_batch_n, existed_n,
            )
            return None

        return {
            "ids_new": ids_new,
            "texts_new": texts_new,
            "metadatas_new": metadatas_new,
            "ids_update": ids_update,
            "texts_update": texts_update,
            "metas_update": metas_update,
            "raw_n": raw_n,
            "after_batch_n": after_batch_n,
            "existed_n": existed_n,
            "uniq": uniq,
        }

    def _index_to_chromadb(
        self,
        ids_new: List[str],
        texts_new: List[str],
        metadatas_new: List[dict],
        ids_update: List[str],
        texts_update: List[str],
        metas_update: List[dict],
    ) -> tuple[int, int]:
        """编码并写入 ChromaDB。

        Returns:
            (written_n, updated_n): 新增和更新的条目数量。
        """
        written_n = 0
        if ids_new:
            embeddings_new = self.embedder.encode_parallel(texts_new)
            self.db.add(
                collection_name=self.config.collection_name,
                documents=texts_new,
                embeddings=embeddings_new,
                metadatas=metadatas_new,
                ids=ids_new,
            )
            written_n = len(texts_new)

        updated_n = 0
        if ids_update:
            embeddings_upd = self.embedder.encode_parallel(texts_update)
            self.db.update(
                collection_name=self.config.collection_name,
                ids=ids_update,
                documents=texts_update,
                embeddings=embeddings_upd,
                metadatas=metas_update,
            )
            updated_n = len(ids_update)

        return written_n, updated_n

    def _sync_to_meilisearch(
        self,
        ids_new: List[str],
        texts_new: List[str],
        metadatas_new: List[dict],
        ids_update: List[str],
        texts_update: List[str],
        metas_update: List[dict],
    ) -> int:
        """同步数据到 Meilisearch（错误隔离：失败不影响 ChromaDB）。

        Returns:
            int: 同步成功的条目数量。
        """
        if not self._meili_indexer:
            return 0

        meili_synced = 0
        try:
            if ids_new:
                synced_new = self._meili_indexer.index_documents(ids_new, texts_new, metadatas_new)
                meili_synced += synced_new

            if ids_update:
                synced_upd = self._meili_indexer.update_documents(ids_update, texts_update, metas_update)
                meili_synced += synced_upd
        except Exception as e:
            self.logger.warning(f"Meilisearch 同步失败: {e}", exc_info=True)

        return meili_synced

    def _sync_to_neo4j(
        self,
        ids_new: List[str],
        metadatas_new: List[dict],
        ids_update: List[str],
        metas_update: List[dict],
    ) -> int:
        """
        同步数据到 Neo4j 图数据库（错误隔离）。

        Args:
            ids_new: 新增分块的 ID 列表。
            metadatas_new: 新增分块的元数据列表。
            ids_update: 需要更新的分块 ID 列表。
            metas_update: 需要更新的分块元数据列表。

        Returns:
            int: 实际写入或更新到图中的条数。
        """
        if not self._neo4j_indexer:
            return 0
        synced = 0
        try:
            if ids_new:
                synced += self._neo4j_indexer.index_chunks(ids_new, metadatas_new)
            if ids_update:
                synced += self._neo4j_indexer.index_chunks(ids_update, metas_update)
        except Exception as e:
            self.logger.warning(f"Neo4j 同步失败: {e}", exc_info=True)
        return synced

    def _log_index_stats(
        self,
        raw_n: int,
        after_batch_n: int,
        existed_n: int,
        written_n: int,
        updated_n: int,
        meili_synced: int,
        neo4j_synced: int,
    ) -> None:
        """
        记录索引统计日志。

        Args:
            raw_n: 原始块数量。
            after_batch_n: 批内去重后的数量。
            existed_n: 库内已存在的数量。
            written_n: 新写入数量。
            updated_n: 更新数量。
            meili_synced: Meilisearch 同步数量。
            neo4j_synced: Neo4j 同步数量。
        """
        log_msg = (
            f"vector index: raw={raw_n}, batch_dedup={after_batch_n}, "
            f"existed={existed_n}, written={written_n}, updated={updated_n}"
        )
        if meili_synced > 0:
            log_msg += f", meilisearch_synced={meili_synced}"
        if neo4j_synced > 0:
            log_msg += f", neo4j_synced={neo4j_synced}"
        self.logger.info(log_msg)


