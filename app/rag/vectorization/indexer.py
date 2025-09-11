from typing import List
from app.rag.processing.interfaces import ProcessedChunk
from app.infra.database.chroma.db_helper import ChromaDBHelper
from .config import VectorizationConfig
from .embedder import Embedder
from .metadata_utils import build_metadata_from_meta
from .utils_writer import normalize_and_build_id, dedup_in_batch, slice_new_records, flatten_metadatas
from app.infra.logging import get_logger


class VectorIndexer:
    """向量索引器：将文本块编码为向量并写入向量库。

    职责：
    - 规范化文本与元数据，生成稳定ID（幂等）
    - 批内去重与库内过滤，避免重复写入
    - 对新增/需更新的条目进行向量编码与持久化

    说明：
    - 文本规范化不进行长度截断，确保可检索文本完整
    - 稳定ID格式："{doc_id or filename}:{chunk_index}:{sha1(norm_text)[:16]}"
    - 统计日志字段：raw / batch_dedup / existed / written
    """

    def __init__(self, config: VectorizationConfig):
        self.config = config
        self.db = ChromaDBHelper()
        self.embedder = Embedder(config)
        self.logger = get_logger(__name__)

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
        # 1) 文本/元数据统一化 + 生成稳定ID
        normalized = [normalize_and_build_id(c.text, c.meta) for c in chunks if isinstance(c.text, str)]

        # 2) 批内去重（按稳定ID）
        uniq = dedup_in_batch(normalized)

        raw_n = len([c for c in chunks if isinstance(c.text, str)])
        after_batch_n = len(uniq)

        ids = list(uniq.keys())
        texts = [uniq[i][0] for i in ids]
        # 扁平化 metadatas（白名单展开）
        metadatas = flatten_metadatas(uniq, ids)

        if not ids:
            return 0

        # 3) 与库对比，过滤已存在的ID
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
            # 读取现有记录的文档与元数据用于对比
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
            # 统计日志
            self.logger.info(
                "vector index: raw=%s, batch_dedup=%s, existed=%s, written=0, updated=0",
                raw_n, after_batch_n, existed_n,
            )
            return 0

        # 4) 编码并写入
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
        self.logger.info(
            "vector index: raw=%s, batch_dedup=%s, existed=%s, written=%s, updated=%s",
            raw_n, after_batch_n, existed_n, written_n, updated_n,
        )
        return written_n + updated_n


