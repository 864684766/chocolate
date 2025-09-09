from typing import List
from app.rag.processing.interfaces import ProcessedChunk
from app.infra.database.chroma.db_helper import ChromaDBHelper
from .config import VectorizationConfig
from .embedder import Embedder
from .metadata_utils import build_metadata_from_meta


class VectorIndexer:
    """向量索引器：将文本块编码为向量并写入向量库。
    """

    def __init__(self, config: VectorizationConfig):
        self.config = config
        self.db = ChromaDBHelper()
        self.embedder = Embedder(config)

    def index_chunks(self, chunks: List[ProcessedChunk]) -> int:
        if not chunks:
            return 0
        texts = [c.text for c in chunks if isinstance(c.text, str)]
        embeddings = self.embedder.encode_parallel(texts)
        # 简化：用行号作为ids
        ids = [f"chunk_{i}" for i in range(len(texts))]
        metadatas = [build_metadata_from_meta(c.meta) for c in chunks][: len(texts)]
        self.db.add(
            collection_name=self.config.collection_name,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        return len(texts)


