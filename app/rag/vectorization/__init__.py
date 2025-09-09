"""
向量化层模块入口。

职责：
- 定义并暴露向量化配置、嵌入器与索引器。
- 仅负责“文本→向量→入库”，不含检索/重排逻辑。
"""

from .config import VectorizationConfig
from .embedder import Embedder
from .indexer import VectorIndexer

__all__ = [
    "VectorizationConfig",
    "Embedder",
    "VectorIndexer",
]


