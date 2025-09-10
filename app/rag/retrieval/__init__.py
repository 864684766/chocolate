"""
检索层模块导出。

说明：
- 该包聚合向量检索、关键词检索、混合融合、上下文构建、（可选）重排等能力。
- 应用层负责“智能路由/编排”，根据需求选择具体检索策略与参数，再调用本层能力执行。
"""

from .schemas import RetrievalQuery, RetrievedItem, RetrievalResult, BuiltContext
from .retriever import VectorRetriever, KeywordRetriever
from .hybrid import HybridSearcher
from .context_builder import ContextBuilder

__all__ = [
    "RetrievalQuery",
    "RetrievedItem",
    "RetrievalResult",
    "BuiltContext",
    "VectorRetriever",
    "KeywordRetriever",
    "HybridSearcher",
    "ContextBuilder",
]


