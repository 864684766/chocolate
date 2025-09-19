from __future__ import annotations

"""
检索器实现（向量检索 / 关键词检索）。

说明：
- 应用层应先进行“智能路由”，选择使用 Vector / Keyword / Hybrid；
- 本文件仅提供执行能力，不负责决策。
"""

"""
统一导出与兼容保留。

说明：
- VectorRetriever 已迁移到 vector_retriever.py
- KeywordRetriever 已移除（建议改用 Meilisearch/自定义关键词检索器）
"""

from .vector_retriever import VectorRetriever  # noqa: F401


