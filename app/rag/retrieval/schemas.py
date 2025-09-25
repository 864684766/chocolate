from __future__ import annotations

"""
检索层数据结构定义（Pydantic 模型）。

设计原则：
- 明确参数含义与单位，便于应用层编排与调试。
- 所有字段使用基础类型，方便序列化与日志记录。
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RetrievalQuery(BaseModel):
    """检索请求参数。

    Attributes:
        query: 用户查询文本（已做基础清洗）。
        where: 元数据过滤条件（ChromaDB 的 where 语法，字段需在 metadata_whitelist 内）。
        top_k: 召回数量上限。
        score_threshold: 分数阈值（低于阈值的结果将被丢弃）。
    """

    query: str = Field(..., description="查询文本")
    where: Optional[Dict[str, Any]] = Field(default=None, description="元数据过滤条件")
    top_k: int = Field(10, ge=1, le=1000, description="召回数量上限")
    score_threshold: float = Field(0.0, ge=0.0, le=1.0, description="分数阈值")


class RetrievedItem(BaseModel):
    """单条命中项。"""

    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """检索结果集。"""

    items: List[RetrievedItem] = Field(default_factory=list)
    latency_ms: int = 0
    debug_info: Dict[str, Any] = Field(default_factory=dict)
    applied_where: Optional[Dict[str, Any]] = Field(default=None, description="本次实际使用的 where 条件")
    matched_count: int = Field(0, ge=0, description="命中条数（未放宽）")


class BuiltContext(BaseModel):
    """上下文拼装结果，供 LLM 使用。"""

    text: str = ""
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    used_tokens: int = 0
    from_items: List[str] = Field(default_factory=list, description="来源 item 的 id 列表")


