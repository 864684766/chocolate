from __future__ import annotations

"""
统一响应基类定义

说明：
- 所有 API 响应均使用 Envelope 外壳，便于统一观测与前端解析。
- 将通用数据结构（基础响应/分页元数据/分页响应）集中在本文件。
"""

from typing import Any, List, Optional
from pydantic import BaseModel, Field


class BaseResponse(BaseModel):
    """统一基础响应模型。

    Attributes:
        code: 业务状态码。0 表示成功，非 0 表示具体错误。
        message: 人类可读的结果描述。
        data: 业务数据载荷，成功时为对象/列表，失败时可为 None。
        request_id: 本次请求唯一 ID，用于链路追踪。
        timestamp: ISO8601 时间戳（UTC）。
        meta: 可选的额外元信息（分页统计等）。
    """

    code: int = Field(0, description="业务状态码，0表示成功")
    message: str = Field("success", description="结果描述")
    data: Optional[Any] = Field(None, description="业务数据")
    request_id: str = Field("", description="请求唯一ID")
    timestamp: str = Field("", description="ISO8601时间戳")
    meta: Optional[dict] = Field(default=None, description="额外元信息，例如分页统计")


class PageMeta(BaseModel):
    """分页元信息。"""

    page: int = Field(1, ge=1, description="当前页码，从1开始")
    size: int = Field(20, ge=1, description="每页条数")
    total: int = Field(0, ge=0, description="总条数")


class PageResponse(BaseResponse):
    """分页响应模型，data 通常为列表。"""

    data: List[Any] = Field(default_factory=list, description="数据列表")
    meta: PageMeta = Field(default_factory=PageMeta, description="分页元信息")


