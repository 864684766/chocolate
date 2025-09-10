from __future__ import annotations

"""
统一响应码与消息枚举

说明：
- 本文件定义了所有 API 的标准响应码与默认消息文案，避免各处硬编码。
- `ResponseCode` 采用 `IntEnum`（整数枚举），便于前端/日志/监控统计；
- `ResponseMessage` 采用 `Enum` + `str` 混合基类，表示“具备字符串值的枚举”。

兼容性：
- Python 3.11 起官方提供 `StrEnum`，在老版本可用 `class MyEnum(str, Enum)` 实现同等效果。
- 与 .NET 的 `enum` 概念类似：一组具名常量；Python 通过 `enum.Enum` 模块提供实现。
"""

from enum import IntEnum, Enum


class ResponseCode(IntEnum):
    """业务级响应码（整数型）。

    约定：
    - 0：通用成功
    - 1xxx：客户端/参数类错误
    - 2xxx：服务端/依赖类错误
    - 9999：未知错误
    """

    OK = 0  # 通用成功

    # 1xxx - 客户端错误/参数错误
    BAD_REQUEST = 1001  # 请求不合法（缺必填、格式不符等）
    VALIDATION_ERROR = 1002  # 语义校验失败（越界/冲突/业务校验不通过）

    # 2xxx - 服务端错误/依赖错误

    HEALTH_ERROR = 2000  # 健康检查失败
    DATABASE_ERROR = 2001  # 数据库/向量库等后端依赖错误
    UPSTREAM_ERROR = 2002  # 上游依赖（LLM/HTTP 服务等）调用异常

    # 9xxx - 兜底
    UNKNOWN_ERROR = 9999  # 未归类错误


class ResponseMessage(str, Enum):
    """默认响应文案（字符串枚举）。

    用途：
    - 统一“顶层 message”字段的默认文案；
    - 具体接口可在合适场景填入更详细的人类可读信息。
    """

    SUCCESS = "success"  # 成功
    BAD_REQUEST = "bad_request"  # 客户端请求不合法
    VALIDATION_ERROR = "validation_error"  # 参数/业务校验失败
    HEALTH_ERROR = "health_check_error"  # 健康检查失败
    DATABASE_ERROR = "database_error"  # 数据库/向量库错误
    UPSTREAM_ERROR = "upstream_error"  # 上游依赖调用异常
    UNKNOWN_ERROR = "unknown_error"  # 未知错误


