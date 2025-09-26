"""时间范围解析器

负责从自然语言查询中识别时间范围条件，并生成对应的 where 条件。
"""

from typing import Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime, timedelta, timezone

if TYPE_CHECKING:
    from ..where_parsers import ParserContext


def created_at_parser(query: str, ctx: "ParserContext") -> Optional[Dict[str, Any]]:
    """解析时间范围条件。

    识别"近一周/近一月/近一个月"，返回起始 ISO8601（UTC）。

    Args:
        query: 原始查询文本。
        ctx: 解析器上下文。

    Returns:
        {"created_at": {"$gte": iso8601}} 或 None。
    """
    if "created_at" not in ctx.whitelist:
        return None
    lower = query.lower()
    now = datetime.now(timezone.utc)
    if "近一周" in lower:
        return {"created_at": {"$gte": (now - timedelta(days=7)).isoformat()}}
    if ("近一月" in lower) or ("近一个月" in lower):
        return {"created_at": {"$gte": (now - timedelta(days=30)).isoformat()}}
    return None
