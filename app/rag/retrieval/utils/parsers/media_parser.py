"""媒体类型解析器

负责从自然语言查询中识别媒体类型关键词，并生成对应的 where 条件。
"""

from typing import Dict, Any, Optional, TYPE_CHECKING
from .common import parse_by_aliases

if TYPE_CHECKING:
    from ..where_parsers import ParserContext


def media_type_parser(query: str, ctx: "ParserContext") -> Optional[Dict[str, Any]]:
    """解析媒体类型条件。

    从关键词映射得到标准媒体类型，校验在受支持集合内。

    Args:
        query: 原始查询文本。
        ctx: 解析器上下文。

    Returns:
        {"media_type": {"$eq": value}} 或 None。
    """
    mt = parse_by_aliases(query, ctx.media_keywords)
    if mt and mt in ctx.media_types and "media_type" in ctx.whitelist:
        return {"media_type": {"$eq": mt}}
    return None
