"""质量分数解析器

负责从自然语言查询中识别质量分数条件，并生成对应的 where 条件。
"""

from typing import Dict, Any, Optional, TYPE_CHECKING
import re

if TYPE_CHECKING:
    from ..where_parsers import ParserContext


# 质量分数匹配模式
_QUALITY_PATTERNS = [
    re.compile(r"(?:质量|quality)\s*(?:>=|大于等于|至少)\s*([0-9.]+)", re.IGNORECASE),
    re.compile(r"(?:质量|quality)\s*(?:>|大于)\s*([0-9.]+)", re.IGNORECASE),
]


def quality_score_parser(query: str, ctx: "ParserContext") -> Optional[Dict[str, Any]]:
    """解析质量阈值条件。

    支持"质量>=x / 大于等于 x / at least x / >x / 大于 x"的形式，范围在 [0,1]。

    Args:
        query: 原始查询文本。
        ctx: 解析器上下文。

    Returns:
        {"quality_score": {"$gte": x}} 或 None。
    """
    if "quality_score" not in ctx.whitelist:
        return None
    for pat in _QUALITY_PATTERNS:
        m = pat.search(query)
        if m:
            try:
                val = float(m.group(1))
                if 0.0 <= val <= 1.0:
                    return {"quality_score": {"$gte": val}}
            except ValueError:
                return None
    return None
