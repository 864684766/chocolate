"""语言解析器

负责从自然语言查询中推断语言类型，并生成对应的 where 条件。
"""

from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..where_parsers import ParserContext


def infer_lang(query: str, ctx: "ParserContext") -> str:
    """通用语言推断函数（仅启发式）。

    不依赖别名映射，直接以中文字符比例与阈值 `ctx.chinese_threshold` 推断；
    当无有效字符时返回 `ctx.fallback_language`。

    Args:
        query: 原始查询文本。
        ctx: 解析器上下文。

    Returns:
        str: 标准语言代码（如 "zh" 或 "en"）。
    """
    chinese = sum(1 for ch in query if '\u4e00' <= ch <= '\u9fff')
    letters = sum(1 for ch in query if ch.isalpha() or ('\u4e00' <= ch <= '\u9fff'))
    if letters == 0:
        return ctx.fallback_language
    return "zh" if (chinese / letters) > ctx.chinese_threshold else "en"


def lang_parser(query: str, ctx: "ParserContext") -> Optional[Dict[str, Any]]:
    """解析语言条件。

    使用通用 `infer_lang` 推断语言，仅当白名单包含 `lang` 时返回 where 子句。

    Args:
        query: 原始查询文本。
        ctx: 解析器上下文。

    Returns:
        {"lang": {"$eq": code}} 或 None。
    """
    code = infer_lang(query, ctx)
    return {"lang": {"$eq": code}} if code and "lang" in ctx.whitelist else None
