"""标签/关键词解析器

负责从自然语言查询中抽取关键词，并生成对应的 where 条件。
"""

from typing import Callable, Dict, Any, Optional, TYPE_CHECKING
from .lang_parser import infer_lang

if TYPE_CHECKING:
    from ..where_parsers import ParserContext


def tags_contains_parser(field: str) -> Callable[[str, "ParserContext"], Optional[Dict[str, Any]]]:
    """动态解析标签/关键词字段（tags 或 keyphrases）。

    策略：
    1) 使用通用语言推断 infer_lang；
    2) 调用系统关键词抽取服务，从 query 中抽取 1~3 个关键词；
    3) 生成 `$and` + `$contains` 条件；
    4) 无关键词时回退到通用键值解析（field:value）。

    Args:
        field: 字段名（如 "tags" 或 "keyphrases"）。

    Returns:
        解析函数，返回 {"$and": [{field: {"$contains": kw}}, ...]} 或通用解析结果，均可能为 None。
    """

    def _parser(query: str, ctx: "ParserContext") -> Optional[Dict[str, Any]]:
        """解析标签/关键词条件。

        Args:
            query: 原始查询文本。
            ctx: 解析器上下文。

        Returns:
            包含关键词条件的 where 子句或 None。
        """
        if field not in ctx.whitelist:
            return None
        try:
            from app.rag.service.keywords_service import extract_keyphrases
            lang = infer_lang(query, ctx)
            kws = extract_keyphrases(query, lang=lang, topk=3)
            kws = [k.strip() for k in (kws or []) if k and isinstance(k, str)]
            if kws:
                return {"$and": [{field: {"$contains": k}} for k in kws]}
        except (ImportError, AttributeError, TypeError, ValueError):
            # 关键词服务不可用或参数错误时静默处理
            pass
        # 无关键词可用时，不构造条件，交由其他解析器或上层逻辑处理
        return None

    return _parser
