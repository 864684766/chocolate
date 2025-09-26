from __future__ import annotations

from typing import Callable, Dict, Any, Optional
from .parsers import (
    lang_parser,
    media_type_parser,
    quality_score_parser,
    created_at_parser,
    tags_contains_parser,
)


class ParserContext:
    """解析器上下文

    保存解析 where 所需的环境信息，避免解析器直接依赖配置读取。

    Attributes:
        whitelist: 允许参与 where 的字段集合
        types_map: 字段到类型的映射（string/number/boolean/array）
        lang_aliases: 语言别名到标准代码的映射
        media_keywords: 关键词到标准媒体类型的映射
        media_types: 支持的媒体类型集合（用于校验）
        chinese_threshold: 中文比例阈值（用于启发式语言识别）
        fallback_language: 语言识别失败时的回退语言
    """

    def __init__(
        self,
        whitelist: set[str],
        types_map: Dict[str, str],
        lang_aliases: Dict[str, str],
        media_keywords: Dict[str, str],
        media_types: set[str],
        chinese_threshold: float,
        fallback_language: str,
    ) -> None:
        """初始化解析器上下文。

        Args:
            whitelist: 允许用于 where 构建的字段集合。
            types_map: 字段→类型 映射（string/number/boolean/array）。
            lang_aliases: 语言别名映射（小写键），用于将自然语言别名映射为标准语言代码。
            media_keywords: 媒体类型关键词映射（小写键），用于将关键词映射为标准媒体类型。
            media_types: 媒体类型允许值集合（text/image/video/audio/pdf 等）。
            chinese_threshold: 中文字符比例阈值（0.0~1.0），用于启发式语言识别。
            fallback_language: 当无法识别语言时的回退语言代码。

        Returns:
            None
        """
        self.whitelist = whitelist
        self.types_map = types_map
        self.lang_aliases = lang_aliases
        self.media_keywords = media_keywords
        self.media_types = media_types
        self.chinese_threshold = chinese_threshold
        self.fallback_language = fallback_language


class ParserRegistry:
    """字段解析器注册表

    负责维护“字段名 → 解析函数”的映射。WhereBuilder 通过注册表获取解析函数，
    从而避免在构建逻辑中出现大量 if/else。

    解析函数签名：
        parser_fn(query: str, ctx: ParserContext) -> Optional[Dict[str, Any]]
    返回值为单个字段的 where 子句，如 {"lang": {"$eq": "zh"}}；
    当无法解析时返回 None。
    """

    def __init__(self) -> None:
        self._parsers: Dict[str, Callable[[str, ParserContext], Optional[Dict[str, Any]]]] = {}

    def register(self, field: str, parser_fn: Callable[[str, ParserContext], Optional[Dict[str, Any]]]) -> None:
        """注册字段解析器。

        Args:
            field: 字段名（需在白名单内）
            parser_fn: 解析函数
        Returns:
            None
        """
        self._parsers[field] = parser_fn

    def parse(self, field: str, query: str, ctx: ParserContext) -> Optional[Dict[str, Any]]:
        """调用已注册的解析器。

        Args:
            field: 字段名
            query: 原始查询文本
            ctx: 解析上下文
        Returns:
            Optional[Dict[str, Any]]: 解析得到的 where 子句；无法解析时返回 None
        """
        func = self._parsers.get(field)
        return func(query, ctx) if func else None


# ----------------- 默认解析器实现 -----------------


def build_default_registry(ctx: ParserContext) -> ParserRegistry:
    """创建默认注册表。

    规则：
    - 为每个白名单字段注册解析器；
    - 已知语义字段（lang/media_type/quality_score/created_at/tags/keyphrases）使用专用解析器；
    - 其他字段默认注册通用键值解析器。

    Args:
        ctx: 解析器上下文。

    Returns:
        ParserRegistry: 预注册好的解析器集合。
    """
    reg = ParserRegistry()
    reg.register("lang", lang_parser)
    reg.register("media_type", media_type_parser)
    reg.register("quality_score", quality_score_parser)
    reg.register("created_at", created_at_parser)
    for array_field in ("tags", "keyphrases"):
        if array_field in ctx.whitelist:
            reg.register(array_field, tags_contains_parser(array_field))
    # 不注册通用键值解析器；自然语言场景下无此需求
    return reg



