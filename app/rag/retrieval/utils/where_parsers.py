from __future__ import annotations

from typing import Callable, Dict, Any, Optional
import re
from datetime import datetime, timedelta, timezone


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

def _parse_by_aliases(query: str, mapping: Dict[str, str]) -> Optional[str]:
    """在给定映射中匹配别名并返回标准值。

    策略：
    1) 优先进行“词边界/分词”匹配（英文/下划线单词、连续中文片段）；
    2) 其次对中文片段做 2~4 字 N-gram 命中；
    3) 最后仍未命中则返回 None。

    Args:
        query: 原始查询文本。
        mapping: 别名到标准值的映射字典（键应为小写或原样中文）。

    Returns:
        命中时返回标准值，未命中返回 None。
    """
    if not mapping:
        return None
    q = query.lower()
    # 1) 英文/下划线 token
    word_tokens = set(re.findall(r"[a-z_]+", q, flags=re.IGNORECASE))
    for alias, val in mapping.items():
        a = alias.lower()
        if a and a.isascii():
            if a in word_tokens:
                return val
    # 2) 中文连续片段与 N-gram
    cjk_spans = re.findall(r"[\u4e00-\u9fff]+", query)
    if cjk_spans:
        # 直接片段包含
        for span in cjk_spans:
            for alias, val in mapping.items():
                if any(ord(ch) >= 0x4e00 and ord(ch) <= 0x9fff for ch in alias):
                    if alias in span:
                        return val
        # N-gram 命中（2~4）
        grams = set()
        for span in cjk_spans:
            n = len(span)
            for g in range(2, min(4, n) + 1):
                for i in range(0, n - g + 1):
                    grams.add(span[i : i + g])
        for alias, val in mapping.items():
            if alias in grams:
                return val
    return None


def infer_lang(query: str, ctx: ParserContext) -> str:
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


def lang_parser(query: str, ctx: ParserContext) -> Optional[Dict[str, Any]]:
    """解析语言条件。

    使用通用 `infer_lang` 推断语言，仅当白名单包含 `lang` 时返回 where 子句。

    Returns:
        {"lang": {"$eq": code}} 或 None。
    """
    code = infer_lang(query, ctx)
    return {"lang": {"$eq": code}} if code and "lang" in ctx.whitelist else None


def media_type_parser(query: str, ctx: ParserContext) -> Optional[Dict[str, Any]]:
    """解析媒体类型条件。

    从关键词映射得到标准媒体类型，校验在受支持集合内。

    Returns:
        {"media_type": {"$eq": value}} 或 None。
    """
    mt = _parse_by_aliases(query, ctx.media_keywords)
    if mt and mt in ctx.media_types and "media_type" in ctx.whitelist:
        return {"media_type": {"$eq": mt}}
    return None


_QUALITY_PATTERNS = [
    re.compile(r"(?:质量|quality)\s*(?:>=|大于等于|至少)\s*([0-9.]+)", re.IGNORECASE),
    re.compile(r"(?:质量|quality)\s*(?:>|大于)\s*([0-9.]+)", re.IGNORECASE),
]


def quality_score_parser(query: str, ctx: ParserContext) -> Optional[Dict[str, Any]]:
    """解析质量阈值条件。

    支持“质量>=x / 大于等于 x / at least x / >x / 大于 x”的形式，范围在 [0,1]。

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


def created_at_parser(query: str, ctx: ParserContext) -> Optional[Dict[str, Any]]:
    """解析时间范围条件。

    识别“近一周/近一月/近一个月”，返回起始 ISO8601（UTC）。

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


def tags_contains_parser(field: str) -> Callable[[str, ParserContext], Optional[Dict[str, Any]]]:
    """动态解析标签/关键词字段（tags 或 keyphrases）。

    策略：
    1) 使用通用语言推断 infer_lang；
    2) 调用系统关键词抽取服务，从 query 中抽取 1~3 个关键词；
    3) 生成 `$and` + `$contains` 条件；
    4) 无关键词时回退到通用键值解析（field:value）。

    Returns:
        {"$and": [{field: {"$contains": kw}}, ...]} 或通用解析结果，均可能为 None。
    """

    def _parser(query: str, ctx: ParserContext) -> Optional[Dict[str, Any]]:
        if field not in ctx.whitelist:
            return None
        try:
            from app.rag.service.keywords_service import extract_keyphrases
            lang = infer_lang(query, ctx)
            kws = extract_keyphrases(query, lang=lang, topk=3)
            kws = [k.strip() for k in (kws or []) if k and isinstance(k, str)]
            if kws:
                return {"$and": [{field: {"$contains": k}} for k in kws]}
        except Exception:
            pass
        # 无关键词可用时，不构造条件，交由其他解析器或上层逻辑处理
        return None

    return _parser


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



