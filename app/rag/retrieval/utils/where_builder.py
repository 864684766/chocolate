from __future__ import annotations

from typing import Dict, Any, Optional, List, Set, Tuple
import re

from app.config import get_config_manager
from .where_parsers import ParserContext, build_default_registry


class WhereBuilder:
    """从自然语言 query 推断严格的 Chroma where 条件。

    约束：
    - 仅使用 `metadata.metadata_whitelist` 中允许的字段
    - 不做任何放宽，未命中则返回 None
    - 方法体不超过 20 行，复杂逻辑拆分为私有辅助函数
    """

    # 质量阈值抽取正则（保留轻量内置规则）
    _QUALITY_PATTERNS = [
        re.compile(r"(?:质量|quality)\s*(?:>=|大于等于|至少)\s*([0-9.]+)", re.IGNORECASE),
        re.compile(r"(?:质量|quality)\s*(?:>|大于)\s*([0-9.]+)", re.IGNORECASE),
    ]

    def __init__(self) -> None:
        """初始化 WhereBuilder。

        动作：
        - 读取 `metadata.metadata_whitelist`（字段与类型）
        - 读取媒体类型映射与关键词表
        - 读取语言别名、中文阈值与回退语言
        不在此读取任何与 where_document 相关配置（保持简单）。
        """
        cfg = get_config_manager()
        self._whitelist, self._types_map = self._load_whitelist(cfg)
        self._media_types, self._media_keywords = self._load_media_types(cfg)
        self._lang_aliases = self._load_lang_aliases(cfg)
        det = (cfg.get_config("metadata") or {}).get("language_detection", {}) or {}
        self._ch_th = float(det.get("chinese_threshold", 0.3))
        self._fallback_lang = str(det.get("fallback_language", (cfg.get_config("metadata") or {}).get("default_language", "zh")))

    def build(self, query: str) -> Optional[Dict[str, Any]]:
        """根据 query 构建严格的 where 条件。

        规则：
        - 遍历白名单字段，通过解析器注册表动态解析；
        - 仅使用白名单字段，未命中则返回 None；
        - 不做放宽（零容忍回退）。

        Returns:
            Optional[Dict[str, Any]]: Chroma where 子句；无条件时返回 None。
        """
        if not isinstance(query, str) or not query.strip():
            return None
        clauses: List[Dict[str, Any]] = []

        # 注册表 + 上下文
        ctx = ParserContext(
            whitelist=self._whitelist,
            types_map=self._types_map,
            lang_aliases=self._lang_aliases,
            media_keywords=self._media_keywords,
            media_types=self._media_types,
            chinese_threshold=self._ch_th,
            fallback_language=self._fallback_lang,
        )
        registry = build_default_registry(ctx)
        for field in self._whitelist:
            clause = registry.parse(field, query, ctx)
            if clause:
                clauses.append(clause)

        if not clauses:
            return None
        where = {"$and": clauses} if len(clauses) > 1 else clauses[0]
        return self._filter_by_whitelist(where)

    def _is_allowed(self, field: str) -> bool:
        """字段是否在白名单内。"""
        return field in self._whitelist

    @staticmethod
    def _load_whitelist(cfg) -> Tuple[Set[str], Dict[str, str]]:
        """从配置读取白名单字段与类型映射。

        Args:
            cfg: 配置管理器实例。

        Returns:
            (fields, types_map):
                fields 为字段集合；types_map 为字段→类型映射（string/number/boolean/array）。
        """
        meta_cfg = cfg.get_config("metadata") or {}
        wl_raw = meta_cfg.get("metadata_whitelist", [])
        if wl_raw and isinstance(wl_raw[0], dict):
            fields = {str(x.get("field")) for x in wl_raw if x.get("field")}
            types = {str(x.get("field")): str(x.get("type")) for x in wl_raw if x.get("field")}
            return fields, types
        fields = set(str(x) for x in (wl_raw or []))
        return fields, {}

    @staticmethod
    def _load_media_types(cfg):
        """读取媒体类型集合与关键词映射（by_keyword）。

        Args:
            cfg: 配置管理器实例。

        Returns:
            (media_types, media_keywords):
                media_types 为允许的媒体类型集合；
                media_keywords 为关键词→标准类型的映射（小写键）。
        """
        meta_cfg = cfg.get_config("metadata") or {}
        m = meta_cfg.get("media_type_mapping", {}) or {}
        values = {str(m.get("default", "text"))}
        values.update(str(v) for v in (m.get("by_extension", {}) or {}).values())
        values.update(str(v) for v in (m.get("by_content_type", {}) or {}).values())
        # 关键词映射（可选）
        kw_map: Dict[str, str] = {}
        by_kw = m.get("by_keyword", {}) or {}
        for keys, val in by_kw.items():
            for k in str(keys).split(","):
                k2 = k.strip().lower()
                if k2:
                    kw_map[k2] = str(val)
        return values, kw_map

    @staticmethod
    def _load_lang_aliases(cfg) -> Dict[str, str]:
        """读取语言别名映射字典（小写）。

        Args:
            cfg: 配置管理器实例。

        Returns:
            Dict[str,str]: 别名→标准语言码的映射。
        """
        meta_cfg = cfg.get_config("metadata") or {}
        det = meta_cfg.get("language_detection", {}) or {}
        aliases: Dict[str, str] = {}
        for keys, val in (det.get("aliases", {}) or {}).items():
            for k in str(keys).split(","):
                k2 = k.strip().lower()
                if k2:
                    aliases[k2] = str(val)
        return aliases

    def _extract_lang(self, query: str) -> Optional[str]:
        """优先用别名，否则回退中文比例启发式。

        Args:
            query: 原始查询文本。

        Returns:
            str|None: 语言代码（zh/en）或 None。
        """
        lower = query.lower()
        for k, v in (self._lang_aliases or {}).items():
            if k in lower:
                return v
        return self._heuristic_lang(lower)

    def _extract_media(self, query: str) -> Optional[str]:
        """从关键词映射中解析媒体类型。

        Args:
            query: 原始查询文本。

        Returns:
            str|None: 标准媒体类型或 None。
        """
        lower = query.lower()
        # 仅使用配置关键词
        for k, v in (self._media_keywords or {}).items():
            if k in lower:
                return v
        return None

    @staticmethod
    def _heuristic_lang(lower_query: str) -> Optional[str]:
        """按中文比例启发式推断语言代码（zh/en）。

        Args:
            lower_query: 已转小写的查询文本。

        Returns:
            str|None: 语言代码（zh/en）或 None。
        """
        # 与 MetadataManager 一致的简化判断：中文字符比例
        meta_cfg = get_config_manager().get_config("metadata") or {}
        det = (meta_cfg.get("language_detection") or {})
        th = float(det.get("chinese_threshold", 0.3))
        # 统计中文与字母数量
        chinese = sum(1 for ch in lower_query if '\u4e00' <= ch <= '\u9fff')
        letters = sum(1 for ch in lower_query if ch.isalpha() or ('\u4e00' <= ch <= '\u9fff'))
        if letters == 0:
            return str(det.get("fallback_language", meta_cfg.get("default_language", "zh")))
        ratio = chinese / letters
        return "zh" if ratio > th else "en"

    # 通用键值解析已搬到注册表的通用解析器，不在此重复

    def _extract_quality_min(self, query: str) -> Optional[float]:
        """从 query 中解析质量阈值（0..1）。

        Args:
            query: 原始查询文本。

        Returns:
            float|None: 解析得到的最小阈值；未命中返回 None。
        """
        for pat in self._QUALITY_PATTERNS:
            m = pat.search(query)
            if m:
                try:
                    val = float(m.group(1))
                    if 0.0 <= val <= 1.0:
                        return val
                except ValueError:
                    return None
        return None

    @staticmethod
    def _extract_created_after(query: str) -> Optional[str]:
        """解析“近一周/近一月”为 ISO8601 起点（UTC）。

        Args:
            query: 原始查询文本。

        Returns:
            str|None: ISO8601 起点字符串，未命中返回 None。
        """
        # 简化实现：识别“近一周/近一月”，返回 ISO8601 起点（UTC 零点）
        from datetime import datetime, timedelta, timezone
        lower = query.lower()
        now = datetime.now(timezone.utc)
        if "近一周" in lower:
            return (now - timedelta(days=7)).isoformat()
        if "近一月" in lower or "近一个月" in lower:
            return (now - timedelta(days=30)).isoformat()
        return None

    def _filter_by_whitelist(self, where: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """递归过滤 where，确保仅包含白名单字段。

        Args:
            where: 构建完成的 where 子句（可能包含组合）。

        Returns:
            Dict|None: 过滤后的 where；若全部被过滤则返回 None。
        """
        def filt(obj: Any) -> Any:
            if isinstance(obj, dict):
                if "$and" in obj or "$or" in obj:
                    key = "$and" if "$and" in obj else "$or"
                    arr = [x for x in (obj.get(key) or [])]
                    return {key: [y for y in (filt(e) for e in arr) if y]}
                # 叶子：{ field: { op: value } }
                if len(obj) == 1:
                    f = list(obj.keys())[0]
                    if f in self._whitelist:
                        return obj
                    return None
            return None

        out = filt(where)
        if not out:
            return None
        # 清理空组合
        if "$and" in out and not out["$and"]:
            return None
        if "$or" in out and not out["$or"]:
            return None
        return out


