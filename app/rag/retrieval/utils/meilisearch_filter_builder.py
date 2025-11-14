"""Meilisearch filter 构建器：直接从元数据字段生成 Meilisearch filter。

职责：
- 根据元数据白名单和查询条件，直接生成 Meilisearch filter 表达式
- 避免依赖 ChromaDB where 格式的转换
- 支持 Meilisearch 的原生过滤能力

说明：
- 与 WhereBuilder 类似，但直接生成 Meilisearch 格式
- 可以复用 WhereBuilder 的解析逻辑，但输出格式不同
"""

from typing import Dict, Any, Optional, List, Set
from app.config import get_config_manager
from app.rag.retrieval.utils.where_parsers import ParserContext, build_default_registry


class MeilisearchFilterBuilder:
    """Meilisearch filter 构建器：从查询文本直接生成 Meilisearch filter。

    用途：
    - 避免 ChromaDB where 格式的转换
    - 直接利用 Meilisearch 的过滤能力
    - 与 WhereBuilder 并行，但输出 Meilisearch 格式

    配置：
    - 使用与 WhereBuilder 相同的元数据白名单和解析器
    """

    def __init__(self):
        """初始化 MeilisearchFilterBuilder。

        读取配置：
        - metadata.metadata_whitelist：元数据字段白名单
        - metadata.language_detection：语言检测配置
        - metadata.media_type_mapping：媒体类型映射
        """
        cfg = get_config_manager()
        self._whitelist, self._types_map = self._load_whitelist(cfg)
        self._media_types, self._media_keywords = self._load_media_types(cfg)
        self._lang_aliases = self._load_lang_aliases(cfg)
        det = (cfg.get_config("metadata") or {}).get("language_detection", {}) or {}
        self._ch_th = float(det.get("chinese_threshold", 0.3))
        self._fallback_lang = str(
            det.get("fallback_language", (cfg.get_config("metadata") or {}).get("default_language", "zh"))
        )

    def build_from_query(self, query: str) -> Optional[List[str]]:
        """从查询文本直接生成 Meilisearch filter。

        Args:
            query: 用户查询文本

        Returns:
            Optional[List[str]]: Meilisearch filter 表达式列表；无条件时返回 None
        """
        if not isinstance(query, str) or not query.strip():
            return None

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

        filters: List[str] = []
        for field in self._whitelist:
            clause = registry.parse(field, query, ctx)
            if clause:
                # 将 ChromaDB 格式的 clause 转换为 Meilisearch filter
                meili_filter = self._clause_to_filter(clause)
                if meili_filter:
                    filters.append(meili_filter)

        return filters if filters else None

    def build_from_where(self, where: Optional[Dict[str, Any]]) -> Optional[List[str]]:
        """从 ChromaDB where 条件生成 Meilisearch filter（兼容模式）。

        Args:
            where: ChromaDB 格式的 where 条件

        Returns:
            Optional[List[str]]: Meilisearch filter 表达式列表；无条件时返回 None
        """
        if not where:
            return None

        filters: List[str] = []
        self._convert_where_node(where, filters)
        return filters if filters else None

    def _clause_to_filter(self, clause: Dict[str, Any]) -> Optional[str]:
        """将单个 ChromaDB clause 转换为 Meilisearch filter 表达式。

        Args:
            clause: ChromaDB 格式的单个字段条件，如 {"lang": {"$eq": "zh"}}

        Returns:
            Optional[str]: Meilisearch filter 表达式，如 "lang = 'zh'"
        """
        if not clause or len(clause) != 1:
            return None

        field = list(clause.keys())[0]
        condition = clause[field]

        if isinstance(condition, dict):
            for op, value in condition.items():
                if op == "$eq":
                    return f"{field} = {self._format_value(value)}"
                elif op == "$ne":
                    return f"{field} != {self._format_value(value)}"
                elif op == "$gt":
                    return f"{field} > {self._format_value(value)}"
                elif op == "$gte":
                    return f"{field} >= {self._format_value(value)}"
                elif op == "$lt":
                    return f"{field} < {self._format_value(value)}"
                elif op == "$lte":
                    return f"{field} <= {self._format_value(value)}"
                elif op == "$in":
                    if isinstance(value, list) and value:
                        values_str = ", ".join(self._format_value(v) for v in value)
                        return f"{field} IN [{values_str}]"
                elif op == "$contains":
                    return f"{field} IN [{self._format_value(value)}]"
        else:
            # 简写形式
            return f"{field} = {self._format_value(condition)}"

        return None

    def _convert_where_node(self, node: Dict[str, Any], filters: List[str]) -> None:
        """递归转换 where 节点（兼容 ChromaDB 格式）。

        Args:
            node: where 条件节点
            filters: 累积的 filter 表达式列表
        """
        if "$and" in node:
            for clause in node["$and"]:
                self._convert_where_node(clause, filters)
            return

        if "$or" in node:
            or_filters: List[str] = []
            for clause in node["$or"]:
                clause_filters: List[str] = []
                self._convert_where_node(clause, clause_filters)
                if clause_filters:
                    if len(clause_filters) == 1:
                        or_filters.append(clause_filters[0])
                    else:
                        or_filters.append(f"({' AND '.join(clause_filters)})")
            if or_filters:
                filters.append(f"({' OR '.join(or_filters)})")
            return

        # 处理字段条件
        for field, condition in node.items():
            if isinstance(condition, dict):
                for op, value in condition.items():
                    if op == "$eq":
                        filters.append(f"{field} = {self._format_value(value)}")
                    elif op == "$ne":
                        filters.append(f"{field} != {self._format_value(value)}")
                    elif op == "$gt":
                        filters.append(f"{field} > {self._format_value(value)}")
                    elif op == "$gte":
                        filters.append(f"{field} >= {self._format_value(value)}")
                    elif op == "$lt":
                        filters.append(f"{field} < {self._format_value(value)}")
                    elif op == "$lte":
                        filters.append(f"{field} <= {self._format_value(value)}")
                    elif op == "$in":
                        if isinstance(value, list) and value:
                            values_str = ", ".join(self._format_value(v) for v in value)
                            filters.append(f"{field} IN [{values_str}]")
                    elif op == "$contains":
                        filters.append(f"{field} IN [{self._format_value(value)}]")
            else:
                filters.append(f"{field} = {self._format_value(condition)}")

    def _format_value(self, value: Any) -> str:
        """格式化值为 Meilisearch filter 中的字符串表示。

        Args:
            value: 要格式化的值

        Returns:
            str: 格式化后的值字符串
        """
        if isinstance(value, str):
            escaped = value.replace("'", "\\'")
            return f"'{escaped}'"
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, (int, float)):
            return str(value)
        else:
            escaped = str(value).replace("'", "\\'")
            return f"'{escaped}'"

    @staticmethod
    def _load_whitelist(cfg) -> tuple[Set[str], Dict[str, str]]:
        """从配置读取白名单字段与类型映射。

        Args:
            cfg: 配置管理器实例

        Returns:
            (fields, types_map): 字段集合和类型映射
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
        """读取媒体类型集合与关键词映射。

        Args:
            cfg: 配置管理器实例

        Returns:
            (media_types, media_keywords): 媒体类型集合和关键词映射
        """
        meta_cfg = cfg.get_config("metadata") or {}
        m = meta_cfg.get("media_type_mapping", {}) or {}
        values = {str(m.get("default", "text"))}
        values.update(str(v) for v in (m.get("by_extension", {}) or {}).values())
        values.update(str(v) for v in (m.get("by_content_type", {}) or {}).values())
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
        """读取语言别名映射字典。

        Args:
            cfg: 配置管理器实例

        Returns:
            Dict[str, str]: 别名到标准语言码的映射
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

