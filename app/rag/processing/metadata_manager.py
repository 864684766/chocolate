from __future__ import annotations

from typing import Dict, Any, List, Optional

from app.config import get_config_manager
from app.rag.service.keywords_service import extract_keyphrases
from app.rag.processing.quality_checker import SimpleQualityAssessor


class MetadataManager:
    """统一的元数据管理器
    
    作用：读取配置中的 metadata_whitelist，根据白名单字段
    将外部传入的 meta 规范化；不会生成白名单之外的字段，也不会兜底。
    """
    
    def __init__(self):
        """
        初始化元数据管理器
        
        加载配置并初始化相关组件
        """
        self.config = self._load_config()
        self.whitelist: List[str] = self.config.get("whitelist_fields", [])
        self.types_map: Dict[str, str] = self.config.get("whitelist_types", {})
        self.quality_assessor = SimpleQualityAssessor()
    
    def build_metadata(self,
                       meta: Dict[str, Any],
                       text: Optional[str] = None) -> Dict[str, Any]:
        """
        按白名单生成元数据
        
        Args:
            meta: 外部传入的原始元数据（任意结构）
            text: 可选的文本内容，用于生成 lang/text_len/keyphrases/quality_score 等派生字段
            
        Returns:
            Dict[str, Any]: 只包含白名单字段的元数据
        """
        if not self.whitelist:
            return {}
        
        base_meta = dict(meta or {})
        normalized: Dict[str, Any] = {}
        
        # 第一遍：处理白名单字段，使用传入值或默认值
        for field in self.whitelist:
            if field in base_meta and base_meta.get(field) not in (None, ""):
                coerced = self._coerce_value(field, base_meta.get(field))
                if coerced is not None and coerced != normalized.get(field):
                    normalized[field] = coerced
                    continue
            else:
                normalized[field] = self._default_value(field)
        
        # 第二遍：如果有文本，生成文本相关字段并覆盖默认值
        if text:
            text_fields = self._build_text_dependent_fields(text, normalized)
            normalized.update(text_fields)
        
        return normalized
    
    def iter_fields(self):
        """
        返回当前白名单字段迭代器
        """
        return iter(self.whitelist)
    
    def build_upload_metadata(self,
                              filename: str,
                              content_type: str,
                              source: str,
                              extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        为上传场景构建统一的基础元数据
        
        Args:
            filename: 文件名
            content_type: MIME 类型
            source: 数据来源
            extra: 额外的元数据（可选）
        
        Returns:
            Dict[str, Any]: 经过白名单过滤的元数据
        """
        payload: Dict[str, Any] = dict(extra or {})
        payload["filename"] = filename
        payload["content_type"] = content_type
        payload["source"] = source
        return self.build_metadata(payload)
    
    def _build_text_dependent_fields(self, text: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成依赖文本内容的字段
        
        Args:
            text: 当前 chunk 的文本
            meta: 已经规范化的元数据（用于质量评估参考）
        
        Returns:
            Dict[str, Any]: 仅包含白名单允许的文本派生字段
        """
        fields: Dict[str, Any] = {}
        lang_value: Optional[str] = meta.get("lang")
        
        if "lang" in self.whitelist:
            lang_value = self._detect_language(text) if not lang_value else str(lang_value)
            fields["lang"] = lang_value
        
        if "text_len" in self.whitelist:
            fields["text_len"] = len(text)
        
        if "keyphrases" in self.whitelist:
            phrases = self._extract_keyphrases(text, lang_value or self.config.get("default_language", "zh"))
            if phrases:
                fields["keyphrases"] = phrases
        
        if "quality_score" in self.whitelist:
            quality = self.quality_assessor.score(text, {**meta, **fields})
            score = quality.get("quality_score")
            if isinstance(score, (int, float)):
                fields["quality_score"] = float(score)
        
        return fields
    
    def _coerce_value(self, field: str, value: Any) -> Any:
        """
        将字段值按白名单类型转换
        
        Args:
            field: 字段名称
            value: 原始值
        
        Returns:
            Any: 转换后的值，若无法转换则返回 None
        """
        if value is None:
            return None
        field_type = self.types_map.get(field)
        if field_type == "number":
            if isinstance(value, (int, float)):
                return value
            try:
                num = float(str(value).strip())
                return int(num) if num.is_integer() else num
            except (ValueError, TypeError):
                return None
        if field_type == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "1", "yes"}:
                    return True
                if lowered in {"false", "0", "no"}:
                    return False
            return None
        if field_type == "array":
            if isinstance(value, (list, tuple)):
                items = [str(item).strip() for item in value if str(item).strip()]
                return items or None
            item = str(value).strip()
            return [item] if item else None
        # string 或未声明类型：统一转字符串
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        return str(value)
    
    def _default_value(self, field: str) -> Any:
        field_type = self.types_map.get(field, "string")
        if field_type == "number":
            return 0
        if field_type == "boolean":
            return False
        if field_type == "array":
            return []
        return ""
    
    def _detect_language(self, text: str) -> str:
        """
        检测文本语言
        
        Args:
            text: 文本内容
            
        Returns:
            str: 语言代码
        """
        min_len = int(self.config.get("language_detection", {}).get("min_text_length", 10))
        fallback_lang = str(self.config.get("language_detection", {}).get(
            "fallback_language", self.config.get("default_language", "zh")
        ))
        if not text or len(text.strip()) < min_len:
            return fallback_lang
        
        # 简单的语言检测逻辑
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        total_chars = len([c for c in text if c.isalpha() or '\u4e00' <= c <= '\u9fff'])
        
        if total_chars == 0:
            return fallback_lang
        
        chinese_ratio = chinese_chars / total_chars
        chinese_threshold = self.config.get("language_detection", {}).get("chinese_threshold", 0.3)
        
        if chinese_ratio > chinese_threshold:
            return "zh"
        else:
            return "en"
    
    @staticmethod
    def _extract_keyphrases(text: str, lang: str) -> list[str]:
        """
        提取关键词
        
        Args:
            text: 文本内容
            lang: 语言代码
            
        Returns:
            list[str]: 关键词列表
        """
        if not text or len(text.strip()) < 20:
            return []
        
        try:
            return extract_keyphrases(text, lang=lang)
        except (ValueError, TypeError, ImportError) as e:
            # 记录具体的异常类型，避免过于宽泛的异常捕获
            print(f"关键词提取失败: {e}")
            return []
    
    @staticmethod
    def _load_config() -> Dict[str, Any]:
        """
        加载配置
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        config_manager = get_config_manager()
        
        # 从配置文件获取元数据管理配置
        metadata_config = config_manager.get_config("metadata") or {}
        whitelist_raw = metadata_config.get("metadata_whitelist", [])
        if whitelist_raw and isinstance(whitelist_raw[0], dict):
            whitelist = [str(item.get("field")) for item in whitelist_raw if item.get("field")]
            types_map = {str(item.get("field")): str(item.get("type")) for item in whitelist_raw if item.get("field")}
        else:
            whitelist = [str(item) for item in whitelist_raw]
            types_map = {}
        
        metadata_config = metadata_config.copy()
        metadata_config["whitelist_fields"] = whitelist
        metadata_config["whitelist_types"] = types_map
        
        if "language_detection" not in metadata_config:
            metadata_config["language_detection"] = {
                "chinese_threshold": 0.3,
                "min_text_length": 10
            }
        if "default_language" not in metadata_config:
            metadata_config["default_language"] = "zh"
        
        return metadata_config
