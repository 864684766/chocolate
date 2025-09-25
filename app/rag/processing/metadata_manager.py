from __future__ import annotations

from typing import Dict, Any, Optional
import hashlib
import time
from datetime import datetime, timezone

from app.config import get_config_manager
from app.rag.service.keywords_service import extract_keyphrases
from app.rag.processing.quality_checker import SimpleQualityAssessor


class MetadataManager:
    """统一的元数据管理器
    
    负责整个数据处理流程中的所有元数据操作：
    - 基础元数据生成 (doc_id, filename, source, created_at)
    - 内容分析元数据 (lang, text_len, keyphrases)
    - 质量评估元数据 (quality_score)
    - 元数据规范化 (白名单过滤, 类型转换)
    - 元数据验证和补全
    """
    
    def __init__(self):
        """
        初始化元数据管理器
        
        加载配置并初始化相关组件
        """
        self.config = self._load_config()
        self.quality_assessor = SimpleQualityAssessor()
    
    def create_metadata(self, 
                       text: str = "",
                       filename: Optional[str] = None,
                       content_type: Optional[str] = None,
                       source: Optional[str] = None) -> Dict[str, Any]:
        """
        创建完整的元数据
        
        Args:
            text: 文本内容
            filename: 文件名
            content_type: 内容类型
            source: 数据源
            
        Returns:
            Dict[str, Any]: 完整的元数据
        """
        meta = {}
        
        # 1. 基础元数据
        meta.update(self._create_basic_metadata(filename, content_type, source))
        
        # 2. 内容分析元数据
        if text:
            meta.update(self._analyze_content_metadata(text))
        
        # 3. 规范化元数据（暂时跳过，避免循环导入）
        # meta = self._normalize_metadata(meta)
        
        return meta
    
    def update_content_metadata(self, text: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新内容相关的元数据
        
        Args:
            text: 文本内容
            meta: 现有元数据
            
        Returns:
            Dict[str, Any]: 更新后的元数据
        """
        updated_meta = meta.copy()
        
        # 更新内容分析元数据
        content_meta = self._analyze_content_metadata(text)
        updated_meta.update(content_meta)
        
        # 质量评估元数据
        quality_meta = self._assess_quality_metadata(text, updated_meta)
        updated_meta.update(quality_meta)
        
        # 规范化元数据（暂时跳过，避免循环导入）
        # updated_meta = self._normalize_metadata(updated_meta)
        
        return updated_meta
    
    def _create_basic_metadata(self, 
                              filename: Optional[str], 
                              content_type: Optional[str], 
                              source: Optional[str]) -> Dict[str, Any]:
        """
        创建基础元数据
        
        Args:
            filename: 文件名
            content_type: 内容类型
            source: 数据源
            
        Returns:
            Dict[str, Any]: 基础元数据
        """
        meta = {
            "doc_id": self._generate_doc_id(filename),
            "source": source or "unknown",
            "created_at": self._get_current_timestamp(),
            "media_type": self._detect_media_type(filename, content_type)
        }
        
        # 文件名
        if filename:
            meta["filename"] = filename
        
        # 内容类型
        if content_type:
            meta["content_type"] = content_type
        
        return meta
    
    def _analyze_content_metadata(self, text: str) -> Dict[str, Any]:
        """
        分析内容相关的元数据
        
        Args:
            text: 文本内容
            
        Returns:
            Dict[str, Any]: 内容分析元数据
        """
        # 语言检测
        lang = self._detect_language(text)
        
        content_meta: Dict[str, Any] = {
            "lang": lang,
            "text_len": len(text)
        }
        
        # 关键词提取
        keyphrases = self._extract_keyphrases(text, lang)
        if keyphrases:
            content_meta["keyphrases"] = keyphrases
        
        return content_meta
    
    def _assess_quality_metadata(self, text: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        质量评估元数据
        
        Args:
            text: 文本内容
            meta: 现有元数据
            
        Returns:
            Dict[str, Any]: 质量评估元数据
        """
        return self.quality_assessor.score(text, meta)
    
    @staticmethod
    def _normalize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        规范化元数据
        
        Args:
            meta: 原始元数据
            
        Returns:
            Dict[str, Any]: 规范化后的元数据
        """
        # 使用现有的规范化逻辑
        from app.rag.vectorization.metadata_utils import normalize_meta_for_vector
        return normalize_meta_for_vector(meta)
    
    @staticmethod
    def _generate_doc_id(filename: Optional[str]) -> str:
        """
        生成文档ID
        
        Args:
            filename: 文件名
            
        Returns:
            str: 文档ID
        """
        # 基于文件名和时间戳生成稳定的文档ID
        if filename:
            content_hash = hashlib.md5(filename.encode('utf-8')).hexdigest()[:8]
        else:
            content_hash = hashlib.md5(str(time.time()).encode('utf-8')).hexdigest()[:8]
        
        timestamp = str(int(time.time()))[-6:]  # 取时间戳后6位
        return f"doc_{content_hash}_{timestamp}"
    
    @staticmethod
    def _get_current_timestamp() -> str:
        """
        获取当前时间戳（ISO8601格式）
        
        Returns:
            str: ISO8601格式的时间戳
        """
        return datetime.now(timezone.utc).isoformat()
    
    def _detect_media_type(self, filename: Optional[str], content_type: Optional[str]) -> str:
        """
        检测媒体类型
        
        Args:
            filename: 文件名
            content_type: 内容类型
            
        Returns:
            str: 媒体类型
        """
        media_config = self.config.get("media_type_mapping", {})
        extension_mapping = media_config.get("by_extension", {})
        content_type_mapping = media_config.get("by_content_type", {})
        default_type = media_config.get("default", "text")
        
        # 基于文件扩展名的媒体类型检测
        if filename:
            ext = filename.lower().split('.')[-1] if '.' in filename else ''
            if ext in extension_mapping:
                return extension_mapping[ext]
        
        # 基于内容类型的媒体类型检测
        if content_type:
            for prefix, media_type in content_type_mapping.items():
                if content_type.startswith(prefix):
                    return media_type
        
        return default_type
    
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
        
        return {
            "default_language": metadata_config.get("default_language", "zh"),
            "language_detection": metadata_config.get("language_detection", {
                "chinese_threshold": 0.3,
                "min_text_length": 10
            }),
            "media_type_mapping": metadata_config.get("media_type_mapping", {
                "by_extension": {},
                "by_content_type": {},
                "default": "text"
            })
        }
