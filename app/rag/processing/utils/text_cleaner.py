"""
通用文本清洗和去重模块

提供统一的文本清洗和去重功能，适用于所有媒体类型（文本、图像、视频、音频、PDF、Word等）。
"""

from __future__ import annotations

from typing import List
import logging
from app.config import get_config_manager
from .quality_utils import (
    filter_captions,
    dedup_captions,
)

logger = logging.getLogger(__name__)


class TextCleaner:
    """通用文本清洗和去重处理器
    
    提供统一的文本清洗和去重功能，适用于所有媒体类型。
    清洗包括：统一换行符、归一化空白字符、去除无效字符等。
    去重包括：精确去重（MD5）、近似去重（向量相似度）、重复片段检测等。
    """

    def __init__(self) -> None:
        """
        初始化文本清洗器
        
        用处：从配置中读取清洗和去重参数，为后续的文本处理做准备。
        
        Returns:
            None
        """
        config_manager = get_config_manager()
        processing_config = config_manager.get_config("processing") or {}
        cleaning_config = processing_config.get("text_cleaning", {})
        dedup_config = processing_config.get("deduplication", {})
        
        # 文本清洗配置
        self.whitespace_pattern = cleaning_config.get("normalize_whitespace_pattern", "[ \\t]+")
        self.whitespace_replacement = cleaning_config.get("normalize_whitespace_replacement", " ")
        self.newlines_pattern = cleaning_config.get("normalize_newlines_pattern", "\\n\\s*\\n\\s*\\n+")
        self.newlines_replacement = cleaning_config.get("normalize_newlines_replacement", "\n\n")
        
        # 去重配置
        self.approx_enabled = dedup_config.get("approximate_enabled", True)
        self.similarity_threshold = float(dedup_config.get("similarity_threshold", 0.95))
        self.embed_model = dedup_config.get("embed_model", None)
        
        # 过滤配置
        filter_config = dedup_config.get("filters", {})
        self.min_len = int(filter_config.get("min_length", 5))
        self.max_len = int(filter_config.get("max_length", 10000))
        self.blacklist_keywords = list(filter_config.get("blacklist_keywords", []))
        self.max_gibberish_ratio = float(filter_config.get("max_gibberish_ratio", 0.3))
        self.forbid_repeat_ngram = int(filter_config.get("forbid_repeat_ngram", 3))
        # 重复比率阈值（基于业界实践，默认 0.5 即 50%）
        self.max_repetition_ratio = float(dedup_config.get("max_repetition_ratio", 0.5))
        # 片段分隔符配置（用于修复片段级重复）
        self.segment_separators = str(dedup_config.get("segment_separators", "，,。.；;！!？?、"))

    def clean(self, text: str) -> str:
        """
        清洗单个文本内容
        
        用处：统一换行符、归一化空白字符、去除无效字符等，
        为后续的分块和去重处理做准备。
        
        Args:
            text: 原始文本内容
            
        Returns:
            str: 清洗后的文本内容
        """
        if not text:
            return ""
        
        import re
        
        # 统一换行符
        t = text.replace("\r\n", "\n").replace("\r", "\n")
        
        # 归一化空白字符（保留单个换行符）
        t = re.sub(self.whitespace_pattern, self.whitespace_replacement, t)
        t = re.sub(self.newlines_pattern, self.newlines_replacement, t)
        
        return t.strip()

    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        批量清洗文本列表
        
        用处：对多个文本内容进行批量清洗，提高处理效率。
        
        Args:
            texts: 原始文本列表
            
        Returns:
            List[str]: 清洗后的文本列表
        """
        return [self.clean(text) for text in texts if text]

    def clean_and_deduplicate(self, texts: List[str]) -> List[str]:
        """
        清洗并去重文本列表（统一处理）
        
        用处：对提取的文本内容进行清洗和去重的完整流程，
        统一处理所有媒体类型的文本列表。内部先清洗再去重，确保处理一致性。
        
        适用场景：
        - 图像描述列表（captions）
        - 视频字幕列表（subtitles）
        - 音频转录片段列表（segments）
        - PDF页面文本列表（pages）
        - Word段落列表（paragraphs）
        - 任何需要清洗和去重的文本列表
        
        Args:
            texts: 原始文本列表
            
        Returns:
            List[str]: 清洗和去重后的文本列表（数量可能减少）
        """
        if not texts:
            return texts
        
        # 第一步：清洗所有文本
        cleaned = self.clean_batch(texts)
        original_count = len(texts)
        cleaned_count = len(cleaned)
        
        # 第二步：规则过滤（先修复片段级重复，再检查长度/黑名单/乱码/重复比率）
        filtered = filter_captions(
            cleaned,
            min_len=self.min_len,
            max_len=self.max_len,
            blacklist_keywords=self.blacklist_keywords,
            max_gibberish_ratio=self.max_gibberish_ratio,
            forbid_repeat_ngram=self.forbid_repeat_ngram,
            max_repetition_ratio=self.max_repetition_ratio,
            segment_separators=self.segment_separators,
        )
        filtered_count = len(filtered)
        
        # 第三步：内容去重（精确+近似）
        try:
            deduplicated = dedup_captions(
                filtered,
                approx=self.approx_enabled,
                threshold=self.similarity_threshold,
                embed_model=self.embed_model
            )
            final_count = len(deduplicated)
            
            # 记录处理结果：只在有变化时记录，避免日志过多
            if final_count < original_count:
                # 数量减少：说明有文本被过滤或去重
                logger.info(f"文本处理完成：{original_count} -> {final_count} 条（清洗+过滤+去重）")
            elif final_count == original_count and (cleaned_count < original_count or filtered_count < cleaned_count):
                # 数量相同但中间有过滤：说明有文本在清洗或过滤阶段被移除，但最终数量相同（可能去重没有效果）
                logger.debug(f"文本处理完成：{original_count} 条（清洗/过滤阶段有移除，但最终数量相同）")
            # 如果 final_count > original_count，这是异常情况，理论上不应该发生
            elif final_count > original_count:
                logger.warning(f"文本处理异常：数量增加 {original_count} -> {final_count}，可能存在逻辑错误")
                
            return deduplicated
        except Exception as e:
            logger.warning(f"文本去重失败，返回过滤后内容: {e}")
            return filtered
