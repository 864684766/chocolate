"""
纯文本内容提取器

将bytes直接按utf-8解码为文本；解码失败则回退到gb18030或latin-1。
适用于.txt等纯文本文件。
"""

import logging
from typing import Dict, Any
from ..base import MediaExtractor

logger = logging.getLogger(__name__)

# 配置常量
DEFAULT_ENCODING = "utf-8"
FALLBACK_ENCODING_GB18030 = "gb18030"
FALLBACK_ENCODING_LATIN1 = "latin-1"


class PlainTextExtractor(MediaExtractor):
    """纯文本内容提取器
    
    将bytes直接按utf-8解码为文本；解码失败则回退到gb18030或latin-1。
    适用于.txt等纯文本文件。
    """
    
    def extract(self, content: bytes, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        从文本文件中提取内容
        
        用处：将二进制内容解码为文本，支持多种编码格式，
        为后续的RAG处理提供文本数据源。
        
        Args:
            content: 文本文件的二进制内容
            meta: 文本文件的元数据信息
            
        Returns:
            Dict[str, Any]: 提取结果字典，包含text和meta
        """
        try:
            text = content.decode(DEFAULT_ENCODING)
        except UnicodeDecodeError:
            # 国标编码优先于latin-1，兼容中文本地文件
            try:
                text = content.decode(FALLBACK_ENCODING_GB18030)
            except UnicodeDecodeError:
                text = content.decode(FALLBACK_ENCODING_LATIN1, errors="ignore")
        
        return {"text": text, "meta": meta}
    
    def is_available(self) -> bool:
        """
        检查文本提取器是否可用
        
        用处：文本提取器不需要额外依赖，始终可用。
        
        Returns:
            bool: 始终返回True
        """
        return True
