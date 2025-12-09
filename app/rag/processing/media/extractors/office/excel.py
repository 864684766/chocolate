"""
Excel内容提取器

使用pandas提取Excel文档的内容。
"""

import logging
from typing import Dict, Any
from .base import OfficeDocumentExtractorBase

logger = logging.getLogger(__name__)


class ExcelExtractor(OfficeDocumentExtractorBase):
    """Excel内容提取器
    
    使用pandas提取Excel文档的数据、表格结构等信息。
    """
    
    def extract(self, content: bytes, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        从Excel文件中提取内容
        
        用处：从Excel文档中提取数据、表格结构等信息，
        为后续的RAG处理提供结构化数据。
        
        Args:
            content: Excel文件的二进制内容
            meta: Excel文件的元数据信息
            
        Returns:
            Dict[str, Any]: 提取结果字典，包含sheets、data、metadata等
        """
        # TODO: 实现Excel提取逻辑
        # 使用pandas读取Excel内容
        raise NotImplementedError("Excel提取器待实现")
    
    def _get_document_type(self) -> str:
        """
        获取文档类型
        
        Returns:
            str: "excel"
        """
        return "excel"
    
    def is_available(self) -> bool:
        """
        检查pandas是否可用
        
        Returns:
            bool: True表示pandas已安装，False表示未安装
        """
        try:
            import pandas as pd
            return True
        except ImportError:
            logger.warning("pandas未安装，Excel提取器不可用")
            return False
