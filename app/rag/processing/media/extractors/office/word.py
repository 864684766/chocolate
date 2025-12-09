"""
Word内容提取器

使用python-docx提取Word文档的内容。
支持.doc转.docx（使用LibreOffice）。
"""

import logging
from typing import Dict, Any
from .base import OfficeDocumentExtractorBase

logger = logging.getLogger(__name__)


class WordExtractor(OfficeDocumentExtractorBase):
    """Word内容提取器
    
    使用python-docx提取Word文档的文本、段落、表格等内容。
    对于.doc旧格式，先转换为.docx再处理。
    """
    
    def extract(self, content: bytes, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        从Word文件中提取内容
        
        用处：从Word文档中提取文本、段落、表格等信息，
        为后续的RAG处理提供结构化数据。
        
        Args:
            content: Word文件的二进制内容
            meta: Word文件的元数据信息
            
        Returns:
            Dict[str, Any]: 提取结果字典，包含paragraphs、tables、metadata等
        """
        # TODO: 实现Word提取逻辑
        # 1. 检查是否为.doc格式，如果是则转换为.docx
        # 2. 使用python-docx提取内容
        raise NotImplementedError("Word提取器待实现")
    
    def _get_document_type(self) -> str:
        """
        获取文档类型
        
        Returns:
            str: "word"
        """
        return "word"
    
    def is_available(self) -> bool:
        """
        检查python-docx是否可用
        
        Returns:
            bool: True表示python-docx已安装，False表示未安装
        """
        try:
            import docx  # python-docx
            return True
        except ImportError:
            logger.warning("python-docx未安装，Word提取器不可用")
            return False
    
    def _convert_doc_to_docx(self, content: bytes) -> bytes:
        """
        将.doc格式转换为.docx格式
        
        用处：处理旧格式的Word文档，使用LibreOffice进行转换。
        
        Args:
            content: .doc文件的二进制内容
            
        Returns:
            bytes: .docx文件的二进制内容
            
        Raises:
            RuntimeError: 如果转换失败或LibreOffice不可用
        """
        # TODO: 实现.doc到.docx的转换逻辑
        # 使用LibreOffice的unoconv或doc2docx
        raise NotImplementedError("doc转docx功能待实现")
