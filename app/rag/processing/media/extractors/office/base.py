"""
Office文档内容提取器基类

提取Office文档（PDF、Word、Excel）处理中的公共逻辑。
"""

import logging
from typing import Dict, Any
from abc import ABC
from ..base import MediaExtractor

logger = logging.getLogger(__name__)


class OfficeDocumentExtractorBase(MediaExtractor, ABC):
    """Office文档内容提取器基类
    
    提供Office文档处理中的公共功能：
    - 元数据规范化
    - 错误处理
    - 内容验证
    
    子类需要实现 extract() 方法来处理特定的文档类型。
    """
    
    def is_available(self) -> bool:
        """
        检查提取器是否可用
        
        用处：检查当前环境是否满足提取器的依赖要求，
        如必要的库是否已安装。
        
        Returns:
            bool: True表示提取器可用，False表示不可用
        """
        # 子类需要实现具体的可用性检查逻辑
        return True
    
    def _normalize_metadata(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        规范化元数据
        
        用处：统一Office文档的元数据格式，
        确保后续处理的一致性。
        
        Args:
            meta: 原始元数据
            
        Returns:
            Dict[str, Any]: 规范化后的元数据
        """
        normalized = meta.copy()
        # 添加文档类型标识
        normalized.setdefault("document_type", self._get_document_type())
        return normalized
    
    def _get_document_type(self) -> str:
        """
        获取文档类型
        
        用处：返回当前提取器处理的文档类型标识。
        
        Returns:
            str: 文档类型，如 "pdf", "word", "excel"
        """
        # 子类需要实现
        raise NotImplementedError("子类必须实现 _get_document_type 方法")
    
    @staticmethod
    def _validate_content(content: Dict[str, Any]) -> bool:
        """
        验证提取的内容
        
        用处：检查提取的内容是否符合预期格式，
        确保后续处理不会出错。
        
        Args:
            content: 提取的内容字典
            
        Returns:
            bool: True表示内容有效，False表示无效
        """
        if not isinstance(content, dict):
            return False
        # 至少应该包含文本内容
        return "text" in content or "pages" in content or "sheets" in content
