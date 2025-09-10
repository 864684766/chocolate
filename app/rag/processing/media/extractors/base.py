"""
媒体内容提取器基类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class MediaExtractor(ABC):
    """媒体内容提取器基类
    
    定义了所有媒体提取器必须实现的接口，包括内容提取和可用性检查。
    """
    
    @abstractmethod
    def extract(self, content: bytes, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        从媒体内容中提取文本信息
        
        用处：从图片、视频、音频等媒体文件中提取可读的文本内容，
        为后续的RAG处理提供文本数据源。
        
        Args:
            content: 媒体文件的二进制内容
            meta: 媒体文件的元数据信息，如文件类型、格式等
            
        Returns:
            Dict[str, Any]: 提取结果字典，包含文本内容和相关元数据
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        检查提取器是否可用
        
        用处：检查当前环境是否满足提取器的依赖要求，
        如必要的库是否已安装、模型是否可用等。
        
        Returns:
            bool: True表示提取器可用，False表示不可用
        """
        pass
