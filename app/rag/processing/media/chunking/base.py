"""
媒体分块策略基类
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class MediaChunkingStrategy(ABC):
    """媒体分块策略基类"""
    
    @abstractmethod
    def chunk(self, content: Any, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        将媒体内容分块，返回包含文本和元数据的块列表
        
        Args:
            content: 媒体内容，类型根据具体策略而定
            meta: 元数据信息
            
        Returns:
            List[Dict[str, Any]]: 分块结果列表，每个字典包含 "text" 和 "meta" 键
        """
        pass
