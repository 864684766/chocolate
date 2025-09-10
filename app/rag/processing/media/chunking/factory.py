"""
分块策略工厂
"""

from .base import MediaChunkingStrategy
from .text import TextChunkingStrategy
from .pdf import PDFChunkingStrategy
from .image import ImageChunkingStrategy
from .video import VideoChunkingStrategy


class ChunkingStrategyFactory:
    """分块策略工厂"""
    
    @staticmethod
    def create_strategy(media_type: str, **kwargs) -> MediaChunkingStrategy:
        """
        根据媒体类型创建对应的分块策略
        
        Args:
            media_type: 媒体类型，如 "text", "pdf", "image", "video", "audio"
            **kwargs: 传递给策略构造函数的参数
            
        Returns:
            MediaChunkingStrategy: 对应的分块策略实例
        """
        strategies = {
            "text": TextChunkingStrategy,
            "pdf": PDFChunkingStrategy,
            "image": ImageChunkingStrategy,
            "video": VideoChunkingStrategy,
            "audio": VideoChunkingStrategy,  # 音频使用视频策略（处理字幕/转写）
        }
        
        strategy_class = strategies.get(media_type.lower(), TextChunkingStrategy)
        return strategy_class(**kwargs)
