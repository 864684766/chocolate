"""
分块策略工厂
"""

from .base import MediaChunkingStrategy
from .text import TextChunkingStrategy
from .pdf import PDFChunkingStrategy
from .image import ImageChunkingStrategy
from .video import VideoChunkingStrategy
from .audio import AudioChunkingStrategy


class ChunkingStrategyFactory:
    """分块策略工厂
    
    用处：根据媒体类型创建对应的分块策略实例，
    统一管理不同媒体类型的分块逻辑。
    """
    
    @staticmethod
    def create_strategy(media_type: str, **kwargs) -> MediaChunkingStrategy:
        """
        根据媒体类型创建对应的分块策略
        
        用处：根据媒体类型选择合适的分块策略，
        支持文本、PDF、图像、视频、音频等多种媒体类型。
        
        Args:
            media_type: 媒体类型，如 "text", "pdf", "image", "video", "audio"
            **kwargs: 传递给策略构造函数的参数（如 chunk_size, overlap）
            
        Returns:
            MediaChunkingStrategy: 对应的分块策略实例
        """
        strategies = {
            "text": TextChunkingStrategy,
            "pdf": PDFChunkingStrategy,
            "image": ImageChunkingStrategy,
            "video": VideoChunkingStrategy,
            "audio": AudioChunkingStrategy,  # 音频使用专门的分块策略
        }
        
        strategy_class = strategies.get(media_type.lower(), TextChunkingStrategy)
        return strategy_class(**kwargs)
