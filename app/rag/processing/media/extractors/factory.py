"""
媒体提取器工厂
"""

from typing import Optional
from .base import MediaExtractor
from .image_ocr import ImageOCRExtractor
from .video import VideoContentExtractor
from .audio import AudioContentExtractor


class MediaExtractorFactory:
    """媒体提取器工厂
    
    根据媒体类型创建对应的提取器实例，
    提供统一的提取器创建接口。
    """
    
    @staticmethod
    def create_extractor(media_type: str) -> Optional[MediaExtractor]:
        """
        根据媒体类型创建对应的提取器
        
        用处：根据媒体文件类型选择合适的提取器，
        支持图像、视频、音频等不同类型的媒体处理。
        
        Args:
            media_type: 媒体类型，如"image"、"video"、"audio"
            
        Returns:
            Optional[MediaExtractor]: 对应的提取器实例，不支持的类型返回None
        """
        if media_type.lower() == "image":
            return ImageOCRExtractor()  # 内置视觉回退
        elif media_type.lower() == "video":
            return VideoContentExtractor()
        elif media_type.lower() == "audio":
            return AudioContentExtractor()
        else:
            return None
