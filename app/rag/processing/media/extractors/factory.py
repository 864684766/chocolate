"""
媒体提取器工厂
"""

from typing import Optional
from .base import MediaExtractor
from .text.plain_text import PlainTextExtractor
from .text.markdown import MarkdownExtractor
from .image.ocr import ImageOCRExtractor
from .audio_video.video import VideoContentExtractor
from .audio_video.audio import AudioContentExtractor
from .office.pdf import PDFExtractor
from .office.word import WordExtractor
from .office.excel import ExcelExtractor


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
        支持文本、Markdown、图像、视频、音频、PDF、Word、Excel等不同类型的媒体处理。
        
        Args:
            media_type: 媒体类型，如"text"、"markdown"、"image"、"video"、"audio"、"pdf"、"word"、"excel"
            
        Returns:
            Optional[MediaExtractor]: 对应的提取器实例，不支持的类型返回None
        """
        media_type_lower = media_type.lower()
        if media_type_lower == "text":
            return PlainTextExtractor()
        elif media_type_lower == "markdown":
            return MarkdownExtractor()
        elif media_type_lower == "image":
            return ImageOCRExtractor()  # 内置视觉回退
        elif media_type_lower == "video":
            return VideoContentExtractor()
        elif media_type_lower == "audio":
            return AudioContentExtractor()
        elif media_type_lower == "pdf":
            return PDFExtractor()
        elif media_type_lower == "word":
            return WordExtractor()
        elif media_type_lower == "excel":
            return ExcelExtractor()
        else:
            return None
