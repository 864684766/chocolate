"""
视频内容提取器
"""

import logging
from typing import Dict, Any, List
from .audio_video_base import AudioVideoExtractorBase
from .subtitle_helper import (
    extract_embedded_subtitles,
    generate_subtitles_with_whisper
)

logger = logging.getLogger(__name__)


class VideoContentExtractor(AudioVideoExtractorBase):
    """视频内容提取器
    
    支持从视频文件中提取字幕和语音转录文本，
    使用Whisper或SpeechRecognition进行语音识别。
    """
    
    def __init__(self):
        """
        初始化视频内容提取器
        
        用处：检查视频处理和语音识别功能的可用性，
        为后续的视频内容提取做准备。
        """
        self._video_processing_available = self._check_video_processing_availability()
        self._speech_recognition_available = self._check_speech_recognition_availability()
    
    def is_available(self) -> bool:
        """
        检查视频内容提取器是否可用
        
        用处：检查是否至少有一种视频处理功能可用（视频处理或语音识别），
        用于判断是否可以处理视频内容提取任务。
        
        Returns:
            bool: True表示至少有一种功能可用，False表示完全不可用
        """
        return self._video_processing_available or self._speech_recognition_available
    
    @staticmethod
    def _check_video_processing_availability() -> bool:
        """
        检查视频处理库是否可用
        
        用处：检查OpenCV库是否已安装，这是视频处理的基础依赖。
        如果库未安装，会记录警告信息。
        
        Returns:
            bool: True表示OpenCV可用，False表示不可用
        """
        try:
            import cv2
            return True
        except ImportError:
            logger.warning("OpenCV not available. Install with: pip install opencv-python")
            return False
    
    @staticmethod
    def _check_speech_recognition_availability() -> bool:
        """
        检查语音识别库是否可用
        
        用处：检查Whisper或SpeechRecognition库是否已安装，
        这些是语音识别功能的基础依赖。优先检查Whisper。
        
        Returns:
            bool: True表示至少有一种语音识别库可用，False表示都不可用
        """
        try:
            import whisper
            return True
        except ImportError:
            try:
                import speech_recognition
                return True
            except ImportError:
                logger.warning(
                    "No speech recognition library available. "
                    "Install with: pip install openai-whisper or SpeechRecognition"
                )
                return False
    
    def extract(self, content: bytes, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        从视频中提取文本内容
        
        用处：从视频文件中同时提取字幕和语音转录文本，
        两者可以互为补充，提供更完整的信息。
        
        优化：只创建一次临时文件，字幕和语音转录共用同一个文件，
        避免重复写入磁盘，提高性能。
        
        Args:
            content: 视频文件的二进制内容
            meta: 视频元数据，如格式、时长等信息
            
        Returns:
            Dict[str, Any]: 包含以下键的字典：
                - subtitles (List[Dict]): 提取的字幕列表
                - transcript (str): 语音转录的文本内容
        """
        if not self.is_available():
            return {"subtitles": [], "transcript": ""}
        
        temp_file_path = None
        
        try:
            # 使用基类方法创建临时文件，供字幕和语音转录共用
            temp_file_path = AudioVideoExtractorBase._create_temp_file(content, meta, default_format="mp4")
            
            # 同时提取字幕和语音转录，共用同一个临时文件
            subtitles = VideoContentExtractor._extract_subtitles(temp_file_path)
            transcript = AudioVideoExtractorBase._extract_transcript(temp_file_path)
            
            return {
                "subtitles": subtitles,
                "transcript": transcript,
            }
        except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError) as e:
            logger.error(f"Video content extraction failed: {e}")
            return {"subtitles": [], "transcript": ""}
        finally:
            # 使用基类方法清理临时文件
            if temp_file_path:
                AudioVideoExtractorBase._cleanup_temp_file(temp_file_path)
    
    @staticmethod
    def _extract_subtitles(video_path: str) -> List[Dict[str, Any]]:
        """
        提取视频字幕
        
        用处：从视频文件中提取字幕信息，优先提取内嵌字幕，
        如果没有内嵌字幕则使用 Whisper 生成带时间戳的字幕。
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            List[Dict[str, Any]]: 字幕列表，每个字典包含：
                - text (str): 字幕文本内容
                - start_time (float): 开始时间（秒）
                - end_time (float): 结束时间（秒）
        """
        subtitles = []
        
        try:
            # 优先尝试提取内嵌字幕
            subtitles = extract_embedded_subtitles(video_path)
            
            # 如果没有内嵌字幕，使用 Whisper 生成
            if not subtitles:
                # 从配置获取模型名称
                try:
                    from app.config import get_config_manager
                    config_manager = get_config_manager()
                    speech_config = config_manager.get_speech_recognition_config()
                    model_name = speech_config.get("model", "base")
                    
                    # 移除 "whisper-" 前缀（如果存在）
                    if model_name.startswith("whisper-"):
                        model_name = model_name.replace("whisper-", "")
                except Exception as e:
                    logger.debug(f"Failed to get config, using defaults: {e}")
                    model_name = "base"
                
                # Whisper 会自动检测语言，无需传递 language 参数
                subtitles = generate_subtitles_with_whisper(
                    video_path, 
                    model_name=model_name
                )
        
        except Exception as e:
            logger.warning(f"Subtitle extraction failed: {e}")
        
        return subtitles
    