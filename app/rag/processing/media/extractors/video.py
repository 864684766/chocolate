"""
视频内容提取器
"""

import logging
from typing import Dict, Any, List
from .base import MediaExtractor

logger = logging.getLogger(__name__)


class VideoContentExtractor(MediaExtractor):
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
        
        用处：从视频文件中提取字幕或语音转录文本，
        优先提取字幕，如果无字幕则进行语音识别转录。
        
        Args:
            content: 视频文件的二进制内容
            meta: 视频元数据，如格式、时长等信息
            
        Returns:
            Dict[str, Any]: 包含以下键的字典：
                - subtitles (List[Dict]): 提取的字幕列表
                - transcript (str): 语音转录的文本内容
                - video_meta (Dict): 视频处理元数据，包含格式、处理状态等
        """
        if not self.is_available():
            return {
                "subtitles": [],
                "transcript": "",
                "video_meta": {"error": "Video processing not available"}
            }
        
        try:
            subtitles = self._extract_subtitles()
            transcript = ""
            if not subtitles:
                transcript = self._extract_transcript(content, meta)
            return {
                "subtitles": subtitles,
                "transcript": transcript,
                "video_meta": {
                    "video_format": meta.get("video_format", "auto"),
                    "has_subtitles": bool(subtitles),
                    "has_transcript": bool(transcript)
                }
            }
        except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError) as e:
            logger.error(f"Video content extraction failed: {e}")
            return {
                "subtitles": [],
                "transcript": "",
                "video_meta": {"error": str(e)}
            }
    
    @staticmethod
    def _extract_subtitles() -> List[Dict[str, Any]]:
        """
        提取视频字幕
        
        用处：从视频文件中提取字幕信息，目前未实现。
        未来可以支持SRT、VTT等字幕格式的解析。
        
        Returns:
            List[Dict[str, Any]]: 字幕列表，每个字典包含时间戳和文本内容
        """
        logger.info("Subtitle extraction not implemented yet")
        return []
    
    def _extract_transcript(self, video_bytes: bytes, meta: Dict[str, Any]) -> str:
        """
        提取视频语音转录文本
        
        用处：使用语音识别技术将视频中的语音转换为文本，
        优先使用Whisper，回退到SpeechRecognition。
        
        Args:
            video_bytes: 视频文件的二进制内容
            meta: 视频元数据
            
        Returns:
            str: 语音转录的文本内容，失败时返回空字符串
        """
        try:
            return self._extract_with_whisper(video_bytes, meta)
        except ImportError:
            pass
        try:
            return self._extract_with_speech_recognition(video_bytes, meta)
        except ImportError:
            pass
        logger.warning("No speech recognition library available")
        return ""
    
    @staticmethod
    def _extract_with_whisper(video_bytes: bytes, meta: Dict[str, Any]) -> str:
        """
        使用Whisper进行语音识别
        
        用处：使用OpenAI的Whisper模型进行高质量的语音识别，
        支持多语言和长音频处理。
        
        Args:
            video_bytes: 视频文件的二进制内容
            meta: 视频元数据，用于确定文件格式
            
        Returns:
            str: 语音识别的文本结果
        """
        import whisper
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix=f".{meta.get('video_format', 'mp4')}", delete=False) as temp_file:
            temp_file.write(video_bytes)
            temp_file_path = temp_file.name
        try:
            model = whisper.load_model("base")
            # 使用模块函数形式，显式传入 model
            result = whisper.transcribe(model, audio=temp_file_path)
            return result["text"]
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    @staticmethod
    def _extract_with_speech_recognition(video_bytes: bytes, meta: Dict[str, Any]) -> str:
        """
        使用SpeechRecognition进行语音识别
        
        用处：使用SpeechRecognition库进行语音识别，
        作为Whisper的备选方案，支持Google语音识别API。
        
        Args:
            video_bytes: 视频文件的二进制内容
            meta: 视频元数据，用于确定文件格式
            
        Returns:
            str: 语音识别的文本结果
        """
        import speech_recognition as sr
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix=f".{meta.get('video_format', 'mp4')}", delete=False) as temp_file:
            temp_file.write(video_bytes)
            temp_file_path = temp_file.name
        try:
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_file_path) as source:
                audio = recognizer.record(source)
            text = recognizer.recognize_google(audio=audio, language='zh-CN')
            return text
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
