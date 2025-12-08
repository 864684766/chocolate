"""
音频内容提取器
"""

import logging
from typing import Dict, Any
from .audio_video_base import AudioVideoExtractorBase

logger = logging.getLogger(__name__)


class AudioContentExtractor(AudioVideoExtractorBase):
    """音频内容提取器
    
    支持从音频文件中提取语音转录文本，
    使用Whisper或SpeechRecognition进行语音识别。
    """
    
    def __init__(self):
        """
        初始化音频内容提取器
        
        用处：检查语音识别功能的可用性，
        为后续的音频内容提取做准备。
        """
        self._speech_recognition_available = self._check_speech_recognition_availability()
    
    def is_available(self) -> bool:
        """
        检查音频内容提取器是否可用
        
        用处：检查语音识别库是否可用，
        用于判断是否可以处理音频内容提取任务。
        
        Returns:
            bool: True表示语音识别功能可用，False表示不可用
        """
        return self._speech_recognition_available
    
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
        从音频中提取文本内容
        
        用处：从音频文件中提取语音转录文本，
        使用Whisper或SpeechRecognition进行语音识别。
        
        优化：只创建一次临时文件，避免重复写入磁盘，提高性能。
        
        Args:
            content: 音频文件的二进制内容
            meta: 音频元数据，如格式、时长等信息
            
        Returns:
            Dict[str, Any]: 包含以下键的字典：
                - transcript (str): 语音转录的文本内容
        """
        if not self.is_available():
            return {"transcript": ""}
        
        temp_file_path = None
        
        try:
            # 使用基类方法创建临时文件
            # 默认格式为 mp3，如果 meta 中有 audio_format 则使用该格式
            temp_file_path = AudioVideoExtractorBase._create_temp_file(content, meta, default_format="mp3")
            
            # 提取语音转录文本（音频没有字幕，只提取转录）
            transcript = AudioVideoExtractorBase._extract_transcript(temp_file_path)
            
            return {
                "transcript": transcript,
            }
        except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError) as e:
            logger.error(f"Audio content extraction failed: {e}")
            return {"transcript": ""}
        finally:
            # 使用基类方法清理临时文件
            if temp_file_path:
                AudioVideoExtractorBase._cleanup_temp_file(temp_file_path)

