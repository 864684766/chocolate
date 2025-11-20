"""
视频内容提取器
"""

import logging
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List
from .base import MediaExtractor
from .subtitle_helper import (
    extract_embedded_subtitles,
    generate_subtitles_with_whisper
)

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
                - video_meta (Dict): 视频处理元数据，包含格式、处理状态等
        """
        if not self.is_available():
            return {
                "subtitles": [],
                "transcript": "",
                "video_meta": {"error": "Video processing not available"}
            }
        
        temp_file_path = None
        
        try:
            # 只创建一次临时视频文件，供字幕和语音转录共用
            temp_file_path = self._create_temp_video_file(content, meta)
            
            # 同时提取字幕和语音转录，共用同一个临时文件
            subtitles = VideoContentExtractor._extract_subtitles(temp_file_path, meta)
            transcript = VideoContentExtractor._extract_transcript(temp_file_path)
            
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
        finally:
            # 清理临时文件
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temp video file: {e}")
    
    @staticmethod
    def _extract_subtitles(video_path: str, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        提取视频字幕
        
        用处：从视频文件中提取字幕信息，优先提取内嵌字幕，
        如果没有内嵌字幕则使用 Whisper 生成带时间戳的字幕。
        
        Args:
            video_path: 视频文件路径
            meta: 视频元数据，包含配置信息
            
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
                # 优先从 meta 中获取配置，如果没有则从全局配置获取
                config = VideoContentExtractor._get_video_config()
                speech_config = meta.get("speech_recognition") or config.get("speech_recognition", {})
                model_name = speech_config.get("model", "base")
                language = speech_config.get("language")
                
                # 移除 "whisper-" 前缀（如果存在）
                if model_name.startswith("whisper-"):
                    model_name = model_name.replace("whisper-", "")
                
                subtitles = generate_subtitles_with_whisper(
                    video_path, 
                    model_name=model_name,
                    language=language
                )
        
        except Exception as e:
            logger.warning(f"Subtitle extraction failed: {e}")
        
        return subtitles
    
    @staticmethod
    def _get_video_config() -> Dict[str, Any]:
        """
        获取视频处理配置
        
        用处：从配置管理器中获取视频处理相关配置。
        
        Returns:
            Dict[str, Any]: 视频处理配置字典
        """
        try:
            from app.config import get_config_manager
            config_manager = get_config_manager()
            return config_manager.get_video_processing_config()
        except Exception as e:
            logger.warning(f"Failed to get video config: {e}")
            return {}
    
    @staticmethod
    def _get_temp_dir() -> Path:
        """
        获取临时文件目录路径
        
        用处：从配置文件中读取临时文件目录路径。
        必须在配置文件中明确指定 temp_dir 路径，如果为空或未配置则抛出异常。
        
        Returns:
            Path: 临时文件目录路径
            
        Raises:
            ValueError: 如果配置中的 temp_dir 为空或未配置
            RuntimeError: 如果无法读取配置
        """
        try:
            from app.config import get_config_manager
            config_manager = get_config_manager()
            video_config = config_manager.get_video_processing_config()
            temp_dir_path = video_config.get("temp_dir", "").strip()
            
            # 如果配置中未指定路径或为空，抛出异常
            if not temp_dir_path:
                raise ValueError(
                    "临时文件目录未配置。请在 config/app_config.json 的 "
                    "media_processing.video_processing.temp_dir 中指定临时文件目录路径。"
                )
            
            # 使用配置的路径，支持 ~ 展开（跨平台）
            temp_dir = Path(temp_dir_path).expanduser().resolve()
            
            # 确保临时目录存在
            if not temp_dir.exists():
                temp_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created temp directory from config: {temp_dir}")
            
            return temp_dir
            
        except ValueError:
            # 重新抛出 ValueError（配置为空的情况）
            raise
        except Exception as e:
            raise RuntimeError(f"无法读取临时文件目录配置: {e}")
    
    @staticmethod
    def _create_temp_video_file(content: bytes, meta: Dict[str, Any]) -> str:
        """
        创建临时视频文件
        
        用处：将视频二进制内容写入临时文件，用于字幕提取。
        临时文件创建在项目根目录下的 temp 文件夹中，方便维护和检查。
        
        Args:
            content: 视频文件的二进制内容
            meta: 视频元数据，用于确定文件扩展名
            
        Returns:
            str: 临时文件路径
        """
        video_format = meta.get("video_format", "mp4")
        temp_dir = VideoContentExtractor._get_temp_dir()
        
        # 从配置中获取临时文件目录，在此目录下创建临时视频文件
        # tempfile.NamedTemporaryFile 官方文档：https://docs.python.org/3/library/tempfile.html#tempfile.NamedTemporaryFile
        # 使用 str(temp_dir) 将 Path 对象转为字符串，Path 会自动处理跨平台路径分隔符
        temp_file = tempfile.NamedTemporaryFile(
            suffix=f".{video_format}",
            delete=False,
            dir=str(temp_dir)
        )
        temp_file.write(content)
        temp_file.flush()  # 确保数据写入磁盘（跨平台）
        os.fsync(temp_file.fileno())  # 强制同步到磁盘（跨平台）
        # 使用 os.path.abspath() 获取绝对路径，自动处理跨平台路径分隔符
        temp_file_path = os.path.abspath(temp_file.name)
        temp_file.close()
        
        # 验证文件确实存在
        if not os.path.exists(temp_file_path):
            raise RuntimeError(f"临时文件创建失败: {temp_file_path}")
        
        logger.debug(f"Created temp video file: {temp_file_path}")
        return temp_file_path
    
    @staticmethod
    def _extract_transcript(video_path: str) -> str:
        """
        提取视频语音转录文本
        
        用处：使用语音识别技术将视频中的语音转换为文本，
        优先使用Whisper，回退到SpeechRecognition。
        
        优化：直接使用文件路径，避免重复创建临时文件。
        
        Args:
            video_path: 视频文件路径（已存在的临时文件）
            
        Returns:
            str: 语音转录的文本内容，失败时返回空字符串
        """
        # 优先尝试使用 Whisper
        whisper_result = VideoContentExtractor._try_extract_with_whisper(video_path)
        if whisper_result:
            return whisper_result
        
        # 如果 Whisper 失败，尝试使用 SpeechRecognition
        speech_recognition_result = VideoContentExtractor._try_extract_with_speech_recognition(video_path)
        if speech_recognition_result:
            return speech_recognition_result
        
        logger.warning("No speech recognition library available or all methods failed")
        return ""
    
    @staticmethod
    def _try_extract_with_whisper(video_path: str) -> str:
        """
        尝试使用 Whisper 提取语音转录
        
        用处：尝试使用 Whisper 进行语音识别，捕获所有异常。
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            str: 语音转录的文本内容，失败时返回空字符串
        """
        try:
            return VideoContentExtractor._extract_with_whisper(video_path)
        except ImportError:
            logger.debug("Whisper not available")
        except Exception as e:
            logger.warning(f"Whisper transcription failed: {e}")
        return ""
    
    @staticmethod
    def _try_extract_with_speech_recognition(video_path: str) -> str:
        """
        尝试使用 SpeechRecognition 提取语音转录
        
        用处：尝试使用 SpeechRecognition 进行语音识别，捕获所有异常。
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            str: 语音转录的文本内容，失败时返回空字符串
        """
        try:
            return VideoContentExtractor._extract_with_speech_recognition(video_path)
        except ImportError:
            logger.debug("SpeechRecognition not available")
        except Exception as e:
            logger.warning(f"SpeechRecognition transcription failed: {e}")
        return ""
    
    @staticmethod
    def _extract_with_whisper(video_path: str) -> str:
        """
        使用Whisper进行语音识别
        
        用处：使用OpenAI的Whisper模型进行高质量的语音识别，
        支持多语言和长音频处理。
        
        优化：直接使用文件路径，不再创建临时文件。
        
        Args:
            video_path: 视频文件路径（已存在的文件）
            
        Returns:
            str: 语音识别的文本结果，失败时抛出异常
            
        Raises:
            ImportError: Whisper 库未安装
            Exception: 其他处理异常
        """
        import whisper
        
        # 获取配置
        model_name = "base"
        language = None
        try:
            from app.config import get_config_manager
            config_manager = get_config_manager()
            video_config = config_manager.get_video_processing_config()
            speech_config = video_config.get("speech_recognition", {})
            model_name = speech_config.get("model", "base")
            language = speech_config.get("language")
            
            # 移除 "whisper-" 前缀（如果存在）
            if model_name.startswith("whisper-"):
                model_name = model_name.replace("whisper-", "")
        except Exception as e:
            logger.debug(f"Failed to get config, using defaults: {e}")
        
        # 加载模型并转录音频
        model = whisper.load_model(model_name)
        transcribe_options = {}
        if language:
            transcribe_options['language'] = language
        
        # 使用 audio 参数明确指定音频文件路径
        result = model.transcribe(audio=video_path, **transcribe_options)
        
        # 验证返回结果
        if not result or not isinstance(result, dict):
            raise ValueError("Whisper returned invalid result format")
        
        text = result.get("text", "")
        if not text:
            logger.warning("Whisper transcription returned empty text")
        
        return text
    
    @staticmethod
    def _extract_with_speech_recognition(video_path: str) -> str:
        """
        使用SpeechRecognition进行语音识别
        
        用处：使用SpeechRecognition库进行语音识别，
        作为Whisper的备选方案，支持Google语音识别API。
        
        优化：直接使用文件路径，不再创建临时文件。
        
        Args:
            video_path: 视频文件路径（已存在的文件）
            
        Returns:
            str: 语音识别的文本结果，失败时抛出异常
            
        Raises:
            ImportError: SpeechRecognition 库未安装
            Exception: 其他处理异常
        """
        import speech_recognition as sr
        
        # 获取语言配置
        language = 'zh-CN'
        try:
            from app.config import get_config_manager
            config_manager = get_config_manager()
            video_config = config_manager.get_video_processing_config()
            speech_config = video_config.get("speech_recognition", {})
            language = speech_config.get("language", "zh-CN")
        except Exception as e:
            logger.debug(f"Failed to get language config, using default: {e}")
        
        # 进行语音识别
        recognizer = sr.Recognizer()
        with sr.AudioFile(video_path) as source:
            audio = recognizer.record(source)
        
        text = recognizer.recognize_google(audio=audio, language=language)
        
        if not text:
            logger.warning("SpeechRecognition returned empty text")
        
        return text
