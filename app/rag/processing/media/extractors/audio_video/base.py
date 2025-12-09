"""
音频和视频内容提取器基类

提取音频和视频处理中的公共逻辑，包括语音识别、临时文件管理等。
"""

import logging
import tempfile
import os
from pathlib import Path
from typing import Dict, Any
from abc import ABC
from ..base import MediaExtractor

logger = logging.getLogger(__name__)


class AudioVideoExtractorBase(MediaExtractor, ABC):
    """音频和视频内容提取器基类
    
    提供音频和视频处理中的公共功能：
    - 语音识别（Whisper 和 SpeechRecognition）
    - 临时文件管理
    - 配置读取
    
    子类需要实现 extract() 方法来处理特定的媒体类型。
    """
    
    @staticmethod
    def _get_temp_dir() -> Path:
        """
        获取临时文件目录路径
        
        用处：从配置文件中读取临时文件目录路径，供视频和音频处理共用。
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
            media_config = config_manager.get_media_processing_config()
            temp_dir_path = media_config.get("temp_dir", "").strip()
            
            # 如果配置中未指定路径或为空，抛出异常
            if not temp_dir_path:
                raise ValueError(
                    "临时文件目录未配置。请在 config/app_config.json 的 "
                    "media_processing.temp_dir 中指定临时文件目录路径。"
                )
            
            # 使用配置的路径，支持 ~ 展开（跨平台）
            # Path() 创建路径对象，自动处理跨平台路径分隔符（Windows: \，Linux/Mac: /）
            # .expanduser() 将 ~ 展开为用户主目录（如 ~/temp -> /home/user/temp 或 C:\Users\user\temp）
            # .resolve() 解析为绝对路径，并解析所有符号链接（如 ../temp -> /absolute/path/temp）
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
    def _create_temp_file(content: bytes, meta: Dict[str, Any], default_format: str = "mp4") -> str:
        """
        创建临时媒体文件
        
        用处：将媒体文件的二进制内容写入临时文件，用于后续处理。
        支持视频和音频格式，根据 meta 中的格式信息或使用默认格式。
        
        Args:
            content: 媒体文件的二进制内容
            meta: 媒体元数据，用于确定文件扩展名
            default_format: 默认文件格式（如果 meta 中未指定）
        
        Returns:
            str: 临时文件路径
            
        Raises:
            RuntimeError: 如果临时文件创建失败
        """
        # 优先从 meta 中获取格式，支持 video_format 和 audio_format
        file_format = (
            meta.get("video_format") or 
            meta.get("audio_format") or 
            meta.get("format") or 
            default_format
        )
        
        temp_dir = AudioVideoExtractorBase._get_temp_dir()
        
        # 创建临时文件
        # tempfile.NamedTemporaryFile 官方文档：https://docs.python.org/3/library/tempfile.html#tempfile.NamedTemporaryFile
        # 
        # 参数说明：
        # - suffix: 文件扩展名（如 ".mp4", ".mp3"），用于指定文件格式
        # - delete=False: 不自动删除文件（因为我们后续还要使用这个文件进行语音识别）
        # - dir: 临时文件目录路径（需要字符串格式，所以用 str(temp_dir) 转换）
        # 
        # 返回值：一个文件对象，可以通过 .name 获取文件路径，通过 .write() 写入数据
        temp_file = tempfile.NamedTemporaryFile(
            suffix=f".{file_format}",
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
        
        logger.debug(f"Created temp media file: {temp_file_path}")
        return temp_file_path
    
    @staticmethod
    def _cleanup_temp_file(file_path: str) -> None:
        """
        清理临时文件
        
        用处：删除不再需要的临时文件，释放磁盘空间。
        应该在 extract() 方法的 finally 块中调用，确保即使发生异常也能清理文件。
        
        Args:
            file_path: 临时文件路径
            
        说明：
            - 如果文件不存在，不会抛出异常（静默处理）
            - 如果删除失败，只记录警告日志，不抛出异常（避免影响主流程）
        """
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
                logger.debug(f"Cleaned up temp file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temp file {file_path}: {e}")
    
    @staticmethod
    def _extract_transcript(file_path: str) -> str:
        """
        提取语音转录文本
        
        用处：使用语音识别技术将音频/视频中的语音转换为文本，
        优先使用Whisper，回退到SpeechRecognition。
        
        优化：直接使用文件路径，避免重复创建临时文件。
        
        Args:
            file_path: 音频/视频文件路径（已存在的临时文件）
            
        Returns:
            str: 语音转录的文本内容，失败时返回空字符串
        """
        # 优先尝试使用 Whisper
        whisper_result = AudioVideoExtractorBase._try_extract_with_whisper(file_path)
        if whisper_result:
            return whisper_result
        
        # 如果 Whisper 失败，尝试使用 SpeechRecognition
        speech_recognition_result = AudioVideoExtractorBase._try_extract_with_speech_recognition(file_path)
        if speech_recognition_result:
            return speech_recognition_result
        
        logger.warning("No speech recognition library available or all methods failed")
        return ""
    
    @staticmethod
    def _try_extract_with_whisper(file_path: str) -> str:
        """
        尝试使用 Whisper 提取语音转录
        
        用处：尝试使用 Whisper 进行语音识别，捕获所有异常。
        
        Args:
            file_path: 音频/视频文件路径
            
        Returns:
            str: 语音转录的文本内容，失败时返回空字符串
        """
        try:
            return AudioVideoExtractorBase._extract_with_whisper(file_path)
        except ImportError:
            logger.debug("Whisper not available")
        except Exception as e:
            logger.warning(f"Whisper transcription failed: {e}")
        return ""
    
    @staticmethod
    def _try_extract_with_speech_recognition(file_path: str) -> str:
        """
        尝试使用 SpeechRecognition 提取语音转录
        
        用处：尝试使用 SpeechRecognition 进行语音识别，捕获所有异常。
        
        Args:
            file_path: 音频/视频文件路径
            
        Returns:
            str: 语音转录的文本内容，失败时返回空字符串
        """
        try:
            return AudioVideoExtractorBase._extract_with_speech_recognition(file_path)
        except ImportError:
            logger.debug("SpeechRecognition not available")
        except Exception as e:
            logger.warning(f"SpeechRecognition transcription failed: {e}")
        return ""
    
    @staticmethod
    def _extract_with_whisper(file_path: str) -> str:
        """
        使用Whisper进行语音识别
        
        用处：使用OpenAI的Whisper模型进行高质量的语音识别，
        支持多语言和长音频处理。Whisper会自动检测语言。
        
        优化：直接使用文件路径，不再创建临时文件。
        
        Args:
            file_path: 音频/视频文件路径（已存在的文件）
            
        Returns:
            str: 语音识别的文本结果，失败时抛出异常
            
        Raises:
            ImportError: Whisper 库未安装
            Exception: 其他处理异常
        """
        # 获取配置
        model_name = "base"
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
        
        # 使用通用模型加载器加载 Whisper 模型（自动缓存）
        from app.infra.models import ModelLoader, ModelType, LoaderConfig
        config = LoaderConfig(
            model_name=model_name,
            device="auto",
            model_type=ModelType.WHISPER
        )
        model = ModelLoader.load_model(config)
        
        # 不传递 language 参数，让 Whisper 自动检测语言
        result = model.transcribe(audio=file_path)
        
        # 验证返回结果
        if not result or not isinstance(result, dict):
            raise ValueError("Whisper returned invalid result format")
        
        text = result.get("text", "")
        if not text:
            logger.warning("Whisper transcription returned empty text")
        
        # 记录检测到的语言（用于调试）
        detected_language = result.get("language")
        if detected_language:
            logger.debug(f"Whisper 自动检测到语言: {detected_language}")
        
        return text
    
    @staticmethod
    def _extract_with_speech_recognition(file_path: str) -> str:
        """
        使用SpeechRecognition进行语音识别
        
        用处：使用SpeechRecognition库进行语音识别，
        作为Whisper的备选方案，支持Google语音识别API。
        
        优化：直接使用文件路径，不再创建临时文件。
        
        Args:
            file_path: 音频/视频文件路径（已存在的文件）
            
        Returns:
            str: 语音识别的文本结果，失败时抛出异常
            
        Raises:
            ImportError: SpeechRecognition 库未安装
            Exception: 其他处理异常
        """
        import speech_recognition as sr
        
        # SpeechRecognition 需要指定语言，使用默认值
        # 
        # 重要说明：
        # - Whisper 支持自动语言检测（不传 language 参数即可）
        # - SpeechRecognition 库（Google API）不支持自动语言检测，必须指定语言代码
        # - 因此这里使用固定默认值 'zh-CN'（中文）
        # - 如果识别其他语言，可能需要根据实际情况调整，或使用 Whisper（推荐）
        language: str = 'zh-CN'
        
        # 进行语音识别
        recognizer = sr.Recognizer()
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
        
        # SpeechRecognition 库的类型提示不完整，recognize_google 方法确实存在
        text = recognizer.recognize_google(audio, language=language)  # type: ignore[attr-defined]
        
        if not text:
            logger.warning("SpeechRecognition returned empty text")
            return ""
        
        return text
