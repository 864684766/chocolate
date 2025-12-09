"""
字幕提取辅助方法
"""

import logging
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any, TYPE_CHECKING

# 尝试导入 ffmpeg-python 包（安装后导入名为 ffmpeg）
# 注意：ffmpeg-python 是 PyPI 包名，导入时使用 import ffmpeg
# 官方文档：https://github.com/kkroening/ffmpeg-python
if TYPE_CHECKING:
    # 类型检查时导入，用于类型提示
    import ffmpeg  # 这是 ffmpeg-python 包
else:
    # 运行时导入，处理 ImportError
    try:
        import ffmpeg  # type: ignore  # 这是 ffmpeg-python 包
    except ImportError:
        ffmpeg = None  # type: ignore

logger = logging.getLogger(__name__)

# 字幕格式常量
SUBTITLE_FORMAT_SRT = "srt"
SUBTITLE_FORMAT_VTT = "vtt"
SUBTITLE_FORMAT_ASS = "ass"
SUBTITLE_FORMAT_SSA = "ssa"

# 默认字幕格式列表（按优先级排序）
DEFAULT_SUBTITLE_FORMATS = [SUBTITLE_FORMAT_SRT, SUBTITLE_FORMAT_VTT, SUBTITLE_FORMAT_ASS, SUBTITLE_FORMAT_SSA]


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


def extract_embedded_subtitles(video_path: str) -> List[Dict[str, Any]]:
    """
    从视频文件中提取内嵌字幕
    
    用处：使用 ffmpeg 从视频文件中提取内嵌字幕流，
    支持 SRT、VTT、ASS、SSA 等格式。
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        List[Dict[str, Any]]: 字幕列表，每个字典包含：
            - text (str): 字幕文本内容
            - start_time (float): 开始时间（秒）
            - end_time (float): 结束时间（秒）
            如果提取失败或没有字幕，返回空列表
    """
    # 检查 ffmpeg 是否可用
    if ffmpeg is None:
        logger.warning("ffmpeg-python not available for subtitle extraction")
        return []
    
    subtitles = []
    temp_subtitle_path = None
    
    try:
        # 确保使用绝对路径（ffmpeg 在 Windows 和 Linux 下都需要绝对路径）
        # 使用 os.path.abspath() 可以跨平台处理路径分隔符
        video_path_abs = os.path.abspath(video_path)
        
        # 检查视频文件是否存在
        if not os.path.exists(video_path_abs):
            logger.warning(f"Video file not found: {video_path_abs} (original: {video_path})")
            return []
        
        # 检查文件是否可读
        if not os.access(video_path_abs, os.R_OK):
            logger.warning(f"Video file is not readable: {video_path_abs}")
            return []
        
        # 检查文件大小（确保文件已完全写入）
        file_size = os.path.getsize(video_path_abs)
        if file_size == 0:
            logger.warning(f"Video file is empty: {video_path_abs}")
            return []
        
        logger.debug(f"Probing video file: {video_path_abs} (size: {file_size} bytes)")
        
        # 尝试提取字幕流（使用绝对路径）
        # ffmpeg.probe() 官方文档：https://github.com/kkroening/ffmpeg-python/tree/master/ffmpeg
        # API 参考：https://github.com/kkroening/ffmpeg-python/blob/master/ffmpeg/_probe.py
        # 用法：probe_result = ffmpeg.probe(video_path)
        probe_result = ffmpeg.probe(video_path_abs)
        subtitle_streams = _find_subtitle_streams(probe_result)
        
        if not subtitle_streams:
            logger.debug(f"No subtitle streams found in video: {video_path}")
            return []
        
        # 从配置中获取临时文件目录，在此目录下创建临时字幕文件
        # tempfile.NamedTemporaryFile 官方文档：https://docs.python.org/3/library/tempfile.html#tempfile.NamedTemporaryFile
        temp_dir = _get_temp_dir()
        temp_subtitle_file = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.srt', 
            delete=False, 
            dir=str(temp_dir),
            encoding='utf-8'
        )
        temp_subtitle_path = temp_subtitle_file.name
        temp_subtitle_file.close()
        logger.debug(f"Created temp subtitle file: {temp_subtitle_path}")
        
        # 提取第一个可用的字幕流
        # 使用字幕流在字幕流列表中的索引（不是全局流索引）
        # map 参数格式为 '0:s:0' 表示输入0的第0个字幕流
        # ffmpeg.input() 和 ffmpeg.output() 官方文档：https://github.com/kkroening/ffmpeg-python/tree/master/ffmpeg
        # API 参考：
        #   - input/output: https://github.com/kkroening/ffmpeg-python/blob/master/ffmpeg/__init__.py
        #   - run: https://github.com/kkroening/ffmpeg-python/blob/master/ffmpeg/_run.py
        # 用法：ffmpeg.input(path).output(path, **kwargs).overwrite_output().run(quiet=True, check=True)
        subtitle_index = 0
        # 使用绝对路径确保 ffmpeg 能找到文件（Windows 和 Linux 都需要绝对路径）
        (
            ffmpeg
            .input(video_path_abs)
            .output(
                temp_subtitle_path,
                **{'map': f'0:s:{subtitle_index}'},
                format='srt'
            )
            .overwrite_output()
            .run(quiet=True, check=True)
        )
        
        # 解析字幕文件
        subtitles = _parse_srt_file(temp_subtitle_path)
        
    except FileNotFoundError as e:
        logger.warning(f"Video file not found for subtitle extraction: {video_path}, error: {e}")
    except Exception as e:
        logger.warning(f"Failed to extract embedded subtitles from {video_path}: {e}", exc_info=True)
    finally:
        # 清理临时文件
        if temp_subtitle_path and os.path.exists(temp_subtitle_path):
            try:
                os.unlink(temp_subtitle_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp subtitle file: {e}")
    
    return subtitles


def _find_subtitle_streams(probe_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    查找视频中的字幕流
    
    用处：从 ffprobe 结果中查找所有字幕流信息。
    
    Args:
        probe_result: ffprobe 探测结果
        
    Returns:
        List[Dict[str, Any]]: 字幕流列表
    """
    subtitle_streams = []
    streams = probe_result.get('streams', [])
    
    for stream in streams:
        if stream.get('codec_type') == 'subtitle':
            subtitle_streams.append(stream)
    
    return subtitle_streams


def _parse_srt_file(srt_path: str) -> List[Dict[str, Any]]:
    """
    解析 SRT 字幕文件
    
    用处：将 SRT 格式的字幕文件解析为结构化数据。
    
    Args:
        srt_path: SRT 文件路径
        
    Returns:
        List[Dict[str, Any]]: 字幕列表，每个字典包含 text、start_time、end_time
    """
    subtitles = []
    
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 简单的 SRT 解析（可以后续优化）
        blocks = content.strip().split('\n\n')
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue
            
            # 解析时间戳行（SRT 格式：序号、时间戳、文本内容）
            # 时间戳格式示例：00:00:01,234 --> 00:00:03,456
            # 其中 "00:00:01,234" 是开始时间，"00:00:03,456" 是结束时间
            time_line = lines[1]
            if '-->' in time_line:
                # 按箭头符号分割时间戳，得到开始时间和结束时间
                time_parts = time_line.split('-->')
                if len(time_parts) == 2:
                    # 将时间戳字符串转换为秒数
                    start_time = _srt_time_to_seconds(time_parts[0].strip())
                    end_time = _srt_time_to_seconds(time_parts[1].strip())
                    # 提取字幕文本（从第3行开始，可能有多行）
                    text = '\n'.join(lines[2:]).strip()
                    
                    if text:
                        subtitles.append({
                            'text': text,
                            'start_time': start_time,
                            'end_time': end_time
                        })
    
    except Exception as e:
        logger.warning(f"Failed to parse SRT file: {e}")
    
    return subtitles


def _srt_time_to_seconds(time_str: str) -> float:
    """
    将 SRT 时间格式转换为秒数
    
    用处：将 SRT 格式的时间字符串（如 "00:01:23,456"）转换为秒数。
    
    Args:
        time_str: SRT 时间字符串，格式为 HH:MM:SS,mmm
        
    Returns:
        float: 对应的秒数
    """
    try:
        # 替换逗号为点，以便处理毫秒
        time_str = time_str.replace(',', '.')
        parts = time_str.split(':')
        
        if len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
    except Exception as e:
        logger.warning(f"Failed to parse time string '{time_str}': {e}")
    
    return 0.0


def generate_subtitles_with_whisper(
    video_path: str, 
    model_name: str = "base"
) -> List[Dict[str, Any]]:
    """
    使用 Whisper 生成带时间戳的字幕
    
    用处：当视频没有内嵌字幕时，使用 Whisper 进行语音识别
    并生成带时间戳的字幕数据。Whisper 会自动检测语言。
    
    Args:
        video_path: 视频文件路径
        model_name: Whisper 模型名称（如 "base", "small", "medium"）
        
    Returns:
        List[Dict[str, Any]]: 字幕列表，每个字典包含 text、start_time、end_time
    """
    try:
        import whisper
    except ImportError:
        logger.warning("Whisper not available for subtitle generation")
        return []
    
    subtitles = []
    
    try:
        # 使用通用模型加载器加载 Whisper 模型（自动缓存）
        from app.infra.models import ModelLoader, ModelType, LoaderConfig
        config = LoaderConfig(
            model_name=model_name,
            device="auto",
            model_type=ModelType.WHISPER
        )
        model = ModelLoader.load_model(config)
        
        # 不传递 language 参数，让 Whisper 自动检测语言
        result = model.transcribe(audio=video_path)
        
        # 从 segments 中提取字幕
        segments = result.get('segments', [])
        for segment in segments:
            text = segment.get('text', '').strip()
            start_time = segment.get('start', 0.0)
            end_time = segment.get('end', 0.0)
            
            if text:
                subtitles.append({
                    'text': text,
                    'start_time': start_time,
                    'end_time': end_time
                })
    
    except Exception as e:
        logger.warning(f"Failed to generate subtitles with Whisper: {e}")
    
    return subtitles
