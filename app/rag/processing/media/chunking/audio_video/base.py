"""
音频和视频分块策略基类

提取音频和视频处理转录文本的公共逻辑。
"""

from typing import List, Dict, Any
from ..base import MediaChunkingStrategy
from ..text import TextChunkingStrategy


class AudioVideoChunkingStrategyBase(MediaChunkingStrategy):
    """音频和视频分块策略基类
    
    用处：提取音频和视频处理转录文本的公共逻辑，
    避免代码重复，提高可维护性。
    """
    
    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        """
        初始化分块策略
        
        Args:
            chunk_size: 分块大小
            overlap: 重叠大小
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, content: Any, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        将媒体内容分块（默认实现，基于转录文本）
        
        用处：提供基于转录文本的默认分块实现，
        子类可以覆盖此方法以实现更复杂的分块逻辑（如视频的字幕分块）。
        
        Args:
            content: 媒体内容，可以是字典（包含 transcript）或字符串（转录文本）
            meta: 元数据信息
            
        Returns:
            List[Dict[str, Any]]: 分块结果列表，每个字典包含 "text" 和 "meta" 键
        """
        # 如果 content 是字典，提取 transcript
        if isinstance(content, dict):
            transcript = content.get("transcript", "")
        elif isinstance(content, str):
            transcript = content
        else:
            return []
        
        # 使用公共方法进行分块
        return self._chunk_by_transcript(transcript, start_index=0, chunk_type="transcript")
    
    def _chunk_by_transcript(
        self, 
        transcript: str, 
        start_index: int,
        chunk_type: str = "transcript"
    ) -> List[Dict[str, Any]]:
        """
        基于语音转写结果分块
        
        用处：将语音转写的文本内容进行分块处理，
        使用文本分块策略，并设置正确的 chunk_type。
        
        Args:
            transcript: 语音转写文本
            start_index: 起始索引
            chunk_type: 分块类型标识，用于区分音频和视频（如 "audio_transcript" 或 "video_transcript"）
            
        Returns:
            List[Dict[str, Any]]: 分块结果列表，每个字典包含 "text" 和 "meta" 键
        """
        if not transcript or not transcript.strip():
            return []
        
        # 使用文本分块策略
        text_strategy = TextChunkingStrategy(self.chunk_size, self.overlap)
        text_chunks = text_strategy.chunk(transcript, {})
        
        # 转换元数据格式，设置正确的 chunk_type
        for i, chunk in enumerate(text_chunks):
            chunk["meta"].update({
                "chunk_index": start_index + i,
                "chunk_type": chunk_type,
            })
        
        return text_chunks
