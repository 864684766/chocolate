"""
音频分块策略
"""

from typing import List, Dict, Any
from .base import AudioVideoChunkingStrategyBase


class AudioChunkingStrategy(AudioVideoChunkingStrategyBase):
    """音频分块策略 - 基于语音转写文本"""
    
    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        """
        初始化音频分块策略
        
        用处：设置音频内容的分块参数，用于后续的文本分块处理。
        
        Args:
            chunk_size: 分块大小
            overlap: 重叠大小
        """
        super().__init__(chunk_size, overlap)
    
    def chunk(self, content: Dict[str, Any], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        音频内容分块，基于语音转写文本
        
        用处：从音频内容中提取语音转写文本进行分块处理，
        音频只有转录文本，没有字幕信息。
        
        Args:
            content: 音频内容，应包含语音转写结果（transcript）
            meta: 元数据
            
        Returns:
            List[Dict[str, Any]]: 分块结果列表，每个字典包含 "text" 和 "meta" 键
        """
        # 音频内容只包含转录文本
        transcript = content.get("transcript", "")
        
        if not transcript:
            return []
        
        # 使用基类的公共方法进行分块，设置正确的 chunk_type
        return self._chunk_by_transcript(transcript, start_index=0, chunk_type="audio_transcript")
