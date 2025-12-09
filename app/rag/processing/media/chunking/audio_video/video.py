"""
视频分块策略
"""

from typing import List, Dict, Any
from .base import AudioVideoChunkingStrategyBase


class VideoChunkingStrategy(AudioVideoChunkingStrategyBase):
    """视频分块策略 - 基于字幕和时间戳"""
    
    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        """
        初始化视频分块策略
        
        用处：设置视频内容的分块参数，用于后续的字幕和转录文本分块处理。
        
        Args:
            chunk_size: 分块大小
            overlap: 重叠大小
        """
        super().__init__(chunk_size, overlap)
    
    def chunk(self, content: Dict[str, Any], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        视频内容分块，基于字幕和时间信息
        
        用处：从视频内容中提取字幕或转录文本进行分块处理，
        优先使用带时间戳的字幕信息，如果没有字幕则使用语音转写结果。
        
        Args:
            content: 视频内容，应包含字幕和可能的语音转写结果
            meta: 元数据
            
        Returns:
            List[Dict[str, Any]]: 分块结果列表，每个字典包含 "text" 和 "meta" 键
        """
        # content 应该包含字幕和可能的语音转写结果
        subtitles = content.get("subtitles", [])
        transcript = content.get("transcript", "")
        
        chunks = []
        chunk_index = 0
        
        if subtitles:
            # 优先使用字幕信息（带时间戳）
            chunks.extend(self._chunk_by_subtitles(subtitles, chunk_index))
        elif transcript:
            # 回退到语音转写结果（使用基类的公共方法）
            chunks.extend(self._chunk_by_transcript(transcript, chunk_index, chunk_type="video_transcript"))
        
        return chunks
    
    def _chunk_by_subtitles(self, subtitles: List[Dict[str, Any]], start_index: int) -> List[Dict[str, Any]]:
        """
        基于字幕分块（视频特有）
        
        用处：根据字幕的时间戳信息进行分块，保留时间信息，
        这是视频特有的功能，音频没有字幕。
        
        Args:
            subtitles: 字幕列表，每个字典包含 text、start_time、end_time
            start_index: 起始索引
            
        Returns:
            List[Dict[str, Any]]: 分块结果列表，每个字典包含 "text" 和 "meta" 键
        """
        chunks = []
        current_chunk = ""
        current_start_time = None
        current_end_time = None
        chunk_index = start_index
        
        for subtitle in subtitles:
            text = subtitle.get("text", "")
            start_time = subtitle.get("start_time", 0)
            end_time = subtitle.get("end_time", 0)
            
            if len(current_chunk) + len(text) <= self.chunk_size:
                # 可以添加到当前块
                if not current_chunk:
                    current_start_time = start_time
                current_chunk += " " + text if current_chunk else text
                current_end_time = end_time
            else:
                # 当前块已满，保存并开始新块
                if current_chunk:
                    chunk_meta = {
                        "chunk_index": chunk_index,
                        "chunk_type": "video_subtitle",
                        "chunk_size": len(current_chunk),
                        "start_time": current_start_time,
                        "end_time": current_end_time,
                    }
                    
                    chunks.append({
                        "text": current_chunk.strip(),
                        "meta": chunk_meta
                    })
                    chunk_index += 1
                
                # 开始新块
                current_chunk = text
                current_start_time = start_time
                current_end_time = end_time
        
        # 处理最后一个块
        if current_chunk:
            chunk_meta = {
                "chunk_index": chunk_index,
                "chunk_type": "video_subtitle",
                "chunk_size": len(current_chunk),
                "start_time": current_start_time,
                "end_time": current_end_time,
            }
            
            chunks.append({
                "text": current_chunk.strip(),
                "meta": chunk_meta
            })
        
        return chunks
