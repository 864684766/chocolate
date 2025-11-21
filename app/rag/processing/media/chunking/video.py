"""
视频分块策略
"""

from typing import List, Dict, Any
from .base import MediaChunkingStrategy
from .text import TextChunkingStrategy


class VideoChunkingStrategy(MediaChunkingStrategy):
    """视频分块策略 - 基于字幕和时间戳"""
    
    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        """
        初始化视频分块策略
        
        Args:
            chunk_size: 分块大小
            overlap: 重叠大小
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, content: Dict[str, Any], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        视频内容分块，基于字幕和时间信息
        
        Args:
            content: 视频内容，应包含字幕和可能的语音转写结果
            meta: 元数据
            
        Returns:
            List[Dict[str, Any]]: 分块结果列表
        """
        # content 应该包含字幕和可能的语音转写结果
        subtitles = content.get("subtitles", [])
        transcript = content.get("transcript", "")
        
        chunks = []
        chunk_index = 0
        
        if subtitles:
            # 优先使用字幕信息
            chunks.extend(self._chunk_by_subtitles(subtitles, chunk_index))
        elif transcript:
            # 回退到语音转写结果
            chunks.extend(self._chunk_by_transcript(transcript, chunk_index))
        
        return chunks
    
    def _chunk_by_subtitles(self, subtitles: List[Dict[str, Any]], start_index: int) -> List[Dict[str, Any]]:
        """
        基于字幕分块
        
        Args:
            subtitles: 字幕列表
            start_index: 起始索引
            
        Returns:
            List[Dict[str, Any]]: 分块结果列表
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
    
    def _chunk_by_transcript(self, transcript: str, start_index: int) -> List[Dict[str, Any]]:
        """
        基于语音转写结果分块
        
        Args:
            transcript: 语音转写文本
            start_index: 起始索引
            
        Returns:
            List[Dict[str, Any]]: 分块结果列表
        """
        # 使用文本分块策略
        text_strategy = TextChunkingStrategy(self.chunk_size, self.overlap)
        text_chunks = text_strategy.chunk(transcript, {})
        
        # 转换元数据格式
        for i, chunk in enumerate(text_chunks):
            chunk["meta"].update({
                "chunk_index": start_index + i,
                "chunk_type": "video_transcript",
            })
        
        return text_chunks
