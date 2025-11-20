"""
文本分块策略
"""

from typing import List, Dict, Any
from .base import MediaChunkingStrategy
from app.config import get_config_manager


class TextChunkingStrategy(MediaChunkingStrategy):
    """文本分块策略 - 使用 LangChain 智能分块"""
    
    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        """
        初始化文本分块策略
        
        Args:
            chunk_size: 分块大小
            overlap: 重叠大小
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, content: str, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        智能文本分块
        
        Args:
            content: 文本内容
            meta: 元数据
            
        Returns:
            List[Dict[str, Any]]: 分块结果列表
        """
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            
            # 从配置文件读取分隔符（language_processing.chinese.chunking.separators）
            config_manager = get_config_manager()
            chinese_config = config_manager.get_chinese_processing_config()
            chunking_config = chinese_config.get("chunking", {})
            separators = chunking_config.get("separators", [
                "\n\n",      # 段落分隔
                "\n",        # 行分隔
                "。",        # 中文句号
                "！",        # 中文感叹号
                "？",        # 中文问号
                "；",        # 中文分号
                "，",        # 中文逗号
                " ",         # 空格
                ""           # 字符级别
            ])
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
                separators=separators,
                length_function=len
            )
            
            chunks = splitter.split_text(content)
            
            # 为每个块添加元数据
            result = []
            for i, chunk in enumerate(chunks):
                chunk_meta = meta.copy()
                chunk_meta.update({
                    "chunk_index": i,
                    "chunk_type": "text",
                    "chunk_size": len(chunk),
                    "total_chunks": len(chunks)
                })
                result.append({
                    "text": chunk,
                    "meta": chunk_meta
                })
            
            return result
            
        except ImportError:
            # 回退到简单分块
            return self._simple_chunk(content, meta)
    
    def _simple_chunk(self, content: str, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        简单分块（回退方案）
        
        Args:
            content: 文本内容
            meta: 元数据
            
        Returns:
            List[Dict[str, Any]]: 分块结果列表
        """
        chunks = []
        start = 0
        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            chunk_text = content[start:end]
            
            chunk_meta = meta.copy()
            chunk_meta.update({
                "chunk_index": len(chunks),
                "chunk_type": "text_simple",
                "chunk_size": len(chunk_text),
                "start_pos": start,
                "end_pos": end
            })
            
            chunks.append({
                "text": chunk_text,
                "meta": chunk_meta
            })
            
            start = end - self.overlap if end - self.overlap > start else end
        
        return chunks
