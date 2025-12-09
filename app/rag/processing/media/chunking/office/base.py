"""
Office文档分块策略基类

提取Office文档（PDF、Word、Excel）处理中的公共分块逻辑。
"""

from typing import List, Dict, Any
from ..base import MediaChunkingStrategy


class OfficeDocumentChunkingStrategyBase(MediaChunkingStrategy):
    """Office文档分块策略基类
    
    提供Office文档处理中的公共分块功能：
    - 段落分块
    - 表格分块
    - 文本转换
    
    子类需要实现或覆盖 chunk() 方法来处理特定的文档类型。
    """
    
    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        """
        初始化分块策略
        
        Args:
            chunk_size: 分块大小（字符数）
            overlap: 重叠大小（字符数）
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def _split_text(self, text: str, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        将文本按指定大小分块
        
        用处：提供通用的文本分块功能，
        子类可以调用此方法进行基础文本分块。
        
        Args:
            text: 要分块的文本
            meta: 元数据信息
            
        Returns:
            List[Dict[str, Any]]: 分块结果列表，每个字典包含 "text" 和 "meta" 键
        """
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            
            chunk_meta = meta.copy()
            chunk_meta.update({
                "chunk_index": chunk_index,
                "chunk_size": len(chunk_text)
                # 注意：start_pos和end_pos不在metadata_whitelist中，已被移除
            })
            
            chunks.append({
                "text": chunk_text,
                "meta": chunk_meta
            })
            
            start = end - self.overlap if end - self.overlap > start else end
            chunk_index += 1
        
        return chunks
    
    def _convert_table_to_text(self, table: List[List[str]]) -> str:
        """
        将表格转换为文本格式
        
        用处：将表格数据转换为可读的文本格式，
        便于后续的文本分块处理。
        
        Args:
            table: 表格数据，二维列表
            
        Returns:
            str: 转换后的文本
        """
        lines = []
        for row in table:
            line = " | ".join(str(cell) for cell in row)
            lines.append(line)
        return "\n".join(lines)
