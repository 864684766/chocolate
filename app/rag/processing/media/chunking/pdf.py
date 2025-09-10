"""
PDF分块策略
"""

from typing import List, Dict, Any
from .base import MediaChunkingStrategy


class PDFChunkingStrategy(MediaChunkingStrategy):
    """PDF 分块策略 - 保留文档结构"""
    
    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        """
        初始化PDF分块策略
        
        Args:
            chunk_size: 分块大小
            overlap: 重叠大小
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, content: Dict[str, Any], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        PDF 内容分块，保留章节结构
        
        Args:
            content: PDF内容，应包含 {"pages": [...], "toc": [...]} 等结构
            meta: 元数据
            
        Returns:
            List[Dict[str, Any]]: 分块结果列表
        """
        # content 应该包含 {"pages": [...], "toc": [...]} 等结构
        pages = content.get("pages", [])
        
        chunks = []
        chunk_index = 0
        
        for page_num, page in enumerate(pages):
            page_text = page.get("text", "")
            page_meta = page.get("meta", {})
            
            if len(page_text) <= self.chunk_size:
                # 页面内容不长，直接作为一个块
                chunk_meta = meta.copy()
                chunk_meta.update({
                    "chunk_index": chunk_index,
                    "chunk_type": "pdf_page",
                    "page_number": page_num + 1,
                    "chunk_size": len(page_text),
                    **page_meta
                })
                
                chunks.append({
                    "text": page_text,
                    "meta": chunk_meta
                })
                chunk_index += 1
            else:
                # 页面内容过长，需要分块
                page_chunks = self._split_page_content(page_text, page_num, page_meta)
                for page_chunk in page_chunks:
                    chunk_meta = meta.copy()
                    chunk_meta.update({
                        "chunk_index": chunk_index,
                        "chunk_type": "pdf_page_part",
                        **page_chunk["meta"]
                    })
                    
                    chunks.append({
                        "text": page_chunk["text"],
                        "meta": chunk_meta
                    })
                    chunk_index += 1
        
        return chunks
    
    def _split_page_content(self, text: str, page_num: int, page_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        分割页面内容
        
        Args:
            text: 页面文本
            page_num: 页面号
            page_meta: 页面元数据
            
        Returns:
            List[Dict[str, Any]]: 分割后的页面块列表
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            
            chunks.append({
                "text": chunk_text,
                "meta": {
                    "page_number": page_num + 1,
                    "chunk_size": len(chunk_text),
                    "start_pos": start,
                    "end_pos": end,
                    **page_meta
                }
            })
            
            start = end - self.overlap if end - self.overlap > start else end
        
        return chunks
