"""
PDF分块策略

保留PDF文档的页面结构和章节信息。
"""

import logging
from typing import List, Dict, Any
from .base import OfficeDocumentChunkingStrategyBase

logger = logging.getLogger(__name__)

# 配置常量
MIN_PAGE_CHUNK_SIZE = 100  # 页面内容最小分块大小，小于此值不分块


class PDFChunkingStrategy(OfficeDocumentChunkingStrategyBase):
    """PDF分块策略
    
    保留PDF文档的页面结构和章节信息进行分块。
    处理页面内容和表格内容，保留页面信息。
    """
    
    def chunk(self, content: Dict[str, Any], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        PDF内容分块，保留文档结构
        
        用处：将PDF提取的内容进行分块处理，
        保留页面和章节信息，便于后续检索。
        
        Args:
            content: PDF内容，应包含 {"pages": [...], "tables": [...]} 等结构
            meta: 元数据
            
        Returns:
            List[Dict[str, Any]]: 分块结果列表，每个字典包含 "text" 和 "meta" 键
        """
        chunks = []
        chunk_index = 0
        
        # 处理页面内容
        pages = content.get("pages", [])
        page_chunks = self._chunk_pages(pages, meta, chunk_index)
        chunks.extend(page_chunks)
        chunk_index += len(page_chunks)
        
        # 处理表格内容
        tables = content.get("tables", [])
        table_chunks = self._chunk_tables(tables, meta, chunk_index)
        chunks.extend(table_chunks)
        chunk_index += len(table_chunks)
        
        # 更新总块数
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk["meta"]["total_chunks"] = total_chunks
        
        return chunks
    
    def _chunk_pages(self, pages: List[Dict[str, Any]], base_meta: Dict[str, Any], start_index: int = 0) -> List[Dict[str, Any]]:
        """
        处理页面内容分块
        
        用处：对PDF页面内容进行分块，如果页面内容过长则进一步分割。
        
        Args:
            pages: 页面数据列表，每个元素包含 {"text": str, "meta": {"page_number": int}}
            base_meta: 基础元数据
            start_index: 起始块索引
            
        Returns:
            List[Dict[str, Any]]: 页面分块结果列表
        """
        chunks = []
        chunk_index = start_index
        
        for page_data in pages:
            page_text = page_data.get("text", "")
            page_meta = page_data.get("meta", {})
            page_number = page_meta.get("page_number", 0)
            
            if not page_text.strip():
                continue
            
            # 合并基础元数据和页面元数据
            combined_meta = {**base_meta, **page_meta}
            
            # 如果页面内容较短，直接作为一个块
            if len(page_text) <= self.chunk_size:
                chunk_meta = combined_meta.copy()
                chunk_meta.update({
                    "chunk_index": chunk_index,
                    "chunk_type": "pdf_page",
                    "chunk_size": len(page_text)
                })
                chunks.append({
                    "text": page_text,
                    "meta": chunk_meta
                })
                chunk_index += 1
            else:
                # 页面内容较长，使用基类方法分块
                page_chunks = self._split_text(page_text, combined_meta)
                for chunk in page_chunks:
                    chunk["meta"]["chunk_type"] = "pdf_page"
                    chunk["meta"]["page_number"] = page_number
                    chunk["meta"]["chunk_index"] = chunk_index
                    chunk_index += 1
                    chunks.append(chunk)
        
        return chunks
    
    def _chunk_tables(self, tables: List[Dict[str, Any]], base_meta: Dict[str, Any], start_index: int = 0) -> List[Dict[str, Any]]:
        """
        处理表格内容分块
        
        用处：将PDF表格转换为文本格式后进行分块。
        
        Args:
            tables: 表格数据列表，每个元素包含 {"page": int, "data": List[List[str]], ...}
            base_meta: 基础元数据
            start_index: 起始块索引
            
        Returns:
            List[Dict[str, Any]]: 表格分块结果列表
        """
        chunks = []
        chunk_index = start_index
        
        for table_data in tables:
            table_rows = table_data.get("data", [])
            page_number = table_data.get("page", 0)
            
            if not table_rows:
                continue
            
            # 将表格转换为文本
            table_text = self._convert_table_to_text(table_rows)
            
            if not table_text.strip():
                continue
            
            # 合并基础元数据和表格元数据
            combined_meta = {**base_meta, "page_number": page_number}
            
            # 如果表格内容较短，直接作为一个块
            if len(table_text) <= self.chunk_size:
                chunk_meta = combined_meta.copy()
                chunk_meta.update({
                    "chunk_index": chunk_index,
                    "chunk_type": "pdf_table",
                    "chunk_size": len(table_text)
                })
                chunks.append({
                    "text": table_text,
                    "meta": chunk_meta
                })
                chunk_index += 1
            else:
                # 表格内容较长，使用基类方法分块
                table_chunks = self._split_text(table_text, combined_meta)
                for chunk in table_chunks:
                    chunk["meta"]["chunk_type"] = "pdf_table"
                    chunk["meta"]["page_number"] = page_number
                    chunk["meta"]["chunk_index"] = chunk_index
                    chunk_index += 1
                    chunks.append(chunk)
        
        return chunks
