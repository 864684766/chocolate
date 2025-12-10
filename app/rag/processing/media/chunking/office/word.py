"""
Word分块策略

保留Word文档的段落和表格结构。
"""

import logging
from typing import List, Dict, Any
from .base import OfficeDocumentChunkingStrategyBase

logger = logging.getLogger(__name__)

# 配置常量
MIN_PARAGRAPH_CHUNK_SIZE = 50  # 段落内容最小分块大小，小于此值不分块


class WordChunkingStrategy(OfficeDocumentChunkingStrategyBase):
    """Word分块策略
    
    保留Word文档的段落和表格结构进行分块。
    处理段落内容和表格内容，保留段落和表格信息。
    """
    
    def chunk(self, content: Dict[str, Any], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Word内容分块，保留段落和表格结构
        
        用处：将Word提取的内容进行分块处理，
        保留段落和表格结构，便于后续检索。
        
        Args:
            content: Word内容，应包含 {"paragraphs": [...], "tables": [...]} 等结构
            meta: 元数据
            
        Returns:
            List[Dict[str, Any]]: 分块结果列表，每个字典包含 "text" 和 "meta" 键
        """
        chunks = []
        chunk_index = 0
        
        # 处理段落内容
        paragraphs = content.get("paragraphs", [])
        para_chunks = self._chunk_paragraphs(paragraphs, meta, chunk_index)
        chunks.extend(para_chunks)
        chunk_index += len(para_chunks)
        
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
    
    def _chunk_paragraphs(self, paragraphs: List[Dict[str, Any]], 
                         base_meta: Dict[str, Any], start_index: int = 0) -> List[Dict[str, Any]]:
        """
        处理段落内容分块
        
        用处：对Word段落内容进行分块，如果段落内容过长则进一步分割。
        
        Args:
            paragraphs: 段落数据列表，每个元素包含 {"text": str, "meta": {"paragraph_index": int}}
            base_meta: 基础元数据
            start_index: 起始块索引
            
        Returns:
            List[Dict[str, Any]]: 段落分块结果列表
        """
        chunks = []
        chunk_index = start_index
        
        for para_data in paragraphs:
            para_text = para_data.get("text", "")
            para_meta = para_data.get("meta", {})
            para_index = para_meta.get("paragraph_index", 0)
            
            if not para_text.strip():
                continue
            
            # 合并基础元数据和段落元数据
            combined_meta = {**base_meta, **para_meta}
            
            # 如果段落内容较短，直接作为一个块
            if len(para_text) <= self.chunk_size:
                chunk_meta = combined_meta.copy()
                chunk_meta.update({
                    "chunk_index": chunk_index,
                    "chunk_type": "word_paragraph",
                    "chunk_size": len(para_text)
                })
                chunks.append({
                    "text": para_text,
                    "meta": chunk_meta
                })
                chunk_index += 1
            else:
                # 段落内容较长，使用基类方法分块
                para_chunks = self._split_text(para_text, combined_meta)
                for chunk in para_chunks:
                    chunk["meta"]["chunk_type"] = "word_paragraph"
                    chunk["meta"]["paragraph_index"] = para_index
                    chunk["meta"]["chunk_index"] = chunk_index
                    chunk_index += 1
                    chunks.append(chunk)
        
        return chunks
    
    def _chunk_tables(self, tables: List[Dict[str, Any]], 
                     base_meta: Dict[str, Any], start_index: int = 0) -> List[Dict[str, Any]]:
        """
        处理表格内容分块
        
        用处：将Word表格转换为文本格式后进行分块。
        
        Args:
            tables: 表格数据列表，每个元素包含 {"data": List[List[str]], "meta": {"table_index": int}}
            base_meta: 基础元数据
            start_index: 起始块索引
            
        Returns:
            List[Dict[str, Any]]: 表格分块结果列表
        """
        chunks = []
        chunk_index = start_index
        
        for table_data in tables:
            table_rows = table_data.get("data", [])
            table_meta = table_data.get("meta", {})
            table_index = table_meta.get("table_index", 0)
            
            if not table_rows:
                continue
            
            # 将表格转换为文本
            table_text = self._convert_table_to_text(table_rows)
            
            if not table_text.strip():
                continue
            
            # 合并基础元数据和表格元数据
            combined_meta = {**base_meta, **table_meta}
            
            # 如果表格内容较短，直接作为一个块
            if len(table_text) <= self.chunk_size:
                chunk_meta = combined_meta.copy()
                chunk_meta.update({
                    "chunk_index": chunk_index,
                    "chunk_type": "word_table",
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
                    chunk["meta"]["chunk_type"] = "word_table"
                    chunk["meta"]["table_index"] = table_index
                    chunk["meta"]["chunk_index"] = chunk_index
                    chunk_index += 1
                    chunks.append(chunk)
        
        return chunks
