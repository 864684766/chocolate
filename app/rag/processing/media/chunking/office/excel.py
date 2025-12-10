"""
Excel分块策略

按工作表和数据区域进行分块。
"""

import logging
from typing import List, Dict, Any
from .base import OfficeDocumentChunkingStrategyBase

logger = logging.getLogger(__name__)

# 配置常量
MIN_SHEET_CHUNK_SIZE = 100  # 工作表内容最小分块大小，小于此值不分块


class ExcelChunkingStrategy(OfficeDocumentChunkingStrategyBase):
    """Excel分块策略
    
    按工作表和数据区域进行分块。
    处理工作表内容，保留工作表信息。
    """
    
    def chunk(self, content: Dict[str, Any], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Excel内容分块，按工作表和数据区域
        
        用处：将Excel提取的内容进行分块处理，
        按工作表和数据区域组织，便于后续检索。
        
        Args:
            content: Excel内容，应包含 {"sheets": [...]} 等结构
            meta: 元数据
            
        Returns:
            List[Dict[str, Any]]: 分块结果列表，每个字典包含 "text" 和 "meta" 键
        """
        chunks = []
        chunk_index = 0
        
        # 处理工作表内容
        sheets = content.get("sheets", [])
        sheet_chunks = self._chunk_sheets(sheets, meta, chunk_index)
        chunks.extend(sheet_chunks)
        chunk_index += len(sheet_chunks)
        
        # 更新总块数
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk["meta"]["total_chunks"] = total_chunks
        
        return chunks
    
    def _chunk_sheets(self, sheets: List[Dict[str, Any]], 
                     base_meta: Dict[str, Any], start_index: int = 0) -> List[Dict[str, Any]]:
        """
        处理工作表内容分块
        
        用处：对Excel工作表内容进行分块，如果工作表内容过长则进一步分割。
        
        Args:
            sheets: 工作表数据列表，每个元素包含 {"data": List[List[str]], "meta": {...}}
            base_meta: 基础元数据
            start_index: 起始块索引
            
        Returns:
            List[Dict[str, Any]]: 工作表分块结果列表
        """
        chunks = []
        chunk_index = start_index
        
        for sheet_data in sheets:
            sheet_rows = sheet_data.get("data", [])
            sheet_meta = sheet_data.get("meta", {})
            sheet_index = sheet_meta.get("sheet_index", 0)
            sheet_name = sheet_meta.get("sheet_name", "")
            
            if not sheet_rows:
                continue
            
            # 将表格转换为文本
            sheet_text = self._convert_table_to_text(sheet_rows)
            
            if not sheet_text.strip():
                continue
            
            # 合并基础元数据和工作表元数据
            combined_meta = {**base_meta, **sheet_meta}
            
            # 如果工作表内容较短，直接作为一个块
            if len(sheet_text) <= self.chunk_size:
                chunk_meta = combined_meta.copy()
                chunk_meta.update({
                    "chunk_index": chunk_index,
                    "chunk_type": "excel_sheet",
                    "chunk_size": len(sheet_text)
                })
                chunks.append({
                    "text": sheet_text,
                    "meta": chunk_meta
                })
                chunk_index += 1
            else:
                # 工作表内容较长，使用基类方法分块
                sheet_chunks = self._split_text(sheet_text, combined_meta)
                for chunk in sheet_chunks:
                    chunk["meta"]["chunk_type"] = "excel_sheet"
                    chunk["meta"]["sheet_index"] = sheet_index
                    chunk["meta"]["sheet_name"] = sheet_name
                    chunk["meta"]["chunk_index"] = chunk_index
                    chunk_index += 1
                    chunks.append(chunk)
        
        return chunks
