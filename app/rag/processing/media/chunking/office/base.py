"""
Office文档分块策略基类

提取Office文档（PDF、Word、Excel）处理中的公共分块逻辑。
"""

from abc import abstractmethod
from typing import List, Dict, Any, Tuple
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
        
        用处：设置分块大小和重叠大小参数。
        
        Args:
            chunk_size: 分块大小（字符数）
            overlap: 重叠大小（字符数）
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    @abstractmethod
    def chunk(self, content: Dict[str, Any], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        将Office文档内容分块
        
        用处：子类必须实现此方法来处理特定类型的Office文档分块。
        
        Args:
            content: Office文档内容，类型根据具体策略而定
            meta: 元数据信息
            
        Returns:
            List[Dict[str, Any]]: 分块结果列表，每个字典包含 "text" 和 "meta" 键
        """
        pass
    
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
    
    @staticmethod
    def _detect_table_header(table: List[List[str]]) -> Tuple[List[str] | None, List[List[str]]]:
        """
        检测表格表头
        
        用处：判断第一行是否为表头，如果是则分离表头和数据行。
        使用简单启发式：第一行平均长度较短且后续有数据行，则认为是表头。
        
        Args:
            table: 表格数据，二维列表
            
        Returns:
            tuple: (表头列表或None, 数据行列表)
        """
        if not table or len(table) < 2:
            return None, table
        
        first_row = table[0]
        data_rows = table[1:]
        
        # 简单启发式：如果第一行都是短文本（可能是列名），且后续行有数据，则认为是表头
        first_row_avg_len = sum(len(str(cell).strip()) for cell in first_row) / len(first_row) if first_row else 0
        if first_row_avg_len < 20 and len(data_rows) > 0:
            return first_row, data_rows
        
        return None, table
    
    @staticmethod
    def _convert_table_to_text(table: List[List[str]]) -> str:
        """
        将表格转换为文本格式
        
        用处：将表格数据转换为可读的文本格式，
        使用自然语言连接符，对语义搜索更友好。
        
        Args:
            table: 表格数据，二维列表
            
        Returns:
            str: 转换后的文本，使用自然语言描述表格内容
        """
        if not table:
            return ""
        
        # 检测表头
        headers, data_rows = OfficeDocumentChunkingStrategyBase._detect_table_header(table)
        
        if headers:
            # 有表头：使用自然语言描述
            return OfficeDocumentChunkingStrategyBase._convert_table_with_headers(headers, data_rows)
        else:
            # 无表头：使用简单格式
            return OfficeDocumentChunkingStrategyBase._convert_table_simple(table)
    
    @staticmethod
    def _convert_table_with_headers(headers: List[str], data_rows: List[List[str]]) -> str:
        """
        将带表头的表格转换为自然语言文本
        
        用处：使用自然语言连接符（如"是"）将表格转换为语义友好的文本。
        格式：列名是值，列名是值
        
        Args:
            headers: 表头列表（列名）
            data_rows: 数据行列表
            
        Returns:
            str: 自然语言描述的文本
        """
        lines = []
        
        for row in data_rows:
            # 确保行数据长度与表头一致
            row_data = row[:len(headers)]
            if len(row_data) < len(headers):
                row_data.extend([""] * (len(headers) - len(row_data)))
            
            # 构建自然语言描述：列名是值
            parts = []
            for i, header in enumerate(headers):
                cell_value = str(row_data[i]).strip() if i < len(row_data) else ""
                if cell_value:
                    # 使用"是"连接：原文是xxx
                    header_str = str(header).strip()
                    if header_str:
                        parts.append(f"{header_str}是{cell_value}")
            
            if parts:
                line = "，".join(parts)
                lines.append(line)
        
        return "\n".join(lines)
    
    @staticmethod
    def _convert_table_simple(table: List[List[str]]) -> str:
        """
        将无表头的表格转换为简单文本格式
        
        用处：当无法识别表头时，使用空格分隔的简单格式。
        
        Args:
            table: 表格数据，二维列表
            
        Returns:
            str: 转换后的文本
        """
        lines = []
        for row in table:
            # 过滤空单元格，使用空格连接
            cells = [str(cell).strip() for cell in row if str(cell).strip()]
            if cells:
                line = " ".join(cells)
                lines.append(line)
        return "\n".join(lines)
