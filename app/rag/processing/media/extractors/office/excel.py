"""
Excel内容提取器

使用pandas提取Excel文档的内容。
支持.xlsx和.xls格式。
"""

import logging
import io
from typing import Dict, Any, List
from .base import OfficeDocumentExtractorBase

logger = logging.getLogger(__name__)


class ExcelExtractor(OfficeDocumentExtractorBase):
    """Excel内容提取器
    
    使用pandas提取Excel文档的数据、表格结构等信息。
    支持.xlsx和.xls格式。
    """
    
    def extract(self, content: bytes, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        从Excel文件中提取内容
        
        用处：从Excel文档中提取数据、表格结构等信息，
        为后续的RAG处理提供结构化数据。
        
        Args:
            content: Excel文件的二进制内容
            meta: Excel文件的元数据信息
            
        Returns:
            Dict[str, Any]: 提取结果字典，包含sheets等
        """
        if not self.is_available():
            logger.error("pandas不可用，无法提取Excel内容")
            raise RuntimeError("pandas未安装，无法提取Excel内容")
        
        # 规范化元数据
        self._normalize_metadata(meta)
        
        try:
            # 检查文件类型（CSV 或 Excel）
            filename = meta.get("filename", "")
            is_csv = filename.lower().endswith(".csv")
            
            if is_csv:
                # CSV 文件使用 read_csv 处理
                sheets_data = self._extract_csv_sheet(content)
            else:
                # Excel 文件使用 ExcelFile 处理
                excel_file = ExcelExtractor._open_excel_file(content, filename)
                sheets_data = self._extract_sheets(excel_file)
            
            result = {
                "sheets": sheets_data,
                "total_sheets": len(sheets_data)
            }
            
            if not self._validate_content(result):
                logger.warning("提取的Excel内容验证失败")
            
            return result
            
        except Exception as e:
            logger.error(f"Excel提取失败: {str(e)}", exc_info=True)
            raise RuntimeError(f"Excel提取失败: {str(e)}") from e
    
    @staticmethod
    def _detect_excel_engine(filename: str) -> str:
        """
        根据文件扩展名检测Excel引擎
        
        用处：根据文件扩展名返回合适的pandas Excel引擎。
        .xlsx 使用 openpyxl，.xls 使用 xlrd。
        
        Args:
            filename: 文件名
            
        Returns:
            str: Excel引擎名称（"openpyxl" 或 "xlrd"）
        """
        filename_lower = filename.lower()
        if filename_lower.endswith(".xlsx"):
            return "openpyxl"
        elif filename_lower.endswith(".xls"):
            return "xlrd"
        else:
            # 默认尝试 openpyxl
            return "openpyxl"
    
    @staticmethod
    def _open_excel_file(content: bytes, filename: str = ""):
        """
        打开Excel文件
        
        用处：从二进制内容创建pandas ExcelFile对象，明确指定引擎。
        
        Args:
            content: Excel文件的二进制内容
            filename: 文件名，用于判断引擎类型
            
        Returns:
            ExcelFile: pandas ExcelFile对象
            
        Raises:
            RuntimeError: 如果无法打开Excel文件
        """
        import pandas as pd
        
        excel_stream = io.BytesIO(content)
        engine = ExcelExtractor._detect_excel_engine(filename)
        
        try:
            return pd.ExcelFile(excel_stream, engine=engine)
        except Exception as e:
            # 如果指定引擎失败，尝试其他引擎
            logger.warning(f"使用引擎 {engine} 打开Excel文件失败: {str(e)}，尝试其他引擎")
            if engine == "openpyxl":
                try:
                    return pd.ExcelFile(excel_stream, engine="xlrd")
                except Exception:
                    pass
            elif engine == "xlrd":
                try:
                    return pd.ExcelFile(excel_stream, engine="openpyxl")
                except Exception:
                    pass
            raise RuntimeError(f"无法打开Excel文件，已尝试所有可用引擎: {str(e)}") from e
    
    @staticmethod
    def _extract_csv_sheet(content: bytes) -> List[Dict[str, Any]]:
        """
        提取CSV文件内容
        
        用处：CSV文件只有一个工作表，使用pandas read_csv读取。
        注意：CSV文件格式本身不支持多工作表，所以只返回一个工作表。
        
        Args:
            content: CSV文件的二进制内容
            
        Returns:
            List[Dict[str, Any]]: 工作表数据列表（CSV只有一个工作表，所以列表长度为1）
        """
        import pandas as pd
        
        csv_stream = io.BytesIO(content)
        
        try:
            # 尝试不同的编码和分隔符
            df = pd.read_csv(csv_stream, header=None, encoding='utf-8')
        except UnicodeDecodeError as e:
            # 如果UTF-8失败，尝试其他编码
            logger.warning(f"CSV文件UTF-8编码读取失败: {str(e)}，尝试GBK编码")
            csv_stream.seek(0)
            try:
                df = pd.read_csv(csv_stream, header=None, encoding='gbk')
            except UnicodeDecodeError as e2:
                # 如果GBK也失败，尝试Latin-1（几乎可以读取任何字节序列）
                logger.warning(f"CSV文件GBK编码读取失败: {str(e2)}，尝试Latin-1编码")
                csv_stream.seek(0)
                df = pd.read_csv(csv_stream, header=None, encoding='latin-1')
        
        # 转换为二维列表
        sheet_rows = ExcelExtractor._dataframe_to_rows(df)
        
        if not sheet_rows:
            return []
        
        # CSV文件只有一个工作表，使用默认名称"Sheet1"
        sheet_meta = ExcelExtractor._build_sheet_meta(0, "Sheet1", df)
        
        return [{
            "data": sheet_rows,
            "meta": sheet_meta
        }]
    
    @staticmethod
    def _extract_sheets(excel_file) -> List[Dict[str, Any]]:
        """
        提取所有工作表内容
        
        用处：遍历Excel文件所有工作表，提取每个工作表的数据和元数据。
        注意：Excel文件可以包含多个工作表，此方法会处理所有工作表。
        
        Args:
            excel_file: pandas ExcelFile对象
            
        Returns:
            List[Dict[str, Any]]: 工作表数据列表，每个元素包含data和meta
                                 列表长度等于Excel文件中的工作表数量
        """
        import pandas as pd
        
        sheets_data = []
        
        # 遍历Excel文件中的所有工作表（Excel支持多工作表）
        for sheet_idx, sheet_name in enumerate(excel_file.sheet_names):
            try:
                # 读取工作表数据
                df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
                
                # 转换为二维列表（保留所有数据，包括空值）
                sheet_rows = ExcelExtractor._dataframe_to_rows(df)
                
                if not sheet_rows:
                    continue
                
                # 构建工作表元数据
                sheet_meta = ExcelExtractor._build_sheet_meta(sheet_idx, sheet_name, df)
                
                sheets_data.append({
                    "data": sheet_rows,
                    "meta": sheet_meta
                })
                
            except Exception as e:
                logger.warning(f"提取工作表 '{sheet_name}' 失败: {str(e)}")
                continue
        
        return sheets_data
    
    @staticmethod
    def _dataframe_to_rows(df) -> List[List[str]]:
        """
        将DataFrame转换为行数据列表
        
        用处：将pandas DataFrame转换为二维列表格式，
        便于后续处理和分块。
        
        Args:
            df: pandas DataFrame对象
            
        Returns:
            List[List[str]]: 行数据列表，二维列表
        """
        import pandas as pd
        
        rows = []
        for _, row in df.iterrows():
            row_data = [str(cell) if pd.notna(cell) else "" for cell in row]
            # 过滤完全空的行
            if any(cell.strip() for cell in row_data):
                rows.append(row_data)
        return rows
    
    @staticmethod
    def _build_sheet_meta(sheet_idx: int, sheet_name: str, df) -> Dict[str, Any]:
        """
        构建工作表元数据
        
        用处：提取工作表的基础信息，这些信息会在分块策略中使用。
        注意：只提取metadata_whitelist中的字段，避免提取无用的字段。
        
        Args:
            sheet_idx: 工作表索引（从0开始）
            sheet_name: 工作表名称
            df: pandas DataFrame对象
            
        Returns:
            Dict[str, Any]: 工作表元数据字典
        """
        return {
            "sheet_index": sheet_idx + 1,  # 工作表索引从1开始
            "sheet_name": sheet_name,
            "max_row": len(df),
            "max_column": len(df.columns) if len(df) > 0 else 0
        }
    
    def _get_document_type(self) -> str:
        """
        获取文档类型
        
        Returns:
            str: "excel"
        """
        return "excel"
    
    def is_available(self) -> bool:
        """
        检查pandas是否可用
        
        Returns:
            bool: True表示pandas已安装，False表示未安装
        """
        try:
            import pandas as pd
            return True
        except ImportError:
            logger.warning("pandas未安装，Excel提取器不可用")
            return False
