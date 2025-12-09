"""
PDF内容提取器

使用PyMuPDF提取PDF文档的内容。
"""

import logging
import io
from typing import Dict, Any, List
from .base import OfficeDocumentExtractorBase

logger = logging.getLogger(__name__)

# 配置常量
DEFAULT_ENCODING = "utf-8"
TABLE_EXTRACTION_ENABLED = True
TOC_EXTRACTION_ENABLED = True


class PDFExtractor(OfficeDocumentExtractorBase):
    """PDF内容提取器
    
    使用PyMuPDF（Fitz）提取PDF文档的文本、表格、图片等内容。
    """
    
    def extract(self, content: bytes, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        从PDF文件中提取内容
        
        用处：从PDF文档中提取文本、表格、元数据等信息，
        为后续的RAG处理提供结构化数据。
        
        Args:
            content: PDF文件的二进制内容
            meta: PDF文件的元数据信息
            
        Returns:
            Dict[str, Any]: 提取结果字典，包含pages、toc、tables等
        """
        if not self.is_available():
            logger.error("PyMuPDF不可用，无法提取PDF内容")
            raise RuntimeError("PyMuPDF未安装，无法提取PDF内容")
        
        # 规范化元数据（虽然当前未直接使用，但保持一致性）
        self._normalize_metadata(meta)
        
        try:
            pdf_doc = PDFExtractor._open_pdf(content)
            pages_data = self._extract_pages(pdf_doc)
            toc_data = PDFExtractor._extract_toc(pdf_doc) if TOC_EXTRACTION_ENABLED else []
            tables_data = self._extract_tables(pdf_doc) if TABLE_EXTRACTION_ENABLED else []
            
            result = {
                "pages": pages_data,
                "toc": toc_data,
                "tables": tables_data,
                "total_pages": len(pages_data)
            }
            
            pdf_doc.close()
            
            if not self._validate_content(result):
                logger.warning("提取的PDF内容验证失败")
            
            return result
            
        except Exception as e:
            logger.error(f"PDF提取失败: {str(e)}", exc_info=True)
            raise RuntimeError(f"PDF提取失败: {str(e)}") from e
    
    @staticmethod
    def _open_pdf(content: bytes) -> Any:
        """
        打开PDF文档
        
        用处：从二进制内容创建PyMuPDF文档对象。
        
        Args:
            content: PDF文件的二进制内容
            
        Returns:
            fitz.Document: PyMuPDF文档对象
        """
        import fitz  # PyMuPDF
        pdf_stream = io.BytesIO(content)
        return fitz.open(stream=pdf_stream, filetype="pdf")
    
    @staticmethod
    def _extract_pages(pdf_doc: Any) -> List[Dict[str, Any]]:
        """
        提取所有页面的文本内容
        
        用处：遍历PDF所有页面，提取每页的文本内容和元数据。
        注意：meta中只包含会被metadata_whitelist保留的字段（如page_number）。
        如果分块策略需要其他信息（如width/height），可以从pages数据中获取。
        
        Args:
            pdf_doc: PyMuPDF文档对象
            
        Returns:
            List[Dict[str, Any]]: 页面数据列表，每个元素包含text和meta
        """
        pages_data = []
        total_pages = len(pdf_doc)
        
        for page_num in range(total_pages):
            page = pdf_doc[page_num]
            page_text = PDFExtractor._extract_page_text(page)
            page_meta = PDFExtractor._build_page_meta(page_num)
            
            pages_data.append({
                "text": page_text,
                "meta": page_meta
            })
        
        return pages_data
    
    @staticmethod
    def _extract_page_text(page: Any) -> str:
        """
        提取单页文本内容
        
        用处：从PDF页面对象中提取文本内容。
        
        Args:
            page: PyMuPDF页面对象
            
        Returns:
            str: 页面文本内容
        """
        return page.get_text()
    
    @staticmethod
    def _build_page_meta(page_num: int) -> Dict[str, Any]:
        """
        构建页面元数据
        
        用处：提取页面的基础信息，这些信息会在分块策略中使用。
        注意：只提取metadata_whitelist中的字段，避免提取无用的字段。
        如果分块策略需要其他信息（如width/height），可以从content的pages数据中获取。
        
        Args:
            page_num: 页码（从0开始）
            
        Returns:
            Dict[str, Any]: 页面元数据字典，只包含会被保留的字段（page_number）
        """
        return {
            "page_number": page_num + 1  # 在metadata_whitelist中，会被保留
        }
    
    @staticmethod
    def _extract_toc(pdf_doc: Any) -> List[Dict[str, Any]]:
        """
        提取PDF目录结构
        
        用处：提取PDF的目录（TOC）信息，用于保留文档结构。
        
        Args:
            pdf_doc: PyMuPDF文档对象
            
        Returns:
            List[Dict[str, Any]]: 目录项列表，每个元素包含level、title、page等
        """
        toc_items = []
        raw_toc = pdf_doc.get_toc()
        
        for item in raw_toc:
            toc_items.append({
                "level": item[0],
                "title": item[1],
                "page": item[2]
            })
        
        return toc_items
    
    def _extract_tables(self, pdf_doc: Any) -> List[Dict[str, Any]]:
        """
        提取PDF中的表格
        
        用处：提取PDF文档中的表格数据，使用PyMuPDF的表格检测功能。
        
        Args:
            pdf_doc: PyMuPDF文档对象
            
        Returns:
            List[Dict[str, Any]]: 表格数据列表，每个元素包含page、table_data等
        """
        tables_data = []
        
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            page_tables = self._extract_page_tables(page, page_num)
            tables_data.extend(page_tables)
        
        return tables_data
    
    @staticmethod
    def _extract_page_tables(page: Any, page_num: int) -> List[Dict[str, Any]]:
        """
        提取单页中的表格
        
        用处：从PDF页面中检测并提取表格数据。
        
        Args:
            page: PyMuPDF页面对象
            page_num: 页码（从0开始）
            
        Returns:
            List[Dict[str, Any]]: 该页的表格数据列表
        """
        page_tables = []
        
        try:
            # PyMuPDF 1.23+ 支持 find_tables()
            tables = page.find_tables()
            
            for table_idx, table in enumerate(tables):
                table_data = PDFExtractor._convert_table_to_list(table)
                page_tables.append({
                    "page": page_num + 1,
                    "table_index": table_idx,
                    "data": table_data,
                    "row_count": len(table_data),
                    "col_count": len(table_data[0]) if table_data else 0
                })
        except AttributeError:
            # 如果PyMuPDF版本不支持find_tables，记录警告
            logger.debug(f"PyMuPDF版本不支持表格提取，页面 {page_num + 1} 跳过表格提取")
        except Exception as e:
            logger.warning(f"提取页面 {page_num + 1} 的表格时出错: {str(e)}")
        
        return page_tables
    
    @staticmethod
    def _convert_table_to_list(table: Any) -> List[List[str]]:
        """
        将表格对象转换为二维列表
        
        用处：将PyMuPDF的表格对象转换为可序列化的列表格式。
        
        Args:
            table: PyMuPDF表格对象
            
        Returns:
            List[List[str]]: 二维列表，每行是一个列表
        """
        table_data = []
        
        for row in table.extract():
            table_data.append([str(cell) if cell else "" for cell in row])
        
        return table_data
    
    def _get_document_type(self) -> str:
        """
        获取文档类型
        
        用处：返回当前提取器处理的文档类型标识。
        
        Returns:
            str: "pdf"
        """
        return "pdf"
    
    def is_available(self) -> bool:
        """
        检查PyMuPDF是否可用
        
        用处：检查当前环境是否已安装PyMuPDF库。
        
        Returns:
            bool: True表示PyMuPDF已安装，False表示未安装
        """
        try:
            import fitz  # PyMuPDF
            return True
        except ImportError:
            logger.warning("PyMuPDF未安装，PDF提取器不可用")
            return False
