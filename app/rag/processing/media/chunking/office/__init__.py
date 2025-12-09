"""
Office文档分块策略模块

提供PDF、Word、Excel文档的分块策略。
"""

from .base import OfficeDocumentChunkingStrategyBase
from .pdf import PDFChunkingStrategy
from .word import WordChunkingStrategy
from .excel import ExcelChunkingStrategy

__all__ = [
    "OfficeDocumentChunkingStrategyBase",
    "PDFChunkingStrategy",
    "WordChunkingStrategy",
    "ExcelChunkingStrategy",
]
