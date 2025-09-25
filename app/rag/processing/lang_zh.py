from __future__ import annotations

from typing import List, Dict, Any
import re

from .interfaces import LanguageProcessor
from app.config import get_config_manager


class ChineseProcessor(LanguageProcessor):
    """改进版中文处理器：使用 LangChain 智能分块 + 中文优化。
    
    特性：
    - 智能分块：基于语义边界，避免破坏句子完整性
    - 中文优化：支持中文标点、段落分割
    - 可配置：支持多种分块策略和参数
    """

    def __init__(self, chunk_size: int = None, overlap: int = None, 
                 use_langchain: bool = None) -> None:
        """
        初始化中文处理器
        
        Args:
            chunk_size: 分块大小，如果为None则从配置中读取
            overlap: 重叠大小，如果为None则从配置中读取  
            use_langchain: 是否使用LangChain，如果为None则从配置中读取
        """
        # 获取配置管理器
        config_manager = get_config_manager()
        chinese_config = config_manager.get_chinese_processing_config()
        chunking_config = chinese_config.get("chunking", {})
        
        # 使用传入参数或配置中的默认值
        self.chunk_size = chunk_size if chunk_size is not None else chunking_config.get("default_chunk_size", 800)
        self.overlap = overlap if overlap is not None else chunking_config.get("default_overlap", 150)
        self.use_langchain = use_langchain if use_langchain is not None else chunking_config.get("use_langchain", True)
        
        # 从配置中获取其他参数
        self._config = chinese_config

    def clean(self, text: str) -> str:
        """
        清洗文本，优化中文处理。
        - 统一换行符和空白字符
        - 保留中文标点符号
        - 归一化全角/半角字符
        """
        if not text:
            return ""
        
        # 从配置中获取文本清洗参数
        text_cleaning_config = self._config.get("text_cleaning", {})
        whitespace_pattern = text_cleaning_config.get("normalize_whitespace_pattern", "[ \\t]+")
        whitespace_replacement = text_cleaning_config.get("normalize_whitespace_replacement", " ")
        newlines_pattern = text_cleaning_config.get("normalize_newlines_pattern", "\\n\\s*\\n\\s*\\n+")
        newlines_replacement = text_cleaning_config.get("normalize_newlines_replacement", "\n\n")
        
        # 统一换行符
        t = text.replace("\r\n", "\n").replace("\r", "\n")
        
        # 归一化空白字符（保留单个换行符）
        t = re.sub(whitespace_pattern, whitespace_replacement, t)  # 多个空格/制表符 -> 单个空格
        t = re.sub(newlines_pattern, newlines_replacement, t)  # 多个连续换行 -> 双换行
        
        # 全角转半角（可选，根据需求调整）
        # t = self._normalize_fullwidth(t)
        
        return t.strip()

    def chunk(self, text: str) -> List[str]:
        """
        智能文本分块。
        优先使用 LangChain 分块器，回退到自定义分块逻辑。
        """
        if not text:
            return []
        
        if self.use_langchain:
            try:
                return self._langchain_chunk(text)
            except ImportError:
                print("Warning: LangChain not available, falling back to custom chunking")
                return self._custom_chunk(text)
        else:
            return self._custom_chunk(text)

    def _langchain_chunk(self, text: str) -> List[str]:
        """使用 LangChain 的智能分块器"""
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            
            # 从配置中获取中文友好的分隔符顺序
            chunking_config = self._config.get("chunking", {})
            separators = chunking_config.get("separators", [
                "\n\n",      # 段落分隔
                "\n",        # 行分隔
                "。",        # 中文句号
                "！",        # 中文感叹号
                "？",        # 中文问号
                "；",        # 中文分号
                "，",        # 中文逗号
                " ",         # 空格
                ""           # 字符级别（最后手段）
            ])
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
                separators=separators,
                length_function=len,
                is_separator_regex=False
            )
            
            return splitter.split_text(text)
            
        except ImportError:
            raise ImportError("LangChain is required for smart chunking. Install with: pip install langchain")

    def _custom_chunk(self, text: str) -> List[str]:
        """自定义分块逻辑（回退方案）"""
        chunks: List[str] = []
        if not text:
            return chunks
        
        # 按段落分割
        paragraphs = text.split("\n\n")
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            if len(para) <= self.chunk_size:
                chunks.append(para)
            else:
                # 段落过长时按句子分割
                sentences = self._split_sentences(para)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= self.chunk_size:
                        current_chunk += sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
        
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """按中文标点符号分割句子"""
        # 从配置中获取中文标点符号模式
        sentence_splitting_config = self._config.get("sentence_splitting", {})
        sentence_endings = sentence_splitting_config.get("sentence_endings_pattern", "[。！？；]")
        return re.split(f'({sentence_endings})', text)


