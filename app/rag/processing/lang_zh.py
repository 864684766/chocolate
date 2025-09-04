from __future__ import annotations

from typing import List, Dict, Any
import re

from .interfaces import LanguageProcessor


class ChineseProcessor(LanguageProcessor):
    """改进版中文处理器：使用 LangChain 智能分块 + 中文优化。
    
    特性：
    - 智能分块：基于语义边界，避免破坏句子完整性
    - 中文优化：支持中文标点、段落分割
    - 可配置：支持多种分块策略和参数
    """

    def __init__(self, chunk_size: int = 800, overlap: int = 150, 
                 use_langchain: bool = True) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.use_langchain = use_langchain

    def clean(self, text: str) -> str:
        """
        清洗文本，优化中文处理。
        - 统一换行符和空白字符
        - 保留中文标点符号
        - 归一化全角/半角字符
        """
        if not text:
            return ""
        
        # 统一换行符
        t = text.replace("\r\n", "\n").replace("\r", "\n")
        
        # 归一化空白字符（保留单个换行符）
        t = re.sub(r"[ \t]+", " ", t)  # 多个空格/制表符 -> 单个空格
        t = re.sub(r"\n\s*\n\s*\n+", "\n\n", t)  # 多个连续换行 -> 双换行
        
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
            
            # 中文友好的分隔符顺序
            separators = [
                "\n\n",      # 段落分隔
                "\n",        # 行分隔
                "。",        # 中文句号
                "！",        # 中文感叹号
                "？",        # 中文问号
                "；",        # 中文分号
                "，",        # 中文逗号
                " ",         # 空格
                ""           # 字符级别（最后手段）
            ]
            
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

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """按中文标点符号分割句子"""
        # 中文标点符号
        sentence_endings = r'[。！？；]'
        return re.split(f'({sentence_endings})', text)

    def extract_meta(self) -> Dict[str, Any]:
        """提取处理器的元数据"""
        return {
            "processor_type": "ChineseProcessor",
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "use_langchain": self.use_langchain,
            "language": "zh"
        }


