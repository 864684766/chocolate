"""
文本分块策略
"""

import logging
import re
from typing import List, Dict, Any
from .base import MediaChunkingStrategy
from app.config import get_config_manager

logger = logging.getLogger(__name__)


class TextChunkingStrategy(MediaChunkingStrategy):
    """文本分块策略 - 使用 LangChain 智能分块"""
    
    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        """
        初始化文本分块策略
        
        Args:
            chunk_size: 分块大小
            overlap: 重叠大小
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, content: str, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        智能文本分块
        
        Args:
            content: 文本内容
            meta: 元数据
            
        Returns:
            List[Dict[str, Any]]: 分块结果列表
        """
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            
            # 从配置文件读取分隔符（language_processing.chinese.chunking.separators）
            config_manager = get_config_manager()
            chinese_config = config_manager.get_chinese_processing_config()
            chunking_config = chinese_config.get("chunking", {})
            separators = chunking_config.get("separators", [
                "\n\n",      # 段落分隔
                "\n",        # 行分隔
                "。",        # 中文句号
                "！",        # 中文感叹号
                "？",        # 中文问号
                "；",        # 中文分号
                "，",        # 中文逗号
                " ",         # 空格
                ""           # 字符级别
            ])
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
                separators=separators,
                length_function=len
            )
            
            chunks = splitter.split_text(content)
            
            # 为每个块添加元数据
            result = []
            for i, chunk in enumerate(chunks):
                chunk_meta: Dict[str, Any] = dict(meta)
                chunk_meta.update({
                    "chunk_index": i,
                    "chunk_type": "text",
                    "chunk_size": len(chunk),
                    "total_chunks": len(chunks)
                })
                result.append({
                    "text": chunk,
                    "meta": chunk_meta
                })
            
            return result
            
        except ImportError as e:
            # 回退到简单分块
            logger.warning(f"langchain-text-splitters未安装，使用简单分块策略: {str(e)}")
            return self._simple_chunk(content, meta)
    
    def _simple_chunk(self, content: str, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        语义边界分块（回退方案）
        
        采用语义边界处理（段落、句子）+ 重叠机制，既保证语义完整性又保证上下文连续性。
        
        Args:
            content: 文本内容
            meta: 元数据
            
        Returns:
            List[Dict[str, Any]]: 分块结果列表
        """
        if not content:
            return []
        
        sentence_endings = self._get_sentence_endings_pattern()
        paragraphs = content.split("\n\n")
        chunks = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_chunks = self._process_paragraph(para, meta, sentence_endings, len(chunks))
            chunks.extend(para_chunks)
        
        chunks = self._apply_overlap_and_update_total(chunks)
        return chunks
    
    @staticmethod
    def _get_sentence_endings_pattern() -> str:
        """
        从配置中获取中文标点符号模式
        
        Returns:
            str: 句子结束标点的正则表达式模式
        """
        config_manager = get_config_manager()
        chinese_config = config_manager.get_chinese_processing_config()
        sentence_splitting_config = chinese_config.get("sentence_splitting", {})
        return sentence_splitting_config.get("sentence_endings_pattern", "[。！？；]")
    
    def _process_paragraph(self, para: str, meta: Dict[str, Any], 
                          sentence_endings: str, start_index: int) -> List[Dict[str, Any]]:
        """
        处理单个段落，根据长度选择不同的分块策略
        
        Args:
            para: 段落文本
            meta: 元数据
            sentence_endings: 句子结束标点的正则表达式模式
            start_index: 起始块索引
            
        Returns:
            List[Dict[str, Any]]: 该段落的分块结果列表
        """
        if len(para) <= self.chunk_size:
            return [self._create_chunk_dict(para, meta, start_index)]
        return self._process_long_paragraph(para, meta, sentence_endings, start_index)
    
    def _process_long_paragraph(self, para: str, meta: Dict[str, Any], 
                               sentence_endings: str, start_index: int) -> List[Dict[str, Any]]:
        """
        处理长段落，按句子分割并组合成块
        
        Args:
            para: 段落文本
            meta: 元数据
            sentence_endings: 句子结束标点的正则表达式模式
            start_index: 起始块索引
            
        Returns:
            List[Dict[str, Any]]: 该段落的分块结果列表
        """
        sentences = self._split_sentences(para, sentence_endings)
        chunks = []
        current_chunk = ""
        chunk_index = start_index
        
        for sentence in sentences:
            sentence_len = len(sentence)
            current_len = len(current_chunk)
            
            if current_len + sentence_len <= self.chunk_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(self._create_chunk_dict(current_chunk.strip(), meta, chunk_index))
                    chunk_index += 1
                
                if sentence_len > self.chunk_size:
                    sentence_chunks = self._force_split_long_sentence(sentence, meta, chunk_index)
                    chunks.extend(sentence_chunks)
                    chunk_index += len(sentence_chunks)
                    current_chunk = ""
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(self._create_chunk_dict(current_chunk.strip(), meta, chunk_index))
        
        return chunks
    
    def _force_split_long_sentence(self, sentence: str, meta: Dict[str, Any], 
                                   start_index: int) -> List[Dict[str, Any]]:
        """
        强制切分超长句子，按字符切分但保留重叠
        
        Args:
            sentence: 超长句子文本
            meta: 元数据
            start_index: 起始块索引
            
        Returns:
            List[Dict[str, Any]]: 切分后的块列表
        """
        chunks = []
        sentence_len = len(sentence)
        chunk_index = start_index
        start = 0
        
        while start < sentence_len:
            end = min(start + self.chunk_size, sentence_len)
            chunk_text = sentence[start:end]
            chunks.append(self._create_chunk_dict(chunk_text, meta, chunk_index))
            chunk_index += 1
            start = end - self.overlap if end - self.overlap > start else end
        
        return chunks
    
    @staticmethod
    def _create_chunk_dict(text: str, meta: Dict[str, Any], chunk_index: int) -> Dict[str, Any]:
        """
        创建块字典，包含文本和元数据
        
        Args:
            text: 块文本内容
            meta: 基础元数据
            chunk_index: 块索引
            
        Returns:
            Dict[str, Any]: 包含text和meta的块字典
        """
        chunk_meta: Dict[str, Any] = dict(meta)
        chunk_meta.update({
            "chunk_index": chunk_index,
            "chunk_type": "text_simple",
            "chunk_size": len(text)
        })
        return {
            "text": text,
            "meta": chunk_meta
        }
    
    def _apply_overlap_and_update_total(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        应用重叠机制并更新total_chunks
        
        Args:
            chunks: 原始分块列表
            
        Returns:
            List[Dict[str, Any]]: 处理后的分块列表
        """
        if len(chunks) > 1 and self.overlap > 0:
            chunks = self._add_overlap_between_chunks(chunks)
        self._update_total_chunks(chunks)
        return chunks
    
    @staticmethod
    def _update_total_chunks(chunks: List[Dict[str, Any]]) -> None:
        """
        更新所有块的total_chunks元数据
        
        Args:
            chunks: 分块列表
        """
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk["meta"]["total_chunks"] = total_chunks
    
    @staticmethod
    def _split_sentences(text: str, sentence_endings: str) -> List[str]:
        """
        按中文标点符号分割句子
        
        Args:
            text: 待分割的文本
            sentence_endings: 句子结束标点的正则表达式模式
            
        Returns:
            List[str]: 分割后的句子列表（包含标点符号）
        """
        return re.split(f'({sentence_endings})', text)
    
    def _add_overlap_between_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        在块之间添加重叠内容
        
        Args:
            chunks: 原始分块列表
            
        Returns:
            List[Dict[str, Any]]: 添加重叠后的分块列表
        """
        if len(chunks) <= 1:
            return chunks
        
        result = [chunks[0]]  # 第一个块不需要前向重叠
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]
            
            prev_text = prev_chunk["text"]
            curr_text = curr_chunk["text"]
            
            # 从上一个块的末尾提取overlap长度的文本
            if len(prev_text) > self.overlap:
                overlap_text = prev_text[-self.overlap:]
                # 尝试在句子边界处截取
                overlap_text = self._trim_to_sentence_boundary(overlap_text, is_end=True)
            else:
                overlap_text = prev_text
            
            # 将重叠文本添加到当前块的开头
            if overlap_text and not curr_text.startswith(overlap_text):
                # 避免重复添加
                curr_chunk = dict(curr_chunk)
                curr_chunk["text"] = overlap_text + curr_text
                curr_chunk["meta"] = dict(curr_chunk["meta"])
                curr_chunk["meta"]["chunk_size"] = len(curr_chunk["text"])
            
            result.append(curr_chunk)
        
        return result
    
    @staticmethod
    def _trim_to_sentence_boundary(text: str, is_end: bool = True) -> str:
        """
        在句子边界处截取文本
        
        Args:
            text: 待截取的文本
            is_end: 是否从末尾截取（True表示从末尾，False表示从开头）
            
        Returns:
            str: 截取后的文本
        """
        if not text:
            return text
        
        # 查找最近的句子边界（中文标点）
        sentence_endings = "[。！？；]"
        if is_end:
            # 从末尾查找
            match = re.search(sentence_endings, text)
            if match:
                return text[match.end():]
        else:
            # 从开头查找
            match = re.search(sentence_endings, text)
            if match:
                return text[:match.end()]
        
        return text
