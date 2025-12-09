"""
Markdown内容提取器

使用markdown-it-py库提取Markdown文档的纯文本内容。
适用于.md等Markdown文件。
"""

import logging
from typing import Dict, Any, List
from .plain_text import PlainTextExtractor

logger = logging.getLogger(__name__)


class MarkdownExtractor(PlainTextExtractor):
    """Markdown内容提取器
    
    使用markdown-it-py库将Markdown转换为纯文本，
    保留文档结构信息，去除Markdown标记。
    适用于.md等Markdown文件。
    """
    
    def extract(self, content: bytes, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        从Markdown文件中提取内容
        
        用处：从Markdown文档中提取纯文本内容，
        使用专门的Markdown库进行解析，保留文档结构。
        
        Args:
            content: Markdown文件的二进制内容
            meta: Markdown文件的元数据信息
            
        Returns:
            Dict[str, Any]: 提取结果字典，包含text和meta
        """
        # 先使用基类提取文本
        base_result = super().extract(content, meta)
        markdown_text = base_result["text"]
        
        # 使用markdown-it-py转换为纯文本
        plain_text = self._markdown_to_text(markdown_text)
        
        base_result["text"] = plain_text
        return base_result
    
    def _markdown_to_text(self, markdown_text: str) -> str:
        """
        将Markdown转换为纯文本
        
        用处：使用markdown-it-py库解析Markdown并提取纯文本内容。
        如果库不可用，回退到简单的正则表达式处理。
        
        Args:
            markdown_text: Markdown格式的文本
            
        Returns:
            str: 转换后的纯文本
        """
        try:
            from markdown_it import MarkdownIt
            
            md = MarkdownIt()
            # 将Markdown解析为tokens，然后提取文本
            tokens = md.parse(markdown_text)
            text_parts = self._extract_text_from_tokens(tokens)
            return "\n".join(text_parts)
        except ImportError:
            logger.warning("markdown-it-py未安装，使用简单正则处理Markdown")
            return self._simple_markdown_clean(markdown_text)
    
    def _extract_text_from_tokens(self, tokens) -> List[str]:
        """
        从Markdown tokens中提取文本
        
        用处：遍历Markdown解析后的tokens，提取所有文本内容，
        包括代码块、段落、列表等所有内容。
        
        Args:
            tokens: markdown-it-py解析后的tokens列表
            
        Returns:
            list[str]: 文本内容列表
        """
        text_parts = []
        
        for token in tokens:
            if token.type == "text":
                # 普通文本内容
                text_parts.append(token.content)
            elif token.type == "code_block" or token.type == "fence":
                # 代码块内容：提取代码块内的文本
                code_content = getattr(token, "content", "")
                if code_content:
                    text_parts.append(code_content)
            elif token.type == "code_inline":
                # 行内代码内容
                inline_code = getattr(token, "content", "")
                if inline_code:
                    text_parts.append(inline_code)
            elif token.type == "heading_open":
                # 标题开始，后续的文本会是标题内容
                pass
            elif token.type == "heading_close":
                # 标题结束，添加换行
                text_parts.append("")
            elif token.type == "paragraph_open":
                # 段落开始
                pass
            elif token.type == "paragraph_close":
                # 段落结束，添加换行
                text_parts.append("")
            elif token.children:
                # 递归处理子tokens
                child_texts = self._extract_text_from_tokens(token.children)
                text_parts.extend(child_texts)
        
        return text_parts
    
    @staticmethod
    def _simple_markdown_clean(text: str) -> str:
        """
        简单的Markdown清理（回退方案）
        
        用处：当markdown-it-py不可用时，使用正则表达式进行基础清理。
        
        Args:
            text: Markdown格式的文本
            
        Returns:
            str: 清理后的文本
        """
        import re
        
        # 移除代码围栏
        text = re.sub(r"```[\s\S]*?```", "\n", text)
        # 移除行内代码
        text = re.sub(r"`([^`]+)`", r"\1", text)
        # 移除标题前缀
        text = re.sub(r"^\s*#+\s*", "", text, flags=re.MULTILINE)
        # 移除链接标记但保留文本 [text](url) -> text
        text = re.sub(r"\[([^]]+)]\([^)]+\)", r"\1", text)
        # 移除图片标记
        text = re.sub(r"!\[([^]]*)]\([^)]+\)", r"\1", text)
        # 移除粗体和斜体标记
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)
        text = re.sub(r"__([^_]+)__", r"\1", text)
        text = re.sub(r"_([^_]+)_", r"\1", text)
        
        return text
    
    def is_available(self) -> bool:
        """
        检查Markdown提取器是否可用
        
        用处：检查markdown-it-py库是否已安装。
        即使库未安装，也可以使用简单的正则处理，所以始终返回True。
        
        Returns:
            bool: 始终返回True（有回退方案）
        """
        return True
