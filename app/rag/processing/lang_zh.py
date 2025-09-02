from __future__ import annotations

from typing import List
import re

from .interfaces import LanguageProcessor


class ChineseProcessor(LanguageProcessor):
    """简化版中文处理：清洗 + 基于字符长度的分块。
    后续可替换为更智能的分块器（如基于句号/段落/Token）。
    """

    def __init__(self, chunk_size: int = 800, overlap: int = 150) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def clean(self, text: str) -> str:
        t = text.replace("\r", "")
        t = re.sub(r"\s+", " ", t)
        return t.strip()

    def chunk(self, text: str) -> List[str]:
        chunks: List[str] = []
        if not text:
            return chunks
        start = 0
        length = len(text)
        while start < length:
            end = min(start + self.chunk_size, length)
            chunks.append(text[start:end])
            start = end - self.overlap if end - self.overlap > start else end
        return chunks


