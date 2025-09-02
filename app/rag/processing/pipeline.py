from __future__ import annotations

from typing import List, Dict, Any

from .interfaces import RawSample, ProcessedChunk, MediaExtractor, LanguageProcessor, QualityAssessor
from .media_text import PlainTextExtractor
from .lang_zh import ChineseProcessor


class ProcessingPipeline:
    """将 raw samples 转换为文本块的最小流水线实现。

    - MediaExtractor: 将二进制样本转成 text + meta
    - LanguageProcessor: 清洗文本并分块
    """

    def __init__(self,
                 extractor: MediaExtractor | None = None,
                 lang_processor: LanguageProcessor | None = None,
                 quality: QualityAssessor | None = None) -> None:
        self.extractor = extractor or PlainTextExtractor()
        self.lang = lang_processor or ChineseProcessor()
        self.quality = quality

    def run(self, samples: List[RawSample]) -> List[ProcessedChunk]:
        chunks: List[ProcessedChunk] = []
        for s in samples:
            extracted = self.extractor.extract(s)
            cleaned = self.lang.clean(extracted["text"])
            parts = self.lang.chunk(cleaned)
            for idx, part in enumerate(parts):
                meta: Dict[str, Any] = dict(extracted["meta"])  # 确保是 Dict[str, Any]
                meta["chunk_index"] = idx  # 避免 update 的类型提示误报
                if self.quality:
                    meta.update(self.quality.score(part, meta))
                chunks.append(ProcessedChunk(text=part, meta=meta))
        return chunks


