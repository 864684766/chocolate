from __future__ import annotations

from typing import List, Dict, Any

from .interfaces import RawSample, ProcessedChunk, MediaExtractor, LanguageProcessor, QualityAssessor
from .media_text import PlainTextExtractor
from .lang_zh import ChineseProcessor
from .media_chunking import ChunkingStrategyFactory
from .media_extractors import MediaExtractorFactory
from app.config import get_config_manager
from app.core.chunking import decide_chunk_params


class ProcessingPipeline:
    """将 raw samples 转换为文本块的最小流水线实现。

    - MediaExtractor: 将二进制样本转成 text + meta 或媒体特定结构
    - LanguageProcessor: 清洗文本并分块
    - 支持多种媒体类型的分块策略
    """

    def __init__(self,
                 extractor: MediaExtractor | None = None,
                 lang_processor: LanguageProcessor | None = None,
                 quality: QualityAssessor | None = None,
                 use_media_chunking: bool = True) -> None:
        self.extractor = extractor or PlainTextExtractor()
        self.lang = lang_processor or ChineseProcessor()
        self.quality = quality
        self.use_media_chunking = use_media_chunking

    def run(self, samples: List[RawSample]) -> List[ProcessedChunk]:
        chunks: List[ProcessedChunk] = []
        for sample in samples:
            chunks.extend(self._process_sample(sample))
        return chunks

    # ---- helpers ----
    def _process_sample(self, sample: RawSample) -> List[ProcessedChunk]:
        extracted = self._extract_sample(sample)
        if not extracted:
            return []
        if self.use_media_chunking and self._should_use_media_chunking(extracted):
            return self._process_with_media_chunking(extracted)
        return self._process_text_chunks(extracted)

    def _extract_sample(self, sample: RawSample) -> Dict[str, Any] | None:
        media_type = str(sample.meta.get("media_type", "text")).lower()
        if media_type in ["image", "video", "audio"]:
            extractor = MediaExtractorFactory.create_extractor(media_type)
            if extractor and extractor.is_available():
                content = extractor.extract(sample.bytes, dict(sample.meta))
                return {"content": content, "meta": dict(sample.meta)}
            return None
        return self.extractor.extract(sample)

    def _process_text_chunks(self, extracted: Dict[str, Any]) -> List[ProcessedChunk]:
        cleaned = self.lang.clean(extracted.get("text", ""))
        if not cleaned:
            return []
        parts = self.lang.chunk(cleaned)
        chunks: List[ProcessedChunk] = []
        for idx, part in enumerate(parts):
            meta: Dict[str, Any] = dict(extracted["meta"])  # type: ignore[index]
            meta["chunk_index"] = idx
            meta["chunk_type"] = "text_traditional"
            if self.quality:
                meta.update(self.quality.score(part, meta))
            chunks.append(ProcessedChunk(text=part, meta=meta))
        return chunks

    @staticmethod
    def _should_use_media_chunking(extracted: Dict[str, Any]) -> bool:
        """判断是否应该使用媒体分块策略"""
        meta = extracted.get("meta", {})
        media_type = str(meta.get("media_type", "")).lower()
        return media_type in ["pdf", "image", "video", "audio"]
    
    def _process_with_media_chunking(self, extracted: Dict[str, Any]) -> List[ProcessedChunk]:
        """使用媒体分块策略处理内容"""
        meta = extracted.get("meta", {})
        media_type = meta.get("media_type", "text")
        content = extracted.get("content", extracted.get("text", ""))

        # 计算分块参数：从核心 chunking 模块获取（可配置+自适应）
        chunk_size, overlap = decide_chunk_params(str(media_type), content, meta)

        strategy = ChunkingStrategyFactory.create_strategy(
            str(media_type),
            chunk_size=chunk_size,
            overlap=overlap
        )

        chunk_results = strategy.chunk(content, meta)  # type: ignore[arg-type]

        chunks: List[ProcessedChunk] = []
        for chunk_result in chunk_results:
            chunk_text = chunk_result["text"]
            chunk_meta = chunk_result["meta"]
            if self.quality:
                chunk_meta.update(self.quality.score(chunk_text, chunk_meta))
            chunks.append(ProcessedChunk(text=chunk_text, meta=chunk_meta))
        return chunks

    # 说明：分块参数算法已抽到 app.core.chunking 模块复用


