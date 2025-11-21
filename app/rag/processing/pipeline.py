from __future__ import annotations

from typing import List, Dict, Any

from .interfaces import RawSample, ProcessedChunk, MediaExtractor, LanguageProcessor, QualityAssessor
from .media_text import PlainTextExtractor
from .lang_zh import ChineseProcessor
from .media.chunking import ChunkingStrategyFactory
from .media.extractors import MediaExtractorFactory
from .utils.chunking import decide_chunk_params
from .metadata_manager import MetadataManager


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
        """
        初始化处理流水线

        Args:
            extractor: 媒体提取器，用于从二进制数据中提取文本内容
                      如果为 None，则使用默认的 PlainTextExtractor
            lang_processor: 语言处理器，用于文本清洗和分块
                          如果为 None，则使用默认的 ChineseProcessor
            quality: 质量评估器，用于评估文本块质量
                   如果为 None，则跳过质量评估
            use_media_chunking: 是否使用媒体分块策略
                               True: 根据媒体类型使用专门的分块策略
                               False: 使用传统的文本分块策略

        Returns:
            None
        """
        self.extractor = extractor or PlainTextExtractor()
        self.lang = lang_processor or ChineseProcessor()
        self.quality = quality
        self.metadata_manager = MetadataManager()
        self.use_media_chunking = use_media_chunking

    def run(self, samples: List[RawSample]) -> List[ProcessedChunk]:
        """
        运行处理流水线，将原始样本转换为处理后的文本块

        Args:
            samples: 原始样本列表，每个样本包含二进制数据和元数据

        Returns:
            List[ProcessedChunk]: 处理后的文本块列表，每个块包含文本内容和元数据
        """
        chunks: List[ProcessedChunk] = []
        for sample in samples:
            chunks.extend(self._process_sample(sample))
        return chunks

    # ---- helpers ----
    def _process_sample(self, sample: RawSample) -> List[ProcessedChunk]:
        """
        处理单个原始样本，根据媒体类型选择合适的分块策略

        Args:
            sample: 原始样本，包含二进制数据和元数据

        Returns:
            List[ProcessedChunk]: 处理后的文本块列表
        """
        extracted = self._extract_sample(sample)
        if not extracted:
            return []
        if self.use_media_chunking and self._should_use_media_chunking(extracted):
            return self._process_with_media_chunking(extracted)
        return self._process_text_chunks(extracted)

    def _extract_sample(self, sample: RawSample) -> Dict[str, Any] | None:
        """
        从原始样本中提取内容，根据媒体类型选择合适的提取器

        Args:
            sample: 原始样本，包含二进制数据和元数据

        Returns:
            Dict[str, Any] | None: 提取的内容字典，包含 "content" 和 "meta" 键
                                 如果提取失败则返回 None
        """
        media_type = str(sample.meta.get("media_type", "text")).lower()
        if media_type in ["image", "video", "audio"]:
            extractor = MediaExtractorFactory.create_extractor(media_type)
            if extractor and extractor.is_available():
                content = extractor.extract(sample.bytes, dict(sample.meta))
                return {"content": content, "meta": dict(sample.meta)}
            return None
        return self.extractor.extract(sample)

    def _process_text_chunks(self, extracted: Dict[str, Any]) -> List[ProcessedChunk]:
        """
        使用传统文本分块策略处理提取的内容

        Args:
            extracted: 提取的内容字典，包含 "text" 和 "meta" 键

        Returns:
            List[ProcessedChunk]: 处理后的文本块列表，每个块包含文本内容和元数据
        """
        cleaned = self.lang.clean(extracted.get("text", ""))
        if not cleaned:
            return []
        parts = self.lang.chunk(cleaned)
        chunks: List[ProcessedChunk] = []
        total_chunks = len(parts)
        base_meta = extracted.get("meta", {})
        for idx, part in enumerate(parts):
            candidate_fields = {
                **base_meta,
                "chunk_index": idx,
                "chunk_type": "text",
                "chunk_size": len(part),
                "total_chunks": total_chunks
            }
            normalized_meta = self.metadata_manager.build_metadata(candidate_fields, text=part)
            chunks.append(ProcessedChunk(text=part, meta=normalized_meta))
        return chunks

    @staticmethod
    def _should_use_media_chunking(extracted: Dict[str, Any]) -> bool:
        """
        判断是否应该使用媒体分块策略

        Args:
            extracted: 提取的内容字典，包含 "meta" 键

        Returns:
            bool: True 表示应该使用媒体分块策略，False 表示使用传统文本分块
        """
        meta = extracted.get("meta", {})
        media_type = str(meta.get("media_type", "")).lower()
        return media_type in ["pdf", "image", "video", "audio"]
    
    def _process_with_media_chunking(self, extracted: Dict[str, Any]) -> List[ProcessedChunk]:
        """
        使用媒体分块策略处理内容，根据媒体类型选择专门的分块策略

        Args:
            extracted: 提取的内容字典，包含 "content"/"text" 和 "meta" 键

        Returns:
            List[ProcessedChunk]: 处理后的文本块列表，每个块包含文本内容和元数据
        """
        meta = extracted.get("meta", {})
        media_type = meta.get("media_type", "text")
        content = extracted.get("content", extracted.get("text", ""))

        # 计算分块参数：从核心 chunking 模块获取（可配置+自适应）
        chunk_size, overlap = decide_chunk_params(str(media_type), content)

        strategy = ChunkingStrategyFactory.create_strategy(
            str(media_type),
            chunk_size=chunk_size,
            overlap=overlap
        )

        chunk_results = strategy.chunk(content, meta)  # type: ignore[arg-type]

        chunks: List[ProcessedChunk] = []
        for chunk_result in chunk_results:
            chunk_text = chunk_result["text"]
            chunk_meta = self.metadata_manager.build_metadata(
                chunk_result.get("meta", {}),
                text=chunk_text
            )
            chunks.append(ProcessedChunk(text=chunk_text, meta=chunk_meta))
        return chunks

    # 说明：分块参数算法已抽到 app.core.chunking 模块复用


