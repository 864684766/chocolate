from __future__ import annotations

from typing import List, Dict, Any

from .interfaces import RawSample, ProcessedChunk, MediaExtractor, QualityAssessor
from .media.extractors.text.plain_text import PlainTextExtractor
from .media.chunking import ChunkingStrategyFactory
from .media.extractors import MediaExtractorFactory
from .utils.chunking import decide_chunk_params
from .metadata_manager import MetadataManager
from .utils.text_cleaner import TextCleaner


class ProcessingPipeline:
    """将 raw samples 转换为文本块的最小流水线实现。

    - MediaExtractor: 将二进制样本转成 text + meta 或媒体特定结构
    - TextCleaner: 统一文本清洗和去重
    - MediaChunkingStrategy: 根据媒体类型选择专门的分块策略
    - 支持多种媒体类型（text、markdown、pdf、word、excel、image、video、audio）的统一处理
    """

    def __init__(self,
                 extractor: MediaExtractor | None = None,
                 quality: QualityAssessor | None = None) -> None:
        """
        初始化处理流水线

        Args:
            extractor: 媒体提取器，用于从二进制数据中提取文本内容
                      如果为 None，则使用默认的 PlainTextExtractor
            quality: 质量评估器，用于评估文本块质量
                   如果为 None，则跳过质量评估

        Returns:
            None
        """
        self.extractor = extractor or PlainTextExtractor()
        self.quality = quality
        self.metadata_manager = MetadataManager()
        self.text_cleaner = TextCleaner()

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
        # 统一使用媒体分块策略处理所有媒体类型
        return self._process_with_media_chunking(extracted)

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
        if media_type in ["text", "markdown", "image", "video", "audio", "pdf", "word", "excel"]:
            extractor = MediaExtractorFactory.create_extractor(media_type)
            if extractor and extractor.is_available():
                content = extractor.extract(sample.bytes, dict(sample.meta))
                # 对于text和markdown类型，返回格式是{"text": str, "meta": dict}
                # 对于Office文档（pdf、word、excel）和其他媒体类型，返回格式是媒体特定结构
                if media_type in ["text", "markdown"]:
                    return {"text": content.get("text", ""), "meta": content.get("meta", dict(sample.meta))}
                else:
                    return {"content": content, "meta": dict(sample.meta)}
            return None
        # 兜底：使用默认文本提取器（统一使用新接口）
        if self.extractor and hasattr(self.extractor, 'extract'):
            try:
                # 尝试新接口
                content = self.extractor.extract(sample.bytes, dict(sample.meta))
                return {"text": content.get("text", ""), "meta": content.get("meta", dict(sample.meta))}
            except TypeError:
                # 兼容旧接口（如果传入的是RawSample）
                return self.extractor.extract(sample)
        return None
    
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
        
        # 对媒体提取的内容进行清洗和去重处理
        content = self._clean_media_content(content, media_type)

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
            # 对每个chunk的文本进行清洗
            chunk_text = self.text_cleaner.clean(chunk_text)
            if not chunk_text:
                continue
            chunk_meta = self.metadata_manager.build_metadata(
                chunk_result.get("meta", {}),
                text=chunk_text
            )
            chunks.append(ProcessedChunk(text=chunk_text, meta=chunk_meta))
        return chunks
    
    def _clean_media_content(self, content: Any, media_type: str) -> Any:
        """
        清洗媒体提取的内容
        
        用处：根据媒体类型对提取的内容进行清洗和去重处理。
        对于文本类型内容，进行清洗和去重；对于结构化内容（如字幕列表），
        提取文本后进行清洗和去重。
        
        Args:
            content: 媒体提取的内容，可能是字符串、列表或字典
            media_type: 媒体类型（image/video/audio/pdf等）
            
        Returns:
            any: 清洗和去重后的内容，保持原有结构
        """
        if isinstance(content, str):
            # 单个文本：清洗即可
            return self.text_cleaner.clean(content)
        elif isinstance(content, list):
            # 列表类型（如字幕列表、描述列表等）
            if not content:
                return content
            # 如果是字符串列表，进行清洗和去重
            if isinstance(content[0], str):
                return self.text_cleaner.clean_and_deduplicate(content)
            # 如果是字典列表（如字幕、OCR结果等），提取文本字段进行清洗和去重
            elif isinstance(content[0], dict):
                # 提取所有文本字段，建立原文本到原项的映射
                original_texts = []
                text_to_item: Dict[str, Dict[str, Any]] = {}
                for item in content:
                    if isinstance(item, dict):
                        # 提取常见的文本字段（支持多种字段名：text/caption/content/transcript）
                        text = item.get("text") or item.get("caption") or item.get("content") or item.get("transcript", "")
                        if text:
                            text_str = str(text).strip()
                            if text_str:
                                original_texts.append(text_str)
                                # 如果同一个文本出现多次，保留第一个
                                if text_str not in text_to_item:
                                    text_to_item[text_str] = item
                
                if not original_texts:
                    return content
                
                # 清洗和去重文本列表
                cleaned_texts = self.text_cleaner.clean_and_deduplicate(original_texts)
                
                # 建立清洗后文本到原文本的映射（通过文本匹配）
                # 由于去重可能导致数量减少，需要找到每个清洗后文本对应的原文本
                cleaned_to_original: Dict[str, str] = {}
                seen_cleaned = set()
                
                # 先清洗所有原文本，建立映射
                for orig_text in original_texts:
                    cleaned = self.text_cleaner.clean(orig_text)
                    if cleaned and cleaned not in seen_cleaned:
                        cleaned_to_original[cleaned] = orig_text
                        seen_cleaned.add(cleaned)
                
                # 构建结果列表：只保留去重后的项，并更新文本字段
                result = []
                for cleaned_text in cleaned_texts:
                    # 找到对应的原文本
                    original_text = cleaned_to_original.get(cleaned_text)
                    if original_text and original_text in text_to_item:
                        # 取对应的原项，更新文本字段
                        item = dict(text_to_item[original_text])
                        # 确定文本字段名
                        text_key = "text" if "text" in item else ("caption" if "caption" in item else ("content" if "content" in item else "transcript"))
                        if text_key in item:
                            item[text_key] = cleaned_text
                        result.append(item)
                
                return result
        elif isinstance(content, dict):
            # 字典类型：递归处理所有文本字段
            result = {}
            for key, value in content.items():
                if isinstance(value, (str, list)):
                    result[key] = self._clean_media_content(value, media_type)
                else:
                    result[key] = value
            return result
        
        return content

    # 说明：分块参数算法已抽到 app.core.chunking 模块复用


