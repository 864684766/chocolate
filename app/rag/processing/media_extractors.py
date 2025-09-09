"""
媒体内容提取器模块

这个模块提供了从图片、视频等媒体文件中提取文本内容的功能。
所有功能都有依赖检查，如果没有安装相应的库，会优雅降级。
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod
from app.config import get_config_manager
from .utils.quality_utils import (
    filter_captions,
    dedup_captions,
    clip_rerank,
)

logger = logging.getLogger(__name__)


class MediaExtractor(ABC):
    """媒体内容提取器基类
    
    定义了所有媒体提取器必须实现的接口，包括内容提取和可用性检查。
    """
    
    @abstractmethod
    def extract(self, content: bytes, meta: Dict[str, Any]) -> Dict[str, Any]:
        """从媒体内容中提取文本信息
        
        用处：从图片、视频、音频等媒体文件中提取可读的文本内容，
        为后续的RAG处理提供文本数据源。
        
        Args:
            content (bytes): 媒体文件的二进制内容
            meta (Dict[str, Any]): 媒体文件的元数据信息，如文件类型、格式等
            
        Returns:
            Dict[str, Any]: 提取结果字典，包含文本内容和相关元数据
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查提取器是否可用
        
        用处：检查当前环境是否满足提取器的依赖要求，
        如必要的库是否已安装、模型是否可用等。
        
        Returns:
            bool: True表示提取器可用，False表示不可用
        """
        pass


class ImageVisionExtractor(MediaExtractor):
    """最小视觉理解提取器：尝试生成 caption + 可选视觉向量。
    依赖可选：
    - transformers[torch]: 'image-to-text' 管道（BLIP）
    - sentence-transformers: 生成 CLIP 文本向量（用于 caption 向量化占位）
    若依赖缺失，优雅降级为返回空结果。
    """

    def __init__(self) -> None:
        """初始化图像视觉提取器
        
        用处：初始化配置管理器、检查依赖库可用性、
        设置CLIP重排分数缓存等组件。
        """
        self._config_manager = get_config_manager()
        self._caption_config = self._config_manager.get_image_captioning_config()
        self._caption_available = self._check_caption()
        self._embed_available = self._check_text_embed()
        self._last_clip_scores: List[Tuple[str, float]] = []

    def is_available(self) -> bool:
        """检查图像视觉提取器是否可用
        
        用处：检查是否至少有一种功能可用（图像描述或文本嵌入），
        用于判断是否可以处理图像内容。
        
        Returns:
            bool: True表示至少有一种功能可用，False表示完全不可用
        """
        return self._caption_available or self._embed_available

    @staticmethod
    def _check_caption() -> bool:
        """检查图像描述功能是否可用
        
        用处：检查transformers库是否已安装，这是图像描述功能的基础依赖。
        如果库未安装，会优雅降级而不影响其他功能。
        
        Returns:
            bool: True表示transformers库可用，False表示不可用
        """
        try:
            from transformers import pipeline  # noqa: F401
            return True
        except (ImportError, ModuleNotFoundError):
            logger.info("Image caption model not available (transformers).")
            return False

    @staticmethod
    def _check_text_embed() -> bool:
        """检查文本嵌入功能是否可用
        
        用处：检查sentence-transformers库是否已安装，这是生成文本嵌入向量的基础依赖。
        如果库未安装，会跳过嵌入向量生成功能。
        
        Returns:
            bool: True表示sentence-transformers库可用，False表示不可用
        """
        try:
            import sentence_transformers  # noqa: F401
            return True
        except (ImportError, ModuleNotFoundError):
            logger.info("Sentence-Transformers not available; skip caption embedding.")
            return False

    def extract(self, content: bytes, meta: Dict[str, Any]) -> Dict[str, Any]:
        """从图像中提取视觉内容
        
        用处：对图像进行多模态处理，包括生成描述、翻译、去重、生成嵌入向量等，
        为RAG系统提供结构化的图像文本表示。
        
        Args:
            content (bytes): 图像的二进制内容
            meta (Dict[str, Any]): 图像元数据，如格式、尺寸等信息
            
        Returns:
            Dict[str, Any]: 包含以下键的字典：
                - captions (List[str]): 生成的图像描述列表（已翻译和去重）
                - caption_embedding (Optional[List[float]]): 描述文本的嵌入向量
                - vision_meta (Dict): 视觉处理元数据，包含模型信息、生成数量等
        """
        captions: List[str] = []
        
        # 初始化变量，避免作用域问题
        model_name = "Salesforce/blip-image-captioning-base"
        language_prompt = "Describe this image:"
        embedding_config = {}

        # 检查是否启用图像描述功能
        if not self._is_captioning_enabled():
            return self._empty_caption_payload()

        if self._caption_available:
            try:
                captions.extend(self._generate_captions_with_backend(content))
            except (ImportError, ModuleNotFoundError, OSError, RuntimeError) as e:
                logger.warning(f"caption generation failed: {e}")

        # 可选：英文→中文翻译
        captions = self._translate_en_to_zh(captions)
        
        # 对翻译后的中文内容进行去重
        captions = self._deduplicate_translated_captions(captions)

        # 生成嵌入向量
        caption_embed, embed_model_name = self._generate_caption_embedding(captions)
        if embed_model_name:
            embedding_config = {"model": embed_model_name}

        return {
            "captions": captions,
            "caption_embedding": caption_embed,
            "vision_meta": {
                "caption_model": model_name if captions else None,
                "embed_model": embedding_config.get("model", "clip-ViT-B-32") if caption_embed is not None else None,
                "num_captions_generated": len(captions),
                "language_prompt": language_prompt if captions else None,
                "caption_backend": "image-to-text" if captions else None,
            },
        }

    # ---- private helpers ----
    def _generate_captions_with_backend(self, content: bytes) -> List[str]:
        """根据配置的后端生成图像描述
        
        用处：使用配置的图像描述模型生成英文描述，并进行基础过滤和重排，
        为后续的翻译和去重做准备。
        
        Args:
            content (bytes): 图像的二进制内容
            
        Returns:
            List[str]: 生成的英文描述列表，已通过基础过滤和重排
        """
        import io
        from PIL import Image

        model_name = self._caption_config.get("model", "Salesforce/blip-image-captioning-base")
        generation_config = self._caption_config.get("generation", {})
        filter_config = self._caption_config.get("filters", {})
        embed_conf = self._caption_config.get("embedding", {})
        num_captions = generation_config.get("num_captions", 1)
        max_length = generation_config.get("max_length", 50)
        temperature = generation_config.get("temperature", 0.7)
        do_sample = generation_config.get("do_sample", True)
        # 移除英文去重逻辑，改为翻译后统一去重

        image = Image.open(io.BytesIO(content)).convert("RGB")

        # 1) 生成英文候选
        captions = self._generate_with_image2text(
            model_name, image, num_captions, max_length, temperature, do_sample
        )

        # 2) 规则过滤（长度/黑名单/乱码）
        if captions:
            captions = filter_captions(
                captions,
                min_len=int(filter_config.get("min_length", 5)),
                max_len=int(filter_config.get("max_length", 120)),
                blacklist_keywords=list(filter_config.get("blacklist_keywords", [])),
                max_gibberish_ratio=float(filter_config.get("max_gibberish_ratio", 0.3)),
                forbid_repeat_ngram=0,  # 英文阶段不检测重复片段，移到翻译后
            )

        # 3) 英文去重已移除，改为翻译后统一去重

        # 4) 可选：CLIP 粗排并阈值过滤，提升图像一致性
        captions = self._rerank_with_clip(image, captions)
        return captions

    @staticmethod
    def _generate_with_image2text(model_name: str,
                                  image,
                                  num_captions: int,
                                  max_length: int,
                                  temperature: float,
                                  do_sample: bool) -> List[str]:
        """使用图像到文本模型生成描述
        
        用处：调用transformers的图像到文本管道，使用指定的模型生成图像描述，
        支持批量生成和参数控制，具有版本兼容性回退机制。
        
        Args:
            model_name (str): 使用的模型名称或路径
            image: PIL图像对象
            num_captions (int): 要生成的描述数量
            max_length (int): 生成文本的最大长度
            temperature (float): 生成温度，控制随机性
            do_sample (bool): 是否使用采样生成
            
        Returns:
            List[str]: 生成的图像描述文本列表
        """
        from transformers import pipeline
        captions: List[str] = []
        cap_pipe = pipeline("image-to-text", model=model_name)
        # 优先使用文档建议的 generate_kwargs 传参；若版本不支持则回退到多次调用
        try:
            out = cap_pipe(
                image,
                max_new_tokens=max_length,
                generate_kwargs={
                    "temperature": temperature,
                    "do_sample": do_sample,
                    "num_return_sequences": num_captions,
                },
            )
            if isinstance(out, list):
                for result in out:
                    text = result.get("generated_text", "").strip()
                    if text:
                        captions.append(text)
        except TypeError:
            # 回退方案：逐次调用聚合
            for _ in range(max(1, int(num_captions))):
                out = cap_pipe(image, max_new_tokens=max_length)
                if isinstance(out, list):
                    for result in out:
                        text = result.get("generated_text", "").strip()
                        if text:
                            captions.append(text)
        logger.info(f"Generated {len(captions)} captions using backend: image-to-text, model: {model_name}")
        return captions

    def _rerank_with_clip(self, image, captions: List[str]) -> List[str]:
        """使用CLIP模型对描述进行重排
        
        用处：根据图像-文本相似度对生成的描述进行重新排序，
        选择与图像最匹配的描述，提升描述质量。
        
        Args:
            image: PIL图像对象
            captions (List[str]): 待重排的描述列表
            
        Returns:
            List[str]: 重排后的描述列表，按相似度排序
        """
        if not captions:
            return captions
        rerank_cfg = self._caption_config.get("rerank", {})
        if not rerank_cfg.get("enabled", False):
            return captions
        try:
            model_name = rerank_cfg.get("model", "openai/clip-vit-base-patch32")
            top_k = int(rerank_cfg.get("top_k", 2))
            min_prob = float(rerank_cfg.get("min_clip_prob", 0.0))
            ranked = clip_rerank(image, captions, model_name=model_name, top_k=max(1, top_k))
            # 记录分数供上层写入 meta
            self._last_clip_scores = ranked
            # 应用最小阈值过滤
            filtered = [c for c, p in ranked if p >= min_prob]
            return filtered or [c for c, _ in ranked]
        except (ImportError, ModuleNotFoundError) as e:
            logger.info(f"CLIP rerank not available: {e}")
            return captions
        except Exception as e:
            logger.warning(f"CLIP rerank failed: {e}")
            return captions

    def _is_captioning_enabled(self) -> bool:
        """检查图像描述功能是否启用
        
        用处：从配置中读取图像描述功能的启用状态，
        用于决定是否执行图像描述生成流程。
        
        Returns:
            bool: True表示功能已启用，False表示已禁用
        """
        enabled = bool(self._caption_config.get("enabled", True))
        if not enabled:
            logger.info("Image captioning is disabled in config")
        return enabled

    @staticmethod
    def _empty_caption_payload() -> Dict[str, Any]:
        """返回空的描述结果
        
        用处：当图像描述功能被禁用或处理失败时，返回标准格式的空结果，
        保持接口一致性。
        
        Returns:
            Dict[str, Any]: 包含空值的标准结果字典
        """
        return {
            "captions": [],
            "caption_embedding": None,
            "vision_meta": {"caption_model": None, "embed_model": None},
        }

    def _generate_caption_embedding(self, captions: List[str]) -> Tuple[Optional[List[float]], Optional[str]]:
        """为描述文本生成嵌入向量
        
        用处：使用sentence-transformers模型将文本描述转换为向量表示，
        用于后续的相似度计算和检索。
        
        Args:
            captions (List[str]): 描述文本列表，通常取第一个进行向量化
            
        Returns:
            Tuple[Optional[List[float]], Optional[str]]: 
                - 第一个元素：嵌入向量列表，失败时返回None
                - 第二个元素：使用的模型名称，失败时返回None
        """
        if not captions or not self._embed_available:
            return None, None
        try:
            embedding_config = self._caption_config.get("embedding", {})
            if not embedding_config.get("enabled", True):
                return None, None
            from sentence_transformers import SentenceTransformer
            embed_model = embedding_config.get("model", "clip-ViT-B-32")
            model = SentenceTransformer(embed_model)
            vec = model.encode(captions[0])
            caption_embed = vec.tolist() if hasattr(vec, "tolist") else None
            logger.info(f"Generated embedding using model: {embed_model}")
            return caption_embed, embed_model
        except (ImportError, ModuleNotFoundError, OSError, RuntimeError) as e:
            logger.warning(f"caption embedding failed: {e}")
            return None, None

    def _translate_en_to_zh(self, captions: List[str]) -> List[str]:
        """将英文描述翻译为中文
        
        用处：使用MarianMT模型将英文图像描述翻译为中文，
        支持批量翻译和错误处理，为中文RAG系统提供本地化内容。
        
        Args:
            captions (List[str]): 英文描述列表
            
        Returns:
            List[str]: 翻译后的中文描述列表，失败时返回原始英文列表
        """
        if not captions:
            return captions
        translation_cfg = self._caption_config.get("translation", {})
        if not translation_cfg.get("enabled", False):
            return captions
        try:
            model_path = translation_cfg.get("model", "Helsinki-NLP/opus-mt-en-zh")
            batch_size = int(translation_cfg.get("batch_size", 16))
            from transformers import pipeline
            # 使用稳定的通用任务名 "translation"，并显式传入 src_lang/tgt_lang，
            # 以匹配 transformers 的类型签名（Literal["translation"], str, ...）
            translator = pipeline("translation", model=model_path, src_lang="en", tgt_lang="zh")

            zh_captions: List[str] = []
            for i in range(0, len(captions), batch_size):
                chunk = captions[i:i+batch_size]
                try:
                    out = translator(chunk)
                except TypeError:
                    out = [translator(text)[0] for text in chunk]
                for item in out:
                    text = item.get("translation_text") if isinstance(item, dict) else str(item)
                    text = (text or "").strip()
                    if text and text not in zh_captions:
                        zh_captions.append(text)
            if zh_captions:
                logger.info("Translated captions EN→ZH via MarianMT")
                return zh_captions
        except (ImportError, ModuleNotFoundError) as e:
            # 典型错误：缺少 sentencepiece
            logger.error(f"translation pipeline not available: {e}")
            raise
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            logger.error(f"caption translation failed: {e}")
            raise
        return captions

    # 对翻译后的中文内容进行去重
    def _deduplicate_translated_captions(self, captions: List[str]) -> List[str]:
        """
        对翻译后的中文描述进行去重处理和质量过滤。
        
        用处：翻译模型可能产生重复或高度相似的中文内容，通过精确去重、近似去重
        和重复片段检测来减少冗余，提升最终内容质量。
        
        Args:
            captions: 翻译后的中文描述列表
            
        Returns:
            去重和过滤后的中文描述列表
        """
        if not captions:
            return captions
        
        # 1) 先进行重复片段检测和质量过滤
        from .utils.quality_utils import dedup_captions, filter_captions
        
        filter_config = self._caption_config.get("filters", {})
        dedup_config = self._caption_config.get("deduplication", {})
        captions = filter_captions(
            captions,
            min_len=int(filter_config.get("min_length", 5)),
            max_len=int(filter_config.get("max_length", 120)),
            blacklist_keywords=list(filter_config.get("blacklist_keywords", [])),
            max_gibberish_ratio=float(filter_config.get("max_gibberish_ratio", 0.3)),
            forbid_repeat_ngram=int(dedup_config.get("forbid_repeat_ngram", 3)),  # 在中文上检测重复片段
        )
        
        # 2) 再进行内容去重
        dedup_config = self._caption_config.get("deduplication", {})
        approx_enabled = dedup_config.get("approximate_enabled", True)
        threshold = dedup_config.get("similarity_threshold", 0.95)
        embed_model = dedup_config.get("embed_model", None)
        
        try:
            deduplicated = dedup_captions(
                captions,
                approx=approx_enabled,
                threshold=threshold,
                embed_model=embed_model
            )
            
            if len(deduplicated) < len(captions):
                logger.info(f"中文描述处理完成：{len(captions)} -> {len(deduplicated)} 条描述")
            
            return deduplicated
            
        except Exception as e:
            logger.warning(f"中文描述去重失败，返回过滤后内容: {e}")
            return captions

    



class ImageOCRExtractor(MediaExtractor):
    """图片OCR提取器（含视觉理解回退）
    
    支持多种OCR引擎（EasyOCR、PaddleOCR、Tesseract），
    当OCR无法提取文本时自动回退到视觉理解功能。
    """
    
    def __init__(self):
        """初始化OCR提取器
        
        用处：初始化配置管理器、检查可用的OCR引擎、
        创建视觉理解回退机制。
        """
        self._config_manager = get_config_manager()
        self._ocr_config = self._config_manager.get_ocr_config()
        self._ocr_engines = self._get_available_engines()
        self._vision_fallback = ImageVisionExtractor()
    
    def is_available(self) -> bool:
        """检查OCR提取器是否可用
        
        用处：检查是否至少有一种OCR引擎可用，或者视觉理解回退功能可用，
        用于判断是否可以处理图像文本提取任务。
        
        Returns:
            bool: True表示至少有一种功能可用，False表示完全不可用
        """
        return len(self._ocr_engines) > 0 or self._vision_fallback.is_available()
    
    def _get_available_engines(self) -> List[str]:
        """获取可用的OCR引擎列表
        
        用处：根据配置检查各种OCR引擎的可用性，
        按配置顺序返回可用的引擎列表，用于后续的文本提取。
        
        Returns:
            List[str]: 可用的OCR引擎名称列表，按配置优先级排序
        """
        engines = []
        configured_engines = self._ocr_config.get("engines", ["easyocr", "paddleocr", "tesseract"])
        
        # 按配置顺序检查引擎
        for engine in configured_engines:
            try:
                if engine == "easyocr":
                    import easyocr
                    engines.append("easyocr")
                elif engine == "paddleocr":
                    import paddleocr
                    engines.append("paddleocr")
                elif engine == "tesseract":
                    import pytesseract
                    engines.append("tesseract")
            except ImportError:
                logger.info(f"OCR engine {engine} not available")
                continue
        
        if not engines:
            logger.info("No OCR library available. Will try vision fallback if available.")
        
        return engines
    
    def extract(self, content: bytes, meta: Dict[str, Any]) -> Dict[str, Any]:
        """从图片中提取文本内容
        
        用处：使用OCR引擎从图像中提取文本，如果OCR失败或结果为空，
        自动回退到视觉理解功能生成图像描述。
        
        Args:
            content (bytes): 图像的二进制内容
            meta (Dict[str, Any]): 图像元数据，如格式、尺寸等信息
            
        Returns:
            Dict[str, Any]: 包含以下键的字典：
                - ocr_results (List[Dict]): OCR识别的文本结果列表
                - image_meta (Dict): 图像处理元数据
                - captions (List[str], 可选): 视觉理解生成的描述（回退时）
                - caption_embedding (List[float], 可选): 描述的嵌入向量（回退时）
                - vision_meta (Dict, 可选): 视觉处理元数据（回退时）
        """
        ocr_payload: Dict[str, Any] = {"ocr_results": [], "image_meta": {}}
 
        if self._ocr_engines:
            for engine in self._ocr_engines:
                try:
                    if engine == "easyocr":
                        ocr_payload = self._extract_with_easyocr(content, meta)
                        break
                    if engine == "paddleocr":
                        ocr_payload = self._extract_with_paddleocr(content, meta)
                        break
                    if engine == "tesseract":
                        ocr_payload = self._extract_with_tesseract(content, meta)
                        break
                except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError) as e:
                    logger.warning(f"OCR engine {engine} failed: {e}")
                    continue
 
        ocr_results = ocr_payload.get("ocr_results", [])
        # 当 OCR 结果为空或极少时，回退到视觉理解
        if not ocr_results and self._vision_fallback.is_available():
            vision = self._vision_fallback.extract(content, meta)
            return {
                **ocr_payload,
                "captions": vision.get("captions", []),
                "caption_embedding": vision.get("caption_embedding"),
                "vision_meta": vision.get("vision_meta", {}),
            }
 
        return ocr_payload
    
    def _extract_with_easyocr(self, image_bytes: bytes, meta: Dict[str, Any]) -> Dict[str, Any]:
        """使用EasyOCR引擎提取文本
        
        用处：使用EasyOCR库从图像中识别文本，支持多语言识别，
        返回带位置信息的文本结果和置信度分数。
        
        Args:
            image_bytes (bytes): 图像的二进制内容
            meta (Dict[str, Any]): 图像元数据
            
        Returns:
            Dict[str, Any]: 包含OCR结果和元数据的字典
        """
        import easyocr
        import cv2
        import numpy as np
        
        # 从配置获取语言设置
        languages = self._ocr_config.get("languages", ['ch_sim', 'en'])
        confidence_threshold = self._ocr_config.get("confidence_threshold", 0.5)
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        reader = easyocr.Reader(languages)
        results = reader.readtext(img)
        ocr_results = []
        for (bbox, text, confidence) in results:
            # 应用置信度阈值过滤
            if confidence >= confidence_threshold:
                x_coordinates = [point[0] for point in bbox]
                y_coordinates = [point[1] for point in bbox]
                ocr_results.append({
                    "text": text,
                    "confidence": confidence,
                    "x": min(x_coordinates),
                    "y": min(y_coordinates),
                    "width": max(x_coordinates) - min(x_coordinates),
                    "height": max(y_coordinates) - min(y_coordinates),
                    "bbox": bbox
                })
        return {
            "ocr_results": ocr_results,
            "image_meta": {
                "ocr_engine": "easyocr",
                "image_format": meta.get("image_format", "auto"),
                "total_texts": len(ocr_results)
            }
        }
    
    def _extract_with_paddleocr(self, image_bytes: bytes, meta: Dict[str, Any]) -> Dict[str, Any]:
        """使用PaddleOCR引擎提取文本
        
        用处：使用PaddleOCR库从图像中识别文本，特别优化中文识别，
        支持角度检测和文本方向校正。
        
        Args:
            image_bytes (bytes): 图像的二进制内容
            meta (Dict[str, Any]): 图像元数据
            
        Returns:
            Dict[str, Any]: 包含OCR结果和元数据的字典
        """
        from paddleocr import PaddleOCR
        import cv2
        import numpy as np
        
        # 从配置获取语言设置
        languages = self._ocr_config.get("languages", ['ch_sim', 'en'])
        confidence_threshold = self._ocr_config.get("confidence_threshold", 0.5)
        
        # 将语言代码转换为PaddleOCR格式
        lang_map = {'ch_sim': 'ch', 'en': 'en', 'ch': 'ch'}
        paddle_lang = 'ch'  # 默认中文
        for lang in languages:
            if lang in lang_map:
                paddle_lang = lang_map[lang]
                break
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        ocr = PaddleOCR(use_angle_cls=True, lang=paddle_lang)
        results = ocr.ocr(img, cls=True)
        ocr_results = []
        for line in results:
            for (bbox, (text, confidence)) in line:
                # 应用置信度阈值过滤
                if confidence >= confidence_threshold:
                    x_coordinates = [point[0] for point in bbox]
                    y_coordinates = [point[1] for point in bbox]
                    ocr_results.append({
                        "text": text,
                        "confidence": confidence,
                        "x": min(x_coordinates),
                        "y": min(y_coordinates),
                        "width": max(x_coordinates) - min(x_coordinates),
                        "height": max(y_coordinates) - min(y_coordinates),
                        "bbox": bbox
                    })
        return {
            "ocr_results": ocr_results,
            "image_meta": {
                "ocr_engine": "paddleocr",
                "image_format": meta.get("image_format", "auto"),
                "total_texts": len(ocr_results)
            }
        }
    
    def _extract_with_tesseract(self, image_bytes: bytes, meta: Dict[str, Any]) -> Dict[str, Any]:
        """使用Tesseract引擎提取文本
        
        用处：使用Tesseract OCR引擎从图像中识别文本，
        支持多语言识别，作为传统OCR的备选方案。
        
        Args:
            image_bytes (bytes): 图像的二进制内容
            meta (Dict[str, Any]): 图像元数据
            
        Returns:
            Dict[str, Any]: 包含OCR结果和元数据的字典
        """
        import pytesseract
        from PIL import Image
        import io
        
        # 从配置获取语言设置
        languages = self._ocr_config.get("languages", ['ch_sim', 'en'])
        
        # 将语言代码转换为Tesseract格式
        lang_map = {'ch_sim': 'chi_sim', 'en': 'eng', 'ch': 'chi_sim'}
        tesseract_langs = []
        for lang in languages:
            if lang in lang_map:
                tesseract_langs.append(lang_map[lang])
        tesseract_lang = '+'.join(tesseract_langs) if tesseract_langs else 'chi_sim+eng'
        
        img = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(img, lang=tesseract_lang)
        ocr_results = [{
            "text": text.strip(),
            "confidence": 0.8,
            "x": 0,
            "y": 0,
            "width": img.width,
            "height": img.height,
            "bbox": [[0, 0], [img.width, 0], [img.width, img.height], [0, img.height]]
        }] if text.strip() else []
        return {
            "ocr_results": ocr_results,
            "image_meta": {
                "ocr_engine": "tesseract",
                "image_format": meta.get("image_format", "auto"),
                "total_texts": len(ocr_results)
            }
        }


class VideoContentExtractor(MediaExtractor):
    """视频内容提取器
    
    支持从视频文件中提取字幕和语音转录文本，
    使用Whisper或SpeechRecognition进行语音识别。
    """
    
    def __init__(self):
        """初始化视频内容提取器
        
        用处：检查视频处理和语音识别功能的可用性，
        为后续的视频内容提取做准备。
        """
        self._video_processing_available = self._check_video_processing_availability()
        self._speech_recognition_available = self._check_speech_recognition_availability()
    
    def is_available(self) -> bool:
        """检查视频内容提取器是否可用
        
        用处：检查是否至少有一种视频处理功能可用（视频处理或语音识别），
        用于判断是否可以处理视频内容提取任务。
        
        Returns:
            bool: True表示至少有一种功能可用，False表示完全不可用
        """
        return self._video_processing_available or self._speech_recognition_available
    
    @staticmethod
    def _check_video_processing_availability() -> bool:
        """检查视频处理库是否可用
        
        用处：检查OpenCV库是否已安装，这是视频处理的基础依赖。
        如果库未安装，会记录警告信息。
        
        Returns:
            bool: True表示OpenCV可用，False表示不可用
        """
        try:
            import cv2
            return True
        except ImportError:
            logger.warning("OpenCV not available. Install with: pip install opencv-python")
            return False
    
    @staticmethod
    def _check_speech_recognition_availability() -> bool:
        """检查语音识别库是否可用
        
        用处：检查Whisper或SpeechRecognition库是否已安装，
        这些是语音识别功能的基础依赖。优先检查Whisper。
        
        Returns:
            bool: True表示至少有一种语音识别库可用，False表示都不可用
        """
        try:
            import whisper
            return True
        except ImportError:
            try:
                import speech_recognition
                return True
            except ImportError:
                logger.warning(
                    "No speech recognition library available. "
                    "Install with: pip install openai-whisper or SpeechRecognition"
                )
                return False
    
    def extract(self, content: bytes, meta: Dict[str, Any]) -> Dict[str, Any]:
        """从视频中提取文本内容
        
        用处：从视频文件中提取字幕或语音转录文本，
        优先提取字幕，如果无字幕则进行语音识别转录。
        
        Args:
            content (bytes): 视频文件的二进制内容
            meta (Dict[str, Any]): 视频元数据，如格式、时长等信息
            
        Returns:
            Dict[str, Any]: 包含以下键的字典：
                - subtitles (List[Dict]): 提取的字幕列表
                - transcript (str): 语音转录的文本内容
                - video_meta (Dict): 视频处理元数据，包含格式、处理状态等
        """
        if not self.is_available():
            return {
                "subtitles": [],
                "transcript": "",
                "video_meta": {"error": "Video processing not available"}
            }
        
        try:
            subtitles = self._extract_subtitles()
            transcript = ""
            if not subtitles:
                transcript = self._extract_transcript(content, meta)
            return {
                "subtitles": subtitles,
                "transcript": transcript,
                "video_meta": {
                    "video_format": meta.get("video_format", "auto"),
                    "has_subtitles": bool(subtitles),
                    "has_transcript": bool(transcript)
                }
            }
        except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError) as e:
            logger.error(f"Video content extraction failed: {e}")
            return {
                "subtitles": [],
                "transcript": "",
                "video_meta": {"error": str(e)}
            }
    
    @staticmethod
    def _extract_subtitles() -> List[Dict[str, Any]]:
        """提取视频字幕
        
        用处：从视频文件中提取字幕信息，目前未实现。
        未来可以支持SRT、VTT等字幕格式的解析。
        
        Returns:
            List[Dict[str, Any]]: 字幕列表，每个字典包含时间戳和文本内容
        """
        logger.info("Subtitle extraction not implemented yet")
        return []
    
    def _extract_transcript(self, video_bytes: bytes, meta: Dict[str, Any]) -> str:
        """提取视频语音转录文本
        
        用处：使用语音识别技术将视频中的语音转换为文本，
        优先使用Whisper，回退到SpeechRecognition。
        
        Args:
            video_bytes (bytes): 视频文件的二进制内容
            meta (Dict[str, Any]): 视频元数据
            
        Returns:
            str: 语音转录的文本内容，失败时返回空字符串
        """
        try:
            return self._extract_with_whisper(video_bytes, meta)
        except ImportError:
            pass
        try:
            return self._extract_with_speech_recognition(video_bytes, meta)
        except ImportError:
            pass
        logger.warning("No speech recognition library available")
        return ""
    
    @staticmethod
    def _extract_with_whisper(video_bytes: bytes, meta: Dict[str, Any]) -> str:
        """使用Whisper进行语音识别
        
        用处：使用OpenAI的Whisper模型进行高质量的语音识别，
        支持多语言和长音频处理。
        
        Args:
            video_bytes (bytes): 视频文件的二进制内容
            meta (Dict[str, Any]): 视频元数据，用于确定文件格式
            
        Returns:
            str: 语音识别的文本结果
        """
        import whisper
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix=f".{meta.get('video_format', 'mp4')}", delete=False) as temp_file:
            temp_file.write(video_bytes)
            temp_file_path = temp_file.name
        try:
            model = whisper.load_model("base")
            # 使用模块函数形式，显式传入 model
            result = whisper.transcribe(model, audio=temp_file_path)
            return result["text"]
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    @staticmethod
    def _extract_with_speech_recognition(video_bytes: bytes, meta: Dict[str, Any]) -> str:
        """使用SpeechRecognition进行语音识别
        
        用处：使用SpeechRecognition库进行语音识别，
        作为Whisper的备选方案，支持Google语音识别API。
        
        Args:
            video_bytes (bytes): 视频文件的二进制内容
            meta (Dict[str, Any]): 视频元数据，用于确定文件格式
            
        Returns:
            str: 语音识别的文本结果
        """
        import speech_recognition as sr
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix=f".{meta.get('video_format', 'mp4')}", delete=False) as temp_file:
            temp_file.write(video_bytes)
            temp_file_path = temp_file.name
        try:
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_file_path) as source:
                audio = recognizer.record(source)
            text = recognizer.recognize_google(audio=audio, language='zh-CN')
            return text
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)


class MediaExtractorFactory:
    """媒体提取器工厂
    
    根据媒体类型创建对应的提取器实例，
    提供统一的提取器创建接口。
    """
    
    @staticmethod
    def create_extractor(media_type: str) -> Optional[MediaExtractor]:
        """根据媒体类型创建对应的提取器
        
        用处：根据媒体文件类型选择合适的提取器，
        支持图像、视频、音频等不同类型的媒体处理。
        
        Args:
            media_type (str): 媒体类型，如"image"、"video"、"audio"
            
        Returns:
            Optional[MediaExtractor]: 对应的提取器实例，不支持的类型返回None
        """
        if media_type.lower() == "image":
            return ImageOCRExtractor()  # 内置视觉回退
        elif media_type.lower() in ["video", "audio"]:
            return VideoContentExtractor()
        else:
            return None
