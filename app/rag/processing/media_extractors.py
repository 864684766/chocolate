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
    """媒体内容提取器基类"""
    
    @abstractmethod
    def extract(self, content: bytes, meta: Dict[str, Any]) -> Dict[str, Any]:
        """从媒体内容中提取文本信息"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查提取器是否可用"""
        pass


class ImageVisionExtractor(MediaExtractor):
    """最小视觉理解提取器：尝试生成 caption + 可选视觉向量。
    依赖可选：
    - transformers[torch]: 'image-to-text' 管道（BLIP）
    - sentence-transformers: 生成 CLIP 文本向量（用于 caption 向量化占位）
    若依赖缺失，优雅降级为返回空结果。
    """

    def __init__(self) -> None:
        self._config_manager = get_config_manager()
        self._caption_config = self._config_manager.get_image_captioning_config()
        self._caption_available = self._check_caption()
        self._embed_available = self._check_text_embed()
        self._last_clip_scores: List[Tuple[str, float]] = []

    def is_available(self) -> bool:
        return self._caption_available or self._embed_available

    @staticmethod
    def _check_caption() -> bool:
        try:
            from transformers import pipeline  # noqa: F401
            return True
        except (ImportError, ModuleNotFoundError):
            logger.info("Image caption model not available (transformers).")
            return False

    @staticmethod
    def _check_text_embed() -> bool:
        try:
            import sentence_transformers  # noqa: F401
            return True
        except (ImportError, ModuleNotFoundError):
            logger.info("Sentence-Transformers not available; skip caption embedding.")
            return False

    def extract(self, content: bytes, meta: Dict[str, Any]) -> Dict[str, Any]:
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
                "caption_backend": str(self._caption_config.get("backend", "image-to-text")) if captions else None,
            },
        }

    # ---- private helpers ----
    def _generate_captions_with_backend(self, content: bytes) -> List[str]:
        """根据配置的 backend 分发到具体实现，返回生成的 captions 列表"""
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
        remove_duplicates = generation_config.get("remove_duplicates", True)

        image = Image.open(io.BytesIO(content)).convert("RGB")

        # 1) 生成英文候选
        captions = self._generate_with_image2text(
            model_name, image, num_captions, max_length, temperature, do_sample, remove_duplicates
        )

        # 2) 规则过滤（长度/黑名单/乱码/重复片段）
        if captions:
            captions = filter_captions(
                captions,
                min_len=int(filter_config.get("min_length", 5)),
                max_len=int(filter_config.get("max_length", 120)),
                blacklist_keywords=list(filter_config.get("blacklist_keywords", [])),
                max_gibberish_ratio=float(filter_config.get("max_gibberish_ratio", 0.3)),
                forbid_repeat_ngram=int(filter_config.get("forbid_repeat_ngram", 3)),
            )

        # 3) 近似去重（可选）
        if captions:
            captions = dedup_captions(
                captions,
                approx=True,
                threshold=0.95,
                embed_model=embed_conf.get("model"),
            )

        # 4) 可选：CLIP 粗排并阈值过滤，提升图像一致性
        captions = self._rerank_with_clip(image, captions)
        return captions

    @staticmethod
    def _generate_with_image2text(model_name: str,
                                  image,
                                  num_captions: int,
                                  max_length: int,
                                  temperature: float,
                                  do_sample: bool,
                                  remove_duplicates: bool) -> List[str]:
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
                    if text and (not remove_duplicates or text not in captions):
                        captions.append(text)
        except TypeError:
            # 回退方案：逐次调用聚合
            for _ in range(max(1, int(num_captions))):
                out = cap_pipe(image, max_new_tokens=max_length)
                if isinstance(out, list):
                    for result in out:
                        text = result.get("generated_text", "").strip()
                        if text and (not remove_duplicates or text not in captions):
                            captions.append(text)
        logger.info(f"Generated {len(captions)} captions using backend: image-to-text, model: {model_name}")
        return captions

    # 使用 CLIP 重排：根据图像-文本相似度挑选更贴近图片的候选
    def _rerank_with_clip(self, image, captions: List[str]) -> List[str]:
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

    # 是否启用图像描述
    def _is_captioning_enabled(self) -> bool:
        enabled = bool(self._caption_config.get("enabled", True))
        if not enabled:
            logger.info("Image captioning is disabled in config")
        return enabled

    # 空结果负载
    @staticmethod
    def _empty_caption_payload() -> Dict[str, Any]:
        return {
            "captions": [],
            "caption_embedding": None,
            "vision_meta": {"caption_model": None, "embed_model": None},
        }

    # 生成嵌入向量（单一职责）
    def _generate_caption_embedding(self, captions: List[str]) -> Tuple[Optional[List[float]], Optional[str]]:
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

    # 翻译：英文→中文（基于配置，可选）
    def _translate_en_to_zh(self, captions: List[str]) -> List[str]:
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

    



class ImageOCRExtractor(MediaExtractor):
    """图片 OCR 提取器（含视觉理解回退）。"""
    
    def __init__(self):
        self._config_manager = get_config_manager()
        self._ocr_config = self._config_manager.get_ocr_config()
        self._ocr_engines = self._get_available_engines()
        self._vision_fallback = ImageVisionExtractor()
    
    def is_available(self) -> bool:
        """检查是否有可用的 OCR 引擎或视觉回退"""
        return len(self._ocr_engines) > 0 or self._vision_fallback.is_available()
    
    def _get_available_engines(self) -> List[str]:
        """获取可用的 OCR 引擎列表"""
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
        """从图片中提取文本；若文本为空则回退到视觉描述。"""
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
        """使用 EasyOCR 提取文本"""
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
    """视频内容提取器"""
    
    def __init__(self):
        self._video_processing_available = self._check_video_processing_availability()
        self._speech_recognition_available = self._check_speech_recognition_availability()
    
    def is_available(self) -> bool:
        """检查视频处理是否可用"""
        return self._video_processing_available or self._speech_recognition_available
    
    @staticmethod
    def _check_video_processing_availability() -> bool:
        """检查视频处理库是否可用"""
        try:
            import cv2
            return True
        except ImportError:
            logger.warning("OpenCV not available. Install with: pip install opencv-python")
            return False
    
    @staticmethod
    def _check_speech_recognition_availability() -> bool:
        """检查语音识别库是否可用"""
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
        """从视频中提取内容"""
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
        logger.info("Subtitle extraction not implemented yet")
        return []
    
    def _extract_transcript(self, video_bytes: bytes, meta: Dict[str, Any]) -> str:
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
    """媒体提取器工厂"""
    
    @staticmethod
    def create_extractor(media_type: str) -> Optional[MediaExtractor]:
        """根据媒体类型创建对应的提取器"""
        if media_type.lower() == "image":
            return ImageOCRExtractor()  # 内置视觉回退
        elif media_type.lower() in ["video", "audio"]:
            return VideoContentExtractor()
        else:
            return None
