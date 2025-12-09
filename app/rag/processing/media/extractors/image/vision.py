"""
图像视觉理解提取器
"""

import logging
from typing import Dict, Any, List, Tuple
from ..base import MediaExtractor
from app.config import get_config_manager
from app.infra.models import ModelLoader, ModelType, LoaderConfig
# 使用相对导入避免路径解析问题
# 从 image/ 到 processing/ 需要 3 个点：.. -> extractors/, ../.. -> media/, ../../.. -> processing/
from ....utils.quality_utils import (
    clip_rerank,
)

logger = logging.getLogger(__name__)


class ImageVisionExtractor(MediaExtractor):
    """最小视觉理解提取器：尝试生成 caption + 可选视觉向量。
    依赖可选：
    - transformers[torch]: 'image-to-text' 管道（BLIP）
    - sentence-transformers: 生成 CLIP 文本向量（用于 caption 向量化占位）
    若依赖缺失，优雅降级为返回空结果。
    """

    def __init__(self) -> None:
        """
        初始化图像视觉提取器
        
        用处：初始化配置管理器、检查依赖库可用性、
        设置CLIP重排分数缓存等组件。
        """
        self._config_manager = get_config_manager()
        self._caption_config = self._config_manager.get_image_captioning_config()
        self._caption_available = self._check_caption()
        self._embed_available = self._check_text_embed()
        self._last_clip_scores: List[Tuple[str, float]] = []

    def is_available(self) -> bool:
        """
        检查图像视觉提取器是否可用
        
        用处：检查是否至少有一种功能可用（图像描述或文本嵌入），
        用于判断是否可以处理图像内容。
        
        Returns:
            bool: True表示至少有一种功能可用，False表示完全不可用
        """
        return self._caption_available or self._embed_available

    @staticmethod
    def _check_caption() -> bool:
        """
        检查图像描述功能是否可用
        
        用处：检查transformers库是否已安装，这是图像描述功能的基础依赖。
        如果库未安装，会优雅降级而不影响其他功能。
        
        Returns:
            bool: True表示transformers库可用，False表示不可用
        """
        try:
            from transformers import pipeline  # type: ignore[import-untyped]  # noqa: F401
            return True
        except (ImportError, ModuleNotFoundError):
            logger.info("Image caption model not available (transformers).")
            return False

    @staticmethod
    def _check_text_embed() -> bool:
        """
        检查文本嵌入功能是否可用
        
        用处：检查sentence-transformers库是否已安装，这是生成文本嵌入向量的基础依赖。
        如果库未安装，会跳过嵌入向量生成功能。
        
        Returns:
            bool: True表示sentence-transformers库可用，False表示不可用
        """
        try:
            import sentence_transformers  # type: ignore[import-untyped]  # noqa: F401
            return True
        except (ImportError, ModuleNotFoundError):
            logger.info("Sentence-Transformers not available; skip embedding availability check.")
            return False

    def extract(self, content: bytes, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        从图像中提取视觉内容
        
        用处：对图像进行多模态处理，包括生成描述、翻译、去重、生成嵌入向量等，
        为RAG系统提供结构化的图像文本表示。
        
        Args:
            content: 图像的二进制内容
            meta: 图像元数据，如格式、尺寸等信息
            
        Returns:
            Dict[str, Any]: 包含以下键的字典：
                - captions (List[str]): 生成的图像描述列表（已翻译和去重）
        """
        captions: List[str] = []

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
        
        # 去重和清洗逻辑已移至 pipeline 统一处理，此处不再处理

        return {
            "captions": captions,
        }

    # ---- private helpers ----
    def _generate_captions_with_backend(self, content: bytes) -> List[str]:
        """
        根据配置的后端生成图像描述
        
        用处：使用配置的图像描述模型生成英文描述，并进行基础过滤和重排，
        为后续的翻译和去重做准备。
        
        Args:
            content: 图像的二进制内容
            
        Returns:
            List[str]: 生成的英文描述列表，已通过基础过滤和重排
        """
        import io
        from PIL import Image  # type: ignore[import-untyped]

        model_name = self._caption_config.get("model", "Salesforce/blip-image-captioning-base")
        generation_config = self._caption_config.get("generation", {})
        filter_config = self._caption_config.get("filters", {})
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

        # 2) 规则过滤和去重已移至 pipeline 统一处理，此处不再处理

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
        """
        使用图像到文本模型生成描述
        
        用处：调用transformers的图像到文本管道，使用指定的模型生成图像描述，
        支持批量生成和参数控制，具有版本兼容性回退机制。
        
        Args:
            model_name: 使用的模型名称或路径
            image: PIL图像对象
            num_captions: 要生成的描述数量
            max_length: 生成文本的最大长度
            temperature: 生成温度，控制随机性
            do_sample: 是否使用采样生成
            
        Returns:
            List[str]: 生成的图像描述文本列表
        """
        captions: List[str] = []
        # 使用统一的模型加载器加载 pipeline
        config = LoaderConfig(
            model_name=model_name,
            model_type=ModelType.TRANSFORMERS,
            task="image-to-text",
            device="auto"
        )
        cap_pipe = ModelLoader.load_model(config)
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
        """
        使用CLIP模型对描述进行重排
        
        用处：根据图像-文本相似度对生成的描述进行重新排序，
        选择与图像最匹配的描述，提升描述质量。
        
        Args:
            image: PIL图像对象
            captions: 待重排的描述列表
            
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
        """
        检查图像描述功能是否启用
        
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
        """
        返回空的描述结果
        
        用处：当图像描述功能被禁用或处理失败时，返回标准格式的空结果，
        保持接口一致性。
        
        Returns:
            Dict[str, Any]: 包含空值的标准结果字典
        """
        return {
            "captions": [],
        }

    def _translate_en_to_zh(self, captions: List[str]) -> List[str]:
        """
        将英文描述翻译为中文
        
        用处：使用MarianMT模型将英文图像描述翻译为中文，
        支持批量翻译和错误处理，为中文RAG系统提供本地化内容。
        
        Args:
            captions: 英文描述列表
            
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
            # 使用统一的模型加载器加载 translation pipeline
            config = LoaderConfig(
                model_name=model_path,
                model_type=ModelType.TRANSFORMERS,
                task="translation",
                device="auto",
                src_lang="en",
                tgt_lang="zh"
            )
            translator = ModelLoader.load_model(config)

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
