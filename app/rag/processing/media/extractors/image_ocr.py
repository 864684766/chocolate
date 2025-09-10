"""
图像OCR提取器
"""

import logging
from typing import Dict, Any, List
from .base import MediaExtractor
from .image_vision import ImageVisionExtractor
from app.config import get_config_manager

logger = logging.getLogger(__name__)


class ImageOCRExtractor(MediaExtractor):
    """图片OCR提取器（含视觉理解回退）
    
    支持多种OCR引擎（EasyOCR、PaddleOCR、Tesseract），
    当OCR无法提取文本时自动回退到视觉理解功能。
    """
    
    def __init__(self):
        """
        初始化OCR提取器
        
        用处：初始化配置管理器、检查可用的OCR引擎、
        创建视觉理解回退机制。
        """
        self._config_manager = get_config_manager()
        self._ocr_config = self._config_manager.get_ocr_config()
        self._ocr_engines = self._get_available_engines()
        self._vision_fallback = ImageVisionExtractor()
    
    def is_available(self) -> bool:
        """
        检查OCR提取器是否可用
        
        用处：检查是否至少有一种OCR引擎可用，或者视觉理解回退功能可用，
        用于判断是否可以处理图像文本提取任务。
        
        Returns:
            bool: True表示至少有一种功能可用，False表示完全不可用
        """
        return len(self._ocr_engines) > 0 or self._vision_fallback.is_available()
    
    def _get_available_engines(self) -> List[str]:
        """
        获取可用的OCR引擎列表
        
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
        """
        从图片中提取文本内容
        
        用处：使用OCR引擎从图像中提取文本，如果OCR失败或结果为空，
        自动回退到视觉理解功能生成图像描述。
        
        Args:
            content: 图像的二进制内容
            meta: 图像元数据，如格式、尺寸等信息
            
        Returns:
            Dict[str, Any]: 包含以下键的字典：
                - ocr_results (List[Dict]): OCR识别的文本结果列表
                - image_meta (Dict): 图像处理元数据
                - captions (List[str], 可选): 视觉理解生成的描述（回退时）
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
            }
 
        return ocr_payload
    
    def _extract_with_easyocr(self, image_bytes: bytes, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用EasyOCR引擎提取文本
        
        用处：使用EasyOCR库从图像中识别文本，支持多语言识别，
        返回带位置信息的文本结果和置信度分数。
        
        Args:
            image_bytes: 图像的二进制内容
            meta: 图像元数据
            
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
        """
        使用PaddleOCR引擎提取文本
        
        用处：使用PaddleOCR库从图像中识别文本，特别优化中文识别，
        支持角度检测和文本方向校正。
        
        Args:
            image_bytes: 图像的二进制内容
            meta: 图像元数据
            
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
        """
        使用Tesseract引擎提取文本
        
        用处：使用Tesseract OCR引擎从图像中识别文本，
        支持多语言识别，作为传统OCR的备选方案。
        
        Args:
            image_bytes: 图像的二进制内容
            meta: 图像元数据
            
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
