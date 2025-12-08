"""
图像OCR提取器
"""

import logging
from typing import Dict, Any, List, Optional
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
                - captions (List[str], 可选): 视觉理解生成的描述（回退时）
        """
        ocr_payload: Dict[str, Any] = {"ocr_results": []}
 
        if self._ocr_engines:
            for engine in self._ocr_engines:
                try:
                    if engine == "easyocr":
                        ocr_payload = self._extract_with_easyocr(content)
                        break
                    if engine == "paddleocr":
                        ocr_payload = self._extract_with_paddleocr(content)
                        break
                    if engine == "tesseract":
                        ocr_payload = self._extract_with_tesseract(content)
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
    
    def _extract_with_easyocr(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        使用EasyOCR引擎提取文本
        
        用处：使用EasyOCR库从图像中识别文本，支持多语言识别，
        返回带位置信息的文本结果和置信度分数。
        
        Args:
            image_bytes: 图像的二进制内容
            
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
        return {"ocr_results": ocr_results}
    
    def _extract_with_paddleocr(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        使用PaddleOCR引擎提取文本
        
        用处：使用PaddleOCR库从图像中识别文本，特别优化中文识别，
        支持角度检测和文本方向校正。
        
        Args:
            image_bytes: 图像的二进制内容
            
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
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang=paddle_lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )
        
        # 使用 PaddleOCR 3.0+ 推荐的 predict() API
        results = ocr.predict(input=img)
        
        # 从 Result 对象中提取 OCR 结果
        ocr_results = self._extract_ocr_from_predict_results(results, confidence_threshold)
        
        return {"ocr_results": ocr_results}
    
    def _extract_ocr_from_predict_results(self, results, confidence_threshold: float) -> List[Dict[str, Any]]:
        """
        从 predict() 返回的 Result 对象列表中提取 OCR 结果
        
        用处：将 PaddleOCR 3.0+ 的 Result 对象转换为标准化的 OCR 结果格式，
        应用置信度阈值过滤，并计算边界框信息。
        
        Args:
            results: predict() 方法返回的 Result 对象列表
            confidence_threshold: 置信度阈值，低于此值的结果将被过滤
            
        Returns:
            List[Dict[str, Any]]: 标准化的 OCR 结果列表，每个结果包含文本、置信度、位置等信息
        """
        if not results:
            return []
        
        ocr_results = []
        for result in results:
            # 尝试多种方式提取 OCR 结果
            extracted = self._extract_from_result_object(result, confidence_threshold)
            ocr_results.extend(extracted)
        
        return ocr_results
    
    def _extract_from_result_object(self, result, confidence_threshold: float) -> List[Dict[str, Any]]:
        """
        从单个 Result 对象中提取 OCR 结果
        
        用处：尝试多种方式从 Result 对象中提取 OCR 数据，支持不同的数据格式。
        
        Args:
            result: Result 对象
            confidence_threshold: 置信度阈值
            
        Returns:
            List[Dict[str, Any]]: OCR 结果列表
        """
        # 方式1: 通过 dt_polys 属性（PaddleOCR 3.0+ 常见格式）
        if hasattr(result, 'dt_polys') and result.dt_polys:
            return self._extract_from_dt_polys(result.dt_polys, confidence_threshold)
        
        # 方式2: 通过 ocr_result 属性（兼容格式）
        if hasattr(result, 'ocr_result') and result.ocr_result:
            return self._extract_from_ocr_result(result.ocr_result, confidence_threshold)
        
        # 方式3: 如果 Result 对象可以直接转换为列表（向后兼容）
        if isinstance(result, (list, tuple)):
            return self._extract_from_ocr_result(result, confidence_threshold)
        
        return []
    
    def _extract_from_dt_polys(self, dt_polys, confidence_threshold: float) -> List[Dict[str, Any]]:
        """
        从 dt_polys 中提取 OCR 结果
        
        用处：处理 PaddleOCR 3.0+ 的 dt_polys 格式，提取文本、置信度和边界框信息。
        
        Args:
            dt_polys: dt_polys 对象列表
            confidence_threshold: 置信度阈值
            
        Returns:
            List[Dict[str, Any]]: OCR 结果列表
        """
        ocr_results = []
        for poly in dt_polys:
            text = getattr(poly, 'rec_text', '') if hasattr(poly, 'rec_text') else ''
            confidence = float(getattr(poly, 'rec_score', 0.0)) if hasattr(poly, 'rec_score') else 0.0
            bbox = getattr(poly, 'points', []) if hasattr(poly, 'points') else []
            
            if confidence >= confidence_threshold and text:
                ocr_result = self._build_ocr_result(text, confidence, bbox)
                if ocr_result:
                    ocr_results.append(ocr_result)
        return ocr_results
    
    def _extract_from_ocr_result(self, ocr_result, confidence_threshold: float) -> List[Dict[str, Any]]:
        """
        从 ocr_result 嵌套结构中提取 OCR 结果
        
        用处：处理类似 ocr() 方法返回的嵌套结构格式，提取文本、置信度和边界框信息。
        
        Args:
            ocr_result: OCR 结果嵌套结构（可能是列表的列表）
            confidence_threshold: 置信度阈值
            
        Returns:
            List[Dict[str, Any]]: OCR 结果列表
        """
        ocr_results = []
        for line in ocr_result if ocr_result else []:
            if not line:
                continue
            for item in line:
                if item and len(item) >= 2:
                    bbox, (text, confidence) = item[0], item[1]
                    if confidence >= confidence_threshold:
                        ocr_result_dict = self._build_ocr_result(text, confidence, bbox)
                        if ocr_result_dict:
                            ocr_results.append(ocr_result_dict)
        return ocr_results
    
    @staticmethod
    def _build_ocr_result(text: str, confidence: float, bbox: List) -> Optional[Dict[str, Any]]:
        """
        构建标准化的 OCR 结果字典
        
        用处：根据文本、置信度和边界框信息构建标准化的 OCR 结果字典，
        计算边界框的位置和尺寸信息。
        
        Args:
            text: 识别的文本内容
            confidence: 识别置信度
            bbox: 边界框坐标列表
            
        Returns:
            Optional[Dict[str, Any]]: 标准化的 OCR 结果字典，如果边界框无效则返回 None
        """
        if not bbox or len(bbox) < 2:
            return None
        
        # 提取 x 和 y 坐标
        x_coordinates = [point[0] for point in bbox if len(point) >= 2]
        y_coordinates = [point[1] for point in bbox if len(point) >= 2]
        
        if not x_coordinates or not y_coordinates:
            return None
        
        return {
            "text": text,
            "confidence": float(confidence),
            "x": min(x_coordinates),
            "y": min(y_coordinates),
            "width": max(x_coordinates) - min(x_coordinates),
            "height": max(y_coordinates) - min(y_coordinates),
            "bbox": bbox
        }
    
    def _extract_with_tesseract(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        使用Tesseract引擎提取文本
        
        用处：使用Tesseract OCR引擎从图像中识别文本，
        支持多语言识别，作为传统OCR的备选方案。
        
        Args:
            image_bytes: 图像的二进制内容
            
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
        return {"ocr_results": ocr_results}
