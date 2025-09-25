from __future__ import annotations

from typing import Dict, Any
from app.config import get_config_manager
from app.infra.logging import get_logger
from .interfaces import QualityAssessor


class SimpleQualityAssessor(QualityAssessor):
    """质量评估器：计算统一 `quality_score ∈ [0,1]` 并按需记录观测日志。

    说明：
    - 基于启发式：长度、有效字符比例、乱码比例、OCR 置信度（可选）
    - 参数与权重从配置读取：ingestion.metadata.quality
    - 不做重依赖；计算简单、可解释
    """

    def __init__(self) -> None:
        """
        初始化质量评估器
        
        从配置文件中读取质量评估相关参数，包括：
        - 长度阈值：最小长度和满意长度
        - 权重配置：乱码、有效字符、长度、OCR置信度的权重
        - 观测配置：是否启用观测、阈值、告警比例、采样率
        
        Returns:
            None
        """
        cfg = get_config_manager().get_config("metadata") or {}
        qcfg = cfg.get("quality") or {}
        self.min_len = int(qcfg.get("min_len", 20))
        self.sat_len = int(qcfg.get("sat_len", 200))
        w = qcfg.get("weights", {}) or {}
        self.w_garbled = float(w.get("garbled", 0.4))
        self.w_valid = float(w.get("valid", 0.2))
        self.w_length = float(w.get("length", 0.2))
        self.w_ocr = float(w.get("ocr", 0.2))
        obs = qcfg.get("observability", {}) or {}
        self.obs_enabled = bool(obs.get("enabled", True))
        self.obs_th = float(obs.get("threshold", 0.6))
        self.obs_alert_ratio = float(obs.get("alert_ratio", 0.2))
        self.obs_sample_rate = float(obs.get("sample_rate", 0.01))
        self.logger = get_logger(__name__)

    @staticmethod
    def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        """
        将数值限制在指定范围内
        
        Args:
            x: 需要限制的数值
            lo: 最小值，默认为 0.0
            hi: 最大值，默认为 1.0
            
        Returns:
            float: 限制在 [lo, hi] 范围内的数值
        """
        return hi if x > hi else (lo if x < lo else x)

    def _valid_char_ratio(self, text: str) -> float:
        """
        计算文本中有效字符的比例
        
        有效字符包括：
        - 字母数字字符 (isalnum())
        - 空白字符 (isspace())
        - 常用标点符号：,.;:!?，。；：！？()（）[]{}-_
        
        Args:
            text: 需要分析的文本字符串
            
        Returns:
            float: 有效字符比例，范围 [0.0, 1.0]
        """
        total = len(text)
        if total <= 0:
            return 0.0
        valid = sum(1 for ch in text if ch.isalnum() or ch.isspace() or ch in ",.;:!?，。；：！？()（）[]{}-_")
        return self._clamp(valid / total)

    def _garbled_ratio(self, text: str) -> float:
        """
        计算文本中乱码字符的比例
        
        乱码字符定义为：
        - 不可打印字符 (not isprintable())
        - 排除正常的空白字符：换行符(\n)、制表符(\t)、空格( )
        
        Args:
            text: 需要分析的文本字符串
            
        Returns:
            float: 乱码字符比例，范围 [0.0, 1.0]
                  空文本返回 1.0（表示完全乱码）
        """
        total = len(text)
        if total <= 0:
            return 1.0
        garbled = sum(1 for ch in text if (not ch.isprintable()) and ch not in ("\n", "\t", " "))
        return self._clamp(garbled / total)

    def _length_score(self, n: int) -> float:
        """
        根据文本长度计算长度得分
        
        长度得分计算规则：
        - 长度 <= min_len：得分为 0.0（太短）
        - 长度 >= sat_len：得分为 1.0（满意长度）
        - 长度在 min_len 和 sat_len 之间：线性插值计算得分
        
        Args:
            n: 文本长度（字符数）
            
        Returns:
            float: 长度得分，范围 [0.0, 1.0]
        """
        if n <= self.min_len:
            return 0.0
        if n >= self.sat_len:
            return 1.0
        return self._clamp((n - self.min_len) / max(1, (self.sat_len - self.min_len)))

    def score(self, text: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算文本质量得分
        
        质量得分基于多个维度的加权计算：
        1. 乱码得分：1.0 - 乱码比例
        2. 有效字符比例：有效字符占总字符的比例
        3. 长度得分：基于文本长度的得分
        4. OCR置信度：可选的OCR平均置信度
        
        当OCR置信度缺失时，会自动调整权重归一化。
        如果启用观测功能，会对低质量样本进行采样日志记录。
        
        Args:
            text: 需要评估质量的文本内容
            meta: 文本的元数据字典，可能包含：
                - ocr_confidence_avg: OCR平均置信度（可选）
                - source: 文本来源（用于日志记录）
                
        Returns:
            Dict[str, Any]: 包含质量评估结果的字典：
                - quality_score: 综合质量得分，范围 [0.0, 1.0]
        """
        t = text.strip()
        n = len(t)
        valid_ratio = self._valid_char_ratio(t)
        garbled_score = 1.0 - self._garbled_ratio(t)
        length_score = self._length_score(n)
        ocr_score = meta.get("ocr_confidence_avg")
        has_ocr = isinstance(ocr_score, (int, float))
        # 权重归一（当 OCR 缺失时）
        total_w = self.w_garbled + self.w_valid + self.w_length + (self.w_ocr if has_ocr else 0.0)
        if total_w <= 0:
            total_w = 1.0
        score = (
            self.w_garbled * garbled_score +
            self.w_valid * valid_ratio +
            self.w_length * length_score +
            (self.w_ocr * float(ocr_score) if has_ocr else 0.0)
        ) / total_w
        score = self._clamp(score)

        # 观测：仅在开启时，对低分样本做采样日志
        if self.obs_enabled and score < self.obs_th:
            # 简单采样：基于哈希的固定概率
            h = abs(hash(t)) % 10000
            if h < int(self.obs_sample_rate * 10000):
                self.logger.info(
                    "quality_low sample: score=%.3f len=%s valid=%.2f garbled_score=%.2f length_score=%.2f src=%s",
                    score, n, valid_ratio, garbled_score, length_score, str(meta.get("source", ""))
                )

        return {"quality_score": score}


