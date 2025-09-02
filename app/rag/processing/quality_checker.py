from __future__ import annotations

from typing import Dict, Any
from .interfaces import QualityAssessor


class SimpleQualityAssessor(QualityAssessor):
    """极简质量评估：给出长度、是否过短等指标，便于过滤或调试。"""

    def __init__(self, min_len: int = 20) -> None:
        self.min_len = min_len

    def score(self, text: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        length = len(text.strip())
        return {
            "text_len": length,
            "too_short": length < self.min_len,
        }


