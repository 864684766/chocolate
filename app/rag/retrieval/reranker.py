from __future__ import annotations

"""
交叉编码器重排器（占位最小实现）。

职责：
- 对检索候选进行精排，返回更高相关性的前 TopN。

说明：
- 当前为占位实现：直接按输入分数排序并裁剪 TopN。
- 后续可接入开源重排模型（如 BAAI/bge-reranker-*、cross-encoder/ms-marco-*）。
"""

from typing import List, Optional

from .schemas import RetrievedItem
from app.infra.logging import get_logger
from app.config import get_config_manager


class CrossEncoderReranker:
    """交叉编码器重排器占位实现。

    方法体不超过 20 行；真实模型推理请拆分到独立辅助类/模块。
    """

    def __init__(self, model: str | None = None):
        self.model = model or "BAAI/bge-reranker-base"
        self._model = None
        self.logger = get_logger(__name__)

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            self._model = CrossEncoder(self.model)
        except Exception as e:
            self.logger.warning(f"CrossEncoder load failed ({self.model}): {e}; fallback to score sort.")
            self._model = None

    def rerank(self, items: List[RetrievedItem], top_n: int = 10, query: Optional[str] = None) -> List[RetrievedItem]:
        """对候选进行重排并返回前 TopN。

        Args:
            items: 检索候选列表。
            top_n: 返回前 N 个结果。
            query: 原始查询文本（为空则使用现有分数排序）。

        Returns:
            List[RetrievedItem]: 重排后的前 N 项。
        """
        if not items:
            return []
        if not query:
            return sorted(items, key=lambda x: float(x.score), reverse=True)[: max(1, int(top_n))]
        self._load_model()
        if self._model is None:
            return sorted(items, key=lambda x: float(x.score), reverse=True)[: max(1, int(top_n))]
        cfg = get_config_manager().get_config("retrieval") or {}
        bs = int((cfg.get("rerank") or {}).get("batch_size", 16))
        pairs = [(query, it.text) for it in items]
        try:
            scores = self._model.predict(pairs, batch_size=bs)  # type: ignore[attr-defined]
            rescored = [RetrievedItem(id=it.id, text=it.text, score=float(s), metadata=it.metadata) for it, s in zip(items, scores)]
            return sorted(rescored, key=lambda x: x.score, reverse=True)[: max(1, int(top_n))]
        except Exception as e:
            self.logger.warning(f"CrossEncoder predict failed: {e}; fallback sort.")
            return sorted(items, key=lambda x: float(x.score), reverse=True)[: max(1, int(top_n))]


