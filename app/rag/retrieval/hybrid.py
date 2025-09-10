from __future__ import annotations

"""
混合检索融合器：提供 RRF 与加权求和两种基础融合策略。

说明：
- 输入为两个 RetrievalResult（vector 与 keyword），输出为融合后的 RetrievalResult。
- 仅做融合与排序，不负责具体检索执行与路由决策。
"""

from typing import Dict
from .schemas import RetrievalResult, RetrievedItem


class HybridSearcher:
    """混合检索融合器。"""

    @staticmethod
    def rrf(vector: RetrievalResult, keyword: RetrievalResult, k: int = 60, top_k: int = 10) -> RetrievalResult:
        """Reciprocal Rank Fusion（稳健、免调参）。"""
        # 构建排名映射
        rank_vec = {it.id: idx for idx, it in enumerate(vector.items)}
        rank_kw = {it.id: idx for idx, it in enumerate(keyword.items)}

        scores: Dict[str, float] = {}
        items_map: Dict[str, RetrievedItem] = {}

        # 合并ID集合
        all_ids = list({*rank_vec.keys(), *rank_kw.keys()})
        for _id in all_ids:
            rv = rank_vec.get(_id)
            rk = rank_kw.get(_id)
            s = 0.0
            if rv is not None:
                s += 1.0 / (k + rv + 1)
            if rk is not None:
                s += 1.0 / (k + rk + 1)
            scores[_id] = s
            # 保存任一来源的 item 信息
            it = next((i for i in vector.items if i.id == _id), None) or next((i for i in keyword.items if i.id == _id), None)
            if it:
                items_map[_id] = it

        merged = sorted(items_map.values(), key=lambda x: scores.get(x.id, 0.0), reverse=True)[: top_k]
        return RetrievalResult(items=merged, latency_ms=vector.latency_ms + keyword.latency_ms, debug_info={"fusion": "rrf"})

    @staticmethod
    def weighted_sum(vector: RetrievalResult, keyword: RetrievalResult, w_vec: float = 0.7, w_kw: float = 0.3, top_k: int = 10) -> RetrievalResult:
        """加权求和融合。"""
        scores: Dict[str, float] = {}
        items_map: Dict[str, RetrievedItem] = {}

        def norm(score: float) -> float:
            # 简单归一化（向量得分已在 0~1，关键词默认 1.0）
            return max(0.0, min(1.0, score))

        for it in vector.items:
            scores[it.id] = scores.get(it.id, 0.0) + w_vec * norm(it.score)
            items_map[it.id] = it
        for it in keyword.items:
            scores[it.id] = scores.get(it.id, 0.0) + w_kw * norm(it.score)
            items_map[it.id] = items_map.get(it.id, it)

        merged = sorted(items_map.values(), key=lambda x: scores.get(x.id, 0.0), reverse=True)[: top_k]
        return RetrievalResult(items=merged, latency_ms=vector.latency_ms + keyword.latency_ms, debug_info={"fusion": "weighted_sum", "w_vec": w_vec, "w_kw": w_kw})


