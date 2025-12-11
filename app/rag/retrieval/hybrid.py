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
    """
    混合检索融合器。

    用处：将向量检索和关键词检索的结果进行融合，生成统一的排序结果。
    提供两种融合策略：RRF（Reciprocal Rank Fusion）和加权求和（Weighted Sum）。

    职责：
    - 融合两种检索方法的结果
    - 对融合后的结果进行排序
    - 返回 Top-K 结果

    不负责：
    - 具体检索执行（由 VectorRetriever 和 MeilisearchRetriever 负责）
    - 路由决策（由 RetrievalOrchestrator 负责）
    """

    @staticmethod
    def rrf(vector: RetrievalResult, keyword: RetrievalResult, k: int = 60, top_k: int = 10) -> RetrievalResult:
        """
        Reciprocal Rank Fusion（RRF）融合算法。

        用处：基于排名融合两种检索结果，不依赖得分绝对值，稳健且免调参。
        适用于两种检索方法质量相近的场景。

        算法原理：
        - 基于排名计算融合得分，公式：score = 1/(k + rank1 + 1) + 1/(k + rank2 + 1)
        - 排名越靠前，得分越高
        - k 参数控制排名差异的影响程度，默认 60

        优势：
        - 稳健：不依赖得分绝对值，只依赖排名
        - 免调参：k 参数通常使用默认值即可
        - 适合两种检索结果质量相近时

        Args:
            vector: 向量检索结果，包含检索到的文档列表和延迟信息
            keyword: 关键词检索结果，包含检索到的文档列表和延迟信息
            k: RRF 算法参数，控制排名差异的影响程度，默认 60
                值越大，排名靠后的结果得分差异越小
                值越小，排名靠前的结果得分差异越大
            top_k: 返回结果数量上限，默认 10

        Returns:
            RetrievalResult: 融合后的检索结果
                - items: 按融合得分排序的文档列表（Top-K）
                - latency_ms: 两种检索方法的延迟总和

        注意：
        - 如果某个文档只在一个检索结果中出现，也会被包含在融合结果中
        - 融合得分 = 向量检索排名得分 + 关键词检索排名得分
        """
        # 构建排名映射：文档ID -> 排名（从0开始）
        rank_vec = {it.id: idx for idx, it in enumerate(vector.items)}
        rank_kw = {it.id: idx for idx, it in enumerate(keyword.items)}

        scores: Dict[str, float] = {}
        items_map: Dict[str, RetrievedItem] = {}

        # 合并两种检索结果的所有文档ID
        all_ids = list({*rank_vec.keys(), *rank_kw.keys()})
        for _id in all_ids:
            # 获取文档在两种检索结果中的排名
            rv = rank_vec.get(_id)
            rk = rank_kw.get(_id)
            s = 0.0
            # 计算向量检索的排名得分
            if rv is not None:
                s += 1.0 / (k + rv + 1)
            # 计算关键词检索的排名得分
            if rk is not None:
                s += 1.0 / (k + rk + 1)
            scores[_id] = s
            # 保存任一来源的 item 信息（优先使用向量检索的结果）
            it = next((i for i in vector.items if i.id == _id), None) or next((i for i in keyword.items if i.id == _id), None)
            if it:
                items_map[_id] = it

        # 按融合得分降序排序，取 Top-K
        merged = sorted(items_map.values(), key=lambda x: scores.get(x.id, 0.0), reverse=True)[: top_k]
        return RetrievalResult(items=merged, latency_ms=vector.latency_ms + keyword.latency_ms)

    @staticmethod
    def weighted_sum(vector: RetrievalResult, keyword: RetrievalResult, w_vec: float = 0.7, w_kw: float = 0.3, top_k: int = 10) -> RetrievalResult:
        """
        加权求和融合算法。

        用处：基于得分加权融合两种检索结果，可以精细调整权重，适合需要偏向某种检索方法的场景。

        算法原理：
        - 对两种检索的得分进行归一化
        - 计算加权和：final_score = w_vec * vector_score + w_kw * keyword_score
        - 按加权得分排序

        适用场景：
        - 两种检索方法质量差异较大时，可通过权重平衡
        - 需要明确偏向某种检索方式时（如精确查询偏向关键词，语义查询偏向向量）
        - 需要基于得分做精细过滤时

        Args:
            vector: 向量检索结果，包含检索到的文档列表和相似度得分
            keyword: 关键词检索结果，包含检索到的文档列表和相关度得分
            w_vec: 向量检索的权重，默认 0.7（70%）
                建议范围：0.0 ~ 1.0
                当向量检索质量明显更好时，可提高此权重
            w_kw: 关键词检索的权重，默认 0.3（30%）
                建议范围：0.0 ~ 1.0
                当关键词检索质量明显更好时，可提高此权重
                注意：w_vec + w_kw 不需要等于 1.0，系统会自动处理
            top_k: 返回结果数量上限，默认 10

        Returns:
            RetrievalResult: 融合后的检索结果
                - items: 按加权得分排序的文档列表（Top-K）
                - latency_ms: 两种检索方法的延迟总和

        注意：
        - 得分会被归一化到 0~1 范围
        - 如果某个文档只在一个检索结果中出现，也会被包含在融合结果中
        - 融合得分 = 向量检索加权得分 + 关键词检索加权得分
        """
        scores: Dict[str, float] = {}
        items_map: Dict[str, RetrievedItem] = {}

        def norm(score: float) -> float:
            """
            得分归一化函数。

            用处：将得分归一化到 0~1 范围，确保两种检索方法的得分在同一尺度。

            Args:
                score: 原始得分
                    - 向量检索得分：通常在 0~1 之间（余弦相似度）
                    - 关键词检索得分：可能超出 0~1 范围，需要归一化

            Returns:
                float: 归一化后的得分，范围 [0.0, 1.0]
            """
            return max(0.0, min(1.0, score))

        # 处理向量检索结果：累加加权得分
        for it in vector.items:
            scores[it.id] = scores.get(it.id, 0.0) + w_vec * norm(it.score)
            items_map[it.id] = it
        # 处理关键词检索结果：累加加权得分
        for it in keyword.items:
            scores[it.id] = scores.get(it.id, 0.0) + w_kw * norm(it.score)
            # 如果文档已在向量检索结果中，保留向量检索的 item 信息
            items_map[it.id] = items_map.get(it.id, it)

        # 按加权得分降序排序，取 Top-K
        merged = sorted(items_map.values(), key=lambda x: scores.get(x.id, 0.0), reverse=True)[: top_k]
        return RetrievalResult(items=merged, latency_ms=vector.latency_ms + keyword.latency_ms)


