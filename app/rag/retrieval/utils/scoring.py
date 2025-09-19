"""
检索打分映射工具。

职责：
- 将不同度量（余弦距离/相似度、L2、内积）的返回值统一映射为 [0,1] 分数，便于阈值判断与排序。

约定：
- 所有函数提供清晰的参数/返回注释，便于复用与测试。
"""

from __future__ import annotations

from math import exp
from typing import Literal


# 类型别名：度量名称
Metric = Literal["cosine_distance", "cosine_similarity", "l2", "inner_product"]


def clamp01(value: float) -> float:
    """将输入数值裁剪到 [0,1] 区间。

    Args:
        value: 任意浮点数。

    Returns:
        被裁剪到 [0,1] 的浮点数。
    """
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def score_from_distance(value: float, metric: Metric, normalize: bool, alpha: float) -> float:
    """将不同度量的返回值映射为统一分数。

    说明：
    - cosine_distance: 距离越小越好。若距离 d∈[0,2]，则使用 1 - d/2；若 d∈[0,1]，也能得到合理结果。
    - cosine_similarity: 相似度越大越好，s∈[-1,1] → (s+1)/2。
    - l2: 欧氏距离越小越好，使用 exp(-alpha * d) 进行单调映射。
    - inner_product: 依赖业务范围，默认仅做 [0,1] 裁剪。

    Args:
        value: 底层检索返回的原始数值（距离或相似度）。
        metric: 度量名称。
        normalize: 是否将结果限制在 [0,1]。
        alpha: 指数衰减的系数，仅在 l2 下有效。

    Returns:
        映射后的分数，默认范围 [0,1]。
    """
    if metric == "cosine_distance":
        # 使用保守归一：score = 1 - d/2，可兼容 d∈[0,1] 与 d∈[0,2]
        score = 1.0 - (float(value) / 2.0)
        return clamp01(score) if normalize else score

    if metric == "cosine_similarity":
        score = (float(value) + 1.0) / 2.0
        return clamp01(score) if normalize else score

    if metric == "l2":
        score = exp(-float(value) * float(alpha))
        return clamp01(score) if normalize else score

    if metric == "inner_product":
        # 不了解分布时仅裁剪；如需更稳健，可在上层做线性归一
        score = float(value)
        return clamp01(score) if normalize else score

    # 兜底：按“越小越好”的距离处理
    score = 1.0 - float(value)
    return clamp01(score) if normalize else score


