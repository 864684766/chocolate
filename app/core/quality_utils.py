"""
质量与重排工具集：规则过滤、去重、重排（CLIP/交叉编码器）。

注意：本模块的第三方依赖均为可选，缺失时会优雅降级，仅抛出具体异常给调用方。
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Iterable, List, Tuple, Optional

logger = logging.getLogger(__name__)


def normalize_caption(text: str) -> str:
    """标准化描述文本。

    - 去除 None 与首尾空白，保证后续规则判断不因空白造成误判。
    """
    return (text or "").strip()


def has_repeated_ngram(text: str, n: int = 3) -> bool:
    """检测是否存在重复的 n-gram 片段（字符级）。

    - 典型用于识别“self self self”这类重复结构。
    - 返回 True 表示存在重复。
    """
    tokens = list(text)
    seen: set = set()
    for i in range(0, max(0, len(tokens) - n + 1)):
        ng = tuple(tokens[i : i + n])
        if ng in seen:
            return True
        seen.add(ng)
    return False


def repetition_ratio(text: str) -> float:
    """计算重复率（0~1）。

    - 粗略度量：唯一字符占比越低，重复率越高。
    - 仅作为启发式阈值使用。
    """
    if not text:
        return 0.0
    total = len(text)
    unique = len(set(text))
    return max(0.0, 1.0 - unique / float(total))


def contains_blacklisted(text: str, keywords: Iterable[str]) -> bool:
    """检测是否包含黑名单关键词（大小写不敏感）。"""
    if not text:
        return False
    lowered = text.lower()
    for kw in keywords:
        if kw and kw.lower() in lowered:
            return True
    return False


_REGEX_URL = re.compile(r"https?://|www\\.", re.I)
_REGEX_PHONE = re.compile(r"(\+?\d[\d\- ]{7,}\d)")
_REGEX_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def hits_blacklist_regex(text: str) -> bool:
    """是否命中常见广告/联系方式正则（网址/电话/邮箱）。"""
    s = text or ""
    return bool(_REGEX_URL.search(s) or _REGEX_PHONE.search(s) or _REGEX_EMAIL.search(s))


def gibberish_ratio(text: str) -> float:
    """估算“乱码占比”（0~1）。

    - 统计非字母数字/空白字符比例，比例越高越可疑。
    """
    if not text:
        return 1.0
    valid = sum(ch.isalnum() or ch.isspace() for ch in text)
    return max(0.0, 1.0 - valid / float(len(text)))


def md5_of(text: str) -> str:
    """计算文本的 MD5 哈希（用于精确去重）。"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def near_duplicate(c1: str, c2: str, threshold: float = 0.95, embed_model: Optional[str] = None) -> bool:
    """近似重复检测（基于向量余弦相似度）。

    - 默认用多语言句向量模型计算相似度；相似度≥threshold 视为重复。
    - 依赖不可用时返回 False 并记录日志。
    """
    try:
        from sentence_transformers import SentenceTransformer, util
        model_name = embed_model or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        model = SentenceTransformer(model_name)
        emb = model.encode([c1, c2], convert_to_tensor=True, normalize_embeddings=True)
        sim = float(util.cos_sim(emb[0], emb[1]).item())
        return sim >= threshold
    except (ImportError, ModuleNotFoundError) as e:
        logger.info(f"sentence-transformers not available for near-dup: {e}")
    except Exception as e:
        logger.warning(f"near-dup computation failed: {e}")
    return False


def filter_captions(
    captions: List[str],
    *,
    min_len: int = 5,
    max_len: int = 120,
    blacklist_keywords: Optional[List[str]] = None,
    max_gibberish_ratio: float = 0.3,
    forbid_repeat_ngram: int = 3,
) -> List[str]:
    """基于规则的候选过滤。

    - 长度阈值、黑名单（关键词/正则）、乱码占比、重复 n-gram。
    - 返回通过过滤的 caption 列表。
    """
    blacklist_keywords = blacklist_keywords or []
    kept: List[str] = []
    for c in captions:
        c = normalize_caption(c)
        if not c:
            continue
        if len(c) < min_len or len(c) > max_len:
            continue
        if hits_blacklist_regex(c) or contains_blacklisted(c, blacklist_keywords):
            continue
        if gibberish_ratio(c) > max_gibberish_ratio:
            continue
        if forbid_repeat_ngram and has_repeated_ngram(c, n=forbid_repeat_ngram):
            continue
        kept.append(c)
    return kept


def dedup_captions(
    captions: List[str],
    *,
    approx: bool = True,
    threshold: float = 0.95,
    embed_model: Optional[str] = None,
) -> List[str]:
    """去重（精确 + 近似）。

    - 精确去重：相同 MD5 只保留一条。
    - 近似去重：与已保留文本的相似度≥threshold 则丢弃。
    """
    seen_hash: set = set()
    result: List[str] = []
    for c in captions:
        h = md5_of(c)
        if h in seen_hash:
            continue
        if approx and any(near_duplicate(c, r, threshold=threshold, embed_model=embed_model) for r in result):
            continue
        seen_hash.add(h)
        result.append(c)
    return result


def clip_rerank(image, captions: List[str], model_name: str, top_k: int = 2) -> List[Tuple[str, float]]:
    """使用 CLIP 对 (image, caption) 做相似度重排。

    - 返回 [(caption, prob)]，按概率从高到低取前 top_k。
    - 依赖缺失或失败时返回原顺序的前 top_k，并给 0.0 分。
    """
    try:
        from transformers import CLIPProcessor, CLIPModel
        import torch
        clip_model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        inputs = processor(text=captions, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0].tolist()
        ranked = sorted(zip(captions, probs), key=lambda x: x[1], reverse=True)
        return ranked[: max(1, top_k)]
    except (ImportError, ModuleNotFoundError) as e:
        logger.info(f"CLIP not available: {e}")
        return [(c, 0.0) for c in captions[: top_k]]
    except Exception as e:
        logger.warning(f"CLIP rerank failed: {e}")
        return [(c, 0.0) for c in captions[: top_k]]


def cross_encoder_rerank(pairs: List[Tuple[str, str]], model_name: str) -> List[Tuple[int, float]]:
    """交叉编码器精排。

    - 输入：pairs = [(query, doc)]。
    - 输出：[(idx, score)]，按 score 降序排序的索引与分数。
    - 失败或依赖缺失时，返回 0 分占位，调用方可据此降级处理。
    """
    try:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(model_name)

        # 核心修正点：确保 scores 是一个纯粹的浮点数列表/数组
        # CrossEncoder.predict() 通常返回 numpy.ndarray (float)
        # 如果模型内部逻辑导致返回类型不是纯浮点数列表，这里需要提取
        raw_scores = ce.predict(pairs)

        # 确认 raw_scores 是一个一维的浮点数序列
        # 如果 raw_scores 的每个元素本身也是一个元组（例如 (idx, score)），
        # 那么需要提取其中的分数部分。
        # 这种情况通常不会发生，除非模型内部有特殊逻辑或用户传入了错误的pairs

        # 这里假设 raw_scores 已经是 List[float] 或 numpy.ndarray
        # 如果 raw_scores 出现异常结构（比如 [(score,), (score,), ...]），
        # 下面 enumerate 后的 x[1] 就会是 (score,) 这样的元组，导致 TypeError

        # 我们可以强制将分数转换为 float，并确保结构正确
        processed_scores = [float(s) for s in raw_scores]  # 确保每个元素都是 float

        ranked = sorted(list(enumerate(processed_scores)), key=lambda x: x[1], reverse=True)
        return ranked
    except (ImportError, ModuleNotFoundError) as e:
        logger.info(f"CrossEncoder not available: {e}")
        # 在这种情况下，返回的默认值也应该符合类型签名
        return [(i, 0.0) for i, _ in enumerate(pairs)]
    except Exception as e:
        logger.warning(f"CrossEncoder rerank failed: {e}")
        # 在这种情况下，返回的默认值也应该符合类型签名
        return [(i, 0.0) for i, _ in enumerate(pairs)]

