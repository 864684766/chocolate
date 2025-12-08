"""
质量与重排工具集：规则过滤、去重、重排（CLIP/交叉编码器）。

注意：本模块的第三方依赖均为可选，缺失时会优雅降级，仅抛出具体异常给调用方。
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Iterable, List, Tuple, Optional

from app.infra.models import ModelLoader, ModelType, LoaderConfig

logger = logging.getLogger(__name__)


def normalize_caption(text: str) -> str:
    """标准化描述文本。

    - 去除 None 与首尾空白，保证后续规则判断不因空白造成误判。
    
    Args:
        text (str): 需要标准化的原始文本，可以为 None 或空字符串
        
    Returns:
        str: 标准化后的文本，去除首尾空白，None 转换为空字符串
    """
    return (text or "").strip()


def fix_repeated_segments(text: str, separators: str = "，,。.；;！!？?、") -> str:
    """修复片段级重复（按分隔符分割，去除连续重复的片段）。
    
    用处：自动识别并去除文本中按分隔符（逗号、句号等）分割的重复片段，
    例如 "山岳湖,前面有树,前面有树," 修复为 "山岳湖,前面有树,"。
    
    Args:
        text (str): 需要修复的原始文本
        separators (str, optional): 分隔符字符串，默认为 "，,。.；;！!？?、"
        
    Returns:
        str: 修复后的文本，去除重复片段
    """
    if not text or len(text) < 2:
        return text
    
    import re
    # 按配置的分隔符分割文本
    # 转义特殊字符并构建正则表达式
    escaped_separators = re.escape(separators)
    pattern = f'[{escaped_separators}]'
    segments = re.split(pattern, text)
    found_separators = re.findall(pattern, text)
    
    # 过滤空字符串，重组为 (内容, 分隔符) 对
    parts: List[Tuple[str, str]] = []
    for i, segment in enumerate(segments):
        content = segment.strip()
        if content:
            # 获取对应的分隔符（如果有）
            separator = found_separators[i] if i < len(found_separators) else ""
            parts.append((content, separator))
    
    if not parts:
        return text
    
    # 去除连续的重复片段
    result_parts: List[Tuple[str, str]] = []
    prev_content = None
    
    for content, separator in parts:
        # 如果当前内容与前一个相同，跳过（去除重复）
        if content == prev_content:
            continue
        result_parts.append((content, separator))
        prev_content = content
    
    # 重组文本
    if not result_parts:
        return text
    
    result = "".join([content + sep for content, sep in result_parts])
    return result.strip() if result.strip() else text


def calculate_ngram_repetition_ratio(text: str, n: int = 3) -> float:
    """计算 N-Gram 重复比率（0~1）。
    
    用处：计算文本中字符级 N-Gram 的重复比率，用于判断文本是否存在过度重复。
    基于业界实践，重复比率 > 0.5 通常视为异常重复。
    
    Args:
        text (str): 需要检测的文本内容
        n (int, optional): n-gram 的长度，默认为 3。例如 n=3 时检测 3 个字符的重复片段
        
    Returns:
        float: 重复比率，范围 0.0-1.0。0.0 表示没有重复，1.0 表示完全重复
    """
    if not text or len(text) < n:
        return 0.0
    
    tokens = list(text)
    ngram_counts: dict = {}
    total_ngrams = 0
    
    # 统计所有 n-gram 的出现次数
    for i in range(0, max(0, len(tokens) - n + 1)):
        ng = tuple(tokens[i : i + n])
        ngram_counts[ng] = ngram_counts.get(ng, 0) + 1
        total_ngrams += 1
    
    if total_ngrams == 0:
        return 0.0
    
    # 计算重复的 n-gram 数量（出现次数 > 1 的 n-gram）
    repeated_count = sum(count - 1 for count in ngram_counts.values() if count > 1)
    
    # 重复比率 = 重复的 n-gram 数量 / 总 n-gram 数量
    return min(1.0, repeated_count / float(total_ngrams))


def has_excessive_ngram_repetition(text: str, n: int = 3, threshold: float = 0.5) -> bool:
    """检测是否存在过度的 N-Gram 重复（基于重复比率）。
    
    用处：判断文本是否存在过度的字符级重复，用于过滤垃圾文本。
    基于业界实践，重复比率 > 0.5 通常视为异常重复。
    
    Args:
        text (str): 需要检测的文本内容
        n (int, optional): n-gram 的长度，默认为 3
        threshold (float, optional): 重复比率阈值，默认为 0.5（50%）。超过此值视为过度重复
        
    Returns:
        bool: True 表示存在过度重复，False 表示重复程度可接受
    """
    ratio = calculate_ngram_repetition_ratio(text, n)
    return ratio > threshold



def contains_blacklisted(text: str, keywords: Iterable[str]) -> bool:
    """检测是否包含黑名单关键词（大小写不敏感）。
    
    Args:
        text (str): 需要检测的文本内容
        keywords (Iterable[str]): 黑名单关键词列表，可以是列表、集合等可迭代对象
        
    Returns:
        bool: True 表示文本包含黑名单关键词，False 表示不包含
    """
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
    """是否命中常见广告/联系方式正则（网址/电话/邮箱）。
    
    Args:
        text (str): 需要检测的文本内容
        
    Returns:
        bool: True 表示文本包含网址、电话或邮箱等联系方式，False 表示不包含
    """
    s = text or ""
    return bool(_REGEX_URL.search(s) or _REGEX_PHONE.search(s) or _REGEX_EMAIL.search(s))

def gibberish_ratio(text: str) -> float:
    """估算"乱码占比"（0~1）。

    - 统计非字母数字/空白字符比例，比例越高越可疑。
    
    Args:
        text (str): 需要计算乱码占比的文本内容
        
    Returns:
        float: 乱码占比，范围 0.0-1.0。0.0 表示没有乱码，1.0 表示完全乱码
    """
    if not text:
        return 1.0
    valid = sum(ch.isalnum() or ch.isspace() for ch in text)
    return max(0.0, 1.0 - valid / float(len(text)))

def repetition_ratio(text: str) -> float:
    """计算重复率（0~1）。

    - 粗略度量：唯一字符占比越低，重复率越高。
    - 仅作为启发式阈值使用。

    Args:
        text (str): 需要计算重复率的文本内容

    Returns:
        float: 重复率，范围 0.0-1.0。0.0 表示没有重复，1.0 表示完全重复
    """
    if not text:
        return 0.0
    total = len(text)
    unique = len(set(text))
    return max(0.0, 1.0 - unique / float(total))

def md5_of(text: str) -> str:
    """计算文本的 MD5 哈希（用于精确去重）。
    
    Args:
        text (str): 需要计算 MD5 哈希的文本内容
        
    Returns:
        str: 文本的 MD5 哈希值，32 位十六进制字符串
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def near_duplicate(c1: str, c2: str, threshold: float = 0.95, embed_model: Optional[str] = None) -> bool:
    """近似重复检测（基于向量余弦相似度）。

    - 默认用多语言句向量模型计算相似度；相似度≥threshold 视为重复。
    - 依赖不可用时返回 False 并记录日志。
    - 使用模块级模型缓存，避免重复加载相同模型。
    
    Args:
        c1 (str): 第一个文本内容
        c2 (str): 第二个文本内容
        threshold (float, optional): 相似度阈值，默认为 0.95。相似度大于等于此值视为重复
        embed_model (Optional[str], optional): 嵌入模型名称，默认为多语言模型
        
    Returns:
        bool: True 表示两个文本近似重复，False 表示不重复或检测失败
    """
    try:
        from sentence_transformers import util
        
        model_name = embed_model

        if not model_name:
            logger.info(f"sentence-transformers not available for near-dup: {model_name}")
            return False
        
        # 使用通用模型加载器加载模型（自动缓存）
        config = LoaderConfig(
            model_name=model_name,
            device="auto",
            model_type=ModelType.SENTENCE_TRANSFORMER
        )
        model = ModelLoader.load_model(config)
        
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
    max_repetition_ratio: float = 0.5,
    segment_separators: str = "，,。.；;！!？?、",
) -> List[str]:
    """基于规则的候选过滤。

    - 先修复片段级重复，再进行质量检查。
    - 长度阈值、黑名单（关键词/正则）、乱码占比、重复 n-gram 比率。
    - 返回通过过滤的 caption 列表。
    
    Args:
        captions (List[str]): 需要过滤的文本列表
        min_len (int, optional): 最小长度阈值，默认为 5
        max_len (int, optional): 最大长度阈值，默认为 120
        blacklist_keywords (Optional[List[str]], optional): 黑名单关键词列表，默认为 None
        max_gibberish_ratio (float, optional): 最大乱码占比阈值，默认为 0.3
        forbid_repeat_ngram (int, optional): N-gram 长度，用于计算重复比率，默认为 3。设为 0 表示不检查重复
        max_repetition_ratio (float, optional): 最大重复比率阈值，默认为 0.5（50%）。超过此值视为过度重复
        segment_separators (str, optional): 片段分隔符字符串，用于修复片段级重复，默认为 "，,。.；;！!？?、"
        
    Returns:
        List[str]: 通过所有过滤条件的文本列表
    """
    blacklist_keywords = blacklist_keywords or []
    kept: List[str] = []
    for c in captions:
        # 第一步：标准化
        c = normalize_caption(c)
        if not c:
            continue
        
        # 第二步：修复片段级重复（使用配置的分隔符）
        c = fix_repeated_segments(c, separators=segment_separators)
        
        # 第三步：长度检查
        if len(c) < min_len or len(c) > max_len:
            continue
        
        # 第四步：黑名单检查
        if hits_blacklist_regex(c) or contains_blacklisted(c, blacklist_keywords):
            continue
        
        # 第五步：乱码占比检查
        if gibberish_ratio(c) > max_gibberish_ratio:
            continue
        
        # 第六步：字符级重复比率检查（修复后仍可能存在的字符级重复）
        if forbid_repeat_ngram > 0:
            if has_excessive_ngram_repetition(c, n=forbid_repeat_ngram, threshold=max_repetition_ratio):
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
    
    Args:
        captions (List[str]): 需要去重的文本列表
        approx (bool, optional): 是否启用近似去重，默认为 True
        threshold (float, optional): 近似去重的相似度阈值，默认为 0.95
        embed_model (Optional[str], optional): 嵌入模型名称，用于近似去重，默认为 None
        
    Returns:
        List[str]: 去重后的文本列表，保持原有顺序
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
    
    Args:
        image: 图像对象，可以是 PIL Image 或图像路径
        captions (List[str]): 需要重排的文本描述列表
        model_name (str): CLIP 模型名称，如 "openai/clip-vit-base-patch32"
        top_k (int, optional): 返回前 k 个结果，默认为 2
        
    Returns:
        List[Tuple[str, float]]: 重排后的结果列表，每个元素为 (文本描述, 相似度概率)
    """
    try:
        import torch
        
        # 使用通用模型加载器加载 CLIP 模型和处理器（自动缓存）
        config = LoaderConfig(
            model_name=model_name,
            device="auto",
            model_type=ModelType.CLIP
        )
        clip_model, processor = ModelLoader.load_model(config)
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



