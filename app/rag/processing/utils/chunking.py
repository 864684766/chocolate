from __future__ import annotations

from typing import Any, Dict, Tuple

from app.config import get_config_manager


def decide_chunk_params(media_type: str, content: Any) -> Tuple[int, int]:
    """决定分块参数。

    用处：综合配置与内容特征，产出 `chunk_size` 与 `overlap`，供上层流水线/策略使用。

    Args:
        media_type (str): 媒体类型（text/pdf/image/video/audio/code）。
        content (Any): 原始内容或其载荷，用于估算长度/密度。

    Returns:
        Tuple[int, int]: (chunk_size, overlap)，单位为近似 token 数（或字符近似）。
    """
    cfg = get_config_manager().get_media_processing_config().get("chunking", {})
    if not cfg:
        base = _params_from_tables({"defaults": {}}, media_type)
        return _auto_tune_params({}, content, base)
    if not bool(cfg.get("auto", True)):
        return _params_from_tables(cfg, media_type)
    base = _params_from_tables(cfg, media_type)
    return _auto_tune_params(cfg, content, base)


def _params_from_tables(cfg: Dict[str, Any], media_type: str) -> Tuple[int, int]:
    """从配置表格获取参数。

    用处：当启用手动配置或作为自适应的基线时，读取 `defaults/by_media_type`。

    Args:
        cfg (Dict[str, Any]): `media_processing.chunking` 配置段。
        media_type (str): 媒体类型键。

    Returns:
        Tuple[int, int]: (chunk_size, overlap)。未配置时返回温和的估算值。
    """
    defaults = cfg.get("defaults", {})
    table = cfg.get("by_media_type", {})
    mt = table.get(media_type.lower(), {})
    size = mt.get("chunk_size", defaults.get("chunk_size"))
    ov = mt.get("overlap", defaults.get("overlap"))
    if size is None or ov is None:
        return _estimate_from_media(media_type)
    return max(50, int(size)), max(0, int(ov))


def _estimate_tokens(text: str) -> int:
    """估算 token 数（无 tokenizer 情况下）。

    Args:
        text (str): 文本内容。

    Returns:
        int: 估算 token 数（1 token≈1.6 字符）。
    """
    if not text:
        return 0
    return int(len(str(text)) / 1.6)


def _auto_tune_params(cfg: Dict[str, Any], content: Any, base: Tuple[int, int]) -> Tuple[int, int]:
    """基于内容自适应计算参数。

    用处：在 `auto=true` 或无配置时，根据目标片长/上下文窗口与内容规模，生成稳健参数。

    Args:
        cfg (Dict[str, Any]): `chunking.targets` 可选参数段。
        content (Any): 内容或载荷，用于估算 token。
        base (Tuple[int,int]): 基线 (size, overlap)，常来自表格配置。

    Returns:
        Tuple[int,int]: 自适应后的 (chunk_size, overlap)。
    """
    targets = cfg.get("targets", {})
    base_size, base_overlap = base
    target = int(targets.get("target_tokens_per_chunk", max(200, int(base_size * 0.8))))
    max_ctx = int(targets.get("max_context_tokens", base_size * 16 or 8192))
    overlap_ratio = float(targets.get("overlap_ratio", max(0.1, min(0.25, (base_overlap or 1) / max(base_size, 1)))))

    if isinstance(content, str):
        tokens = _estimate_tokens(content)
    elif isinstance(content, list):
        joined = "\n".join(map(lambda x: str(x)[:2000], content))
        tokens = _estimate_tokens(joined)
    else:
        tokens = _estimate_tokens(str(content))

    chunk_size = min(max(200, target), max(200, max_ctx // 4))
    overlap = int(chunk_size * overlap_ratio)

    if tokens < chunk_size * 1.2:
        overlap = min(overlap, max(10, int(base_overlap * 0.5) if base_overlap else 50))

    upper_size = max(base_size * 2, int(base_size + base_size * 0.5) or chunk_size)
    lower_size = max(100, int(base_size * 0.6) if base_size else 200)
    chunk_size = max(min(chunk_size, upper_size), lower_size)

    upper_ov = max(base_overlap * 2, int(base_overlap + base_overlap * 0.5) if base_overlap else overlap)
    lower_ov = max(0, int(base_overlap * 0.5) if base_overlap else 0)
    overlap = max(min(overlap, upper_ov), lower_ov)
    return int(chunk_size), int(overlap)


def _estimate_from_media(media_type: str) -> Tuple[int, int]:
    """依据媒体类型给出温和估计。

    用处：当配置缺省时，避免固定魔法数；通过简单启发式给出基线。

    Args:
        media_type (str): 媒体类型。

    Returns:
        Tuple[int,int]: 估算的 (chunk_size, overlap)。
    """
    mt = media_type.lower()
    base_map = {
        "text": (700, 120),
        "pdf": (900, 170),
        "image": (320, 60),
        "video": (1000, 180),
        "audio": (1000, 180),
        "code": (420, 80),
    }
    size, ov = base_map.get(mt, (700, 120))
    return int(size), int(ov)


