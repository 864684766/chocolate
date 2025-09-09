from __future__ import annotations

from typing import Any, Dict, Union, Iterable
from app.config import get_config_manager

Primitive = Union[str, int, float, bool, None]


def _is_primitive(value: Any) -> bool:
    """判断是否为基础类型。

    用处：ChromaDB 的 metadata 值必须是基础类型。

    Args:
        value (Any): 任意值。

    Returns:
        bool: 是否为 str/int/float/bool/None。
    """
    return isinstance(value, (str, int, float, bool)) or value is None


def _extract_image_meta(meta: Dict[str, Any]) -> Dict[str, Primitive]:
    """从 image_meta 提取可检索的原子键。

    提取：ocr_engine、image_format、total_texts。
    """
    out: Dict[str, Primitive] = {}
    im = meta.get("image_meta")
    if isinstance(im, dict):
        for k in ("ocr_engine", "image_format", "total_texts"):
            v = im.get(k)
            if _is_primitive(v):
                out[k] = v  # type: ignore[assignment]
    return out


def _extract_bounds(meta: Dict[str, Any]) -> Dict[str, Primitive]:
    """从 region_bounds/dict 中提取边界为原子键。"""
    out: Dict[str, Primitive] = {}
    bounds = meta.get("region_bounds")
    if isinstance(bounds, dict):
        for k in ("min_x", "max_x", "min_y", "max_y"):
            v = bounds.get(k)
            if _is_primitive(v):
                out[k] = v  # type: ignore[assignment]
    return out


def build_metadata_from_meta(meta: Dict[str, Any]) -> Dict[str, Primitive]:
    """构造“通用最小集” metadatas（仅基础类型）。

    入参为块的 meta（可能较丰富），输出为可直接写入 Chroma 的扁平字典：
    - 通用：doc_id/source/filename/content_type/media_type
            chunk_index/chunk_type/chunk_size/created_at
    - 媒体：page_number/start_pos/end_pos/region_index
    - 图片：ocr_engine/image_format/total_texts/min_x/max_x/min_y/max_y
    """
    whitelist: Iterable[str] = get_config_manager().get_config("vectorization").get("metadata_whitelist", [])

    out: Dict[str, Primitive] = {
        k: v for k, v in meta.items() if k in whitelist and _is_primitive(v)
    }
    # 如白名单包含图片/边界字段，则尝试从嵌套结构提取
    if any(k in whitelist for k in ("ocr_engine", "image_format", "total_texts")):
        out.update(_extract_image_meta(meta))
    if any(k in whitelist for k in ("min_x", "max_x", "min_y", "max_y")):
        out.update(_extract_bounds(meta))
    return out


