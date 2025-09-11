from __future__ import annotations

from typing import Dict, Tuple, List, Iterable
from hashlib import sha1

from .metadata_utils import normalize_text_for_vector, build_metadata_from_meta


def normalize_and_build_id(text: str, meta: Dict) -> Tuple[str, str, Dict]:
    """规范化文本并生成稳定ID（不再二次归一化 meta）。

    Args:
        text: 原始文本
        meta: 原始元数据

    Returns:
        (uid, norm_text, norm_meta)
    """
    norm_text = normalize_text_for_vector(text)
    # 保留入口与流水线产出的扁平 meta，不做二次归一化；
    # 最终写库前由 build_metadata_from_meta 统一白名单扁平化。
    norm_meta = dict(meta)
    doc_id = str(norm_meta.get("doc_id") or norm_meta.get("filename") or "")
    chunk_index = int(norm_meta.get("chunk_index", -1) or -1)
    sig = sha1(norm_text.encode("utf-8")).hexdigest()[:16]
    uid = f"{doc_id}:{chunk_index}:{sig}"
    return uid, norm_text, norm_meta


def dedup_in_batch(items: Iterable[Tuple[str, str, Dict]]) -> Dict[str, Tuple[str, Dict]]:
    """批内去重：按 uid 去重。

    Args:
        items: 可迭代的 (uid, norm_text, norm_meta)

    Returns:
        dict: uid -> (norm_text, norm_meta)
    """
    uniq: Dict[str, Tuple[str, Dict]] = {}
    for uid, t, m in items:
        if uid not in uniq:
            uniq[uid] = (t, m)
    return uniq


def slice_new_records(ids: List[str], exist_ids: set, texts: List[str], metadatas: List[Dict]) -> Tuple[List[str], List[str], List[Dict]]:
    """根据已存在ID集合切片出新增记录。

    Returns:
        (ids_new, texts_new, metadatas_new)
    """
    new_mask = [i for i, _id in enumerate(ids) if _id not in exist_ids]
    return [ids[i] for i in new_mask], [texts[i] for i in new_mask], [metadatas[i] for i in new_mask]


def flatten_metadatas(uniq: Dict[str, Tuple[str, Dict]], ids: List[str]) -> List[Dict]:
    """将规范化后的 meta 进一步按白名单规则扁平化，生成写库用 metadatas 列表。"""
    return [build_metadata_from_meta(uniq[i][1]) for i in ids]


