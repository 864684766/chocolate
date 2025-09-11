from __future__ import annotations

from typing import List, Tuple, Dict
from fastapi import UploadFile
from app.config import get_config_manager


def _get_supported_extensions() -> List[str]:
    cfg = get_config_manager().get_config()
    return list((cfg.get("ingestion", {}) or {}).get("supported_extensions", []))


def classify_files(files: List[UploadFile]) -> Tuple[List[UploadFile], List[Dict[str, str]]]:
    """将文件按是否支持的后缀分类。

    返回: (accepted, rejected)
    rejected 条目包含 filename、reason。
    """
    accepted: List[UploadFile] = []
    rejected: List[Dict[str, str]] = []
    seen: set = set()  # 依据 (文件名, content_type) 做幂等去重，避免前端重复上传
    supported = set(_get_supported_extensions())
    for f in files:
        name = f.filename or ""
        key = (name.lower(), str(getattr(f, "content_type", "")).lower())
        if key in seen:
            # 跳过重复条目
            continue
        ext = ""
        if "." in name:
            ext = name[name.rfind("."):].lower()
        if ext in supported:
            accepted.append(f)
            seen.add(key)
        else:
            rejected.append({
                "filename": name,
                "reason": f"unsupported extension: '{ext or 'N/A'}'",
            })
    return accepted, rejected


