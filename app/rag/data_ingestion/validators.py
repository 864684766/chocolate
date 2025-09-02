from __future__ import annotations

from typing import List, Tuple, Dict
from fastapi import UploadFile


# 支持的文件后缀（小写）
SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".pdf", ".docx",
    ".png", ".jpg", ".jpeg", ".webp",
    ".mp3", ".wav",
    ".mp4", ".mov", ".avi", ".mkv",
}


def classify_files(files: List[UploadFile]) -> Tuple[List[UploadFile], List[Dict[str, str]]]:
    """将文件按是否支持的后缀分类。

    返回: (accepted, rejected)
    rejected 条目包含 filename、reason。
    """
    accepted: List[UploadFile] = []
    rejected: List[Dict[str, str]] = []
    for f in files:
        name = f.filename or ""
        ext = ""
        if "." in name:
            ext = name[name.rfind("."):].lower()
        if ext in SUPPORTED_EXTENSIONS:
            accepted.append(f)
        else:
            rejected.append({
                "filename": name,
                "reason": f"unsupported extension: '{ext or 'N/A'}'",
            })
    return accepted, rejected


