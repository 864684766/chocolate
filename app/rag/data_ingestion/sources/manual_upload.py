"""手动上传数据源

职责：接收来自 API 的已上传文件（路径/字节流/文本），
标准化为原始样本（raw sample），供后续 processing 管道消费。

说明：本模块不直接暴露 HTTP 路由，所有对外 API 由 app/api 下实现。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class UploadItem:
    """单个上传条目。"""
    filename: str
    content_type: str
    bytes_data: bytes
    metadata: Optional[Dict[str, Any]] = None


class ManualUploadSource:
    """手动上传数据源的最小实现。

    只做两件事：
    1) 校验并解析上传的文件
    2) 返回标准化原始样本 raw samples：[{"text"/"bytes", "meta": {...}}]
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def process_items(items: List[UploadItem]) -> List[Dict[str, Any]]:
        raw_samples: List[Dict[str, Any]] = []
        for item in items:
            meta = {
                "source": "manual_upload",
                "filename": item.filename,
                "content_type": item.content_type,
            }
            if item.metadata:
                meta.update(item.metadata)

            # 暂不解析文件内容为文本，交由 processing/media/* 处理
            raw = {
                "bytes": item.bytes_data,
                "meta": meta,
            }
            raw_samples.append(raw)

        return raw_samples


