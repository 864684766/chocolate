from __future__ import annotations

from typing import List, Optional, Dict, Any, Tuple
from fastapi import UploadFile, HTTPException

from app.rag.data_ingestion.validators import classify_files, SUPPORTED_EXTENSIONS
from app.rag.data_ingestion.sources.manual_upload import ManualUploadSource, UploadItem
from app.rag.processing.interfaces import RawSample
from app.rag.processing.pipeline import ProcessingPipeline


def classify_or_400(files: List[UploadFile]) -> Tuple[List[UploadFile], List[Dict[str, str]]]:
    """
    将文件按是否支持的后缀分类。
    返回: (accepted, rejected)
    """
    accepted, rejected = classify_files(files)
    if accepted:
        return accepted, rejected
    raise HTTPException(
        status_code=400,
        detail={
            "error": "no supported files",
            "supported_extensions": sorted(list(SUPPORTED_EXTENSIONS)),
            "rejected": rejected,
        },
    )


async def build_upload_items(accepted: List[UploadFile], dataset: Optional[str]) -> List[UploadItem]:
    """
    构建上传条目列表。
    返回: List[UploadItem]
    """
    items: List[UploadItem] = []
    for f in accepted:
        content = await f.read()
        items.append(UploadItem(
            filename=f.filename,
            content_type=f.content_type or "application/octet-stream",
            bytes_data=content,
            metadata={"dataset": dataset} if dataset else None,
        ))
    return items


def run_manual_upload(items: List[UploadItem]) -> List[Dict[str, Any]]:
    """
    运行手动上传。
    返回: List[Dict[str, Any]]
    """
    source = ManualUploadSource()
    return source.process_items(items)


def build_response(files_cnt: int,
                   accepted: List[UploadFile],
                   rejected: List[Dict[str, str]],
                   raw_samples: List[Dict[str, Any]],
                   dataset: Optional[str],
                   chunks_cnt: Optional[int] = None) -> Dict[str, Any]:
    """
    构建响应。
    返回: Dict[str, Any]
    """
    resp = {
        "received": files_cnt,
        "accepted": len(accepted),
        "rejected": rejected,
        "raw_samples": len(raw_samples),
        "dataset": dataset,
        "supported_extensions": sorted(list(SUPPORTED_EXTENSIONS)),
    }
    if chunks_cnt is not None:
        resp["chunks"] = chunks_cnt
    return resp


def make_raw_sample_objects(raw_list: List[Dict[str, Any]]) -> List[RawSample]:
    return [RawSample(bytes=raw["bytes"], meta=raw["meta"]) for raw in raw_list]


def run_processing(samples: List[RawSample]) -> int:
    pipeline = ProcessingPipeline()
    chunks = pipeline.run(samples)
    return len(chunks)


