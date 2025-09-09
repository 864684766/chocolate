from __future__ import annotations

from typing import List, Optional, Dict, Any, Tuple
from fastapi import UploadFile, HTTPException
import os

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


# ---- media type helpers ----
_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv", ".webm"}
_AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
_PDF_EXT = {".pdf"}
_TEXT_EXT = {".txt", ".md", ".rst", ".csv", ".tsv", ".json"}


def detect_media_type(filename: str, content_type: Optional[str]) -> str:
    """根据 content_type 与文件后缀推断媒体类型。
    结果仅用于路由处理：image | video | audio | pdf | text
    """
    if content_type:
        ct = content_type.lower()
        if ct.startswith("image/"):
            return "image"
        if ct.startswith("video/"):
            return "video"
        if ct.startswith("audio/"):
            return "audio"
        if ct == "application/pdf":
            return "pdf"
        if ct.startswith("text/"):
            return "text"

    _, ext = os.path.splitext(filename.lower())
    if ext in _IMAGE_EXT:
        return "image"
    if ext in _VIDEO_EXT:
        return "video"
    if ext in _AUDIO_EXT:
        return "audio"
    if ext in _PDF_EXT:
        return "pdf"
    # 默认按文本处理
    return "text"


async def build_upload_items(accepted: List[UploadFile], dataset: Optional[str]) -> List[UploadItem]:
    """
    构建上传条目列表。
    返回: List[UploadItem]
    """
    items: List[UploadItem] = []
    for f in accepted:
        content = await f.read()
        media_type = detect_media_type(f.filename, f.content_type)
        meta: Dict[str, Any] = {"media_type": media_type}
        if dataset:
            meta["dataset"] = dataset
        items.append(UploadItem(
            filename=f.filename,
            content_type=f.content_type or "application/octet-stream",
            bytes_data=content,
            metadata=meta,
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



def process_and_vectorize(samples: List[RawSample]) -> Dict[str, int]:
    """处理并向量化入口。

    用处：在文件上传后串联 Processing 与 Vectorization 层；
    将 `ProcessedChunk.text` 编码为向量并写入向量库。

    Args:
        samples (List[RawSample]): 标准化后的原始样本列表。

    Returns:
        Dict[str, int]: {"chunks": 分块数量, "embedded": 已向量化数量}。
    """
    from app.rag.vectorization import VectorIndexer, VectorizationConfig

    pipeline = ProcessingPipeline()
    chunks = pipeline.run(samples)
    if not chunks:
        return {"chunks": 0, "embedded": 0}

    cfg = VectorizationConfig.from_config_manager()
    indexer = VectorIndexer(cfg)
    embedded = indexer.index_chunks(chunks)
    return {"chunks": len(chunks), "embedded": embedded}