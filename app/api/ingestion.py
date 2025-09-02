from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List, Optional

from app.data_ingestion.sources.manual_upload import ManualUploadSource, UploadItem


router = APIRouter()


from app.data_ingestion.validators import classify_files, SUPPORTED_EXTENSIONS


@router.post("/upload", summary="手动上传文件，进入RAG喂养流水线（原始样本阶段）")
async def upload_files(
    files: List[UploadFile] = File(..., description="一个或多个文件"),
    dataset: Optional[str] = Form(None),
):
    """接受文件并返回标准化原始样本的统计信息。

    说明：这里只完成标准化封装，真正的解析/分块/向量化应由后续
    processing 与 vectorization 层异步完成（后续可接 Celery 任务）。
    """
    accepted, rejected = classify_files(files)
    if not accepted:
        # 全部不支持，直接 400
        raise HTTPException(
            status_code=400,
            detail={
                "error": "no supported files",
                "supported_extensions": sorted(list(SUPPORTED_EXTENSIONS)),
                "rejected": rejected,
            },
        )

    source = ManualUploadSource()
    items: List[UploadItem] = []

    for f in accepted:
        content = await f.read()
        items.append(
            UploadItem(
                filename=f.filename,
                content_type=f.content_type or "application/octet-stream",
                bytes_data=content,
                metadata={"dataset": dataset} if dataset else None,
            )
        )

    raw_samples = source.process_items(items)

    return {
        "received": len(files),
        "accepted": len(accepted),
        "rejected": rejected,
        "raw_samples": len(raw_samples),
        "dataset": dataset,
        "supported_extensions": sorted(list(SUPPORTED_EXTENSIONS)),
    }


