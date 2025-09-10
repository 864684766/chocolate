from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, Form
from typing import List, Optional

from app.rag.service.ingestion_helpers import (
    classify_or_400,
    build_upload_items,
    run_manual_upload,
    build_response,
    make_raw_sample_objects,
    process_and_vectorize,
)
from .schemas import BaseResponse, ResponseCode, ResponseMessage
from datetime import datetime


router = APIRouter()


@router.post("/upload", summary="手动上传文件，进入RAG喂养流水线（原始样本阶段）")
async def upload_files(
    files: List[UploadFile] = File(..., description="一个或多个文件"),
    dataset: Optional[str] = Form(None),
):
    """接受文件并返回标准化原始样本的统计信息。

    说明：这里只完成标准化封装，真正的解析/分块/向量化应由后续
    processing 与 vectorization 层异步完成（后续可接 Celery 任务）。
    """
    accepted, rejected = classify_or_400(files)
    items = await build_upload_items(accepted, dataset)
    raw_list = run_manual_upload(items)
    samples = make_raw_sample_objects(raw_list)
    result = process_and_vectorize(samples)
    # 仅返回与本接口强相关的字段，遵循 REST 语义
    payload = {
        "received": len(files),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "dataset": dataset,
        "chunks": result.get("chunks", 0),
        "embedded": result.get("embedded", 0),
    }
    return BaseResponse(
        code=ResponseCode.OK,
        message=ResponseMessage.SUCCESS,
        data=payload,
        request_id="",
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


