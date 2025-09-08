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
    return build_response(len(files), accepted, rejected, raw_list, dataset, result.get("chunks"))


