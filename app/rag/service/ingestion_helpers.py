from __future__ import annotations

from typing import List, Optional, Dict, Any, Tuple
from fastapi import UploadFile, HTTPException
import os

from app.rag.data_ingestion.validators import classify_files
from app.rag.data_ingestion.sources.manual_upload import ManualUploadSource, UploadItem
from app.rag.processing.interfaces import RawSample
from app.rag.processing.pipeline import ProcessingPipeline
from app.config import get_config_manager


def classify_or_400(files: List[UploadFile]) -> Tuple[List[UploadFile], List[Dict[str, str]]]:
    """
    将文件按是否支持的后缀分类。
    返回: (accepted, rejected)
    """
    accepted, rejected = classify_files(files)
    if accepted:
        return accepted, rejected
    # 读取配置中的受支持后缀
    cfg = get_config_manager().get_config()
    supported = sorted(list((cfg.get("ingestion", {}) or {}).get("supported_extensions", [])))
    raise HTTPException(
        status_code=400,
        detail={
            "error": "no supported files",
            "supported_extensions": supported,
            "rejected": rejected,
        },
    )


def _media_map_from_config() -> Dict[str, str]:
    """从配置加载后缀到媒体类型的映射。

    支持两种写法：
    - 简单数组 ingestion.supported_extensions 仅声明后缀（默认按文本处理）
    - 对象数组 ingestion.media_map: [{"ext": ".png", "media_type": "image"}, ...]
    """
    cfg = get_config_manager().get_config()
    ing = (cfg.get("ingestion", {}) or {})
    mapping_list = ing.get("media_map", [])
    mapping: Dict[str, str] = {}
    if isinstance(mapping_list, list) and mapping_list and isinstance(mapping_list[0], dict):
        for it in mapping_list:
            ext = str(it.get("ext", "")).lower()
            mtype = str(it.get("media_type", "text"))
            if ext:
                mapping[ext] = mtype
    # 回退：把 supported_extensions 全部标为 text
    if not mapping:
        exts: List[str] = list(ing.get("supported_extensions", []))
        for e in exts:
            mapping[str(e).lower()] = "text"
    return mapping


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
    mapping = _media_map_from_config()
    return mapping.get(ext, "text")


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
        "supported_extensions": sorted(list((get_config_manager().get_config().get("ingestion", {}) or {}).get("supported_extensions", []))),
    }
    if chunks_cnt is not None:
        resp["chunks"] = chunks_cnt
    return resp


def make_raw_sample_objects(raw_list: List[Dict[str, Any]]) -> List[RawSample]:
    """
    将原始数据字典列表转换为RawSample对象列表。
    
    用处：将从数据源（如手动上传）获取的原始数据字典转换为处理流水线
    所需的标准化RawSample对象，为后续的内容提取和处理做准备。
    
    Args:
        raw_list: 原始数据字典列表，每个字典必须包含以下键：
            - bytes (bytes): 文件的二进制内容
            - meta (Dict[str, Any]): 文件的元数据，如媒体类型、文件名等
            
    Returns:
        List[RawSample]: RawSample对象列表，每个对象包含二进制数据和元数据，
                        可直接用于ProcessingPipeline处理
    """
    return [RawSample(bytes=raw["bytes"], meta=raw["meta"]) for raw in raw_list]

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
    from app.rag.processing.quality_checker import SimpleQualityAssessor
    pipeline = ProcessingPipeline(quality=SimpleQualityAssessor())
    chunks = pipeline.run(samples)
    if not chunks:
        return {"chunks": 0, "embedded": 0}

    cfg = VectorizationConfig.from_config_manager()
    indexer = VectorIndexer(cfg)
    embedded = indexer.index_chunks(chunks)
    return {"chunks": len(chunks), "embedded": embedded}