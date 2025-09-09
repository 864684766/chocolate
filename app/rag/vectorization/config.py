from dataclasses import dataclass
from typing import Dict, Any, Optional
from app.config import get_config_manager


@dataclass
class VectorizationConfig:
    """向量化配置

    - 模型与批处理参数
    - 数据库连接（从 vectorization.database 读取）
    """

    # 模型/批处理
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    device: str = "auto"
    batch_size: int = 32
    max_sequence_length: int = 512
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff_factor: float = 2.0

    # 集合/质量
    collection_name: str = "documents"
    collection_metadata: Optional[Dict[str, Any]] = None
    min_text_length: int = 10
    max_text_length: int = 10000
    skip_empty_text: bool = True
    max_workers: int = 4

    # 数据库
    database: Dict[str, Any] = None

    @classmethod
    def from_config_manager(cls) -> "VectorizationConfig":
        cfg = get_config_manager().get_config().get("vectorization", {})
        return cls(
            model_name=cfg.get("model_name", cls.model_name),
            device=cfg.get("device", cls.device),
            batch_size=cfg.get("batch_size", cls.batch_size),
            max_sequence_length=cfg.get("max_sequence_length", cls.max_sequence_length),
            max_retries=cfg.get("max_retries", cls.max_retries),
            retry_delay=cfg.get("retry_delay", cls.retry_delay),
            retry_backoff_factor=cfg.get("retry_backoff_factor", cls.retry_backoff_factor),
            collection_name=cfg.get("collection_name", cls.collection_name),
            collection_metadata=cfg.get("collection_metadata"),
            min_text_length=cfg.get("min_text_length", cls.min_text_length),
            max_text_length=cfg.get("max_text_length", cls.max_text_length),
            skip_empty_text=cfg.get("skip_empty_text", cls.skip_empty_text),
            max_workers=cfg.get("max_workers", cls.max_workers),
            database=cfg.get("database", {}),
        )


