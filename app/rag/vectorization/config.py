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

    # 集合
    collection_name: str = "documents"

    # 数据库
    database: Dict[str, Any] = None

    @classmethod
    def from_config_manager(cls) -> "VectorizationConfig":
        cfg = get_config_manager().get_config().get("vectorization", {})
        return cls(
            model_name=cfg.get("model_name", cls.model_name),
            device=cfg.get("device", cls.device),
            batch_size=cfg.get("batch_size", cls.batch_size),
            collection_name=cfg.get("collection_name", cls.collection_name),
            database=cfg.get("database", {}),
        )


