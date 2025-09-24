from dataclasses import dataclass
from typing import Dict, Any
from app.config import get_config_manager


@dataclass
class VectorizationConfig:
    """向量化配置

    - 模型与批处理参数
    - 数据库连接（从 vectorization.database 读取）
    """

    # 模型/批处理
    model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    device: str = "auto"
    batch_size: int = 32

    # 数据库
    database: Dict[str, Any] = None
    collection_name: str = None

    @classmethod
    def from_config_manager(cls) -> "VectorizationConfig":
        cfg = get_config_manager().get_config().get("vectorization", {})
        
        # 从 database.collection_name 读取集合名
        database = cfg.get("database", {})
        collection_name = database.get("collection_name")
        if not collection_name:
            raise ValueError("vectorization.database.collection_name 未配置")
        
        instance = cls(
            model=cfg.get("model", cls.model),
            device=cfg.get("device", cls.device),
            batch_size=cfg.get("batch_size", cls.batch_size),
            database=database,
            collection_name=collection_name,
        )
        
        return instance


