from dataclasses import dataclass
from typing import Dict, Any
from app.config import get_config_manager


@dataclass
class VectorizationConfig:
    """向量化配置

    - 模型与批处理参数
    - 数据库连接（读取 databases.chroma）
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
        cfg_manager = get_config_manager()
        vec_cfg = cfg_manager.get_config().get("vectorization", {})
        # 从 databases.chroma 读取集合名
        database = cfg_manager.get_vector_database_config()
        collection_name = database.get("collection_name")
        if not collection_name:
            raise ValueError("vectorization.database.collection_name 未配置")
        
        instance = cls(
            model=vec_cfg.get("model", cls.model),
            device=vec_cfg.get("device", cls.device),
            batch_size=vec_cfg.get("batch_size", cls.batch_size),
            database=database,
            collection_name=collection_name,
        )
        
        return instance


