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
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    device: str = "auto"
    batch_size: int = 32

    # 数据库
    database: Dict[str, Any] = None

    @classmethod
    def from_config_manager(cls) -> "VectorizationConfig":
        cfg = get_config_manager().get_config().get("vectorization", {})
        instance = cls(
            model_name=cfg.get("model_name", cls.model_name),
            device=cfg.get("device", cls.device),
            batch_size=cfg.get("batch_size", cls.batch_size),
            database=cfg.get("database", {}),
        )

        # 仅从 database.collection_name 读取集合名（不再回落旧路径）
        db = instance.database or {}
        collection_name = db.get("collection_name")
        if not collection_name:
            raise ValueError("vectorization.database.collection_name 未配置")
        # 动态设置为实例属性，保持下游 self.config.collection_name 可用
        setattr(instance, "collection_name", collection_name)
        return instance


