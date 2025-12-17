from __future__ import annotations

"""
Meilisearch 数据库助手：集中管理客户端与索引获取，统一配置读取。
"""

from typing import Iterable, List, Optional

from app.config import get_config_manager
from app.infra.logging import get_logger


class MeilisearchDBHelper:
    """Meilisearch 助手，负责客户端与索引的获取与创建。"""

    def __init__(self, config_manager=None) -> None:
        """初始化助手，读取 databases.meilisearch 配置。"""
        self.config_manager = config_manager or get_config_manager()
        cfg = self.config_manager.get_meilisearch_database_config() or {}
        self.host: str = str(cfg.get("host", "")).strip()  # Meilisearch 服务地址
        self.api_key: Optional[str] = str(cfg.get("api_key", "")) or None  # API 密钥
        self.default_index: str = str(cfg.get("index", "documents"))  # 默认索引名
        self._client = None  # Meilisearch 客户端实例
        self.logger = get_logger(__name__)  # 日志记录器

    def has_config(self) -> bool:
        """判断是否配置了 Meilisearch。"""
        return bool(self.host)

    def get_client(self):
        """获取或创建 Meilisearch 客户端。"""
        if self._client is not None:
            return self._client
        if not self.host:
            raise RuntimeError("Meilisearch 未配置 host")
        try:
            import meilisearch
        except ImportError as exc:
            raise ImportError("meilisearch 模块未安装，请运行: poetry add meilisearch") from exc
        self._client = meilisearch.Client(url=self.host, api_key=self.api_key)
        return self._client

    def get_index(self, index_name: Optional[str] = None):
        """获取索引对象，不存在时仅返回句柄。"""
        name = index_name or self.default_index
        client = self.get_client()
        return client.index(name)

    def ensure_index(
        self,
        index_name: Optional[str] = None,
        searchable: Optional[Iterable[str]] = None,
        filterable: Optional[Iterable[str]] = None,
        primary_key: str = "id",
    ) -> None:
        """确保索引存在并配置搜索/过滤字段。"""
        name = index_name or self.default_index
        client = self.get_client()
        index = client.index(name)
        try:
            index.get_stats()
        except Exception:
            client.create_index(name, {"primaryKey": primary_key})
        if searchable:
            index.update_searchable_attributes(list(searchable))
        if filterable:
            index.update_filterable_attributes(list(filterable))


