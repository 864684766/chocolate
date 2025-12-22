"""Meilisearch 索引器：处理 Meilisearch 索引的创建、配置和文档写入。

职责：
- 管理 Meilisearch 索引（创建、配置搜索字段和过滤字段）
- 将 ProcessedChunk 数据转换为 Meilisearch 文档格式
- 批量写入 Meilisearch（支持 upsert）
- 错误处理和日志记录

说明：
- 与 VectorIndexer 配合使用，在数据入库时同步写入 Meilisearch
- 数据格式：展开 metadata 到顶层，数组字段保持数组格式
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Set
from app.config import get_config_manager
from app.infra.logging import get_logger
from app.infra.database.meilisearch.db_helper import MeilisearchDBHelper


class MeilisearchIndexer:
    """Meilisearch 索引器：管理索引并写入文档。

    用途：
    - 在数据入库时同步写入 Meilisearch
    - 自动创建和配置索引
    - 处理数据格式转换（ChromaDB → Meilisearch）

    配置：
    - 从 config/app_config.json > retrieval.meilisearch 读取连接信息
    - 从 metadata.metadata_whitelist 读取字段白名单
    """

    def __init__(self):
        """初始化 MeilisearchIndexer。

        读取配置并初始化客户端。
        """
        self.logger = get_logger(__name__)
        self._db = MeilisearchDBHelper()
        self._config_manager = get_config_manager()
        self._index_name = self._db.default_index
        self._whitelist: Set[str] = set()
        self._types_map: Dict[str, str] = {}
        self._load_config()

    def _load_config(self) -> None:
        """从配置读取 Meilisearch 连接信息和元数据白名单。

        Returns:
            None
        """
        meili_cfg = self._config_manager.get_meilisearch_database_config() or {}
        self._index_name = str(meili_cfg.get("index", self._db.default_index))
        self._sync_on_index = bool(meili_cfg.get("sync_on_index", True))
        # 如果未配置 Meilisearch，提前返回
        if not self._db.has_config():
            return
        # 读取元数据白名单
        meta_cfg = self._config_manager.get_config("metadata") or {}
        wl_raw = meta_cfg.get("metadata_whitelist", [])
        if wl_raw and isinstance(wl_raw[0], dict):
            self._whitelist = {str(x.get("field")) for x in wl_raw if x.get("field")}
            self._types_map = {str(x.get("field")): str(x.get("type")) for x in wl_raw if x.get("field")}
        else:
            self._whitelist = set(str(x) for x in (wl_raw or []))

    def is_enabled(self) -> bool:
        """检查 Meilisearch 是否启用。

        判断标准：host 配置存在且不为空。

        Returns:
            bool: 如果启用返回 True，否则返回 False
        """
        return self._db.has_config()

    def _get_client(self):
        """获取或创建 Meilisearch 客户端。

        Returns:
            meilisearch.Client: Meilisearch 客户端实例；如果未启用返回 None

        Raises:
            ImportError: meilisearch 模块未安装
            RuntimeError: 客户端初始化失败
        """
        return self._db.get_client() if self._db.has_config() else None

    def ensure_index(self) -> None:
        """确保索引存在并正确配置。

        如果索引不存在则创建，如果存在则更新配置。

        Returns:
            None

        Raises:
            RuntimeError: 索引创建或配置失败
        """
        if not self._db.has_config():
            return
        try:
            searchable = ["text"]
            filterable = list(self._whitelist)
            self._db.ensure_index(
                index_name=self._index_name,
                searchable=searchable,
                filterable=filterable,
                primary_key="id",
            )
            self.logger.info(
                f"Meilisearch 索引配置完成: 搜索字段={searchable}, 过滤字段={filterable[:5]}..." 
                if len(filterable) > 5 else f"Meilisearch 索引配置完成: 搜索字段={searchable}, 过滤字段={filterable}"
            )
        except Exception as e:
            self.logger.error(f"配置 Meilisearch 索引失败: {e}", exc_info=True)
            raise RuntimeError(f"配置 Meilisearch 索引失败: {e}") from e

    def index_documents(
        self,
        ids: List[str],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> int:
        """批量写入文档到 Meilisearch。

        Args:
            ids: 文档 ID 列表
            texts: 文档文本列表
            metadatas: 元数据列表（ChromaDB 格式，扁平化）

        Returns:
            int: 成功写入的文档数量

        Notes:
            - 如果 Meilisearch host 未配置或同步开关关闭，直接返回 0
            - 错误不会抛出异常，只记录日志
        """
        if not self._db.has_config() or not self._sync_on_index:
            return 0

        if not ids or not texts:
            return 0

        try:
            # 确保索引存在
            self.ensure_index()

            client = self._get_client()
            if not client:
                return 0

            index = client.index(self._index_name)

            # 转换为 Meilisearch 文档格式
            documents = []
            for _id, text, meta in zip(ids, texts, metadatas):
                doc = self._convert_to_meilisearch_doc(_id, text, meta)
                if doc:
                    documents.append(doc)
                else:
                    self.logger.warning(f"跳过无效文档: id={_id}, text_len={len(text) if text else 0}")

            if not documents:
                self.logger.warning("没有有效的 Meilisearch 文档可写入")
                return 0

            self.logger.debug(f"准备写入 {len(documents)} 条文档到 Meilisearch 索引 '{self._index_name}'")

            # 批量写入（支持 upsert）
            result = index.add_documents(documents, primary_key="id")
            
            # 等待索引完成（确保数据立即可见）
            try:
                task = index.wait_for_task(result.task_uid, timeout_in_ms=30000)
                if task.status == "failed":
                    error_msg = task.error.get("message", "未知错误") if task.error else "索引任务失败"
                    self.logger.error(f"Meilisearch 索引任务失败: {error_msg}")
                    return 0
                self.logger.info(f"已同步 {len(documents)} 条文档到 Meilisearch (task_uid={result.task_uid})")
            except Exception as e:
                # 等待超时或其他错误，但文档可能已经提交成功
                self.logger.warning(f"等待 Meilisearch 索引任务完成时出错: {e}，但文档可能已提交")
            
            return len(documents)

        except Exception as e:
            # 错误隔离：不影响主流程
            self.logger.error(f"同步到 Meilisearch 失败: {e}", exc_info=True)
            return 0

    def _convert_to_meilisearch_doc(
        self, doc_id: str, text: str, metadata: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """将 ChromaDB 格式的文档转换为 Meilisearch 格式。

        Args:
            doc_id: 文档 ID（ChromaDB 格式，可能包含冒号）
            text: 文档文本
            metadata: 元数据字典（ChromaDB 格式，扁平化）

        Returns:
            Optional[Dict[str, Any]]: Meilisearch 文档格式；转换失败返回 None

        Notes:
            - Meilisearch 格式：展开 metadata 到顶层
            - 数组字段：ChromaDB 存储为逗号分隔字符串，需要转换回数组
            - 文档 ID 已经是纯 hash 格式（16 位十六进制），符合 Meilisearch 要求
        """
        if not doc_id or not text:
            return None

        # ID 已经是纯 hash 格式（16 位十六进制），完全符合 Meilisearch 要求
        doc: Dict[str, Any] = {
            "id": doc_id,
            "text": text,
        }

        # 展开 metadata 到顶层
        for key, value in metadata.items():
            if key == "id":  # 避免覆盖主键
                continue

            # 处理数组字段：ChromaDB 存储为逗号分隔字符串，需要转换回数组
            field_type = self._types_map.get(key)
            if field_type == "array":
                if isinstance(value, str):
                    # 从逗号分隔字符串转换回数组
                    items = [item.strip() for item in value.split(",") if item.strip()]
                    doc[key] = items if items else []
                elif isinstance(value, (list, tuple)):
                    doc[key] = list(value)
                else:
                    # 其他类型转为字符串数组
                    doc[key] = [str(value)] if value else []
            else:
                # 其他类型直接使用
                doc[key] = value

        return doc

    def update_documents(
        self,
        ids: List[str],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> int:
        """批量更新文档到 Meilisearch。

        Args:
            ids: 文档 ID 列表
            texts: 文档文本列表
            metadatas: 元数据列表

        Returns:
            int: 成功更新的文档数量

        Notes:
            - Meilisearch 的 add_documents 支持 upsert，所以更新和新增使用相同方法
        """
        return self.index_documents(ids, texts, metadatas)

