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
from hashlib import sha1
import re
from app.config import get_config_manager
from app.infra.logging import get_logger

# 尝试导入 Meilisearch 异常类型（如果可用）
try:
    from meilisearch.errors import MeilisearchApiError
    MEILISEARCH_ERROR = MeilisearchApiError
except ImportError:
    # 如果导入失败，使用通用异常
    MEILISEARCH_ERROR = Exception


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
        self._client = None
        self._index_name = None
        self._host = ""
        self._whitelist: Set[str] = set()
        self._types_map: Dict[str, str] = {}
        self._load_config()

    def _load_config(self) -> None:
        """从配置读取 Meilisearch 连接信息和元数据白名单。

        Returns:
            None
        """
        cfg = get_config_manager()
        
        # 读取 Meilisearch 配置
        meili_cfg = (cfg.get_config("retrieval") or {}).get("meilisearch", {}) or {}
        self._host = str(meili_cfg.get("host", "")).strip()
        self._api_key = str(meili_cfg.get("api_key", "")) or None
        self._index_name = str(meili_cfg.get("index", "documents"))
        self._sync_on_index = bool(meili_cfg.get("sync_on_index", True))

        # 如果 host 为空，则不启用 Meilisearch
        if not self._host:
            return

        # 读取元数据白名单
        meta_cfg = cfg.get_config("metadata") or {}
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
        return bool(self._host)

    def _get_client(self):
        """获取或创建 Meilisearch 客户端。

        Returns:
            meilisearch.Client: Meilisearch 客户端实例；如果未启用返回 None

        Raises:
            ImportError: meilisearch 模块未安装
            RuntimeError: 客户端初始化失败
        """
        if self._client is not None:
            return self._client

        if not self._host:
            return None

        try:
            import meilisearch
        except ImportError:
            raise ImportError(
                "meilisearch 模块未安装，请运行: poetry add meilisearch"
            )

        try:
            self._client = meilisearch.Client(
                url=self._host,
                api_key=self._api_key,
            )
            return self._client
        except Exception as e:
            raise RuntimeError(f"初始化 Meilisearch 客户端失败: {e}") from e

    def ensure_index(self) -> None:
        """确保索引存在并正确配置。

        如果索引不存在则创建，如果存在则更新配置。

        Returns:
            None

        Raises:
            RuntimeError: 索引创建或配置失败
        """
        if not self._host:
            return

        try:
            client = self._get_client()
            if not client:
                return

            # 检查索引是否存在
            index_exists = False
            try:
                # 尝试获取索引统计信息，如果不存在会抛出异常
                index = client.index(self._index_name)
                index.get_stats()
                index_exists = True
                self.logger.info(f"Meilisearch 索引 '{self._index_name}' 已存在")
            except MEILISEARCH_ERROR as check_error:
                # 捕获 Meilisearch API 错误（索引不存在通常是 404）
                error_str = str(check_error).lower()
                if "not found" in error_str or "404" in error_str:
                    index_exists = False
                else:
                    # 其他 API 错误，记录日志但继续尝试创建
                    self.logger.warning(f"检查索引时出错: {check_error}，将尝试创建新索引")
                    index_exists = False
            except (ConnectionError, RuntimeError) as check_error:
                # 捕获连接错误或运行时错误
                self.logger.warning(f"检查索引时发生连接或运行时错误: {check_error}，将尝试创建新索引")
                index_exists = False
            
            if not index_exists:
                # 索引不存在，创建新索引
                try:
                    client.create_index(self._index_name, {"primaryKey": "id"})
                    self.logger.info(f"已创建 Meilisearch 索引 '{self._index_name}'")
                except MEILISEARCH_ERROR as create_error:
                    # 创建失败，可能是索引已存在（并发情况）
                    error_str = str(create_error).lower()
                    if "already exists" in error_str or "409" in error_str:
                        self.logger.info(f"Meilisearch 索引 '{self._index_name}' 已存在（并发创建）")
                    else:
                        raise RuntimeError(f"创建 Meilisearch 索引失败: {create_error}") from create_error
                except (ConnectionError, RuntimeError) as create_error:
                    # 连接错误或运行时错误
                    raise RuntimeError(f"创建 Meilisearch 索引失败: {create_error}") from create_error

            # 配置搜索字段和过滤字段
            index = client.index(self._index_name)
            
            # 设置搜索字段：text 为主要搜索字段
            searchable_attributes = ["text"]
            index.update_searchable_attributes(searchable_attributes)
            
            # 设置过滤字段：所有元数据字段都可以用于过滤
            filterable_attributes = list(self._whitelist)
            if filterable_attributes:
                index.update_filterable_attributes(filterable_attributes)
            
            self.logger.info(
                f"Meilisearch 索引配置完成: 搜索字段={searchable_attributes}, "
                f"过滤字段={filterable_attributes[:5]}..." if len(filterable_attributes) > 5 else f"过滤字段={filterable_attributes}"
            )

        except (RuntimeError, ValueError, ConnectionError) as e:
            # 捕获具体的异常类型，避免过于宽泛的 Exception
            self.logger.error(f"配置 Meilisearch 索引失败: {e}", exc_info=True)
            raise RuntimeError(f"配置 Meilisearch 索引失败: {e}") from e
        except Exception as e:
            # 捕获其他未预期的异常
            self.logger.error(f"配置 Meilisearch 索引时发生未预期错误: {type(e).__name__}: {e}", exc_info=True)
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
        if not self._host or not self._sync_on_index:
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

    def _sanitize_doc_id(self, doc_id: str) -> str:
        """清理文档 ID，使其符合 Meilisearch 要求。

        Meilisearch 要求文档 ID：
        - 只能是整数或字符串
        - 只能包含字母数字字符（a-z A-Z 0-9）、连字符（-）和下划线（_）
        - 不能超过 511 字节

        Args:
            doc_id: 原始文档 ID（可能包含冒号等不允许的字符）

        Returns:
            str: 清理后的文档 ID
        """
        if not doc_id:
            return doc_id

        # 将冒号替换为下划线（保持可读性）
        sanitized = doc_id.replace(":", "_")
        
        # 移除其他不允许的字符，只保留字母数字、连字符和下划线
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", sanitized)
        
        # 确保不超过 511 字节
        if len(sanitized.encode("utf-8")) > 511:
            # 如果超过，截断并添加哈希后缀以确保唯一性
            max_len = 400  # 预留空间给哈希
            hash_suffix = sha1(doc_id.encode("utf-8")).hexdigest()[:16]
            sanitized = sanitized[:max_len] + "_" + hash_suffix
        
        return sanitized

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
            - 文档 ID 会被清理以符合 Meilisearch 要求（冒号等字符会被替换）
        """
        if not doc_id or not text:
            return None

        # 清理文档 ID 以符合 Meilisearch 要求
        sanitized_id = self._sanitize_doc_id(doc_id)

        doc: Dict[str, Any] = {
            "id": sanitized_id,
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

