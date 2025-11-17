from __future__ import annotations

"""
Meilisearch 检索器：基于 BM25 算法的关键词检索实现。

职责：
- 与 VectorRetriever.search 保持相同方法签名
- 执行 BM25 关键词检索，支持容错搜索
- 将 ChromaDB where 条件转换为 Meilisearch filter
- 返回格式化的检索结果

说明：
- 需要 Meilisearch 服务运行并已创建索引
- 配置项：retrieval.meilisearch (enabled/host/api_key/index)
"""

from typing import List, Dict, Any
import time

from .schemas import RetrievalQuery, RetrievalResult, RetrievedItem
from app.config import get_config_manager
from app.infra.logging import get_logger
from app.rag.retrieval.utils.query_cleaner import clean_query_basic
from app.rag.retrieval.utils.where_to_meilisearch_filter import convert_where_to_filter
from app.rag.retrieval.utils.meilisearch_filter_builder import MeilisearchFilterBuilder


class MeilisearchRetriever:
    """Meilisearch 关键词检索器：基于 BM25 算法的全文搜索。

    用途：
    - 提供精确关键词匹配和容错搜索能力
    - 与向量检索互补，用于混合检索场景

    配置：
    - 从 config/app_config.json > retrieval.meilisearch 读取连接信息
    - host: Meilisearch 服务地址（如果为空则不启用）
    - api_key: API 密钥（可选）
    - index: 索引名称
    """

    def __init__(self, index_name: str | None = None, use_direct_filter: bool = True):
        """初始化 MeilisearchRetriever。

        Args:
            index_name: 索引名称，如果为 None 则从配置读取
            use_direct_filter: 是否直接从查询生成 filter（True），还是转换 ChromaDB where（False）
                              默认 True，直接从元数据生成，避免转换开销
        """
        self.logger = get_logger(__name__)
        self._client = None
        self._index_name = index_name
        self._use_direct_filter = use_direct_filter
        self._filter_builder = MeilisearchFilterBuilder() if use_direct_filter else None
        self._load_config()

    def _load_config(self) -> None:
        """从配置读取 Meilisearch 连接信息。

        Returns:
            None
        """
        cfg = get_config_manager()
        meili_cfg = (cfg.get_config("retrieval") or {}).get("meilisearch", {}) or {}
        
        self._host = str(meili_cfg.get("host", "")).strip()
        self._api_key = str(meili_cfg.get("api_key", "")) or None
        self._index_name = self._index_name or str(meili_cfg.get("index", "documents"))

    def _get_client(self):
        """获取或创建 Meilisearch 客户端。

        Returns:
            meilisearch.Client: Meilisearch 客户端实例

        Raises:
            ImportError: meilisearch 模块未安装
            RuntimeError: 客户端初始化失败
        """
        if self._client is not None:
            return self._client

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

    def search(self, q: RetrievalQuery) -> RetrievalResult:
        """执行 Meilisearch BM25 关键词检索。

        Args:
            q: 检索请求（含 query/where/top_k/score_threshold）

        Returns:
            RetrievalResult: 命中项、耗时等信息

        Raises:
            RuntimeError: Meilisearch 未配置或检索失败
        """
        if not self._host:
            self.logger.warning("Meilisearch host 未配置，返回空结果")
            return RetrievalResult(
                items=[],
                latency_ms=0,
                matched_count=0,
            )

        t0 = time.time()
        try:
            # 清洗查询文本
            query_text = clean_query_basic(q.query)
            if not query_text:
                return RetrievalResult(
                    items=[],
                    latency_ms=0,
                    matched_count=0,
                )

            # 获取客户端和索引
            client = self._get_client()
            index = client.index(self._index_name)

            # 生成 Meilisearch filter
            # 优先使用直接生成方式（从查询文本直接生成），避免转换开销
            if self._use_direct_filter and self._filter_builder:
                # 直接从查询文本生成 filter（推荐方式）
                filter_list = self._filter_builder.build_from_query(q.query)
                # 如果直接生成失败，回退到转换 where 条件
                if not filter_list and q.where:
                    filter_list = self._filter_builder.build_from_where(q.where)
            else:
                # 兼容模式：转换 ChromaDB where 条件
                filter_list = convert_where_to_filter(q.where)

            # 执行搜索
            # 构建可选参数字典（根据 Meilisearch Python SDK API，使用 opt_params 作为第二个位置参数）
            opt_params: Dict[str, Any] = {
                "limit": q.top_k,
            }
            if filter_list:
                # Meilisearch 的 filter 可以是字符串列表（自动 AND 连接）
                # 或者单个字符串（包含复杂表达式）
                # 如果只有一个 filter，直接使用字符串；多个则使用列表
                if len(filter_list) == 1:
                    opt_params["filter"] = filter_list[0]
                else:
                    opt_params["filter"] = filter_list

            # 查询字符串作为第一个位置参数，可选参数字典作为第二个位置参数
            search_result = index.search(query_text, opt_params)

            # 解析结果
            items: List[RetrievedItem] = []
            hits = search_result.get("hits", [])

            for hit in hits:
                # Meilisearch 返回的分数范围通常是 0-1，但可能超出
                # 需要归一化到 [0, 1] 范围
                raw_score = float(hit.get("_rankingScore", 0.0))
                # Meilisearch 的 _rankingScore 通常在 0-1 之间，但可能超过 1
                score = min(1.0, max(0.0, raw_score))

                # 应用分数阈值过滤
                if score < q.score_threshold:
                    continue

                # 提取文档信息
                doc_id = str(hit.get("id", ""))
                # Meilisearch 的文档内容可能在 "text" 字段或其他字段
                # 优先使用 "text"，否则尝试其他常见字段
                text = hit.get("text") or hit.get("content") or hit.get("document") or ""
                if not text:
                    # 如果没有找到文本字段，尝试从所有字段中提取
                    for key, value in hit.items():
                        if key not in ("id", "_rankingScore", "_formatted") and isinstance(value, str):
                            text = value
                            break

                # 提取元数据（排除 Meilisearch 内部字段）
                metadata: Dict[str, Any] = {}
                for key, value in hit.items():
                    if key not in ("id", "text", "content", "document", "_rankingScore", "_formatted"):
                        metadata[key] = value

                items.append(
                    RetrievedItem(
                        id=doc_id,
                        text=str(text),
                        score=score,
                        metadata=metadata,
                    )
                )

            latency_ms = int((time.time() - t0) * 1000)
            return RetrievalResult(
                items=items,
                latency_ms=latency_ms,
                applied_where=q.where,
                matched_count=len(items),
            )

        except Exception as e:
            self.logger.error(f"Meilisearch 检索失败: {e}", exc_info=True)
            latency_ms = int((time.time() - t0) * 1000)
            return RetrievalResult(
                items=[],
                latency_ms=latency_ms,
                applied_where=q.where,
                matched_count=0,
            )


