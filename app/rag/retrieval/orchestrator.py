from __future__ import annotations

"""
检索编排服务：向量召回 → 重排 → 预览拼接 →（可选）LLM 生成。

单一职责：
- 对外提供 run(query, options) 一个入口；内部协调依赖。
"""

from typing import Any, Dict, List, Optional

from .vector_retriever import VectorRetriever
from .meilisearch_retriever import MeilisearchRetriever
from .hybrid import HybridSearcher
from .reranker import CrossEncoderReranker
from .schemas import RetrievalQuery, RetrievedItem, RetrievalResult
from .graph_retriever import GraphRetriever
from ...config import get_config_manager
from ...llm_adapters.factory import LLMProviderFactory
from ...infra.logging import get_logger


class RetrievalOrchestrator:
    """
    编排器：封装检索、重排与默认生成流程。

    用途：
    - 对外暴露 run(query, options) 单一入口，内部完成“向量召回→重排→预览→生成”。

    注意：
    - 生成阶段所用的 provider 与 model 来自请求或默认设置，并读取 providers.<provider>.models.<model> 的参数。
    """

    def __init__(self) -> None:
        """
        初始化检索编排器，读取配置并初始化依赖检索器。

        用处：
        - 装配向量检索、Meilisearch 检索、可选图扩展与重排。
        - 读取检索、重排、图扩展的配置开关和参数。
        """
        self.cfg = get_config_manager()
        self.logger = get_logger(__name__)
        
        # 检查 Meilisearch 是否配置（通过 host 是否存在判断）
        # 如果 Meilisearch 已配置，则启用混合检索
        meili_cfg = (self.cfg.get_config("retrieval") or {}).get("meilisearch", {}) or {}
        meili_host = str(meili_cfg.get("host", "")).strip()
        self._meili_enabled = bool(meili_host)
        
        # 总是尝试初始化图检索（失败时降级处理）
        graph_cfg = (self.cfg.get_config("retrieval") or {}).get("neo4j", {}) or {}
        self._graph_hops = int(graph_cfg.get("max_hops", 1))
        self._graph_limit = int(graph_cfg.get("max_neighbors", 10))
        self._graph: Optional[GraphRetriever] = None
        try:
            gr = GraphRetriever()
            if gr.is_enabled():
                self._graph = gr
            else:
                self.logger.debug("Neo4j 未配置，图检索将跳过")
        except Exception as e:
            self.logger.warning(f"初始化图检索失败，将跳过图检索: {e}")

    def run(self, query: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        执行编排流程并返回结果。

        参数：
        - query (str): 查询文本。
        - options (dict|None): 可选项：
          - top_k/top_n/score_threshold/max_preview：检索与预览控制
          - provider/ai_type：生成使用的提供商与模型名（或本地路径）
          （已取消 use_llm 开关，默认执行生成）

        返回：
        - dict: {"preview": str, "content": str(optional), "items": List[RetrievedItem]}
        """
        opts = options or {}
        top_k = int(opts.get("top_k", (self._cfg_rerank().get("top_n") or 10)))
        top_n = int(opts.get("top_n", (self._cfg_rerank().get("top_n") or 10)))
        score_th = float(opts.get("score_threshold", 0.0))
        max_preview = int(opts.get("max_preview", 3))

        # 构建严格 where（零放宽，白名单校验）
        from .utils.where_builder import WhereBuilder
        wb = WhereBuilder()
        applied_where = wb.build(query)

        items = self._retrieve(query, top_k, score_th, applied_where)
        # 使用 Neo4j 扩展邻居块（如果启用）
        if self._graph and self._graph.is_enabled():
            items = self._expand_with_graph(items)
        items = self._rerank(query, items, top_n)
        preview = self._build_preview(items, max_preview)

        # 始终执行生成阶段（如需仅预览，可在上层另设接口）
        content = self._generate(query, items, opts)
        return {
            "preview": preview,
            "content": content,
            "items": items,
            "applied_where": applied_where,
            "matched_count": len(items),
        }

    def _cfg_rerank(self) -> Dict[str, Any]:
        """
        读取重排配置块。

        返回：
        - Dict[str,Any]: `retrieval.rerank` 配置字典（缺省为空字典）。
        """
        return (self.cfg.get_config("retrieval") or {}).get("rerank", {})

    def _retrieve(self, query: str, top_k: int, score_th: float, where: Optional[Dict[str, Any]]) -> List[RetrievedItem]:
        """
        执行检索（向量检索或混合检索）。

        参数:
        - query (str): 查询文本。
        - top_k (int): 召回候选数量。
        - score_th (float): 最低得分阈值，低于阈值的结果被过滤。
        - where (Optional[Dict[str, Any]]): 元数据过滤条件。

        返回:
        - List[RetrievedItem]: 召回的候选项列表。

        说明:
        - 如果混合检索启用，并行执行向量检索和关键词检索，然后融合结果
        - 如果未启用或失败，降级为仅向量检索
        - Neo4j 图扩展在后续步骤中执行（基于 RRF 结果）
        """
        q = RetrievalQuery(query=query, where=where, top_k=top_k, score_threshold=score_th)
        
        # 执行向量检索
        vector_retriever = VectorRetriever()
        vector_result = vector_retriever.search(q)
        
        # 如果 Meilisearch 未配置，直接返回向量检索结果
        if not self._meili_enabled:
            return vector_result.items
        
        # 执行混合检索：并行调用 MeilisearchRetriever
        try:
            meili_retriever = MeilisearchRetriever()
            meili_result = meili_retriever.search(q)
            
            # 如果 Meilisearch 没有结果，直接返回向量检索结果
            if not meili_result.items:
                return vector_result.items
            
            # 使用 HybridSearcher 融合结果
            hybrid_cfg = (self.cfg.get_config("retrieval") or {}).get("hybrid", {}) or {}
            method = str(hybrid_cfg.get("method", "rrf"))
            
            if method == "rrf":
                # RRF 融合
                rrf_k = int(hybrid_cfg.get("rrf_k", 60))
                fused_result = HybridSearcher.rrf(
                    vector_result, meili_result, k=rrf_k, top_k=top_k
                )
            else:
                # 加权求和融合
                w_vec = float(hybrid_cfg.get("vector_weight", 0.7))
                w_kw = float(hybrid_cfg.get("keyword_weight", 0.3))
                fused_result = HybridSearcher.weighted_sum(
                    vector_result, meili_result,
                    w_vec=w_vec, w_kw=w_kw, top_k=top_k
                )
            
            return fused_result.items
            
        except Exception as e:
            # 错误降级：Meilisearch 失败时返回向量检索结果
            self.logger.warning(f"混合检索失败，降级为仅向量检索: {e}", exc_info=True)
            return vector_result.items

    def _expand_with_graph(self, items: List[RetrievedItem]) -> List[RetrievedItem]:
        """
        使用图数据库扩展邻居块，增强上下文覆盖。

        用处：
        - 基于 RRF 融合后的结果，在 Neo4j 中查找相邻/相关的块
        - 通过 NEXT 关系找到同一文档中的相邻块，提供更完整的上下文

        Args:
            items: RRF 融合后的检索结果列表。

        Returns:
            List[RetrievedItem]: 合并图扩展后的结果（保持去重）。

        说明:
        - 如果图检索未启用或失败，直接返回原始结果
        - 扩展的邻居块会与原始结果合并去重
        """
        if not items or not self._graph:
            return items
        try:
            return self._graph.expand_neighbors(
                items,
                max_hops=self._graph_hops,
                limit=self._graph_limit,
            )
        except Exception as e:
            self.logger.warning(f"图扩展失败，返回原始结果: {e}", exc_info=True)
            return items

    def _rerank(self, query: str, items: List[RetrievedItem], top_n: int) -> List[RetrievedItem]:
        """
        对候选进行交叉编码器重排。

        参数：
        - query (str): 原始查询文本。
        - items (List[RetrievedItem]): 召回候选。
        - top_n (int): 重排后保留的条数。

        返回：
        - List[RetrievedItem]: 重排后的前 N 项；输入为空时返回空列表。
        """
        if not items:
            return []
        model = str(self._cfg_rerank().get("model") or "") or None
        reranker = CrossEncoderReranker(model=model)
        return reranker.rerank(items, top_n=top_n, query=query)


    @staticmethod
    def _build_preview(items: List[RetrievedItem], max_items: int) -> str:
        """
        拼接预览文本。

        参数：
        - items (List[RetrievedItem]): 候选项列表。
        - max_items (int): 参与拼接的最大条数。

        返回：
        - str: 以两个换行分隔的预览文本；无内容时为空串。
        """
        texts = [it.text for it in items[: max(1, int(max_items))]]
        return "\n\n".join(texts) if texts else ""

    def _generate(self, query: str, items: List[RetrievedItem], opts: Dict[str, Any]) -> str:
        """
        基于检索上下文调用大模型生成答案。

        参数：
        - query (str): 原始查询。
        - items (List[RetrievedItem]): 用于构造上下文的候选项。
        - opts (Dict[str,Any]): 生成控制项（provider、ai_type、enable_thinking、max_context）。

        返回：
        - str: 生成的最终内容；异常或空结果时返回空串。

        说明：
        - max_new_tokens 从模型配置中读取（providers.<provider>.models.<model>.max_new_tokens）
        """
        provider = opts.get("provider") or self.cfg.get_settings().provider
        ai_type = opts.get("ai_type") or self.cfg.get_settings().model
        # 读取模型节点默认参数
        model_cfg = self.cfg.get_model_config(provider, ai_type)
        # backend 路由由工厂内部处理，这里直接调用即可
        model = LLMProviderFactory.get_chat_model(ai_type, provider)
        ctx = self._build_preview(items, max_items=opts.get("max_context", 6))
        
        # 处理空上下文情况
        prompts_cfg = (self.cfg.get_prompts_config() or {}).get("retrieval", {})
        if not ctx or not ctx.strip():
            # 如果没有检索到相关内容，直接返回明确的提示（从配置文件读取）
            empty_message = str(prompts_cfg.get(
                "empty_result_message",
                "抱歉，我在知识库中没有找到与您的问题相关的内容。请尝试使用其他关键词或检查问题是否正确。"
            ))
            return empty_message
        system_text = str(prompts_cfg.get(
            "system", 
            "你是一个中文助理，请基于提供的上下文回答问题。如果上下文为空或与问题无关，请明确说明无法回答。"
        ))
        user_tmpl = str(prompts_cfg.get("user_template", "问题：{query}\n\n上下文：\n{context}"))
        user_text = user_tmpl.format(query=query, context=ctx)
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        
        # 从模型配置读取 max_new_tokens，如果模型配置中没有则使用 llm 默认值
        llm_cfg = self.cfg.get_config("llm") or {}
        default_max_tokens = llm_cfg.get("default_max_new_tokens", 1024)
        max_new_tokens = model_cfg.get("max_new_tokens") or default_max_tokens
        gen = model.generate(
            messages,
            max_new_tokens=int(max_new_tokens),
            enable_thinking=bool(opts.get("enable_thinking", model_cfg.get("enable_thinking", False))),
        )
        return gen.get("content") or gen.get("raw_text") or ""


