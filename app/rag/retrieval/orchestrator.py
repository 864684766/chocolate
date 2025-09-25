from __future__ import annotations

"""
检索编排服务：向量召回 → 重排 → 预览拼接 →（可选）LLM 生成。

单一职责：
- 对外提供 run(query, options) 一个入口；内部协调依赖。
"""

from typing import Any, Dict, List, Optional

from .vector_retriever import VectorRetriever
from .reranker import CrossEncoderReranker
from .schemas import RetrievalQuery, RetrievedItem
from ...config import get_config_manager
from ...llm_adapters.factory import LLMProviderFactory


class RetrievalOrchestrator:
    """
    编排器：封装检索、重排与默认生成流程。

    用途：
    - 对外暴露 run(query, options) 单一入口，内部完成“向量召回→重排→预览→生成”。

    注意：
    - 生成阶段所用的 provider 与 model 来自请求或默认设置，并读取 providers.<provider>.models.<model> 的参数。
    """

    def __init__(self) -> None:
        self.cfg = get_config_manager()

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

    @staticmethod
    def _retrieve(query: str, top_k: int, score_th: float, where: Optional[Dict[str, Any]]) -> List[RetrievedItem]:
        """
        执行向量召回。

        参数：
        - query (str): 查询文本。
        - top_k (int): 召回候选数量。
        - score_th (float): 最低得分阈值，低于阈值的结果被过滤。

        返回：
        - List[RetrievedItem]: 召回的候选项列表。
        """
        retriever = VectorRetriever()
        q = RetrievalQuery(query=query, where=where, top_k=top_k, score_threshold=score_th)
        return retriever.search(q).items

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
        - opts (Dict[str,Any]): 生成控制项（provider、ai_type、max_new_tokens、enable_thinking、max_context）。

        返回：
        - str: 生成的最终内容；异常或空结果时返回空串。
        """
        provider = opts.get("provider") or self.cfg.get_settings().provider
        ai_type = opts.get("ai_type") or self.cfg.get_settings().model
        # 读取模型节点默认参数
        model_cfg = self.cfg.get_model_config(provider, ai_type)
        # backend 路由由工厂内部处理，这里直接调用即可
        model = LLMProviderFactory.get_chat_model(ai_type, provider)
        ctx = self._build_preview(items, max_items=opts.get("max_context", 6))
        prompts_cfg = (self.cfg.get_prompts_config() or {}).get("retrieval", {})
        system_text = str(prompts_cfg.get("system", "你是一个中文助理，请基于提供的上下文回答。"))
        user_tmpl = str(prompts_cfg.get("user_template", "问题：{query}\n\n上下文：\n{context}"))
        user_text = user_tmpl.format(query=query, context=ctx)
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        gen = model.generate(
            messages,
            max_new_tokens=int(opts.get("max_new_tokens", model_cfg.get("max_new_tokens", 1024))),
            enable_thinking=bool(opts.get("enable_thinking", model_cfg.get("enable_thinking", False))),
        )
        return gen.get("content") or gen.get("raw_text") or ""


