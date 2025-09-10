"""
检索层集成测试（直连当前 ChromaDB）。

说明：
- 本测试不打桩，直接连接 `config/app_config.json` 中配置的 ChromaDB 与集合；
- 目的：帮助你调试与学习 end-to-end 流程（召回→融合→上下文构建）；
- 若环境不满足（无法连接/集合无数据），测试会被跳过而不是失败。
"""

from typing import Optional
import pytest

from app.rag.retrieval import (
    RetrievalQuery,
    VectorRetriever,
    KeywordRetriever,
    HybridSearcher,
    ContextBuilder,
)
from app.infra.database.chroma.db_helper import ChromaDBHelper
from app.rag.vectorization.config import VectorizationConfig
from app.infra.exceptions.exceptions import DatabaseConnectionError


def _can_connect() -> bool:
    """检查 ChromaDB 是否可达。

    Returns:
        bool: 可连返回 True；否则 False。
    """
    try:
        ChromaDBHelper().connect()
        return True
    except Exception:
        return False


def _get_collection_name() -> str:
    """读取当前集合名（来源于 vectorization.database.collection_name）。

    Returns:
        str: 集合名。
    """
    return VectorizationConfig.from_config_manager().collection_name


@pytest.mark.integration
def test_vector_retriever_live_query():
    """向量检索：最小集成验证。

    - 直接对真实集合执行 query_embeddings 检索；
    - 若无结果或环境不可用，使用 pytest.skip 进行跳过。
    """
    if not _can_connect():
        pytest.skip("ChromaDB 不可达，跳过集成测试")

    retriever = VectorRetriever()
    q = RetrievalQuery(query="测试", where=None, top_k=10, score_threshold=0.0)
    result = retriever.search(q)

    if not result.items:
        pytest.skip("集合暂无可检索数据，跳过")

    assert result.latency_ms >= 0
    assert all(it.text for it in result.items)


@pytest.mark.integration
def test_keyword_retriever_live_query():
    """关键词/结构化检索：最小集成验证。

    注意：where 可按你的 `metadata_whitelist` 字段调整；这里先不传 where，验证基本返回能力。
    """
    if not _can_connect():
        pytest.skip("ChromaDB 不可达，跳过集成测试")

    retriever = KeywordRetriever()
    q = RetrievalQuery(query="", where=None, top_k=5, score_threshold=0.0)
    result = retriever.search(q)

    if not result.items:
        pytest.skip("集合暂无可检索数据（或 where 过严），跳过")

    # KeywordRetriever 当前实现固定 score=1.0
    assert all(abs(it.score - 1.0) < 1e-6 for it in result.items)


@pytest.mark.integration
def test_hybrid_rrf_with_live_candidates():
    """混合融合（RRF）：基于真实候选做一次融合排序。

    - 分别取向量与关键词候选；
    - 若某一路为空则跳过；
    - 执行 RRF 并断言有序、不报错。
    """
    if not _can_connect():
        pytest.skip("ChromaDB 不可达，跳过集成测试")

    vec = VectorRetriever().search(RetrievalQuery(query="测试", where=None, top_k=10, score_threshold=0.0))
    kw = KeywordRetriever().search(RetrievalQuery(query="", where=None, top_k=10, score_threshold=0.0))

    if not vec.items or not kw.items:
        pytest.skip("候选不足，跳过融合测试")

    fused = HybridSearcher.rrf(vec, kw, top_k=10)
    assert fused.items
    # RRF 不保证严格分数递减，但应返回有序结果与有效文本
    assert all(it.text for it in fused.items)


@pytest.mark.integration
def test_context_builder_live_sample():
    """上下文构建：对真实检索结果进行拼装，验证结构正确。

    - 不引入 tokenizer，仅验证基础结构与字段；
    - 二期可在此接入 tokenizer 并验证 token 预算逻辑。
    """
    if not _can_connect():
        pytest.skip("ChromaDB 不可达，跳过集成测试")

    vec = VectorRetriever().search(RetrievalQuery(query="测试", where=None, top_k=8, score_threshold=0.0))
    if not vec.items:
        pytest.skip("集合暂无可检索数据，跳过")

    ctx = ContextBuilder().build(vec, max_tokens=1500, citation=True)
    assert isinstance(ctx.text, str)
    assert ctx.text != ""
    assert len(ctx.citations) == len(ctx.from_items)


if __name__ == "__main__":
    """
    便捷调试入口：直接运行本文件可快速观察各环节输出。

    用法：python -m tests.test_retrieval
    注意：仍然依赖你已配置好的 ChromaDB 与集合。
    """
    import pprint

    pp = pprint.PrettyPrinter(indent=2, width=120)

    if not _can_connect():
        print("[SKIP] ChromaDB 不可达，无法进行现场调试。")
    else:
        print("[INFO] 集合：", _get_collection_name())

        # Vector
        vr = VectorRetriever()
        qv = RetrievalQuery(query="测试", where=None, top_k=5, score_threshold=0.0)
        rv = vr.search(qv)
        print("\n[Vector] items:")
        for it in rv.items:
            pp.pprint({"id": it.id, "score": it.score, "text": it.text[:80], "meta": {k: it.metadata.get(k) for k in list(it.metadata)[:5]}})
        print("latency_ms:", rv.latency_ms)

        # Keyword
        kr = KeywordRetriever()
        qk = RetrievalQuery(query="", where=None, top_k=5, score_threshold=0.0)
        rk = kr.search(qk)
        print("\n[Keyword] items:")
        for it in rk.items:
            pp.pprint({"id": it.id, "score": it.score, "text": it.text[:80], "meta": {k: it.metadata.get(k) for k in list(it.metadata)[:5]}})
        print("latency_ms:", rk.latency_ms)

        # Fusion (RRF)
        if rv.items and rk.items:
            fused = HybridSearcher.rrf(rv, rk, top_k=5)
            print("\n[RRF Fusion] items:")
            for it in fused.items:
                pp.pprint({"id": it.id, "score": it.score, "text": it.text[:80]})
            print("latency_ms:", fused.latency_ms)
        else:
            print("\n[RRF Fusion] 候选不足，跳过")

        # Context
        if rv.items:
            ctx = ContextBuilder().build(rv, max_tokens=1500, citation=True)
            print("\n[Context] text preview:\n", ctx.text[:300])
            print("citations:")
            pp.pprint(ctx.citations[:5])
        else:
            print("\n[Context] 向量检索无结果，跳过")


