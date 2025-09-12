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

@pytest.mark.integration
def test_vector_retriever_live_query():
    """向量检索：最小集成验证。

    - 直接对真实集合执行 query_embeddings 检索；
    - 若无结果或环境不可用，使用 pytest.skip 进行跳过。
    """
    if not _can_connect():
        pytest.skip("ChromaDB 不可达，跳过集成测试")

    retriever = VectorRetriever()
    
    # 测试1: 无过滤条件的基础检索
    print("\n=== 测试1: 无过滤条件检索 ===")
    q1 = RetrievalQuery(query="长发女孩", where=None, top_k=10, score_threshold=0.0)
    result1 = retriever.search(q1)
    
    if not result1.items:
        pytest.skip("集合暂无可检索数据，跳过")
    
    print(f"无过滤条件检索结果: {len(result1.items)} 条记录")
    for i, item in enumerate(result1.items[:3]):  # 显示前3条
        print(f"  {i+1}. 相似度: {item.score:.4f}, 文本: {item.text[:50]}...")
    
    # 测试2: 按媒体类型过滤
    print("\n=== 测试2: 按媒体类型过滤 ===")
    q2 = RetrievalQuery(query="八箭将是哪几个", where={"media_type": "text"}, top_k=10, score_threshold=0.0)
    result2 = retriever.search(q2)
    
    print(f"媒体类型过滤结果: {len(result2.items)} 条记录")
    for i, item in enumerate(result2.items[:3]):
        print(f"  {i+1}. 相似度: {item.score:.4f}, 媒体类型: {item.metadata.get('media_type', 'N/A')}")
    
    # 测试3: 按块类型过滤
    print("\n=== 测试3: 按块类型过滤 ===")
    q3 = RetrievalQuery(query="八箭将是哪几个", where={"chunk_type": "text_traditional"}, top_k=10, score_threshold=0.0)
    result3 = retriever.search(q3)
    
    print(f"块类型过滤结果: {len(result3.items)} 条记录")
    for i, item in enumerate(result3.items[:3]):
        print(f"  {i+1}. 相似度: {item.score:.4f}, 块类型: {item.metadata.get('chunk_type', 'N/A')}")
    
    # 测试4: 按内容类型过滤
    print("\n=== 测试4: 按内容类型过滤 ===")
    q4 = RetrievalQuery(query="八箭将是哪几个", where={"content_type": "text/plain"}, top_k=10, score_threshold=0.0)
    result4 = retriever.search(q4)
    
    print(f"内容类型过滤结果: {len(result4.items)} 条记录")
    for i, item in enumerate(result4.items[:3]):
        print(f"  {i+1}. 相似度: {item.score:.4f}, 内容类型: {item.metadata.get('content_type', 'N/A')}")
    
    # 测试5: 按块索引范围过滤
    print("\n=== 测试5: 按块索引范围过滤 ===")
    q5 = RetrievalQuery(query="八箭将是哪几个", where={"chunk_index": {"$lt": 3}}, top_k=10, score_threshold=0.0)
    result5 = retriever.search(q5)
    
    print(f"块索引范围过滤结果: {len(result5.items)} 条记录")
    for i, item in enumerate(result5.items[:3]):
        print(f"  {i+1}. 相似度: {item.score:.4f}, 块索引: {item.metadata.get('chunk_index', 'N/A')}")
    
    # 测试6: 按来源过滤
    print("\n=== 测试6: 按来源过滤 ===")
    q6 = RetrievalQuery(query="八箭将是哪几个", where={"source": "manual_upload"}, top_k=10, score_threshold=0.0)
    result6 = retriever.search(q6)
    
    print(f"来源过滤结果: {len(result6.items)} 条记录")
    for i, item in enumerate(result6.items[:3]):
        print(f"  {i+1}. 相似度: {item.score:.4f}, 来源: {item.metadata.get('source', 'N/A')}")
    
    # 测试7: 高相似度阈值过滤
    print("\n=== 测试7: 高相似度阈值过滤 ===")
    q7 = RetrievalQuery(query="八箭将是哪几个", where={"media_type": "text"}, top_k=10, score_threshold=0.7)
    result7 = retriever.search(q7)
    
    print(f"高相似度阈值过滤结果: {len(result7.items)} 条记录")
    for i, item in enumerate(result7.items[:3]):
        print(f"  {i+1}. 相似度: {item.score:.4f}")
    
    # 测试8: 查询不存在的字段（验证不会报错）
    print("\n=== 测试8: 查询不存在字段 ===")
    q8 = RetrievalQuery(query="八箭将是哪几个", where={"nonexistent_field": "some_value"}, top_k=10, score_threshold=0.0)
    result8 = retriever.search(q8)
    
    print(f"查询不存在字段结果: {len(result8.items)} 条记录")
    
    # 验证所有测试的基本要求
    all_results = [result1, result2, result3, result4, result5, result6, result7, result8]
    
    for i, result in enumerate(all_results, 1):
        assert result.latency_ms >= 0, f"测试{i}延迟时间异常"
        assert all(it.text for it in result.items), f"测试{i}存在空文本"
        assert all(it.score >= 0 for it in result.items), f"测试{i}存在负分数"
    
    print(f"\n=== 测试总结 ===")
    print(f"✅ 所有 {len(all_results)} 个测试通过")
    print(f"✅ 无过滤条件: {len(result1.items)} 条")
    print(f"✅ 媒体类型过滤: {len(result2.items)} 条")
    print(f"✅ 块类型过滤: {len(result3.items)} 条")
    print(f"✅ 内容类型过滤: {len(result4.items)} 条")
    print(f"✅ 块索引范围过滤: {len(result5.items)} 条")
    print(f"✅ 来源过滤: {len(result6.items)} 条")
    print(f"✅ 高相似度阈值: {len(result7.items)} 条")
    print(f"✅ 不存在字段查询: {len(result8.items)} 条")
