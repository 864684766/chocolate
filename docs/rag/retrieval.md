# 检索层（Retrieval）

定位：提供“可被应用层编排调用”的检索能力，包含向量检索、关键词检索、混合融合与上下文构建；可选接入重排。应用层负责“智能路由/策略选择”。

## 能力清单

- 向量检索 VectorRetriever（已提供最小实现）

  - 输入：query、where、top_k、score_threshold
  - 流程：编码 → ChromaDB 相似度查询（n_results=top_k）
  - 输出：`RetrievalResult(items=[{id,text,score,metadata}...])`

- 关键词检索 KeywordRetriever（最小实现）

  - 输入：where 过滤（基于 metadata）
  - 流程：Chroma `get(where=...)` 返回文档
  - 输出：固定分数 1.0，后续由融合器/应用层排序

- 混合融合 HybridSearcher（RRF/加权）

  - RRF：稳健、免调参；`score = Σ 1/(k+rank+1)`
  - 加权：`score = w_vec*sim + w_kw*match`

- 上下文构建 ContextBuilder（最小实现）
  - 去重/截断/拼接，输出 `BuiltContext(text, citations, used_tokens, from_items)`
  - 正式实现：结合中文阶段去重与“真实 token 预算”（复用应用层模型分词器）

> 注：重排（Cross-encoder）可在二期引入；建议归应用层策略管理，检索层仅提供接口实现。

## 目录结构（本仓库）

```
app/rag/retrieval/
  __init__.py         # 导出能力
  schemas.py          # Pydantic 模型
  retriever.py        # VectorRetriever / KeywordRetriever
  hybrid.py           # HybridSearcher（RRF/加权）
  context_builder.py  # ContextBuilder（拼接/引用）
app/core/tokenization/
  provider.py         # 复用应用层模型配置，提供 TokenCounter（openai:tiktoken / hf:AutoTokenizer）
```

## 接口约定

- 输入模型：`RetrievalQuery(query, where, top_k, score_threshold)`
- 结果模型：`RetrievalResult(items, latency_ms, debug_info)`
- 上下文：`BuiltContext(text, citations, used_tokens, from_items)`
  - `ContextBuilder.build(result, max_tokens, citation, token_counter=None, ai_type=None, provider=None)`
    - 若提供 `token_counter` 或 `ai_type/provider`，将用真实 tokenizer 进行预算；否则使用字符近似兜底。

## 与应用层的配合

- 应用层做“智能路由”，输出策略：`{ mode: vector|keyword|hybrid, where, top_k, method/weights }`
- 检索层据此执行并返回 `BuiltContext`：
  1. Vector/Keyword 召回
  2. （可选）HybridSearcher 融合
  3. ContextBuilder 生成上下文与引用
- 应用层再把 `context.text` 注入 Prompt，交给 LLM 生成答案。

## TopK 与阈值建议

- 典型：召回 top_k=30~50，最终输出 top_k_out=5~10
- score_threshold：0.2~0.4（按模型与数据分布微调）

## 开发待办（Todo）

为保证功能单一、便于二期扩展，按阶段实施：

### V1（最小可用，先跑通 where→vector）

- [ ] RetrievalQuery/Result/BuiltContext 模型补充字段注释与示例
- [ ] VectorRetriever.search：支持 where 预过滤、score_threshold、生效的 include 字段
- [ ] ContextBuilder：接入 tokenizer，新增严格 token 预算（max_tokens 与回复预留）
- [ ] 观测最小打点：检索耗时 ms、构建耗时 ms、最高分、结果条数
- [ ] 配置项（app_config.json > retrieval）：vector.top_k、vector.score_threshold、context.max_tokens

### V1.5（稳健性增强）

- [ ] KeywordRetriever：完善 where 过滤（常用字段示例与校验）
- [ ] HybridSearcher：增加并行执行样例与 RRF 融合入口（保持可选）
- [ ] 兜底策略：无结果/低分时返回友好提示，或放宽 where 重试一次（开关可配）
- [ ] 启动热身：Embedder 预加载；必要时增加小缓存（可关）
- [ ] 错误处理：超时/网络异常 → 标准化错误与日志

### V2（效果提升，可选开启）

- [ ] Reranker 接口：抽象 cross-encoder 重排器（不与检索耦合）
- [ ] RerankerRouter：按语言选择 BGE(BAAI/bge-reranker-large（精度高）或 BAAI/bge-reranker-base
      )/MiniLM(cross-encoder/ms-marco-MiniLM-L-6-v2)/多语重排模型
- [ ] 融合加权：在 RRF 之外提供 weighted_sum 与业务加权扩展点
- [ ] 指标完善：命中分布、空结果率、RRF/重排开关使用率
- [ ] 文档与示例：端到端示例（路由 → 检索 → 构建 →LLM）

## 目录规划与扩展点（单一职责）

当前结构已满足单一职责，建议保留并预留二期扩展文件：

```
app/rag/retrieval/
  __init__.py           # 统一导出
  schemas.py            # Pydantic 数据模型
  retriever.py          # VectorRetriever / KeywordRetriever（各自仅负责检索）
  hybrid.py             # HybridSearcher（只做融合与排序）
  context_builder.py    # ContextBuilder（只做去重/合并/预算/引用）
  # 预留（V2）：
  reranker.py           # Cross-encoder 重排接口与实现
  router.py             # （可选）应用层路由示例或接口契约
```

说明：

- 应用层只负责“是否并行/是否融合/是否重排”的策略与参数；检索层只执行。两层通过 `RetrievalQuery` 与 `BuiltContext` 解耦。
- 每个文件仅包含一个职责，二期新增直接按文件落位，不会影响既有实现。
