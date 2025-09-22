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
  retriever.py        # 统一导出（VectorRetriever 已拆分）
  vector_retriever.py # 向量检索器（ChromaDB）
  meilisearch_retriever.py # 关键词检索器占位（后续扩展）
  hybrid.py           # HybridSearcher（RRF/加权）
  reranker.py         # CrossEncoderReranker（占位）
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

- 应用层做“智能路由”，输出策略：`{ mode: vector|hybrid, where, top_k, method/weights, rerank? }`
- 检索层据此执行并返回 `BuiltContext`：
  1. Vector 召回（仅向量时，融合层透传）
  2. （可选）HybridSearcher 融合（当接入第二路时启用 RRF/加权）
  3. （可选）CrossEncoderReranker 重排
  4. ContextBuilder 生成上下文与引用
- 应用层再把 `context.text` 注入 Prompt，交给 LLM 生成答案。

## TopK 与阈值建议

- 典型：召回 top_k=30~50，最终输出 top_k_out=5~10
- score_threshold：0.2~0.4（按模型与数据分布微调）

## 评分配置（retrieval.scoring）

目的：不同向量后端/索引返回“距离/相似度”的语义与范围不一，需要一个统一的映射把它们转成 [0,1] 的分数，便于与 `score_threshold` 协同。

配置位置：`config/app_config.json` → `retrieval.scoring`

示例：

```json
{
  "retrieval": {
    "scoring": {
      "metric": "cosine_distance",
      "normalize": true,
      "params": { "alpha": 1.0 }
    }
  }
}
```

字段说明：

- metric：度量类型（见下表）；不同后端可返回不同度量/数值语义。
- normalize：是否将分数限制到 [0,1]（建议开启，便于统一阈值语义）。
- params.alpha：当 metric=l2 时使用的指数衰减系数（`score = exp(-alpha * d)`）。

常见度量与适用后端（参考）：

- cosine_distance（余弦距离，通常为 1 - cosine_similarity）

  - 常见：ChromaDB（基于 FAISS/AnnLite 时可配置为 cosine）、Faiss（IndexFlatIP 配合归一化也可近似余弦）、Milvus（设置 metric_type=COSINE）。
  - 映射：`score = 1 - d/2`（兼容 d∈[0,1] 与 d∈[0,2]，再 clamp 到 [0,1]）。

- cosine_similarity（余弦相似度，s∈[-1,1] 或已归一）

  - 常见：某些后端直接返回相似度；或你自行计算相似度再查询。
  - 映射：`score = (s + 1)/2`。

- l2（欧氏距离，d≥0）

  - 常见：Faiss L2、Milvus L2、部分 Elastic kNN 配置。
  - 映射：`score = exp(-alpha * d)`，alpha 默认 1.0，可按分布微调。

- inner_product（内积/点积）
  - 常见：Faiss IP、Milvus IP、Elasticsearch 8.x kNN 的 dot_product。
  - 映射：范围依数据分布，默认仅裁剪到 [0,1]；生产中建议结合观测分布做线性归一。

说明：实际后端是否返回“距离”或“相似度”需以后端文档/配置为准。若不确定，做一次“用同一向量检索自身”的探针测试：

- 若返回值接近上限且越大越好 → 相似度；
- 若返回值接近 0 且越小越好 → 距离。

实现落点：

- 代码文件 `app/rag/retrieval/utils/scoring.py` 提供 `score_from_distance(value, metric, normalize, alpha)`；
- `app/rag/retrieval/retriever.py::VectorRetriever.search` 读取配置并调用该工具，移除了硬编码 `1.0 - dist`。

## 重排配置（retrieval.rerank）

目的：交叉编码器对候选进行精排，提升最终相关性（现已默认启用）。

配置位置：`config/app_config.json` → `retrieval.rerank`

示例：

```json
{
  "retrieval": {
    "rerank": {
      "model_name": "BAAI/bge-reranker-base",
      "top_n": 10,
      "timeout_ms": 800,
      "batch_size": 16
    }
  }
}
```

字段说明：

- model_name：重排模型名称（示例为 BGE 系列；也可用 `cross-encoder/ms-marco-*`）。
- top_n：重排后保留的候选数（建议与召回 top_k 对齐或略小）。
- timeout_ms：重排超时（用于控制端到端时延）。
- batch_size：重排推理批大小（默认 16；按显存/内存与吞吐取值 8–64 压测确定）。

实现落点：

- `app/rag/retrieval/reranker.py::CrossEncoderReranker`。
- `app/api/agent.py::/retrieval/search` 始终执行重排并返回前 N 条预览；若模型加载失败，将退化为按向量分数排序。

## Meilisearch 配置（retrieval.meilisearch）

目的：为后续引入关键词检索（BM25/容错搜索）预留配置，便于与向量召回做混合融合（RRF/加权）。

配置位置：`config/app_config.json` → `retrieval.meilisearch`

示例：

```json
{
  "retrieval": {
    "meilisearch": {
      "enabled": false,
      "host": "http://localhost:7700",
      "api_key": "",
      "index": "documents"
    }
  }
}
```

字段说明：

- enabled：是否启用 Meilisearch 检索通道（默认关闭）。
- host/api_key/index：Meilisearch 连接信息与索引名。

实现与扩展点：

- 占位检索器：`app/rag/retrieval/meilisearch_retriever.py`（当前返回空结果）。
- 接入后可与 `HybridSearcher` 做 RRF 融合；仅向量时融合层透传（不增加时延）。

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

## 查询侧轻量清洗（已接入）

目的：保证“查询字符串”与“入库文本”的规范一致，提升关键词匹配与向量编码稳定性，避免语种/空白/标点差异带来的召回抖动。

规则（按顺序执行，均为无损规范化）：

- Unicode NFKC 归一化（全角 → 半角、兼容字符统一）
- 去控制字符（仅保留可打印字符）
- 常见中英标点统一（中英文引号、破折号、括号等映射为半角）
- 空白归一化（连续空白折叠为单空格，去首尾空白）

生效位置：

- `app/rag/retrieval/utils/query_cleaner.py::clean_query_basic`
- 在 `VectorRetriever.search` 与 `KeywordRetriever.search` 入口对 `q.query` 无条件应用一次；不会修改库内文档。

说明：

- 该清洗仅作用于查询字符串，入库侧不重复清洗；若将来升级规则，建议通过离线重建索引而非在线改库内文本。
