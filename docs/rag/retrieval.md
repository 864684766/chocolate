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
      "model": "BAAI/bge-reranker-base",
      "top_n": 10,
      "timeout_ms": 800,
      "batch_size": 16
    }
  }
}
```

字段说明：

- model：重排模型名称或本地路径（示例为 BGE 系列；也可用 `cross-encoder/ms-marco-*`）。
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

## 编排提示词配置（prompts.retrieval）

## 元数据与过滤（白名单与关键词）

本项目采用“精简白名单（9 项）”，聚焦可过滤与可审计字段；入库与检索共用同一套关键词抽取逻辑，保证 where 过滤一致性。

配置示例：

```json
"vectorization": {
  "metadata_whitelist": [
    { "field": "doc_id", "type": "string" },
    { "field": "source", "type": "string" },
    { "field": "filename", "type": "string" },
    { "field": "content_type", "type": "string" },
    { "field": "media_type", "type": "string" },
    { "field": "lang", "type": "string" },
    { "field": "tags", "type": "array" },
    { "field": "keyphrases", "type": "array" },
    { "field": "created_at", "type": "string" },
    { "field": "text_len", "type": "number" },
    { "field": "too_short", "type": "boolean" }
  ]
}
```

字段说明：

- doc_id：文档/片段唯一标识（溯源）
- source：数据来源（系统/目录/库）
- filename：原始文件名（定位与展示）
- content_type：MIME 类型（application/pdf、text/markdown、image/png…）
- media_type：模态（text|image|audio|video）
- lang：语言（zh|en…）
- tags：业务标签（人工/规则/模型）
- keyphrases：关键词（TextRank 抽取）
- created_at：创建/入库时间（ISO8601）
- text_len：文本长度（用于质量过滤）
- too_short：是否过短（用于质量过滤）

关键词抽取参数（ingestion.metadata.keywords）：

```json
"ingestion": {
  "metadata": {
    "keywords": {
      "method": "textrank",
      "topk": 10,
      "lang": "auto",
      "stopwords": { "path": "" }
    }
  }
}
```

统一服务：`app/rag/service/keywords_service.py`（接口：`extract_keyphrases(text, lang="zh", topk=10, method="textrank")`）。

字段说明：

- method：关键词抽取方法。
  - textrank：基于 TextRank 的中量方案（中文友好，默认）
  - light：分词 + 去停用词 + 频次/TF‑IDF 的轻量方案
  - keybert：向量/嵌入方案（性能较慢，当前实现回退为 light 占位）
  - none：关闭关键词抽取
- topk：抽取关键词的最大数量（建议 5–15）。
- lang：处理语言（auto/zh/en）。auto 将按调用方传入或默认值决定语言路径。
- stopwords.path：停用词文件路径（UTF‑8，每行一词，支持以 # 开头的注释）。若留空则不做停用词过滤。

质量评分配置（ingestion.metadata.quality）：

```json
"ingestion": {
  "metadata": {
    "quality": {
      "min_len": 20,
      "sat_len": 200,
      "weights": { "garbled": 0.4, "valid": 0.2, "length": 0.2, "ocr": 0.2 },
      "observability": {
        "enabled": true,
        "threshold": 0.6,
        "alert_ratio": 0.2,
        "sample_rate": 0.01
      }
    }
  }
}
```

**配置参数详细说明**：

配置位置：`metadata.quality`

- **min_len**: 最小文本长度阈值（字符数），低于此长度的文本长度得分为 0.0
- **sat_len**: 满意文本长度阈值（字符数），达到或超过此长度的文本长度得分为 1.0
- **weights**: 质量评估各维度权重配置
  - `garbled`: 乱码得分权重，基于不可打印字符比例计算
  - `valid`: 有效字符比例权重，基于字母数字和标点符号比例计算
  - `length`: 长度得分权重，基于文本长度计算
  - `ocr`: OCR 置信度权重，基于 OCR 识别的平均置信度计算
  - 注意：当 OCR 置信度缺失时，权重会自动归一化到其他三个维度
- **observability**: 质量观测与采样日志配置
  - `enabled`: 是否启用质量观测功能
  - `threshold`: 质量得分阈值，低于此值的样本会被记录日志
  - `alert_ratio`: 告警比例阈值（当前未使用，预留给监控系统）
  - `sample_rate`: 采样率，控制低质量样本的日志记录频率（0.01 = 1%）

where 过滤示例：

```json
{
  "media_type": { "$eq": "text" },
  "lang": { "$eq": "zh" },
  "created_at": { "$gte": "2025-01-01T00:00:00Z" },
  "tags": { "$contains": "发票" },
  "keyphrases": { "$contains": "报销" }
}
```

## 元数据 where 过滤与自动推断（WhereBuilder）

系统在编排层内置 WhereBuilder，使用户仅通过自然语言即可得到受限的元数据过滤：

- 输入：`/retrieval/search` 仅接收 `input`（query），不接收 where
- 过程：WhereBuilder 依据 query 自动推断严格 where 条件
- 约束：仅使用 `metadata.metadata_whitelist` 中允许的字段；不做放宽，0 命中即返回空
- 输出：返回体中附带 `applied_where` 与 `matched_count`

### 推断能力（当前）

- 语言（lang）：将自然语言关键词映射为 `zh/en`
- 媒体类型（media_type）：将自然语言关键词映射为 `text/image/video/audio/pdf`
- 质量阈值（quality_score）：解析 “质量>=0.8 / 大于 0.8 / at least 0.8”等表达为 `$gte`
- 时间（created_at）：解析“近一周/近一月”为 ISO8601 起点（UTC），输出 `{ "$gte": "..." }`
- 标签/关键词（tags/keyphrases）：若字段在白名单中，优先使用 `$contains`

仅当字段在 `metadata.metadata_whitelist` 中出现时才会被构造到 where；否则忽略。

### 新增配置项（用于多语言/别名解析）

1. 语言别名（可选）

配置位置：`metadata.language_detection.aliases`

示例：

```json
{
  "metadata": {
    "language_detection": {
      "aliases": {
        "中文,简体中文,zh,zh_cn": "zh",
        "英文,english,en": "en"
      }
    }
  }
}
```

用途：将自然语言/多写法映射为标准语言代码，WhereBuilder 解析时优先使用。

2. 媒体类型关键词映射（可选）

配置位置：`metadata.media_type_mapping.by_keyword`

示例：

```json
{
  "metadata": {
    "media_type_mapping": {
      "by_keyword": {
        "文本,文档,text": "text",
        "图片,图像,image": "image",
        "视频,video": "video",
        "音频,audio": "audio",
        "pdf": "pdf"
      }
    }
  }
}
```

用途：将多语言/别名映射为标准媒体类型值，WhereBuilder 解析时优先使用。

上述两个映射均为可选；未配置时，WhereBuilder 将回退到最小解析（覆盖范围较小）。

### 返回体中的诊断字段

- `applied_where`: 实际提交给向量库的 where 条件（严格白名单，不放宽）
- `matched_count`: 本次命中条数（未重排前的数量）

未命中时：`matched_count=0`，`items=[]`；系统不会做放宽，仅在日志中保留采样记录以便后续调优规则。

目的：将 `/retrieval/search` 生成阶段的提示词从代码中抽离为配置，便于按场景/语言/模型调整。

配置位置：`config/app_config.json` → `prompts.retrieval`

示例：

```json
{
  "prompts": {
    "retrieval": {
      "system": "你是一个中文助理，请基于提供的上下文回答。若信息不足，请明确说明。",
      "user_template": "问题：{query}\n\n上下文：\n{context}"
    }
  }
}
```

字段说明：

- system：作为第一条 system 消息注入模型，设定回答风格与安全边界。
- user_template：用户消息模板，支持占位符 `{query}` 与 `{context}`，分别由用户查询与检索拼接上下文替换。

运行时生效点：`app/rag/retrieval/orchestrator.py::_generate`

注意：

- 角色集合遵循各模型的 `chat_template`（通用安全组合：system、user、assistant）。
- 若模板中出现未定义角色（如 tool/function），请先验证对应模型的 `tokenizer_config.json` 中 `chat_template` 是否支持。
