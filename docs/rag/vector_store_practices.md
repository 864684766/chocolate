# 向量库工程化与扩展实践（ChromaDB）

本文档给出在 ChromaDB 上的工程实践方案，解决“现在小规模即可运行，将来数据增大也便于演进”的问题。

## 目标

- 提前约定最小规则，避免后期大规模迁移的痛点
- 支持按来源/时间/模型版本平滑扩容与切换
- 保持与现有代码最小耦合（通过配置与元数据实现）

## 一、集合命名约定（先做）

建议按“业务或数据域 + 时间/版本”命名，示例：

- `documents_2025Q1`
- `docs_{yyyyMM}`（如 `docs_202509`）
- `kb_{datasetName}`（如 `kb_finance`）
- 模型版本切换：`documents_v2`（与旧集合并存，只读）

命名作用：

- 控制单集合规模，避免无限增长
- 支持“新建集合承接增量 + 旧集合只读”
- 检索层可以一次查询多个集合后合并 TopK（读写隔离的基础）

## 二、元数据埋点（先做）

“埋点”= 每条向量的 `metadata` 字段中写入用于运营/迁移/审计的关键键值。

建议字段：

- `dataset`：数据集/来源（如 `finance_report`）
- `source`：文件/URL/上传批次标识
- `doc_id`：文档 ID（见下文 ID 规则）
- `chunk_index`：分块序号（int）
- `created_at`：生成月份 `YYYY-MM` 或时间戳
- `media_type`：text/pdf/image/video/audio/code
- `embed_model`：嵌入模型名（便于排查维度/效果问题）
- 可选：`quality_score`、`caption_model` 等领域特定信息

作用：

- 迁移/清理时可以 `where` 过滤
- 多集合查询后的结果分析与审计

## 三、ID 规则（先做）

保证可幂等写入与去重，建议：

- 文档 ID `doc_id`：
  - 优先使用稳定来源（如文件名+路径的 MD5 前缀 12 位），或外部业务 ID
- 向量条目 ID `id`：
  - 组合规则：`{doc_id}_{chunk_index}_{md5_8(text)}`
  - 含义：同一文档同一块的同内容只会写入一次，重复写入直接覆盖/跳过

## 四、维度稳定策略（先做）

- 同一集合内向量维度必须一致 → 同一集合内固定使用同一个嵌入模型
- 如需切换模型（维度不同或想 A/B），新建集合：
  - 旧集合：`documents_v1`（只读）
  - 新集合：`documents_v2`（新增写入）
  - 检索层短期“并查合并 TopK”，稳定后逐步下线旧集合

## 五、读写隔离（需要理解）

目标：写入/重建不影响在线检索。

- 写入到“新集合”，在线检索仍查“旧集合”
- 稳定后逐步将查询切换为“新集合 + 旧集合并查”
- 观察延迟/召回后，仅查新集合；旧集合只读保留或下线

实现方式：

- 在配置中支持“集合列表”给检索层（并查）
- 向量化/入库仅指定“当前写入集合”

## 六、哪些事可在 Chroma 端做？

- “索引结构/ANN/持久化/基础分片（取决于部署形态）”由 Chroma 管理
- 我们需要做的：
  - 选择/命名集合
  - 维度一致性（由我们控制模型）
  - 元数据规划（用于 where 过滤）
  - 批量写入大小与重试策略

大型分片/集群/容灾等能力，取决于你的 Chroma 部署与后端存储方案；不是应用层代码里实现的“分库分表”，但目标一致（扩容/高可用）。

## 七、落地步骤

1. 立即执行的轻成本项（本仓库已支持）：

- `collection_name` 采用带时间/版本的命名
- 写入 `metadata` 包含上述埋点键
- `id` 采用 `{doc_id}_{chunk_index}_{md5_8}`
- `embed_model` 字段记录模型名

2. 数据增大后的演进：

- 新建集合承接增量
- 后台重建/迁移（可重嵌入）
- 检索层临时并查多集合
- 观察后逐步切流；旧集合归档或下线

## 八、与当前代码的对齐

- 写入集合：`config.app_config.json > vectorization.collection_name`
- 元数据：`app/rag/vectorization/indexer.py` 的 `metadatas` 来源于 `ProcessedChunk.meta`，可在上游补充埋点键
- 模型名：在向量化配置中记录；`indexer` 可将 `embed_model` 写入 `metadata`

## 九、示例（概念演示）

- 配置（节选）：

```json
{
  "vectorization": {
    "collection_name": "documents_2025Q1",
    "database": { "storage": { "persistent": true } }
  }
}
```

- 元数据（每条向量）：

```json
{
  "doc_id": "a1b2c3d4e5f6",
  "chunk_index": 12,
  "dataset": "finance_report",
  "source": "upload_batch_2025_02_01",
  "created_at": "2025-02",
  "media_type": "text",
  "embed_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
}
```

---

如需，我可以把检索层改为“支持多集合并查”的接口，并在 `app_config.json` 增加 `retrieval.vector_collections: ["documents_2025Q1", "documents_2024Q4"]`。
