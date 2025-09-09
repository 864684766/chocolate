# 向量化层（Vectorization）

职责：将数据处理层（Processing）输出的 `ProcessedChunk(text, meta)` 批量转换为“向量（embeddings）”，并写入向量数据库（ChromaDB）。不包含检索与重排逻辑。

## 目录（当前实现）

```
app/rag/vectorization/
  __init__.py         # 导出 VectorizationConfig / Embedder / VectorIndexer
  config.py           # 读取 app_config.json 中的 vectorization 配置
  embedder.py         # 文本→向量（占位实现，后续替换为 sentence-transformers）
  indexer.py          # 批量编码并写入 ChromaDB（documents/embeddings/metadatas/ids）
```

## 与 Processing 的衔接

入口：`app/rag/service/ingestion_helpers.py::process_and_vectorize`

流程：

- `chunks = ProcessingPipeline().run(samples)` 得到 `List[ProcessedChunk]`
- `VectorIndexer(cfg).index_chunks(chunks)` 生成向量并写入集合 `cfg.collection_name`

## 配置（app_config.json）

位置：`config/app_config.json > vectorization`

```json
{
  "vectorization": {
    "model_name": "D:/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "device": "auto",
    "batch_size": 32,
    "max_sequence_length": 512,
    "max_retries": 3,
    "retry_delay": 1.0,
    "retry_backoff_factor": 2.0,
    "collection_name": "documents_2025Q1",
    "collection_metadata": { "description": "文档向量集合", "version": "1.0" },
    "min_text_length": 10,
    "max_text_length": 10000,
    "skip_empty_text": true,
    "max_workers": 4,
    "database": {
      "host": "124.71.135.104",
      "port": 8000,
      "storage": {
        "type": "remote",
        "path": "/data/chromadb",
        "persistent": true
      },
      "connection": { "timeout": 30, "retry_attempts": 3, "pool_size": 10 }
    }
  }
}
```

参数说明（面向初学者，逐项解释）：

- 模型与批处理（决定“如何把文本变成向量”）

  - `model_name`
    - 它是什么：使用的嵌入模型名或本地模型路径。
    - 推荐：多语言模型，便于同时处理中英文（例如 `paraphrase-multilingual-MiniLM-L12-v2`）。
    - 何时需要改：想提升中文/多语言效果，或改为离线本地模型时。
    - 常见坑：不同模型的“向量维度”可能不同，切换模型建议写入“新集合”。
  - `device`
    - 它是什么：计算设备，`auto`/`cpu`/`cuda`（GPU）。
    - 建议：有 GPU 则用 `cuda`；否则 `auto` 即可。
    - 影响：GPU 通常更快；CPU 更通用但较慢。
  - `batch_size`
    - 它是什么：一次送入模型的文本条数（批大小）。
    - 如何取值：32 起步，根据显存/内存增减。
    - 常见坑：过大容易 OOM（内存/显存溢出）；过小吞吐低。
  - `max_sequence_length`
    - 它是什么：单条文本允许的最大长度（超过可能截断/丢弃）。
    - 作用：限制超长文本，防止极端样本拖慢/卡死。
  - `max_retries` / `retry_delay` / `retry_backoff_factor`
    - 它是什么：失败重试（次数/初始间隔/指数回退因子）。
    - 何时需要改：网络不稳或偶发模型错误时适度增大以提高稳定性。

- 集合与质量（决定“存到哪、哪些文本能入库”）

  - `collection_name`
    - 它是什么：向量写入的集合名（类似“表名”）。
    - 建议：带时间或版本号，如 `documents_2025Q1`，便于后续扩容/迁移。
  - `collection_metadata`
    - 它是什么：集合级元信息（说明/版本）。
    - 用处：团队协作与排查问题时可查看集合背景。
  - `min_text_length` / `max_text_length`
    - 它是什么：允许写入的文本长度范围。
    - 作用：过滤过短的噪声文本和异常超长文本，保证质量与性能。
  - `skip_empty_text`
    - 它是什么：是否跳过空字符串。
    - 建议：保持 `true`，避免生成无意义向量。
  - `max_workers`
    - 它是什么：并发工作线程数（应用层并发）。
    - 建议：根据机器 CPU/IO 能力调整；越大越快，但占用更多资源。

- 数据库（决定“如何连接与存储到 ChromaDB”）

  - `database.host` / `database.port`
    - 它是什么：ChromaDB 服务的地址与端口。
    - 何时需要改：远程部署或端口变更时。
  - `database.storage.type`
    - 它是什么：存储类型（如 `remote`/`local`）。
    - 影响：决定数据落地方式与路径行为。
  - `database.storage.path`
    - 它是什么：数据文件存放路径（仅本地/持久化相关）。
  - `database.storage.persistent`
    - 它是什么：是否持久化（`true` 推荐）。
    - 影响：`false` 可能导致重启后数据丢失。
  - `database.connection.timeout`
    - 它是什么：连接超时（秒）。超时会报错并触发重试。
  - `database.connection.retry_attempts`
    - 它是什么：连接失败的重试次数。
  - `database.connection.pool_size`
    - 它是什么：连接池大小（并发能力）。
    - 提示：过小会排队变慢；过大可能占用过多资源。
  - 说明：索引与底层 ANN 由 Chroma 负责；集合不存在会在首次写入时自动创建。

- 元数据白名单（用于 where 过滤）
  - `metadata_whitelist`：控制哪些字段会写入向量库的 metadatas，必须全部是基础类型。
  - 典型默认值（可按需裁剪/扩展）：
    - 通用：`doc_id, source, filename, content_type, media_type, chunk_index, chunk_type, chunk_size, created_at`
    - 文档/PDF：`page_number, start_pos, end_pos`
    - 图片：`region_index, ocr_engine, image_format, total_texts, min_x, max_x, min_y, max_y`
  - 修改方式：`config/app_config.json > vectorization.metadata_whitelist`

备注：集合命名/元数据/ID/维度稳定等工程实践，见《向量库工程化与扩展实践（ChromaDB）》：`docs/rag/vector_store_practices.md`。

## 使用示例

在服务层一次完成处理+向量化（已接入）：

```python
from app.rag.service.ingestion_helpers import process_and_vectorize

result = process_and_vectorize(samples)
# result = {"chunks": 分块数, "embedded": 已入库向量数}
```

## 真实嵌入实现（说明）

当前 `embedder.py` 为“占位实现”（返回简化向量以打通流程）。
上线前应替换为 sentence-transformers：

- 加载：`SentenceTransformer(config.model_name, device=config.device)`
- 批量编码：`model.encode(texts, batch_size=config.batch_size, normalize_embeddings=True)`
- 并行：必要时在应用层做分片/线程池，并合并结果
