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
- `VectorIndexer(cfg).index_chunks(chunks)` 生成向量并写入集合 `cfg.collection_name`（其值来自 `vectorization.database.collection_name`）

### 统一规范化与去重（新增）

为保证 where 可预期、入库幂等与检索质量，向量化前新增统一处理：

1. 文本规范化 `normalize_text_for_vector(text)`

- NFKC 规范化；移除控制字符；将所有空白折叠为单空格并 trim；不做长度截断
- 该结果用于写入 `documents`、生成稳定 ID 与向量编码

2. 元数据规范化 `normalize_meta_for_vector(meta)`

- 只保留 `vectorization.metadata_whitelist` 中的键；缺省值按类型自动补齐
- 配置格式：`[{ field: string, type: string }]`，示例见下；全局约定 meta 为扁平结构（不做嵌套展开）

3. 稳定 ID、去重与 upsert

- 稳定 ID：`{doc_id or filename}:{chunk_index}:{sha1(norm_text)[:16]}`
- 批内去重：按稳定 ID 去重
- 库内过滤：`get(ids=...)` 过滤已存在 ID 后再写入
- upsert：已存在且内容/元数据变化则 update，否则 add
- 索引器会输出统计日志：`raw / batch_dedup / existed / written / updated`

> 提示：若希望 where 生效，请确保需要过滤的键被加入 `metadata_whitelist`，并在入口/管线阶段保证这些键有值（即使为空串/0/False）。

## 配置（app_config.json）

位置：`config/app_config.json > vectorization`

```jsonc
{
  "vectorization": {
    "model_name": "D:/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "device": "auto",
    "batch_size": 32,
    "metadata_whitelist": [
      { "field": "doc_id", "type": "string" },
      { "field": "source", "type": "string" },
      { "field": "filename", "type": "string" },
      { "field": "content_type", "type": "string" },
      { "field": "media_type", "type": "string" },
      { "field": "chunk_index", "type": "number" },
      { "field": "chunk_type", "type": "string" },
      { "field": "chunk_size", "type": "number" },
      { "field": "created_at", "type": "string" },
      { "field": "page_number", "type": "number" },
      { "field": "start_pos", "type": "number" },
      { "field": "end_pos", "type": "number" },
      { "field": "region_index", "type": "number" },
      { "field": "ocr_engine", "type": "string" },
      { "field": "image_format", "type": "string" },
      { "field": "total_texts", "type": "number" },
      { "field": "min_x", "type": "number" },
      { "field": "max_x", "type": "number" },
      { "field": "min_y", "type": "number" },
      { "field": "max_y", "type": "number" }
    ],
    "database": {
      "host": "124.71.135.104",
      "port": 8000,
      "collection_name": "documents"
    }
  }
}
```

参数说明（面向初学者，逐项解释）：

- 模型与批处理（决定"如何把文本变成向量"）

  - `model_name`
    - **它是什么**：告诉系统用哪个"翻译器"把文字变成数字向量
    - **形象比喻**：就像选择不同的"翻译官"，有的擅长中文，有的擅长英文
    - **推荐选择**：`paraphrase-multilingual-MiniLM-L12-v2`（多语言专家，中英文都行）
    - **本地 vs 在线**：
      - 本地路径（如 `D:/models/...`）：从你电脑加载，速度快但占用空间
      - 在线名称（如 `sentence-transformers/...`）：从网上下载，首次慢但省空间
    - **重要提醒**：换模型就像换翻译官，之前的数据可能不兼容，建议用新集合
  - `device`
    - **它是什么**：选择用哪个"计算器"来算向量
    - **选项说明**：
      - `auto`：自动选择最好的（推荐）
      - `cpu`：用电脑的中央处理器（慢但稳定）
      - `cuda`：用显卡（快但需要 NVIDIA 显卡）
      - `mps`：用苹果芯片（Mac 用户）
    - **速度对比**：显卡 > 苹果芯片 > CPU（大概快 5-10 倍）
  - `batch_size`
    - **它是什么**：一次处理多少条文本
    - **形象比喻**：就像工厂流水线，一次处理 32 个产品 vs 一次处理 1 个
    - **如何选择**：
      - 电脑内存大：可以设 64 或 128（更快）
      - 电脑内存小：设 16 或 32（更稳定）
      - 有显卡：可以设更大（如 128）
    - **常见问题**：设太大电脑会卡死，设太小处理很慢

- 数据库（决定"如何连接与存储到 ChromaDB"）

  - `database.host` / `database.port`
  - `database.collection_name`
    - **它是什么**：向量数据库集合名（更合理地放到 database 下，表示“属于数据库配置的一部分”）
    - **命名建议**：加上时间或版本，如 `documents_2025Q1`、`images_v2`
    - **注意**：更换模型时建议使用新集合，避免维度或语义不兼容
    - **它是什么**：向量数据库的"地址和门牌号"
    - **形象比喻**：就像寄快递，需要知道收件人的地址和门牌号
    - **本地 vs 远程**：
      - 本地：`host: "localhost"`（就在你电脑上）
      - 远程：`host: "124.71.135.104"`（在别的服务器上）

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

## 初学者概念解释

### 什么是向量化？

**简单理解**：把文字变成数字，让电脑能"理解"文字的意思。

**具体例子**：

- 输入：`"我喜欢吃苹果"`
- 输出：`[0.1, 0.3, -0.2, 0.8, ...]`（384 个数字组成的向量）

**为什么需要向量化**？

- 电脑只能处理数字，不能直接理解文字
- 相似的文字会产生相似的向量
- 通过比较向量，就能找到相似的内容

### L2 归一化是什么？

**简单理解**：把向量"标准化"，让所有向量的长度都变成 1。

**具体例子**：

- 原始向量：`[3, 4]`（长度 = 5）
- 归一化后：`[0.6, 0.8]`（长度 = 1）

**为什么要归一化**？

- 在搜索时，我们只关心向量的"方向"，不关心"长度"
- 归一化后，相似的内容向量方向更接近，搜索更准确
- 就像比较两个人的"性格方向"，而不是"性格强度"

### numpy 数组是什么？

**简单理解**：numpy 是 Python 中处理数字的"超级计算器"。

**具体例子**：

```python
# 普通 Python 列表（慢）
normal_list = [1, 2, 3, 4]

# numpy 数组（快）
import numpy as np
numpy_array = np.array([1, 2, 3, 4])
```

**为什么用 numpy**？

- 向量计算需要大量数学运算
- numpy 比普通 Python 快几十倍
- 就像用计算器 vs 用笔算的区别

### get_model_info 的 status 字段

**它是什么**：告诉 you 模型是否已经准备好使用

**可能的值**：

- `"loaded"`：模型已成功加载，可以正常使用
- `"not_loaded"`：模型未加载，可能出错或正在加载中

**用途**：检查模型状态，确保向量化功能正常

## 真实嵌入实现

`embedder.py` 已实现基于 sentence-transformers 的真实向量化：

### 核心功能

- **模型加载**：自动加载配置指定的 sentence-transformers 模型
- **设备选择**：支持 `auto`/`cpu`/`cuda`/`mps`（Apple Silicon）
- **批量编码**：使用配置的 `batch_size` 进行高效批量处理
- **向量归一化**：自动进行 L2 归一化，提升检索效果
- **错误处理**：完善的异常处理和日志记录

### 使用示例

```python
from app.rag.vectorization import VectorizationConfig, Embedder

# 从配置创建
config = VectorizationConfig.from_config_manager()
embedder = Embedder(config)

# 编码文本
texts = ["这是第一段文本", "这是第二段文本"]
vectors = embedder.encode(texts)
print(f"向量维度: {len(vectors[0])}")  # 例如: 384

# 获取模型信息
info = embedder.get_model_info()
print(info)  # 包含模型名、维度、设备等信息

# status 字段的作用：
# - "loaded": 模型已成功加载，可以正常使用
# - "not_loaded": 模型未加载，可能出错或正在加载中
# 用途：检查模型状态，确保向量化功能正常
```

### 性能优化

- **批处理**：通过 `batch_size` 控制内存使用和速度平衡
- **设备选择**：GPU 加速（如有 CUDA），CPU 兼容性
- **模型缓存**：首次下载后自动缓存，后续启动更快
