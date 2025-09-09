# 数据处理层（Processing）

将原始样本标准化、清洗、分块、抽取元数据，并进行语言/媒体特定处理。

## 目录结构（多语言+多媒体）

```
app/rag/processing/
  interfaces.py               # RawSample/ProcessedChunk + 协议接口
  pipeline.py                 # 管道编排（责任链/管道模式）
  quality_checker.py          # 质量评估（SimpleQualityAssessor）
  media_text.py               # 纯文本/Markdown基础解码（utf-8→gb18030→latin-1）
  media_markdown.py           # Markdown 提取（去围栏/标题号）
  lang_zh.py                  # 中文处理器（clean + chunk）
  media_extractors.py         # 媒体内容提取器（图像、视频、音频）
  media_chunking.py           # 媒体分块策略
  config_usage_example.py     # 配置使用示例
  # 预留：media/pdf_extractor.py, media/image_ocr.py, media/audio_asr.py, media/video_extractor.py 等
```

说明：

- 语言与媒体解耦在不同目录，通过统一接口注入到管道。
- 语言选择：运行时通过语言检测（fasttext/pycld3）或外部标注，选择 `lang/*` 下对应实现。

## Text Splitters 与多媒体

- 文本：使用 LangChain Text Splitters（Recursive/Token）
- PDF/图片/音视频：先在 `media/*` 转成"可切分文本"（正文/表格序列化/字幕），再交给 Text Splitters。

## 统一接口

- `LanguageProcessor`：clean(text)->str, chunk(text)->List[str], extract_meta()->dict
- `MediaExtractor`：extract(RawSample)->{"text": str, "meta": {...}}
- `QualityAssessor`：score(text, meta)->Dict[str, Any]

## 分块参数决策（新）

为适配不同媒体与上下文窗口，已将"分块参数决策"独立为处理工具模块：`app/rag/processing/utils/chunking.py`。

- 入口函数：`decide_chunk_params(media_type: str, content: Any, meta: dict) -> (chunk_size, overlap)`
- 用途：基于配置与内容特征，输出稳健的 `chunk_size/overlap`；不直接做切块，便于被 `media_chunking` 的具体策略复用。
- 支持媒体类型：`text/pdf/image/video/audio/code`（可扩展）
- 工作方式：
  - 读取 `config/app_config.json > media_processing.chunking`；
  - 当 `auto=false` 使用表驱动默认值；当 `auto=true` 或缺省时，依据目标 token、上下文上限、内容规模自适应计算；
  - 可通过 `meta` 注入更细粒度的信号做微调（如 OCR 噪声、语速、字幕密度等）。

### app_config.json 新增配置

```json
{
  "media_processing": {
    "chunking": {
      "auto": true,
      "defaults": { "chunk_size": 800, "overlap": 150 },
      "by_media_type": {
        "text": { "chunk_size": 800, "overlap": 120 },
        "pdf": { "chunk_size": 900, "overlap": 180 },
        "image": { "chunk_size": 300, "overlap": 80 },
        "video": { "chunk_size": 1000, "overlap": 200 },
        "audio": { "chunk_size": 1000, "overlap": 200 },
        "code": { "chunk_size": 400, "overlap": 160 }
      },
      "targets": {
        "target_tokens_per_chunk": 512,
        "overlap_ratio": 0.15,
        "max_context_tokens": 8192
      }
    }
  }
}
```

说明：

- **auto**: 是否启用自适应；关闭时走表驱动；
- **defaults/by_media_type**: 提供基线值，亦作为自适应上下限保护；
- **targets**: 目标片长、overlap 比例、上下文窗口（按使用模型的 context 调整）。

### 在管道中的接入

`app/rag/processing/pipeline.py` 中通过：

```python
from app.rag.processing.utils.chunking import decide_chunk_params

chunk_size, overlap = decide_chunk_params(str(media_type), content, meta)
strategy = ChunkingStrategyFactory.create_strategy(
    str(media_type),
    chunk_size=chunk_size,
    overlap=overlap,
)
```

这样即可在不修改业务代码的前提下，通过配置或自适应策略动态控制分块。

## 中文优化要点

- 分词：jieba + 自定义词典
- 标点与空白归一化、繁简转换（opencc）
- 章节/小节/标题保留为元数据，利于重排与引用

## 媒体处理配置

### 配置文件位置

配置文件位于：`config/app_config.json`

### 配置结构

#### 1. 图像描述配置 (image_captioning)

```json
{
  "media_processing": {
    "image_captioning": {
      "enabled": true,
      "model": "D:/models/Salesforce/blip-image-captioning-base",
      "generation": {
        "num_captions": 1,
        "max_length": 50,
        "temperature": 0.7,
        "do_sample": true
      },
      "filters": {
        "min_length": 5,
        "max_length": 120,
        "max_gibberish_ratio": 0.3,
        "blacklist_keywords": ["优惠", "扫码", "购买", "VX"]
      },
      "rerank": {
        "enabled": true,
        "model": "D:/models/openai/clip-vit-base-patch32",
        "top_k": 2,
        "min_clip_prob": 0.2
      },
      "embedding": {
        "enabled": true,
        "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "dimension": 384
      },
      "translation": {
        "enabled": true,
        "model": "D:/models/Helsinki-NLP/opus-mt-en-zh",
        "batch_size": 16
      },
      "deduplication": {
        "approximate_enabled": true,
        "similarity_threshold": 0.95,
        "embed_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "forbid_repeat_ngram": 3
      }
    }
  }
}
```

**配置参数说明（与当前代码实现一致）**：

- **enabled**: 是否启用图像描述功能
- **model**: 使用的图像描述模型（单一模型，建议本地路径）
- **prompt**: 描述生成提示词（完全由配置决定，可写中文或英文），例如：
  - 中文：`"请用中文简洁描述这张图片："`
  - 英文：`"Describe this image in concise English:"`
- **generation**: 生成参数
  - `num_captions`: 生成描述的数量（默认：1）
  - `max_length`: 描述的最大长度（默认：50）
  - `temperature`: 生成温度，控制随机性（默认：0.7）
  - `do_sample`: 是否使用采样（默认：true）
- **rerank**: 粗排配置（提高“图像 ↔ 文本”一致性）

  - `enabled`: 是否启用 CLIP 粗排
  - `model`: CLIP 模型（默认示例：`openai/clip-vit-base-patch32`，可指向本地路径）
  - `top_k`: 粗排后保留的候选数

- **embedding**: 嵌入向量配置

  - `enabled`: 是否生成嵌入向量（默认：true）
  - `model`: 推荐 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`（多语言效果更稳）。注意：代码中的“默认回退值”为 `clip-ViT-B-32`，仅在配置缺失时作为兜底；若你在 `app_config.json` 指定了 `sentence-transformers/...`，则实际以配置为准。
  - `dimension`: 向量维度（与所选模型一致，`paraphrase-multilingual-MiniLM-L12-v2` 为 384）

- **translation**: EN→ZH 翻译配置（用于将英文描述稳定翻译为中文）

  - `enabled`: 是否启用翻译
  - `model`: 推荐 `Helsinki-NLP/opus-mt-en-zh`，支持本地路径
  - `batch_size`: 批量翻译时单批大小

- **deduplication**: 翻译后去重配置（在中文内容上进行去重和质量检测）

  - `approximate_enabled`: 是否启用近似去重（基于向量相似度，默认：true）
  - `similarity_threshold`: 相似度阈值，超过此值认为是重复内容（默认：0.95）
  - `embed_model`: 用于近似去重的嵌入模型（默认：sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2）
  - `forbid_repeat_ngram`: 禁止重复的 n-gram 长度，用于检测单文本内部的重复片段（默认：3）

- **filters**: 英文阶段过滤配置（仅用于英文内容的基础过滤）
  - `min_length`: 最小长度阈值（默认：5）
  - `max_length`: 最大长度阈值（默认：120）
  - `max_gibberish_ratio`: 最大乱码占比（默认：0.3）
  - `blacklist_keywords`: 黑名单关键词列表

#### 4. 分块参数配置 (chunking)

位置：`media_processing.chunking`

- **auto**: 是否启用自适应分块。true=依据内容与目标自动估算；false=按表驱动。
- **defaults**: 通用基线 `chunk_size/overlap`，类型未命中或用于自适应上下限保护时使用。
- **by_media_type**: 按媒体类型细化的基线参数（text/pdf/image/video/audio/code）。
- **targets**:
  - `target_tokens_per_chunk`: 期望的每块 token 目标（近似）。
  - `overlap_ratio`: 相邻块重叠比例（0~1）。
  - `max_context_tokens`: 模型上下文窗口上限（一般取 context 的 1/4 作为 chunk 上限）。

说明：分块参数由 `app/core/chunking.py` 的 `decide_chunk_params()` 统一计算，管道端只接收计算结果并创建具体的分块策略。

#### 2. OCR 配置 (ocr)

```json
{
  "media_processing": {
    "ocr": {
      "engines": ["easyocr", "paddleocr", "tesseract"],
      "languages": ["ch_sim", "en"],
      "confidence_threshold": 0.5
    }
  }
}
```

**配置参数说明**：

- **engines**: OCR 引擎优先级列表
  - `easyocr`: EasyOCR 引擎（推荐，支持多语言）
  - `paddleocr`: PaddleOCR 引擎（中文优化）
  - `tesseract`: Tesseract 引擎（传统 OCR）
- **languages**: 支持的语言列表
  - `ch_sim`: 简体中文
  - `en`: 英文
  - `ch`: 中文（通用）
- **confidence_threshold**: 置信度阈值（0.0-1.0）
  - 低于此阈值的识别结果将被过滤

#### 3. 视频处理配置 (video_processing)

```json
{
  "media_processing": {
    "video_processing": {
      "speech_recognition": {
        "model": "whisper-base",
        "language": "zh-CN"
      }
    }
  }
}
```

**配置参数说明**：

- **speech_recognition**: 语音识别配置
  - `model`: Whisper 模型大小（base, small, medium, large）
  - `language`: 识别语言（zh-CN, en, 等）

### 使用方法

#### 1. 在代码中使用配置（提示词由配置控制）

```python
from app.config import get_config_manager
from app.rag.processing.media_extractors import ImageVisionExtractor

# 获取配置管理器
config_manager = get_config_manager()

# 获取图像描述配置
caption_config = config_manager.get_image_captioning_config()

# 创建提取器
extractor = ImageVisionExtractor()

result = extractor.extract(image_bytes, {"media_type": "image"})
```

#### 3. 分块参数的使用示例（文本/PDF/媒体）

```python
from app.rag.processing.utils.chunking import decide_chunk_params

media_type = "pdf"
content = large_pdf_plain_text  # 经过提取后的可切分文本
meta = {"language": "zh"}

chunk_size, overlap = decide_chunk_params(media_type, content, meta)
# 然后将参数传入具体的 chunking 策略
```

#### 2. 控制输出风格/语言

通过配置文件中的 `image_captioning.prompt` 控制，无需在代码中传 `meta`。运行时修改提示词需重载配置。
注意：当前实现使用 `image-to-text` 管线，部分模型（如 BLIP）不支持指令，`prompt` 将被忽略。

#### 3. 模型与语言策略

- 图像字幕建议使用 `Salesforce/blip-image-captioning-base` 通过 `image-to-text` 管线生成英文描述。
- 若需要中文向量检索，建议在应用层对英文描述进行稳定翻译（EN→ZH），再用多语言嵌入模型生成向量。
- 检索查询亦建议统一语种（例如全部中文），必要时对查询做相同翻译归一化。

#### 4. 后端依赖与建议

- `image-to-text`: 依赖 `transformers`, `torch`, `pillow`

#### 5. 控制生成数量

要生成多个描述，修改配置文件中的`num_captions`参数：

```json
{
  "generation": {
    "num_captions": 3 // 生成3个描述
  }
}
```

系统会在翻译后自动进行去重处理，无需手动配置。

### 新的处理流程说明

图像描述处理现在采用"翻译后统一去重"的策略：

1. **英文生成阶段**：生成英文描述，仅进行基础过滤（长度、黑名单、乱码）
2. **翻译阶段**：将英文描述翻译为中文
3. **中文去重阶段**：在中文内容上进行：
   - 重复片段检测（`forbid_repeat_ngram`）
   - 精确去重（MD5）
   - 近似去重（向量相似度）

这种策略的优势：

- **避免重复工作**：不在英文和中文上都做去重
- **最终质量保证**：确保最终输出的中文内容质量
- **逻辑一致性**：与翻译后去重的策略保持一致

## 当前处理阶段产物与下一步

- 处理阶段产物：`ProcessingPipeline` 将 `RawSample` 规范化并输出 `ProcessedChunk(text, meta)` 列表（在媒体场景下会先经 `MediaExtractor` → 文本/字幕，再进入分块策略）。
- 向量化状态：当前仓库尚未在处理阶段内直接完成“向量化”；向量数据库访问层已具备（`app/core/chroma/db_helper.py` 提供 Chroma 连接与集合操作），但需要在“向量化层”补充：
  1. 选择/配置嵌入模型（推荐多语言 `sentence-transformers` 系列）；
  2. 为每个 `ProcessedChunk.text` 生成向量；
  3. 调用 `ChromaDBHelper.add(collection_name, documents, embeddings, metadatas, ids)` 入库；
  4. 统一 collection 命名与字段规范（如 `doc_id`, `chunk_index`, `media_type`）。

### 向量化层开发建议（下一步）

- 目录：`app/rag/vectorization/`（建议新建）
  - `embedder.py`：封装嵌入模型加载与批量编码（支持本地/云端模型，带重试与批处理）；
  - `indexer.py`：将 `ProcessedChunk` 批量写入 Chroma；
  - `config.py`：向量化配置（模型名、批大小、并发、重试策略、集合命名规则）。
- 简要流程：`chunks = ProcessingPipeline.run(samples)` → `embeddings = Embedder.encode([c.text for c in chunks])` → `ChromaDBHelper.add(...)`。
- 回收策略：对空文本、低质量片段（由 `quality_checker` 标记）可跳过或降权；重入时做 `id`/`md5` 去重。

---

### 常见问题

#### Q: 为什么生成了重复的描述？

A: 这是 BLIP 模型的正常行为。系统会在翻译后自动进行去重处理：

1. 翻译后的中文内容会进行精确去重和近似去重
2. 调整`temperature`参数增加随机性
3. 设置`num_captions: 1`只生成一个描述

#### Q: 如何让模型生成中文描述？

A: 使用`chinese_friendly`模型类型：

```python
meta = {
    "media_type": "image",
    "caption_model_type": "chinese_friendly"  # 会使用中文提示词
}
```

#### Q: 如何提高 OCR 识别准确率？

A: 可以尝试以下方法：

1. 调整`confidence_threshold`参数
2. 修改`engines`顺序，优先使用更准确的引擎
3. 根据文档语言调整`languages`配置

#### Q: 如何禁用某些功能？

A: 在配置文件中设置相应的`enabled`参数为`false`：

```json
{
  "image_captioning": {
    "enabled": false // 禁用图像描述
  },
  "embedding": {
    "enabled": false // 禁用嵌入向量生成
  }
}
```

### 性能优化建议

1. **模型选择**: 根据需求选择合适的模型大小
2. **批量处理**: 对于大量图像，考虑批量处理
3. **缓存**: 对于相同图像，可以缓存结果
4. **并发**: 可以并行处理多个图像

### 依赖库要求

确保安装以下依赖库：

```bash
# 图像描述
pip install transformers torch pillow

# OCR
pip install easyocr paddleocr pytesseract opencv-python

# 嵌入向量
pip install sentence-transformers

# 视频处理
pip install openai-whisper speechrecognition
```
