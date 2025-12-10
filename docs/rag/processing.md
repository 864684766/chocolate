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
  media/                      # 媒体处理模块
    extractors/               # 媒体内容提取器（文本、图像、视频、音频、PDF、Word、Excel）
    chunking/                 # 媒体分块策略（文本、PDF、Word、Excel、图像、视频、音频）
  utils/                      # 工具模块（文本清洗、分块参数决策等）
  config_usage_example.py     # 配置使用示例
```

说明：

- 语言与媒体解耦在不同目录，通过统一接口注入到管道。
- 语言选择：运行时通过语言检测（fasttext/pycld3）或外部标注，选择 `lang/*` 下对应实现。

## Text Splitters 与多媒体

- 文本：使用 LangChain Text Splitters（Recursive/Token）
- PDF/图片/音视频：先在 `media/*` 转成"可切分文本"（正文/表格序列化/字幕），再交给 Text Splitters。

## 统一接口

- `MediaExtractor`：extract(bytes, meta)->{"text": str, "meta": {...}} 或媒体特定结构
- `MediaChunkingStrategy`：chunk(content, meta)->List[Dict[str, Any]]
- `QualityAssessor`：score(text, meta)->Dict[str, Any]

## 元数据管理

系统现在使用统一的 `MetadataManager` 来管理所有元数据操作，包括基础元数据生成、内容分析、质量评估等。

### MetadataManager 功能

`MetadataManager` 负责以下元数据操作：

1. **基础元数据生成**：`doc_id`, `filename`, `source`, `created_at`, `media_type`
2. **内容分析元数据**：`lang`（语言检测）, `text_len`, `keyphrases`
3. **质量评估元数据**：`quality_score`
4. **元数据规范化**：白名单过滤、类型转换

### 统一配置结构

所有元数据相关配置现在统一在 `metadata` 配置节中：

```json
{
  "metadata": {
    "default_language": "zh",
    "language_detection": {
      "enabled": true,
      "chinese_threshold": 0.3,
      "min_text_length": 10,
      "fallback_language": "zh"
    },
    "media_type_mapping": {
      "by_extension": {
        "pdf": "pdf",
        "doc": "document",
        "docx": "document",
        "txt": "text",
        "md": "text",
        "png": "image",
        "jpg": "image",
        "jpeg": "image",
        "gif": "image",
        "bmp": "image",
        "webp": "image",
        "mp4": "video",
        "avi": "video",
        "mov": "video",
        "mkv": "video",
        "mp3": "audio",
        "wav": "audio",
        "flac": "audio"
      },
      "by_content_type": {
        "image/": "image",
        "video/": "video",
        "audio/": "audio",
        "application/pdf": "pdf",
        "text/": "text"
      },
      "default": "text"
    },
    "required_fields": {
      "source": "unknown",
      "media_type": "text"
    },
    "keywords": {
      "method": "textrank",
      "topk": 10,
      "lang": "auto",
      "stopwords": { "path": "D:/test/chocolate/docs/stopWord.txt" }
    },
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

### 配置说明

**基础配置**：

- `default_language`: 默认语言代码
- `required_fields`: 必需字段的默认值

**语言检测配置**：

- `chinese_threshold`: 中文字符比例阈值
  - 含义：按实现，统计文本中中文字符数量与有效字符数量（字母[a-zA-Z]或中文字符[\u4e00-\u9fff]）之比；当该比例 > chinese_threshold 时判定为中文（"zh"），否则为英文（"en"）。
  - 取值范围：0.0 ~ 1.0
  - 当前实现要点（与代码对齐）：
    - 短文本保护：当 `len(text.strip()) < 10` 时，不进行比例计算，直接返回 `default_language`（当前默认 `zh`）。注意：这一阈值目前在代码中是固定值 10，尚未读取配置。
    - 空/无效文本处理：当有效字符总数为 0 时，直接返回 `default_language`。
    - 有效字符定义：仅统计字母与中文字符；标点、空白、表情等不计入有效字符总数。
  - 推荐值：
    - 中文为主的数据集：0.20 ~ 0.35（建议默认 0.30）
    - 中英均衡或混排内容：0.30 ~ 0.50
    - 英文为主的数据集：0.40 ~ 0.60
  - 调参建议：
    - 误判为英文较多：适当下调阈值（例：0.30 → 0.25）
    - 误判为中文较多：适当上调阈值（例：0.30 → 0.40）
    - 标题/短句等极短文本：可考虑后续将短文本阈值改为配置项，或在上游对短文本跳过检测
- `min_text_length`: 最小文本长度
  - 说明：已由代码读取并生效（`MetadataManager._detect_language`）。当文本长度小于该值时直接返回回退语言。
- `fallback_language`: 检测失败时的回退语言
  - 说明：已由代码读取并生效。用于短文本或有效字符为 0 的回退；若缺省则回退到 `default_language`。

**媒体类型映射配置**：

- `by_extension`: 基于文件扩展名的映射
- `by_content_type`: 基于内容类型的映射
- `default`: 默认媒体类型

**关键词提取配置**：

- `method`: 提取方法（textrank/light/keybert/none）
- `topk`: 最大关键词数量
- `lang`: 处理语言
- `stopwords.path`: 停用词文件路径

**质量评估配置**：

- `min_len/sat_len`: 长度阈值
- `weights`: 各维度权重
- `observability`: 观测和日志配置

### 使用示例

```python
from app.rag.processing.metadata_manager import MetadataManager

# 创建元数据管理器
manager = MetadataManager()

# 创建完整元数据
meta = manager.create_metadata(
    text="这是测试文本",
    filename="test.pdf",
    content_type="application/pdf",
    source="manual_upload"
)

# 更新内容元数据
updated_meta = manager.update_content_metadata(
    "更新后的文本内容",
    meta
)
```

### 元数据白名单字段（通用+媒体特定）

当前所有媒体类型共用一份白名单，并按照字段语义分为三层。即使某些字段在特定媒体中暂时没有值，也会在入库之前被补为默认值（字符串 → 空串，数字 →0）以保持结构统一。

**通用文档字段（任何媒体都应具备）：**

- `doc_id`/`source`/`filename`/`content_type`/`media_type`/`created_at`
- `lang`/`text_len`/`keyphrases`/`tags`
- `quality_score`

**通用分块字段（所有 chunk 必备）：**

- `chunk_index`/`chunk_type`/`chunk_size`/`total_chunks`

**媒体特定字段：**

- PDF/Office：`page_number`
- 视频/音频：`start_time`、`end_time`
- 图像 Caption：`caption_count`
- 图像 OCR：`region_index`

在 `config/app_config.json > metadata.metadata_whitelist` 中可以新增或调整字段，流程中的生成器会自动跟进；检索侧（Meilisearch/Chroma `where` 过滤）也会基于同一份白名单进行校验。

## 质量评估配置

质量评估器 (`SimpleQualityAssessor`) 通过配置文件进行参数调整，配置位置：`metadata.quality`

### 配置结构

```json
{
  "metadata": {
    "quality": {
      "min_len": 20,
      "sat_len": 200,
      "weights": {
        "garbled": 0.4,
        "valid": 0.2,
        "length": 0.2,
        "ocr": 0.2
      },
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

### 配置参数说明

**基础参数**：

- `min_len`: 最小文本长度阈值（字符数），低于此长度的文本长度得分为 0.0
- `sat_len`: 满意文本长度阈值（字符数），达到或超过此长度的文本长度得分为 1.0

**权重配置**：

- `garbled`: 乱码得分权重，基于不可打印字符比例计算
- `valid`: 有效字符比例权重，基于字母数字和标点符号比例计算
- `length`: 长度得分权重，基于文本长度计算
- `ocr`: OCR 置信度权重，基于 OCR 识别的平均置信度计算
- 注意：当 OCR 置信度缺失时，权重会自动归一化到其他三个维度

**观测配置**：

- `enabled`: 是否启用质量观测功能
- `threshold`: 质量得分阈值，低于此值的样本会被记录日志
- `alert_ratio`: 告警比例阈值（当前未使用，预留给监控系统）
- `sample_rate`: 采样率，控制低质量样本的日志记录频率（0.01 = 1%）

### 使用示例

```python
from app.rag.processing.quality_checker import SimpleQualityAssessor

# 使用默认配置
assessor = SimpleQualityAssessor()

# 评估文本质量
result = assessor.score("这是一段测试文本", {"source": "test"})
print(f"质量得分: {result['quality_score']}")
```

## 分块参数决策（新）

为适配不同媒体与上下文窗口，已将"分块参数决策"独立为处理工具模块：`app/rag/processing/utils/chunking.py`。

- 入口函数：`decide_chunk_params(media_type: str, content: Any) -> (chunk_size, overlap)`
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

chunk_size, overlap = decide_chunk_params(str(media_type), content)
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

### 文本分块策略配置

文本分块策略 (`TextChunkingStrategy`) 支持通过配置文件进行参数调整：

#### 配置结构

```json
{
  "language_processing": {
    "chinese": {
      "chunking": {
        "default_chunk_size": 800,
        "default_overlap": 150,
        "separators": ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
      },
      "sentence_splitting": {
        "sentence_endings_pattern": "[。！？；]"
      }
    }
  },
  "processing": {
    "text_cleaning": {
      "normalize_whitespace_pattern": "[ \\t]+",
      "normalize_whitespace_replacement": " ",
      "normalize_newlines_pattern": "\\n\\s*\\n\\s*\\n+",
      "normalize_newlines_replacement": "\n\n"
    }
  }
}
```

#### 配置参数说明

**chunking 分块配置**（`language_processing.chinese.chunking`）：

- `default_chunk_size`: 默认分块大小（字符数）
- `default_overlap`: 默认重叠大小（字符数）
- `separators`: 分块分隔符优先级列表，按优先级从高到低排列
  - 该配置被 `TextChunkingStrategy` 使用（LangChain 智能分块）
  - 如果未配置，将使用默认分隔符列表（段落、行、中文标点、空格、字符级别）

**sentence_splitting 句子分割配置**（`language_processing.chinese.sentence_splitting`）：

- `sentence_endings_pattern`: 中文句子结束标点符号正则表达式
  - 用于回退方案（当 LangChain 不可用时）的语义边界分块

**text_cleaning 文本清洗配置**（`processing.text_cleaning`）：

- `normalize_whitespace_pattern`: 空白字符归一化正则表达式
- `normalize_whitespace_replacement`: 空白字符替换字符串
- `normalize_newlines_pattern`: 换行符归一化正则表达式
- `normalize_newlines_replacement`: 换行符替换字符串

#### 使用示例

```python
from app.rag.processing.media.chunking.text import TextChunkingStrategy

# 使用默认配置
strategy = TextChunkingStrategy()

# 覆盖特定参数
strategy = TextChunkingStrategy(chunk_size=1000, overlap=200)

# 分块处理
meta = {"media_type": "text", "source": "test"}
chunks = strategy.chunk("这是测试文本内容", meta)
```

#### 文本分块策略说明

`TextChunkingStrategy` 提供两种分块方式：

1. **LangChain 智能分块**（优先）：

   - 使用 `langchain-text-splitters` 的 `RecursiveCharacterTextSplitter`
   - 支持中文友好的分隔符优先级
   - 自动处理重叠

2. **语义边界分块**（回退方案）：
   - 当 LangChain 不可用时自动回退
   - 按段落和句子边界分割
   - 保留重叠机制，确保上下文连续性

### Office 文档分块策略配置

Office 文档（PDF、Word、Excel）的分块策略基于 `OfficeDocumentChunkingStrategyBase` 基类，提供统一的表格转换和文本分块功能。

#### 表格转换策略（自然语言格式）

系统采用**自然语言连接符**方式将表格转换为文本，以优化语义搜索效果：

**转换规则**：

1. **表头检测**：

   - 自动检测第一行是否为表头（启发式：第一行平均长度 < 20 且后续有数据行）
   - 如果检测到表头，则分离表头和数据行

2. **有表头的表格转换**：

   - 格式：`"列名是值，列名是值"`
   - 示例：
     ```
     原文是信息的特征，速记词是扑克舞动，解释是想象一张扑克
     ```
   - 优势：更符合自然语言，对语义搜索和向量检索更友好

3. **无表头的表格转换**：
   - 格式：使用空格分隔单元格
   - 过滤空单元格，保留有效内容

**实现位置**：

- 基类：`app/rag/processing/media/chunking/office/base.py`
- 方法：`_convert_table_to_text()`、`_detect_table_header()`、`_convert_table_with_headers()`、`_convert_table_simple()`

**使用示例**：

```python
from app.rag.processing.media.chunking.office.base import OfficeDocumentChunkingStrategyBase

# 表格数据
table = [
    ["原文", "速记词", "解释"],
    ["信息的特征", "扑克舞动", "想象一张扑克"],
    ["另一个例子", "记忆宫殿", "构建空间记忆"]
]

# 转换为自然语言文本
text = OfficeDocumentChunkingStrategyBase._convert_table_to_text(table)
# 输出：
# 原文是信息的特征，速记词是扑克舞动，解释是想象一张扑克
# 原文是另一个例子，速记词是记忆宫殿，解释是构建空间记忆
```

**优势说明**：

- ✅ **语义友好**：使用自然语言连接符，查询"原文是什么"时能更好匹配
- ✅ **检索优化**：对向量检索和关键词检索都有帮助
- ✅ **自动检测**：无需手动配置表头，系统自动识别
- ✅ **灵活处理**：同时支持有表头和无表头的表格

#### 分块策略说明

**PDF 分块策略** (`PDFChunkingStrategy`)：

- 按页面分块，保留 `page_number` 元数据
- 表格单独分块，使用自然语言格式转换
- 支持连续的分块索引

**Word 分块策略** (`WordChunkingStrategy`)：

- 段落分块：添加 `word_paragraph` 类型和 `paragraph_index` 元数据
- 表格分块：添加 `word_table` 类型和 `table_index` 元数据
- 长段落自动分割

**Excel 分块策略** (`ExcelChunkingStrategy`)：

- 按工作表分块，每个工作表单独处理
- 添加 `excel_sheet` 类型和 `sheet_index`/`sheet_name` 元数据
- 支持 `.xlsx`、`.xls`、`.csv` 格式

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

#### 3. 语音识别配置 (speech_recognition)

```json
{
  "media_processing": {
    "temp_dir": "D:/test/chocolate/temp",
    "speech_recognition": {
      "model": "whisper-base",
      "description": "语音识别配置，供视频和音频共用。Whisper 会自动检测语言，无需手动指定。"
    },
    "video_processing": {
      "description": "视频处理配置，包含视频特有的处理选项"
    }
  }
}
```

**配置参数说明**：

- **temp_dir**: 临时文件目录路径（必填）
  - 供视频和音频处理共用，用于创建临时文件进行语音识别
  - 示例：Windows: `"D:/test/chocolate/temp"`，Linux: `"/tmp/chocolate"` 或 `"~/chocolate/temp"`
- **speech_recognition**: 语音识别配置（供视频和音频共用）

  - `model`: Whisper 模型大小（base, small, medium, large），支持 "whisper-" 前缀或直接使用模型名
  - **自动语言检测**: Whisper 会自动检测音频/视频中的语言，无需手动配置 `language` 参数
  - 支持 99 种语言的自动识别，准确率高

- **video_processing**: 视频处理配置
  - 包含视频特有的处理选项（当前主要为占位，未来可扩展）

**视频和音频提取逻辑说明**：

系统现在提供了专门的提取器来处理视频和音频：

1. **视频内容提取器** (`VideoContentExtractor`)：

   - **字幕提取**：
     - 优先使用 `ffmpeg-python` 提取视频内嵌字幕（支持 SRT、VTT、ASS、SSA 等格式）
     - 如果没有内嵌字幕，使用 Whisper 生成带时间戳的字幕
     - 返回格式：`List[Dict[str, Any]]`，每个字典包含 `text`、`start_time`、`end_time`
   - **语音转录**：
     - 优先使用 Whisper 进行语音识别（自动语言检测）
     - 如果 Whisper 失败，回退到 SpeechRecognition
     - 返回格式：`str`，完整的语音转录文本
   - 返回格式：`{"subtitles": [...], "transcript": "..."}`

2. **音频内容提取器** (`AudioContentExtractor`)：

   - **语音转录**：
     - 优先使用 Whisper 进行语音识别（自动语言检测）
     - 如果 Whisper 失败，回退到 SpeechRecognition
     - 返回格式：`str`，完整的语音转录文本
   - 返回格式：`{"transcript": "..."}`（音频没有字幕）

3. **自动语言检测**：

   - Whisper 支持自动检测 99 种语言，无需手动配置
   - 检测到的语言会在日志中记录，便于调试和监控
   - 如果使用 SpeechRecognition（备选方案），则使用固定默认值 `zh-CN`

4. **异常处理**：
   - 所有方法都包含完整的异常处理和日志记录
   - 确保异常时能正确回退到备选方案
   - 临时文件会在 `finally` 块中确保清理

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

chunk_size, overlap = decide_chunk_params(media_type, content)
# 然后将参数传入具体的 chunking 策略
```

#### 2. 控制输出风格/语言

通过配置文件中的 `image_captioning.prompt` 控制，无需在代码中传入额外的类型标记。运行时修改提示词需重载配置。
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
    "num_captions": 3
  }
}
```

系统会在翻译后自动进行去重处理，无需在代码里额外传参。

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

A: 通过在配置中设置 `image_captioning.translation.enabled=true` 使用统一的 EN→ZH 翻译策略，或直接选择支持中文输出的图像描述模型；无需在代码中传入 `caption_model_type` 之类的标记字段。

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
