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
      "model": "Salesforce/blip-image-captioning-base",
      "backend": "image-to-text",
      "prompt": "Describe this image briefly:",
      "generation": {
        "num_captions": 1,
        "max_length": 50,
        "temperature": 0.7,
        "do_sample": true,
        "remove_duplicates": true
      },
      "embedding": {
        "enabled": true,
        "model": "clip-ViT-B-32",
        "dimension": 512
      }
    }
  }
}
```

**配置参数说明**：

- **enabled**: 是否启用图像描述功能
- **model**: 使用的图像描述模型（单一模型）
- **backend**: 字幕后端选择，提升可读性与可控性
  - `image-to-text`: 传统图像字幕管线（如 BLIP），忽略 prompt
  - `qwen-vl-chat`: 多模态对话模型（如 Qwen/Qwen-VL-Chat），使用 prompt 作为指令
- **prompt**: 描述生成提示词（完全由配置决定，可写中文或英文），例如：
  - 中文：`"请用中文简洁描述这张图片："`
  - 英文：`"Describe this image in concise English:"`
- **generation**: 生成参数
  - `num_captions`: 生成描述的数量（默认：1）
  - `max_length`: 描述的最大长度（默认：50）
  - `temperature`: 生成温度，控制随机性（默认：0.7）
  - `do_sample`: 是否使用采样（默认：true）
  - `remove_duplicates`: 是否移除重复描述（默认：true）
- **embedding**: 嵌入向量配置
  - `enabled`: 是否生成嵌入向量（默认：true）
  - `model`: 嵌入模型名称（默认：clip-ViT-B-32）
  - `dimension`: 向量维度（默认：512）

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

#### 2. 控制输出风格/语言

通过配置文件中的 `image_captioning.prompt` 控制，无需在代码中传 `meta`。运行时修改提示词需重载配置。
当 `backend` 为 `image-to-text` 时，部分模型（如 BLIP）不支持指令，`prompt` 将被忽略；当 `backend` 为 `qwen-vl-chat` 时，`prompt` 会作为文本指令参与生成。

#### 3. 如何选择 model（与 backend 的对应关系）

- 想用 BLIP 等“图像字幕”模型：

  - 配置示例：
    ```json
    {
      "media_processing": {
        "image_captioning": {
          "backend": "image-to-text",
          "model": "Salesforce/blip-image-captioning-base"
        }
      }
    }
    ```
  - 说明：此类模型经 `pipeline("image-to-text")` 调用，通常默认产出英文，`prompt` 多数会被忽略。

- 想用 Qwen 的“多模态对话”模型：
  - 配置示例：
    ```json
    {
      "media_processing": {
        "image_captioning": {
          "backend": "qwen-vl-chat",
          "model": "Qwen/Qwen-VL-Chat"
        }
      }
    }
    ```
  - 说明：必须把 `model` 配成 Qwen 家族（名称中包含 `qwen` 与 `vl` 的模型），否则将回退到 `image-to-text` 后端；该后端支持 `prompt`，可直接中文指令生成中文描述。

#### 4. 后端依赖与建议

- `image-to-text`: 依赖 `transformers`, `torch`, `pillow`
- `qwen-vl-chat`: 需较新 `transformers` 且开启 `trust_remote_code=True`，显卡上建议 bfloat16/float16 与 `device_map="auto"`

#### 5. 控制生成数量

要生成多个描述，修改配置文件中的`num_captions`参数：

```json
{
  "generation": {
    "num_captions": 3, // 生成3个描述
    "remove_duplicates": true // 移除重复描述
  }
}
```

### 常见问题

#### Q: 为什么生成了重复的描述？

A: 这是 BLIP 模型的正常行为。可以通过以下方式解决：

1. 设置`remove_duplicates: true`（默认已启用）
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
