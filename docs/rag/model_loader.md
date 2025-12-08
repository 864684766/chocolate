# 通用模型加载器

## 概述

通用模型加载器提供统一的模型加载和缓存机制，支持多种模型类型（SentenceTransformer、Transformers、Whisper、CLIP、CrossEncoder 等）。该模块是 RAG 系统中处理各种 AI 模型的基础设施，确保模型加载的高效性和一致性。

## 特性

- ✅ **统一接口**：所有模型通过相同的接口加载
- ✅ **自动缓存**：模块级缓存，避免重复加载
- ✅ **线程安全**：使用锁保证并发安全
- ✅ **LRU 策略**：自动管理缓存大小，删除最旧的模型
- ✅ **可扩展**：支持注册自定义加载器
- ✅ **设备管理**：自动检测和配置设备（CPU/GPU）

## 架构设计

### 文件结构

每个模型类型的加载器都单独放在一个文件中，便于维护和扩展：

```
app/infra/models/
├── __init__.py              # 导出核心组件
├── registry.py              # 注册默认加载器
├── examples.py              # 使用示例
└── loaders/
    ├── __init__.py              # 导出所有加载器类
    ├── base.py                  # 基础类和核心加载逻辑
    ├── sentence_transformer.py  # SentenceTransformer 模型加载器
    ├── transformers.py          # Transformers 模型加载器
    ├── whisper.py               # Whisper 模型加载器
    ├── clip.py                  # CLIP 模型加载器
    └── cross_encoder.py         # CrossEncoder 模型加载器
```

### 设计优势

- ✅ **单一职责**：每个文件只负责一种模型类型
- ✅ **易于维护**：修改某个加载器不影响其他文件
- ✅ **易于扩展**：添加新加载器只需创建新文件
- ✅ **代码清晰**：文件结构一目了然

## 使用方法

### 基本用法

```python
from app.infra.models import ModelLoader, ModelType, LoaderConfig

# 加载 SentenceTransformer 模型
config = LoaderConfig(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device="auto",
    model_type=ModelType.SENTENCE_TRANSFORMER
)
model = ModelLoader.load_model(config)

# 加载 Transformers 模型（带 Tokenizer）
config = LoaderConfig(
    model_name="Qwen/Qwen3-0.6B",
    device="cpu",
    model_type=ModelType.TRANSFORMERS,
    model_class="AutoModelForCausalLM",
    tokenizer_class="AutoTokenizer",
    return_tokenizer=True,
    torch_dtype="float16"
)
model, tokenizer = ModelLoader.load_model(config)

# 加载 Whisper 模型
config = LoaderConfig(
    model_name="base",
    model_type=ModelType.WHISPER
)
model = ModelLoader.load_model(config)

# 加载 CLIP 模型
config = LoaderConfig(
    model_name="openai/clip-vit-base-patch32",
    model_type=ModelType.CLIP
)
clip_model, processor = ModelLoader.load_model(config)

# 加载 CrossEncoder 模型
config = LoaderConfig(
    model_name="BAAI/bge-reranker-base",
    model_type=ModelType.CROSS_ENCODER
)
model = ModelLoader.load_model(config)
```

### 缓存管理

```python
# 获取缓存信息
info = ModelLoader.get_cache_info()
print(f"缓存大小: {info['cache_size']}/{info['max_cache_size']}")

# 清除所有缓存
ModelLoader.clear_cache()

# 移除特定模型
config = LoaderConfig(model_name="...", model_type=ModelType.SENTENCE_TRANSFORMER)
ModelLoader.remove_from_cache(config)
```

### 注册自定义加载器

如果需要添加新的模型加载器，按以下步骤操作：

#### 1. 创建新的加载器文件

例如：`loaders/stable_diffusion.py`

```python
"""
Stable Diffusion 模型加载器

用处：加载 Stable Diffusion 图像生成模型。
"""

from __future__ import annotations

import logging
from typing import Any

from .base import LoaderConfig, ModelLoaderBase, ModelType

logger = logging.getLogger(__name__)


class StableDiffusionLoader(ModelLoaderBase):
    """Stable Diffusion 模型加载器"""

    def load(self, config: LoaderConfig) -> Any:
        """
        加载 Stable Diffusion 模型

        参数:
            config: 加载器配置对象，包含模型名称、设备等参数

        返回:
            加载的 Stable Diffusion 模型管道对象

        异常:
            RuntimeError: 当 diffusers 库未安装或模型加载失败时抛出
        """
        try:
            from diffusers import StableDiffusionPipeline
            return StableDiffusionPipeline.from_pretrained(config.model_name)
        except ImportError:
            raise RuntimeError("diffusers 库未安装，请安装: pip install diffusers")
        except Exception as e:
            raise RuntimeError(f"加载 Stable Diffusion 模型失败: {e}")

    def get_supported_type(self) -> ModelType:
        """
        返回该加载器支持的模型类型

        返回:
            ModelType.STABLE_DIFFUSION (需要在 ModelType 枚举中添加)
        """
        return ModelType.STABLE_DIFFUSION
```

#### 2. 在 `__init__.py` 中导出

```python
from .stable_diffusion import StableDiffusionLoader

__all__ = [
    # ... 其他加载器
    "StableDiffusionLoader",
]
```

#### 3. 在 `registry.py` 中注册

```python
from .loaders import (
    # ... 其他加载器
    StableDiffusionLoader,
)

def register_default_loaders() -> None:
    # ... 其他注册
    ModelLoader.register_loader(ModelType.STABLE_DIFFUSION, StableDiffusionLoader)
```

#### 4. 使用自定义加载器

```python
from app.infra.models import ModelLoader, ModelType, ModelLoaderBase, LoaderConfig

# 注册自定义加载器（通常在应用启动时完成）
ModelLoader.register_loader(ModelType.CUSTOM, MyCustomLoader)

# 使用自定义加载器
config = LoaderConfig(
    model_name="my-model",
    model_type=ModelType.CUSTOM
)
model = ModelLoader.load_model(config)
```

## 支持的模型类型

- `SENTENCE_TRANSFORMER`: sentence-transformers 模型（用于文本嵌入）
- `TRANSFORMERS`: HuggingFace Transformers 模型（用于 LLM 推理）
- `WHISPER`: OpenAI Whisper 模型（用于语音转文字）
- `CLIP`: CLIP 模型和处理器（用于图像理解）
- `CROSS_ENCODER`: CrossEncoder 模型（用于重排序）
- `CUSTOM`: 自定义模型（需要注册加载器）

## 配置

缓存大小通过 `config/app_config.json` 中的 `cache.max_cache_size` 配置，默认为 10。

```json
{
  "cache": {
    "max_cache_size": 10
  }
}
```

## 线程安全

所有缓存操作都是线程安全的，可以在多线程环境中安全使用。内部使用 `threading.Lock` 确保并发访问的安全性。

## 在 RAG 系统中的应用

模型加载器在 RAG 系统的多个环节中发挥作用：

1. **向量化层**：加载 SentenceTransformer 模型生成文本嵌入
2. **数据处理层**：加载 Whisper 模型处理音频/视频转文字
3. **检索层**：加载 CrossEncoder 模型进行重排序
4. **应用层**：加载 Transformers 模型进行 LLM 推理

通过统一的模型加载接口，确保了整个 RAG 系统在模型管理上的一致性和高效性。
