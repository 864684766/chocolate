"""
模型加载器使用示例

展示如何在实际代码中使用通用模型加载器替换现有的模型加载逻辑。
"""

from typing import Any, Tuple
from .loaders import ModelLoader, ModelType, LoaderConfig


# ========== 示例 1: 替换 Embedder 中的模型加载 ==========
def load_embedding_model_example(model_name: str, device: str = "auto") -> Any:
    """加载向量化模型（替换 Embedder._load_model）
    
    用处：使用通用模型加载器加载 SentenceTransformer 模型，自动处理缓存。
    
    Args:
        model_name: 模型名称或路径
        device: 设备类型
        
    Returns:
        SentenceTransformer 模型实例
    """
    config = LoaderConfig(
        model_name=model_name,
        device=device,
        model_type=ModelType.SENTENCE_TRANSFORMER
    )
    return ModelLoader.load_model(config)


# ========== 示例 2: 替换 quality_utils 中的模型加载 ==========
def load_similarity_model_example(model_name: str = None) -> Any:
    """加载相似度计算模型（替换 near_duplicate 中的模型加载）
    
    用处：使用通用模型加载器加载 SentenceTransformer 模型用于相似度计算。
    
    Args:
        model_name: 模型名称，默认为 None（使用默认模型）
        
    Returns:
        SentenceTransformer 模型实例
    """
    default_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    config = LoaderConfig(
        model_name=model_name or default_model,
        device="auto",
        model_type=ModelType.SENTENCE_TRANSFORMER
    )
    return ModelLoader.load_model(config)


def load_clip_model_example(model_name: str) -> Tuple[Any, Any]:
    """加载 CLIP 模型（替换 clip_rerank 中的模型加载）
    
    用处：使用通用模型加载器加载 CLIP 模型和处理器。
    
    Args:
        model_name: CLIP 模型名称
        
    Returns:
        (CLIPModel, CLIPProcessor) 元组
    """
    config = LoaderConfig(
        model_name=model_name,
        device="auto",
        model_type=ModelType.CLIP
    )
    return ModelLoader.load_model(config)


# ========== 示例 3: 替换 reranker 中的模型加载 ==========
def load_reranker_model_example(model_name: str) -> Any:
    """加载重排模型（替换 CrossEncoderReranker._load_model）
    
    用处：使用通用模型加载器加载 CrossEncoder 模型用于检索结果重排。
    
    Args:
        model_name: CrossEncoder 模型名称或路径
        
    Returns:
        CrossEncoder 模型实例
    """
    config = LoaderConfig(
        model_name=model_name,
        device="auto",
        model_type=ModelType.CROSS_ENCODER
    )
    return ModelLoader.load_model(config)


# ========== 示例 4: 替换 Whisper 模型加载 ==========
def load_whisper_model_example(model_name: str = "base") -> Any:
    """加载 Whisper 模型（替换 video.py 和 subtitle_helper.py 中的模型加载）
    
    用处：使用通用模型加载器加载 Whisper 模型用于音频/视频转录。
    
    Args:
        model_name: Whisper 模型大小（base, small, medium, large等）
        
    Returns:
        Whisper 模型实例
    """
    config = LoaderConfig(
        model_name=model_name,
        device="auto",
        model_type=ModelType.WHISPER
    )
    return ModelLoader.load_model(config)


# ========== 示例 5: 替换 LLM 模型加载 ==========
def load_llm_model_example(
    model_name: str,
    device_map: str = "auto",
    torch_dtype: str = "auto"
) -> Tuple[Any, Any]:
    """加载 LLM 模型和 Tokenizer（替换 QwenChat 和 HFChat 中的模型加载）
    
    用处：使用通用模型加载器加载 Transformers LLM 模型和 Tokenizer。
    
    Args:
        model_name: 模型名称或路径
        device_map: 设备映射策略
        torch_dtype: 数据类型
        
    Returns:
        (model, tokenizer) 元组
    """
    config = LoaderConfig(
        model_name=model_name,
        device="auto",
        model_type=ModelType.TRANSFORMERS,
        model_class="AutoModelForCausalLM",
        tokenizer_class="AutoTokenizer",
        return_tokenizer=True,
        torch_dtype=torch_dtype,
        device_map=device_map
    )
    return ModelLoader.load_model(config)


# ========== 示例 6: 缓存管理 ==========
def cache_management_example():
    """缓存管理示例
    
    用处：展示如何管理模型缓存，包括查看缓存信息、清除缓存等。
    """
    # 获取缓存信息
    info = ModelLoader.get_cache_info()
    print(f"当前缓存大小: {info['cache_size']}")
    print(f"最大缓存大小: {info['max_cache_size']}")
    print(f"已缓存的模型: {info['cached_models']}")
    
    # 清除所有缓存
    ModelLoader.clear_cache()
    print("已清除所有模型缓存")
    
    # 移除特定模型
    config = LoaderConfig(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_type=ModelType.SENTENCE_TRANSFORMER
    )
    removed = ModelLoader.remove_from_cache(config)
    print(f"移除模型: {removed}")

