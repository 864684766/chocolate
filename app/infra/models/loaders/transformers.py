"""
Transformers 模型加载器

用处：加载 HuggingFace Transformers 库的模型，支持 AutoModel、AutoTokenizer 等。
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from .base import LoaderConfig, ModelLoaderBase, ModelType

logger = logging.getLogger(__name__)


class TransformersLoader(ModelLoaderBase):
    """Transformers 模型加载器
    
    用处：加载 HuggingFace Transformers 库的模型，支持 AutoModel、AutoTokenizer 等。
    """
    
    def load(self, config: LoaderConfig) -> Any:
        """加载 Transformers 模型或 Pipeline
        
        Args:
            config: 加载器配置
                - model_name: 模型路径或名称
                - device: 设备类型
                - kwargs 可包含:
                    - task: Pipeline 任务类型（如 "image-to-text", "translation"），如果提供则创建 Pipeline
                    - model_class: 模型类名（如 "AutoModelForCausalLM"），默认为 "AutoModel"
                    - tokenizer_class: Tokenizer 类名（如 "AutoTokenizer"），默认为 "AutoTokenizer"
                    - torch_dtype: 数据类型（如 "float16", "bfloat16"）
                    - device_map: 设备映射策略
                    - return_tokenizer: 是否同时返回 tokenizer，默认为 False
                    - src_lang: 翻译任务的源语言（仅用于 translation pipeline）
                    - tgt_lang: 翻译任务的目标语言（仅用于 translation pipeline）
        
        Returns:
            Any: 
                - 如果 task 参数存在，返回 Pipeline 实例
                - 否则返回模型实例，如果 return_tokenizer=True 则返回 (model, tokenizer) 元组
        """
        try:
            from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, pipeline
            
            # 如果指定了 task，则创建 Pipeline
            task = config.kwargs.get("task")
            if task:
                pipeline_kwargs = {
                    "task": task,
                    "model": config.model_name,
                }
                
                # 处理设备
                if config.device and config.device != "auto":
                    pipeline_kwargs["device"] = config.device
                
                # 处理翻译任务的特殊参数
                if task == "translation":
                    if "src_lang" in config.kwargs:
                        pipeline_kwargs["src_lang"] = config.kwargs["src_lang"]
                    if "tgt_lang" in config.kwargs:
                        pipeline_kwargs["tgt_lang"] = config.kwargs["tgt_lang"]
                
                pipe = pipeline(**pipeline_kwargs)
                logger.debug(f"Transformers Pipeline 加载成功: task={task}, model={config.model_name}")
                return pipe
            
            # 否则加载模型和 tokenizer
            model_class_name = config.kwargs.get("model_class", "AutoModel")
            tokenizer_class_name = config.kwargs.get("tokenizer_class", "AutoTokenizer")
            return_tokenizer = config.kwargs.get("return_tokenizer", False)
            
            # 选择模型类
            model_class_map = {
                "AutoModel": AutoModel,
                "AutoModelForCausalLM": AutoModelForCausalLM,
            }
            model_class = model_class_map.get(model_class_name, AutoModel)
            
            # 选择 tokenizer 类
            tokenizer_class = AutoTokenizer if tokenizer_class_name == "AutoTokenizer" else AutoTokenizer
            
            # 准备加载参数
            load_kwargs = {}
            if "torch_dtype" in config.kwargs:
                dtype_str = config.kwargs["torch_dtype"]
                dtype_map = {
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "float32": torch.float32,
                }
                load_kwargs["torch_dtype"] = dtype_map.get(dtype_str, "auto")
            
            if "device_map" in config.kwargs:
                load_kwargs["device_map"] = config.kwargs["device_map"]
            
            # 加载模型和 tokenizer
            model = model_class.from_pretrained(config.model_name, **load_kwargs)
            
            if return_tokenizer:
                tokenizer = tokenizer_class.from_pretrained(config.model_name)
                logger.debug(f"Transformers 模型和 Tokenizer 加载成功: {config.model_name}")
                return model, tokenizer
            else:
                logger.debug(f"Transformers 模型加载成功: {config.model_name}")
                return model
                
        except ImportError:
            raise RuntimeError("transformers 库未安装，请安装: pip install transformers")
        except Exception as e:
            raise RuntimeError(f"加载 Transformers 模型失败: {e}")
    
    def get_supported_type(self) -> ModelType:
        return ModelType.TRANSFORMERS

