from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

from ...base import ChatModel, ChatProviderBase
from ....config import Settings, get_config_manager


class HFChat(ChatModel):
    """
    通用 HF 聊天包装器（支持任意 Transformers CausalLM）。
    仅提供通用能力，个性化模板/解析由各模型包实现。
    """

    def __init__(self, model: str, device_map: str = "auto", torch_dtype: str = "auto") -> None:
        self.model = model
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        """懒加载 tokenizer 与模型
        
        用处：使用通用模型加载器加载模型和 tokenizer，自动处理缓存。
        """
        if self._tokenizer is not None and self._model is not None:
            return
        
        from app.infra.models import ModelLoader, ModelType, LoaderConfig
        
        dtype = _dtype_from_str(self.torch_dtype)
        dtype_str = "float16" if dtype == torch.float16 else ("bfloat16" if dtype == torch.bfloat16 else "float32")
        
        # 使用通用模型加载器加载模型和 tokenizer（自动缓存）
        config = LoaderConfig(
            model_name=self.model,
            device="auto",
            model_type=ModelType.TRANSFORMERS,
            model_class="AutoModelForCausalLM",
            tokenizer_class="AutoTokenizer",
            return_tokenizer=True,
            torch_dtype=dtype_str,
            device_map=self.device_map
        )
        self._model, self._tokenizer = ModelLoader.load_model(config)

    #  noinspection PyProtocol
    def generate(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> Mapping[str, Any]:
        self._load()
        max_new_tokens = int(kwargs.get("max_new_tokens", 1024))
        text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        outputs = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        output_ids = outputs[0][len(inputs.input_ids[0]) :].tolist()
        raw_text = self._tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        return {"content": raw_text, "raw_text": raw_text}


class HFBackendProvider(ChatProviderBase):
    """通用本地推理后端 Provider。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def build_chat_model(self) -> ChatModel:
        cfg = get_config_manager().get_model_config(self.settings.provider, self.settings.model)
        device_map = str(cfg.get("device_map", "auto"))
        torch_dtype = str(cfg.get("torch_dtype", "auto"))
        model_path = cfg.get("model", self.settings.model)
        return HFChat(model_path, device_map=device_map, torch_dtype=torch_dtype)


def _dtype_from_str(name: str) -> Any:
    mapping = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(str(name).lower(), "auto")


