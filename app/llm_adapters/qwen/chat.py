from __future__ import annotations

"""
Qwen 本地推理适配（Transformers）。

单一职责：
- 封装 Qwen 聊天生成（messages → generate）。
"""

from typing import Any, List, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

from ..base import ChatModel, ChatProviderBase
from ...config import Settings, get_config_manager


class QwenChat(ChatModel):
    """
    Qwen 聊天包装器。

    用途：
    - 封装 Qwen 系 Transformers 聊天模型的生成流程，支持 thinking 解析。

    参数：
    - model_name (str): 模型名称或本地路径（如 "Qwen/Qwen3-0.6B"、"D:/models/Qwen/Qwen3-0.6B"）。
    - device_map (str): 设备映射策略，常见取值：
        - "auto": 由 Transformers 自动选择设备/切分策略
        - "cpu"/"cuda"/"cuda:0": 固定到 CPU/GPU/指定 GPU
    - torch_dtype (str): 张量精度，常见取值：
        - "auto": 由模型与硬件自动选择
        - "float16"/"bfloat16"/"float32": 半精度/脑浮点/单精度
    """

    def __init__(self, model_name: str, device_map: str = "auto", torch_dtype: str = "auto") -> None:
        self.model_name = model_name
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        """
        懒加载 tokenizer 与模型，避免重复初始化成本。
        """
        if self._tokenizer is not None and self._model is not None:
            return
        dtype = _dtype_from_str(self.torch_dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=dtype, device_map=self.device_map
        )

    #  noinspection PyProtocol
    def generate(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> Mapping[str, Any]:
        """
        执行一次对话生成并解析 thinking。

        参数：
        - messages (List[Dict[str,str]]): 消息数组，每项包含：
            - role: "system" | "user" | "assistant"
            - content: 文本内容
        - max_new_tokens (int): 最大生成 token 数，默认 2048，建议按显存与吞吐调优。
        - enable_thinking (bool): 是否启用 Qwen 的思维模式（会输出隐藏思考片段），默认 True。

        返回：
        - Dict[str,str]:
            - thinking: Qwen 的隐藏思考内容（若开启 thinking）
            - content: 最终回答内容
            - raw_text: 原始解码文本（含/不含思考，取决于解析规则）
        """
        max_new_tokens = int(kwargs.get("max_new_tokens", 2048))
        enable_thinking = bool(kwargs.get("enable_thinking", True))
        self._load()
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        outputs = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        output_ids = outputs[0][len(inputs.input_ids[0]) :].tolist()
        thinking, content = _split_qwen_thinking(self._tokenizer, output_ids)
        raw_text = self._tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        # 显式地为结果提供类型注解
        result: Mapping[str, Any] = {
            "thinking": thinking,
            "content": content,
            "raw_text": raw_text
        }
        return result


class QwenProvider(ChatProviderBase):
    """
    Qwen 提供商适配器：构建 QwenChat 聊天模型。

    用途：
    - 从 providers.qwen.models.<model> 读取设备/精度等参数，实例化 QwenChat。
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def build_chat_model(self) -> ChatModel:
        """
        构建聊天模型实例。

        返回：
        - QwenChat: 已根据配置设置好的聊天模型。
        """
        cfg = get_config_manager().get_model_config("qwen", self.settings.model)
        device_map = str(cfg.get("device_map", "auto"))
        torch_dtype = str(cfg.get("torch_dtype", "auto"))
        return QwenChat(self.settings.model, device_map=device_map, torch_dtype=torch_dtype)


def _dtype_from_str(name: str) -> Any:
    """将字符串精度名映射为 torch.dtype 或 "auto"。"""
    mapping = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(str(name).lower(), "auto")


def _split_qwen_thinking(tokenizer: Any, output_ids: List[int]) -> tuple[str, str]:
    """
    切分 Qwen 输出的思考与最终内容。

    规则：根据特殊 token（</think> 的 id，示例 151668）自尾部回溯切分；未命中则全部视为 content。

    返回：
    - (thinking, content)
    """
    if not output_ids:
        return "", ""
    try:
        end_token_id = 151668
        index = len(output_ids) - output_ids[::-1].index(end_token_id)
    except ValueError:
        index = 0
    thinking_ids = output_ids[:index]
    content_ids = output_ids[index:]
    thinking = tokenizer.decode(thinking_ids, skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(content_ids, skip_special_tokens=True).strip("\n")
    return thinking, content


