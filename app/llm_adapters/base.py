from __future__ import annotations

"""
LLM Provider 适配器基类与类型约定。

职责：
- 统一各 Provider 的最小接口与交互契约，便于工厂分发与替换。
"""

from typing import Any,Protocol, Sequence, Mapping


class ChatModel(Protocol):
    """
    聊天模型协议：要求实现 `generate(messages, **kwargs)`。

    方法：
    - generate(messages, **kwargs): 执行一次对话生成。
    """

    def generate(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> Mapping[str, Any]:
        """
        执行一次对话生成。

        参数:
        - messages (List[Dict[str,Any]]): 聊天消息列表，每项至少包含 `role` 与 `content` 字段，其它字段按模型自定义。
        - kwargs: 可变关键字参数（不同 Provider 支持不同可选项），常见：
          - max_new_tokens (int): 最大新增 token 数；
          - enable_thinking (bool): 是否启用思维模式（部分模型支持）。

        返回:
        - Mapping[str,Any]: 至少包含 "content" 字段；可按适配器增加 "thinking"、"raw_text" 等字段用于调试或观测。
        """
        ...


class ChatProviderBase:
    """
    Provider 适配器基类：只定义构建聊天模型实例的方法。

    说明：
    - 子类负责读取配置中心与 Settings，构建具体的 ChatModel 实例。
    - ChatModel 实例需满足 ChatModel 协议，提供 generate 方法。
    """

    def build_chat_model(self) -> Any:
        """
        构建并返回聊天模型实例。

        返回：
        - Any: 具备 generate(messages, **kwargs) 方法的对象（满足 ChatModel 协议）。
        """
        raise NotImplementedError


