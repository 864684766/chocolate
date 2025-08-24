from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from ..agent import build_agent
from langchain_core.runnables import RunnableWithMessageHistory, RunnableConfig, RunnableParallel

from ..core.session_manager import get_session_history
from ..llm import get_chat_model

_global_agent_chain_instance: Optional[RunnableWithMessageHistory] = None
_global_runnable_config_template: Optional[RunnableConfig] = None

def initialize_agent_chain(ai_type:Optional[str]=None,provider:Optional[str]=None) -> RunnableWithMessageHistory:
    """
    初始化并返回带有会话历史的完整 Agent 链。
    此函数设计为在应用启动时只调用一次，以避免重复初始化开销。
    """
    global _global_agent_chain_instance
    global _global_runnable_config_template

    if _global_agent_chain_instance is None:
        print("Initializing Agent Chain...")
        # 1. 构建核心 AgentExecutor
        _base_agent_executor = build_agent(ai_type,provider)

        # 2. 定义最终输出格式化的 LLM 提示词
        _final_output_formatter_prompt = PromptTemplate.from_template("""
你是一个友好的智能助手。请将以下信息转化为对用户的友好、简洁的最终答案。
确保你的回答自然、流畅，像是真人对话。

用户最初的问题：{input}
原始答案（可能来自工具或代理的直接输出，可能不完整或格式化）：{raw_agent_output}
如果你有任何相关历史，它会在这里：
{chat_history}

请用自然语言，结合用户问题和原始答案，组织一个完整的回复。
例如，如果原始答案是一个数字（如“2.0”），你可以回答“计算结果是 2。”或者“1 加 1 等于 2。”
如果原始答案是搜索结果，请总结后友好地回答。

最终回复：
""")

        # 3. 获取用于最终格式化的 LLM 实例
        _formatting_llm = get_chat_model(ai_type,provider)

        # 4. 构建包含 Agent 执行和最终格式化步骤的完整 Runnable 链
        _full_chain_with_formatting = (
            RunnableParallel(
                # 从 RunnableWithMessageHistory 传入的变量
                input=lambda x: x["input"],
                chat_history=lambda x: x["chat_history"],
                # 运行 AgentExecutor，其输出将作为 raw_agent_output 传入格式化提示词
                raw_agent_output=_base_agent_executor,
            )
            | _final_output_formatter_prompt # 将字典传入提示词
            | _formatting_llm # 将填充好的提示词传入 LLM
            | StrOutputParser() # 将 LLM 的输出解析为纯字符串
        )

        # 5. 使用 RunnableWithMessageHistory 包装这个完整的链
        _global_agent_chain_instance = RunnableWithMessageHistory(
            _full_chain_with_formatting,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        print("Agent Chain Initialized.")

    return _global_agent_chain_instance

def get_runnable_config(session_id: str) -> RunnableConfig:
    """
    根据会话ID获取 RunnableConfig 对象。
    """
    return RunnableConfig(configurable={"session_id": session_id})