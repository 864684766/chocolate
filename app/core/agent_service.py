from typing import Optional, Dict, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from .dict_helper import touch_cache_key, pop_lru_item
from ..agent import build_agent
from langchain_core.runnables import RunnableWithMessageHistory, RunnableConfig, RunnableParallel

from ..core.session_manager import get_session_history
from ..llm import get_chat_model


class AgentService:
    """Agent服务类，管理Agent链的创建和缓存"""
    
    def __init__(self):
        self._agent_chain_cache:dict = dict()  # LRU缓存实现
        self._max_cache_size: int = 5  # 最大缓存数量（Agent链比模型更重）
        self._runnable_config_template: Optional[RunnableConfig] = None
    
    def get_agent_chain(self, ai_type: Optional[str] = None, provider: Optional[str] = None) -> RunnableWithMessageHistory:
        """获取Agent链实例，支持缓存"""
        cache_key = (ai_type or "default", provider or "default")
        
        if cache_key in self._agent_chain_cache:
            # 移动到末尾（最近使用）
            touch_cache_key(self._agent_chain_cache,cache_key)
            return self._agent_chain_cache[cache_key]
        
        # 创建新的Agent链
        agent_chain = self._create_agent_chain(ai_type, provider)
        
        # LRU缓存管理
        self._agent_chain_cache[cache_key] = agent_chain
        touch_cache_key(self._agent_chain_cache, cache_key) # 移动到末尾
        
        # 如果缓存超过最大大小，删除最旧的
        if len(self._agent_chain_cache) > self._max_cache_size:
            pop_lru_item(self._agent_chain_cache)  # 移动到末尾
        return agent_chain
    
    @staticmethod
    def _create_agent_chain(ai_type: Optional[str] = None, provider: Optional[str] = None) -> RunnableWithMessageHistory:
        """创建Agent链实例"""
        print(f"Creating Agent Chain for ai_type={ai_type}, provider={provider}...")
        
        # 1. 构建核心 AgentExecutor
        base_agent_executor = build_agent(ai_type, provider)

        # 2. 定义最终输出格式化的 LLM 提示词
        final_output_formatter_prompt = PromptTemplate.from_template("""
你是一个友好的智能助手。请将以下信息转化为对用户的友好、简洁的最终答案。
确保你的回答自然、流畅，像是真人对话。

用户最初的问题：{input}
原始答案（可能来自工具或代理的直接输出，可能不完整或格式化）：{raw_agent_output}
如果你有任何相关历史，它会在这里：
{chat_history}

请用自然语言，结合用户问题和原始答案，组织一个完整的回复。
例如，如果原始答案是一个数字（如"2.0"），你可以回答"计算结果是 2。"或者"1 加 1 等于 2。"
如果原始答案是搜索结果，请总结后友好地回答。

最终回复：
""")

        # 3. 获取用于最终格式化的 LLM 实例
        formatting_llm = get_chat_model(ai_type, provider)

        # 4. 构建包含 Agent 执行和最终格式化步骤的完整 Runnable 链
        full_chain_with_formatting = (
            RunnableParallel(
                # 从 RunnableWithMessageHistory 传入的变量
                input=lambda x: x["input"],
                chat_history=lambda x: x["chat_history"],
                # 运行 AgentExecutor，其输出将作为 raw_agent_output 传入格式化提示词
                raw_agent_output=base_agent_executor,
            )
            | final_output_formatter_prompt # 将字典传入提示词
            | formatting_llm # 将填充好的提示词传入 LLM
            | StrOutputParser() # 将 LLM 的输出解析为纯字符串
        )

        # 5. 使用 RunnableWithMessageHistory 包装这个完整的链
        agent_chain = RunnableWithMessageHistory(
            full_chain_with_formatting,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        
        print(f"Agent Chain created successfully for ai_type={ai_type}, provider={provider}")
        return agent_chain
    
    def get_runnable_config(self, session_id: str) -> RunnableConfig:
        """根据会话ID获取 RunnableConfig 对象"""
        if self._runnable_config_template is None:
            self._runnable_config_template = RunnableConfig()
        return RunnableConfig(configurable={"session_id": session_id})
    
    def clear_cache(self) -> None:
        """清除Agent链缓存"""
        self._agent_chain_cache.clear()
    
    def get_cached_chains(self) -> Dict[Tuple[str, str], RunnableWithMessageHistory]:
        """获取当前缓存的所有Agent链"""
        return self._agent_chain_cache.copy()


# 全局Agent服务实例
_agent_service = AgentService()


def initialize_agent_chain(ai_type: Optional[str] = None, provider: Optional[str] = None) -> RunnableWithMessageHistory:
    """
    初始化并返回带有会话历史的完整 Agent 链。
    此函数设计为在应用启动时只调用一次，以避免重复初始化开销。
    """
    return _agent_service.get_agent_chain(ai_type, provider)


def get_runnable_config(session_id: str) -> RunnableConfig:
    """
    根据会话ID获取 RunnableConfig 对象。
    """
    return _agent_service.get_runnable_config(session_id)


def get_agent_service() -> AgentService:
    """获取Agent服务实例"""
    return _agent_service