from typing import Optional, Dict, Tuple

from .dict_helper import touch_cache_key, pop_lru_item
from ..agent import build_agent
from langchain_core.runnables import RunnableWithMessageHistory, RunnableConfig

from ..core.session_manager import get_session_history
from ..config import get_config_manager


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
        
        # 0. 按配置启用 LangSmith 追踪（无需改业务代码）
        cfg = get_config_manager().get_config()
        smith = (cfg.get("observability", {}) or {}).get("langsmith", {})
        if smith.get("enabled"):
            import os
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            # 兼容老/新两套环境变量命名（LangSmith 控制台可能展示 LANGSMITH_*，
            # LangChain SDK 推荐使用 LANGCHAIN_*）。这里同时设置，避免版本差异带来的困扰。
            os.environ["LANGSMITH_TRACING"] = "true"
            if smith.get("api_key"):
                os.environ["LANGCHAIN_API_KEY"] = str(smith.get("api_key"))
                os.environ["LANGSMITH_API_KEY"] = str(smith.get("api_key"))
            if smith.get("project"):
                os.environ["LANGCHAIN_PROJECT"] = str(smith.get("project"))
                os.environ["LANGSMITH_PROJECT"] = str(smith.get("project"))
            if smith.get("endpoint"):
                os.environ["LANGCHAIN_ENDPOINT"] = str(smith.get("endpoint"))
                os.environ["LANGSMITH_ENDPOINT"] = str(smith.get("endpoint"))

        # 1. 构建核心 AgentExecutor
        base_agent_executor = build_agent(ai_type, provider)

        # 2. 直接使用 AgentExecutor，不再需要额外的格式化步骤
        agent_chain = RunnableWithMessageHistory(
            base_agent_executor,
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