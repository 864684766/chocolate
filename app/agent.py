from __future__ import annotations
from typing import Optional, Any
from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate

from .llm import get_chat_model
from .tools import search_docs, http_get, calc
from .config import get_config_manager


def build_agent(ai_type: Optional[str] = None, provider: Optional[str] = None) -> Any:
    """
    构建一个完整的 ReAct Agent，支持多步思考-行动-观察循环。
    
    用处：创建一个支持工具调用的智能Agent，Agent会自主决定何时使用哪个工具，
    可以进行多轮推理：
    - 思考：分析问题，决定下一步行动
    - 行动：调用合适的工具
    - 观察：分析工具结果
    - 重复：继续思考或给出最终答案
    
    Args:
        ai_type: AI类型，可选
        provider: 提供商，可选
        
    Returns:
        Any: LangChain V1 的 Agent 对象，可以直接被 RunnableWithMessageHistory 包装
    """
    llm = get_chat_model(ai_type, provider)
    
    # 可用工具列表
    tools = [search_docs, http_get, calc]
    
    # 从配置文件获取提示词模板
    config_manager = get_config_manager()
    prompts_config = config_manager.get_prompts_config()
    react_template = prompts_config.get("react_template", "")
    
    # 如果模板为空，使用默认模板
    if not react_template:
        react_template = """你是一个有用的AI助手。你可以使用以下工具来回答问题：
- search_docs: 搜索知识库文档
- http_get: 发送HTTP GET请求获取网页内容
- calc: 计算数学表达式

请根据用户的问题，选择合适的工具来获取信息，然后给出准确的回答。"""
    
    # 将 PromptTemplate 转换为字符串（LangChain V1 需要字符串格式的 system_prompt）
    if isinstance(react_template, str):
        system_prompt = react_template
    else:
        # 如果配置中传入的是模板对象，尝试转换为字符串
        prompt_template = PromptTemplate.from_template(react_template) if react_template else None
        if prompt_template:
            # 使用空字典格式化，因为 system_prompt 不需要变量
            system_prompt = prompt_template.format()
        else:
            system_prompt = react_template
    
    # 从配置文件获取Agent配置
    agent_config = config_manager.get_agent_config()
    
    # 创建 Agent（LangChain V1 使用 create_agent，不再需要 AgentExecutor）
    # create_agent 返回的对象可以直接使用，支持 invoke 方法
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        # LangChain V1 的 create_agent 支持以下参数（如果可用）
        # verbose=agent_config.get("verbose", False),  # 详细输出
        # max_iterations=agent_config.get("max_iterations", 5),  # 最大迭代次数
        # handle_parsing_errors=agent_config.get("handle_parsing_errors", True),  # 处理解析错误
    )
    
    return agent