from __future__ import annotations
from typing import Optional
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

from .llm import get_chat_model
from .tools import search_docs, http_get, calc
from .config import get_config_manager


def build_agent(ai_type:Optional[str]=None,provider:Optional[str]=None) -> AgentExecutor:
    """构建一个完整的 ReAct Agent 执行器，支持多步思考-行动-观察循环。
    
    Agent 会自主决定何时使用哪个工具，可以进行多轮推理：
    - 思考：分析问题，决定下一步行动
    - 行动：调用合适的工具
    - 观察：分析工具结果
    - 重复：继续思考或给出最终答案
    """
    llm = get_chat_model(ai_type,provider)
    
    # 可用工具列表
    tools = [search_docs, http_get, calc]
    
    # 从配置文件获取提示词模板
    config_manager = get_config_manager()
    prompts_config = config_manager.get_prompts_config()
    react_template = prompts_config.get("react_template", "")
    
    # ReAct 提示词模板
    react_prompt = PromptTemplate.from_template(react_template)
    
    # 创建 ReAct agent
    agent = create_react_agent(llm, tools, react_prompt)
    
    # 从配置文件获取Agent配置
    agent_config = config_manager.get_agent_config()
    
    # 创建 Agent 执行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=agent_config.get("verbose", False),  # 设为 True 可看到详细的思考过程
        max_iterations=agent_config.get("max_iterations", 5),  # 最大迭代次数，防止无限循环
        max_execution_time=agent_config.get("max_execution_time", 30),  # 最大执行时间（秒）
        handle_parsing_errors=agent_config.get("handle_parsing_errors", True),  # 处理解析错误
        return_intermediate_steps=agent_config.get("return_intermediate_steps", False),  # 是否返回中间步骤
    )
    
    return agent_executor