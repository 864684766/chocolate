from __future__ import annotations
from typing import Any, Optional
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from .llm import get_chat_model
from .tools import search_docs, http_get, calc


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
    
    # ReAct 提示词模板
    react_prompt = PromptTemplate.from_template("""
你是一个智能助理，能够使用工具来帮助回答问题。你有以下工具可用：

{tools}
可以使用的工具名称有：{tool_names} # <--- **新添加的行**

这是你和用户之前的对话历史：{chat_history} 

使用以下格式：

Question: 用户的问题
Thought: 我需要思考如何回答这个问题
Action: 要使用的工具名称
Action Input: 工具的输入参数
Observation: 工具返回的结果
... (这个 Thought/Action/Action Input/Observation 可以重复多次)
Thought: 现在我知道最终答案了
Final Answer: 给用户的最终回答

重要提示：
1. 用中文思考和回答
2. 如果需要计算，使用 calc 工具
3. 如果需要搜索信息，使用 search_docs 工具  
4. 如果需要获取网页内容，使用 http_get 工具
5. 可以组合使用多个工具来解决复杂问题
6. 最终答案要简洁明了

Question: {input}
Thought: {agent_scratchpad}
""")
    
    # 创建 ReAct agent
    agent = create_react_agent(llm, tools, react_prompt)
    
    # 创建 Agent 执行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,  # 设为 True 可看到详细的思考过程
        max_iterations=5,  # 最大迭代次数，防止无限循环
        max_execution_time=30,  # 最大执行时间（秒）
        handle_parsing_errors=True,  # 处理解析错误
        return_intermediate_steps=False,  # 是否返回中间步骤
    )
    
    return agent_executor