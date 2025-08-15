from typing import List
from langchain_core.tools import tool

@tool
def search_docs(query: str) -> str:
    """示例工具：查询内置'知识库'（这里简单用列表代替）。"""
    documents: List[str] = [
        "秦始皇：中国历史上第一位皇帝，统一六国，修筑万里长城雏形。",
        "故宫：位于北京的皇家宫殿，明清两代的皇宫，现为博物院。",
    ]
    matches = [d for d in documents if query in d]
    return "\n".join(matches) if matches else "未找到相关内容。"