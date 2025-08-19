from typing import Dict

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

_session_store:Dict[str, BaseChatMessageHistory] = {}

"""
获取会话ID
"""
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    获取指定会话ID的聊天历史。
    如果使用内存存储，则从内存字典中获取；
    如果未来切换到 Redis/数据库，则在这里实现从对应后端加载历史。
    """
    if session_id not in _session_store:
       _session_store[session_id] = ChatMessageHistory()
       print(f'New session created for ID: {session_id} in memory.')
    return _session_store[session_id]