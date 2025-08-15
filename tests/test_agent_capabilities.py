"""
测试 ReAct Agent 执行器的多步推理和会话记忆功能
"""
import pytest
from unittest.mock import Mock, patch
from app.agent import build_agent
from app.api.agent import get_session_history, _session_store
from langchain_core.runnables.history import RunnableWithMessageHistory


class TestReActAgentCapabilities:
    """测试 ReAct Agent 的核心能力"""

    def _llm_dep_missing(self) -> bool:
        try:
            import langchain_google_genai  # noqa: F401
            return False
        except Exception:
            return True

    def test_agent_executor_creation(self):
        """测试能够正常创建 AgentExecutor"""
        if self._llm_dep_missing():
            pytest.skip("缺少 langchain-google-genai 依赖，跳过需要构建 Agent 的测试")
        executor = build_agent()
        assert hasattr(executor, 'agent')
        assert hasattr(executor, 'tools')
        assert hasattr(executor, 'max_iterations')
        assert executor.max_iterations == 5

    def test_agent_tools_available(self):
        """测试 Agent 有可用的工具"""
        if self._llm_dep_missing():
            pytest.skip("缺少 langchain-google-genai 依赖，跳过需要构建 Agent 的测试")
        executor = build_agent()
        tool_names = [tool.name for tool in executor.tools]
        expected_tools = ['search_docs', 'http_get', 'calc']
        for tool in expected_tools:
            assert tool in tool_names

    @patch('app.llm.get_chat_model')
    def test_agent_invoke_basic(self, mock_get_chat_model):
        """测试 Agent 基本调用功能"""
        # Mock LLM 返回简单回答
        mock_llm = Mock()
        mock_llm.invoke.return_value = "Test response"
        mock_get_chat_model.return_value = mock_llm

        if self._llm_dep_missing():
            pytest.skip("缺少 langchain-google-genai 依赖，跳过需要构建 Agent 的测试")
        executor = build_agent()
        
        # 注意：由于我们 mock 了 LLM，实际不会执行 ReAct 循环
        # 这里主要测试接口是否正常
        assert callable(executor.invoke)

    def test_session_history_creation(self):
        """测试会话历史管理"""
        # 清空全局存储
        _session_store.clear()
        
        # 获取新会话历史
        history1 = get_session_history("test_session_1")
        assert "test_session_1" in _session_store
        
        # 再次获取同一会话，应该返回相同对象
        history2 = get_session_history("test_session_1")
        assert history1 is history2
        
        # 获取不同会话，应该返回不同对象
        history3 = get_session_history("test_session_2")
        assert history1 is not history3
        assert "test_session_2" in _session_store

    def test_session_history_message_storage(self):
        """测试会话历史消息存储"""
        _session_store.clear()
        
        history = get_session_history("test_session")
        
        # 初始状态应该没有消息
        assert len(history.messages) == 0
        
        # 添加消息
        from langchain_core.messages import HumanMessage, AIMessage
        history.add_message(HumanMessage(content="Hello"))
        history.add_message(AIMessage(content="Hi there!"))
        
        # 验证消息已存储
        assert len(history.messages) == 2
        assert history.messages[0].content == "Hello"
        assert history.messages[1].content == "Hi there!"

    def test_runnable_with_history_creation(self):
        """测试带历史的可运行对象创建"""
        if self._llm_dep_missing():
            pytest.skip("缺少 langchain-google-genai 依赖，跳过需要构建 Agent 的测试")
        executor = build_agent()
        
        chain_with_history = RunnableWithMessageHistory(
            executor,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        
        assert hasattr(chain_with_history, 'invoke')
        assert callable(chain_with_history.invoke)


class TestAgentTools:
    """测试 Agent 工具的基本功能"""

    def test_calc_tool_basic(self):
        """测试计算器工具基本功能"""
        from app.tools.calculator import calc
        
        result = calc.invoke("2 + 3")
        assert "5" in result or result == "5"

    def test_search_docs_tool_basic(self):
        """测试文档搜索工具基本功能"""
        from app.tools.search import search_docs
        
        # 应该能正常调用，不抛异常
        result = search_docs.invoke("test query")
        assert isinstance(result, str)

    @patch('requests.get')
    def test_http_get_tool_basic(self, mock_get):
        """测试 HTTP GET 工具基本功能"""
        from app.tools.http import http_get
        
        # Mock HTTP 响应
        mock_response = Mock()
        mock_response.text = "Test webpage content"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = http_get.invoke("https://example.com")
        assert "Test webpage content" in result


if __name__ == "__main__":
    # 简单运行测试
    test_capabilities = TestReActAgentCapabilities()
    test_tools = TestAgentTools()
    
    print("Running basic tests...")
    
    try:
        test_capabilities.test_agent_executor_creation()
        print("✓ Agent executor creation test passed")
        
        test_capabilities.test_agent_tools_available()
        print("✓ Agent tools availability test passed")
        
        test_capabilities.test_session_history_creation()
        print("✓ Session history creation test passed")
        
        test_capabilities.test_session_history_message_storage()
        print("✓ Session history message storage test passed")
        
        test_capabilities.test_runnable_with_history_creation()
        print("✓ Runnable with history creation test passed")
        
        test_tools.test_calc_tool_basic()
        print("✓ Calculator tool test passed")
        
        test_tools.test_search_docs_tool_basic()
        print("✓ Search docs tool test passed")
        
        print("\nAll tests passed! ✓")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        raise