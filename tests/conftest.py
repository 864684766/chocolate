# -*- coding: utf-8 -*-
"""
Pytest 全局配置：
- 确保 tests 运行时可以导入项目根目录下的 app 包
- 为缺失 LLM 依赖的环境提供一个默认的 Mock LLM，避免构建 ReAct Agent 时失败
  （单测中如需自定义 mock，会覆盖此默认 mock）
"""
import sys
from pathlib import Path

import pytest


# 1) 确保将项目根目录加入 sys.path，便于 `from app ...` 导入
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# 2) 为缺少真实 LLM 依赖的环境提供默认 Mock（可被测试内的 patch 覆盖）
@pytest.fixture(autouse=True)
def _mock_llm_for_agent(monkeypatch):
    try:
        # 延迟导入，避免在未使用时触发
        import app.llm as app_llm
    except Exception:
        # 若导入失败，交由具体测试用例报错
        return

    # 使用简单的 Mock 满足 create_react_agent 期望的接口
    from unittest.mock import Mock

    dummy_llm = Mock(name="DummyLLM")
    dummy_llm.bind_tools.return_value = dummy_llm  # create_react_agent 会调用该方法
    dummy_llm.invoke.return_value = "Test response"  # 基本占位返回

    # 将 app.llm.get_chat_model 替换为返回 dummy_llm
    monkeypatch.setattr(app_llm, "get_chat_model", lambda: dummy_llm, raising=False)