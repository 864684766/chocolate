# 架构与目录

## 目录结构

```text
chocolate/
  app/
    llm_adapters/
      base.py           # 适配器基类/协议
      factory.py        # 提供商工厂与缓存
      google/           # Google 提供商包（native）
        __init__.py
        chat.py
      openai/           # OpenAI 提供商包（native）
        __init__.py
        chat.py
      backends/
        hf/             # 通用本地推理后端（Hugging Face）
          __init__.py
          chat.py
    tools/
    api/
    core/
    ...
  scripts/
  tests/
  docs/
```

## 关键组件

- LLM 适配层：工厂+注册表，按配置选择模型与后端（native/transformers）
- Agent 执行器：基于 ReAct，多步推理
- 工具系统：search/http/calc 等
- FastAPI：模块化路由
- 缓存与观测：LRU + LangSmith（可选）

更多细节见根 README。
