# 架构与目录

## 目录结构

```text
chocolate/
  app/
    llm_adapters/ (factory + providers)
    tools/
    api/
    core/
    ...
  scripts/
  tests/
  docs/
```

## 关键组件

- LLM 适配层：工厂+注册表，按配置选择模型
- Agent 执行器：基于 ReAct，多步推理
- 工具系统：search/http/calc 等
- FastAPI：模块化路由
- 缓存与观测：LRU + LangSmith（可选）

更多细节见根 README。
