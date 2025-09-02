# Chocolate: 可扩展的 LangChain Agent 工程

一个基于 LangChain 与 FastAPI 的可扩展智能对话系统工程模板，专注于：

- 多模型适配（Gemini / OpenAI 等）与工厂化接入
- ReAct Agent 执行器与工具系统（搜索/HTTP/计算器）
- 可观测性与缓存、模块化 API、可维护的配置中心
- 企业级 RAG（检索增强生成）实践与多语言/多媒体数据喂养

## 文档索引（请从这里开始）

- 项目概览与目录：[docs/architecture.md](docs/architecture.md)
- 快速开始：[docs/getting_started.md](docs/getting_started.md)
- Web API：[docs/api.md](docs/api.md)
- 测试指南：[docs/tests.md](docs/tests.md)
- 扩展与示例：[docs/extensions.md](docs/extensions.md)
- 常见问题（FAQ）：[docs/faq.md](docs/faq.md)

## RAG 分章节文档

- 总索引：[docs/rag/README.md](docs/rag/README.md)
- 总览：[docs/rag/overview.md](docs/rag/overview.md)
- 数据接入层：[docs/rag/ingestion.md](docs/rag/ingestion.md)
- 数据处理层（多语言/多媒体）：[docs/rag/processing.md](docs/rag/processing.md)
- 向量化层（多语言嵌入）：[docs/rag/vectorization.md](docs/rag/vectorization.md)
- 检索层（召回+重排，含语言路由）：[docs/rag/retrieval.md](docs/rag/retrieval.md)
- 应用层（Gemini 集成）：[docs/rag/application.md](docs/rag/application.md)
- 管理与运维：[docs/rag/management.md](docs/rag/management.md)
- 设计模式与原则：[docs/rag/patterns.md](docs/rag/patterns.md)

欢迎根据业务需要进行裁剪与扩展。
