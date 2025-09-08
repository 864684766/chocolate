# RAG 文档索引

- 总览: [overview.md](overview.md)
- 数据接入层: [ingestion.md](ingestion.md)
- 数据处理层: [processing.md](processing.md)
- 向量化层: [vectorization.md](vectorization.md)
- 检索层: [retrieval.md](retrieval.md)
- 应用层: [application.md](application.md)
- 管理与运维: [management.md](management.md)
- 设计模式: [patterns.md](patterns.md)

## 体系结构与流程

```mermaid
flowchart TD
  A[数据接入层 ingestion\n(上传/采集/标准化)] --> B[数据处理层 processing\n(解析/清洗/分块/质检)]
  B --> C[向量化层 vectorization\n(嵌入生成/写入向量库)]
  C --> D[检索层 retrieval\n(召回/重排/候选拼装)]
  D --> E[应用层 application\n(RAG编排/LLM生成/API)]

  subgraph 后台/离线管线
    A --> B --> C
  end
  subgraph 前台/在线服务
    C --> D --> E
  end
```

说明：

- A/B/C 为后台管线，负责把原始数据转为“可检索的向量化知识”。
- D/E 为在线路径，面对实时问答，完成检索增强与生成。

## 管理与运维、设计模式在 RAG 中的作用

- 管理与运维 ([management.md]):

  - 覆盖配置治理、权限与审计、观测与告警、版本与回滚、数据生命周期等跨层能力。
  - 目标是“让管线与服务稳定可控”，不改变 RAG 业务流程，仅保障其可靠性与可维护性。

- 设计模式 ([patterns.md]):
  - 给出接口抽象、模块解耦、可插拔策略（如分块、重排、嵌入、检索器）的实践建议。
  - 目的是“指导工程落地与扩展”，避免在实现时误把它当作独立的 RAG 层。

请将以上两个文档视作“全链路治理与工程方法论”的补充读物，而非 RAG 六层中的独立功能层。
