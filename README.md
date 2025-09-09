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

## 项目结构

```
app/
├── core/                    # 核心业务逻辑
│   ├── agent_service.py     # Agent服务管理
│   └── __init__.py
├── infra/                   # 基础设施层
│   ├── database/           # 数据库访问
│   │   ├── chroma/         # 向量数据库
│   │   │   ├── db_helper.py      # ChromaDB操作封装
│   │   │   ├── example_usage.py  # 使用示例
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── session/            # 会话管理
│   │   ├── session_manager.py    # 会话历史管理
│   │   └── __init__.py
│   ├── cache/              # 缓存工具
│   │   ├── dict_helper.py        # LRU缓存实现
│   │   └── __init__.py
│   ├── exceptions/         # 异常定义
│   │   ├── exceptions.py         # 自定义异常类
│   │   └── __init__.py
│   ├── management/         # 管理与运维工具
│   │   ├── metrics.py            # 系统监控指标收集
│   │   ├── config_watcher.py     # 配置热更新管理
│   │   ├── audit.py              # 审计日志与操作记录
│   │   ├── healthchecks.py       # 系统健康检查
│   │   └── __init__.py
│   └── __init__.py
├── rag/                    # RAG相关功能
│   ├── data_ingestion/     # 数据接入层
│   │   ├── sources/        # 数据源
│   │   │   └── manual_upload.py  # 手动上传
│   │   └── validators.py   # 数据验证
│   ├── processing/         # 数据处理层
│   │   ├── utils/          # 处理工具
│   │   │   ├── chunking.py       # 分块参数决策
│   │   │   ├── quality_utils.py  # 质量检测与去重
│   │   │   └── __init__.py
│   │   ├── interfaces.py   # 接口定义
│   │   ├── pipeline.py     # 处理流水线
│   │   ├── media_text.py   # 文本处理
│   │   ├── media_markdown.py # Markdown处理
│   │   ├── media_extractors.py # 媒体内容提取
│   │   ├── media_chunking.py # 媒体分块策略
│   │   ├── lang_zh.py      # 中文处理器
│   │   ├── quality_checker.py # 质量评估
│   │   └── ...
│   └── service/            # RAG服务层
│       └── ingestion_helpers.py # 接入辅助函数
├── api/                    # API接口层
│   ├── __init__.py         # FastAPI应用创建
│   ├── agent.py            # Agent对话接口
│   ├── health.py           # 健康检查接口
│   └── ingestion.py        # 数据接入接口
├── llm_adapters/          # LLM适配器
│   ├── __init__.py
│   ├── factory.py          # LLM工厂模式
│   ├── google.py           # Google Gemini适配器
│   └── openai.py           # OpenAI适配器
├── tools/                 # 工具集
│   ├── __init__.py
│   ├── calculator.py       # 计算器工具
│   ├── http.py            # HTTP请求工具
│   └── search.py          # 搜索工具
├── agent.py               # Agent构建器
├── api.py                 # API路由（已废弃）
├── config.py              # 配置管理
├── llm.py                 # LLM统一接口
└── tools.py               # 工具统一接口

scripts/                   # 启动脚本
├── __init__.py
├── run_agent.py           # 启动Agent服务
└── run_api.py             # 启动API服务

tests/                     # 测试文件
├── conftest.py            # 测试配置
├── test_agent_capabilities.py # Agent能力测试
├── test_chromadb.py       # 数据库测试
├── test_comprehensive.py  # 综合测试
└── test_config_system.py  # 配置系统测试

config/                    # 配置文件
└── app_config.json        # 应用配置

docs/                      # 文档
├── README.md              # 项目说明
├── architecture.md        # 架构文档
├── getting_started.md     # 快速开始
├── api.md                 # API文档
├── tests.md               # 测试指南
├── extensions.md          # 扩展指南
├── faq.md                 # 常见问题
└── rag/                   # RAG相关文档
    ├── README.md          # RAG总索引
    ├── overview.md        # RAG总览
    ├── ingestion.md       # 数据接入
    ├── processing.md      # 数据处理
    ├── vectorization.md   # 向量化
    ├── retrieval.md       # 检索
    ├── application.md     # 应用层
    ├── management.md      # 管理与运维
    └── patterns.md        # 设计模式
```

## 架构特点

- **分层清晰**：核心业务、基础设施、RAG 功能、API 接口各司其职
- **基础设施完备**：数据库、会话、缓存、异常、监控等基础设施齐全
- **管理与运维**：内置监控、配置热更新、审计日志、健康检查等运维工具
- **RAG 工具化**：数据处理、质量检测、分块等工具模块化，便于复用

## 管理与运维工具使用方式

### 1. 接口形式（推荐）

管理与运维工具可以通过 API 接口暴露给外部运维平台：

```python
# 在 API 层添加运维端点
from app.infra.management import get_metrics_summary, get_system_health

@app.get("/metrics")
async def get_metrics():
    """获取系统监控指标"""
    return get_metrics_summary()

@app.get("/health")
async def health_check():
    """系统健康检查"""
    return get_system_health()
```

### 2. 代码中直接集成

在业务代码中集成监控和审计：

```python
# 在关键操作中添加监控
from app.infra.management.metrics import monitor_performance, record_processing_rate
from app.infra.management.audit import log_audit_action, AuditAction

@monitor_performance("file_upload")
async def upload_files(files):
    # 记录审计日志
    log_audit_action(
        action=AuditAction.UPLOAD,
        resource_type="file",
        user_id=user_id,
        details={"file_count": len(files)}
    )

    # 业务逻辑...
    record_processing_rate("file_processing", duration, len(files))
```

### 3. 配置热更新

支持运行时配置更新：

```python
# 应用启动时初始化
from app.infra.management.config_watcher import initialize_config_watcher

# 初始化配置监控
config_watcher = initialize_config_watcher("config/app_config.json")

# 添加配置变更回调
def on_config_change(new_config):
    # 处理配置更新
    update_model_settings(new_config)

config_watcher.add_callback(on_config_change)
```

欢迎根据业务需要进行裁剪与扩展。
