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
- 性能优化指南：[docs/performance_optimization.md](docs/performance_optimization.md)

## RAG 分章节文档

- 总索引：[docs/rag/README.md](docs/rag/README.md)
- 总览：[docs/rag/overview.md](docs/rag/overview.md)
- 数据接入层：[docs/rag/ingestion.md](docs/rag/ingestion.md)
- 数据处理层（多语言/多媒体）：[docs/rag/processing.md](docs/rag/processing.md)
- 向量化层（多语言嵌入）：[docs/rag/vectorization.md](docs/rag/vectorization.md)
- 向量库工程化实践（ChromaDB）：[docs/rag/vector_store_practices.md](docs/rag/vector_store_practices.md)
- 检索层（召回+重排，含语言路由、复用应用层分词器做预算）：[docs/rag/retrieval.md](docs/rag/retrieval.md)
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
│   ├── logging/            # 日志配置
│   │   ├── __init__.py
│   │   ├── config.py             # 日志配置管理器
│   │   └── example.py            # 日志使用示例
│   ├── tokenization/       # 分词计数（复用应用层模型配置）
│   │   └── provider.py           # TokenCounter：openai→tiktoken；hf→AutoTokenizer
│   ├── models/             # 通用模型加载器
│   │   ├── __init__.py
│   │   ├── registry.py           # 模型加载器注册表
│   │   ├── examples.py           # 使用示例
│   │   └── loaders/              # 模型加载器实现
│   │       ├── __init__.py
│   │       ├── base.py           # 加载器基类和核心类
│   │       ├── sentence_transformer.py # SentenceTransformer 加载器
│   │       ├── transformers.py   # Transformers 加载器
│   │       ├── whisper.py        # Whisper 加载器
│   │       ├── clip.py           # CLIP 加载器
│   │       └── cross_encoder.py   # CrossEncoder 加载器
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
│   │   ├── media/          # 媒体处理模块
│   │   │   ├── chunking/         # 媒体分块策略
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py       # 分块策略基类
│   │   │   │   ├── factory.py    # 分块策略工厂
│   │   │   │   ├── text.py       # 文本分块策略
│   │   │   │   ├── pdf.py        # PDF分块策略
│   │   │   │   ├── image.py      # 图像分块策略
│   │   │   │   └── video.py      # 视频分块策略
│   │   │   └── extractors/       # 媒体内容提取器
│   │   │       ├── __init__.py
│   │   │       ├── base.py       # 提取器基类
│   │   │       ├── factory.py    # 提取器工厂
│   │   │       ├── image_vision.py # 图像视觉理解提取器
│   │   │       ├── image_ocr.py  # 图像OCR提取器
│   │   │       ├── video.py      # 视频内容提取器
│   │   │       ├── audio.py      # 音频内容提取器
│   │   │       └── audio_video_base.py # 音频和视频提取器基类
│   │   ├── interfaces.py   # 接口定义
│   │   ├── pipeline.py     # 处理流水线
│   │   ├── media_text.py   # 文本处理
│   │   ├── media_markdown.py # Markdown处理
│   │   ├── lang_zh.py      # 中文处理器
│   │   ├── quality_checker.py # 质量评估
│   │   └── ...
│   └── service/            # RAG服务层
│       └── ingestion_helpers.py # 接入辅助函数
│   ├── retrieval/          # 检索层
│   │   ├── __init__.py
│   │   ├── schemas.py             # 数据模型
│   │   ├── retriever.py           # 向量/关键词检索
│   │   ├── hybrid.py              # 融合（RRF/加权）
│   │   └── context_builder.py     # 上下文拼接（真实 token 预算）
├── api/                    # API接口层
│   ├── __init__.py         # FastAPI应用创建
│   ├── agent.py            # Agent对话接口
│   ├── health.py           # 健康检查接口
│   └── ingestion.py        # 数据接入接口
├── llm_adapters/          # LLM适配器（包化）
│   ├── base.py             # 适配器基类/协议
│   ├── factory.py          # 提供商工厂与缓存
│   ├── google/             # Google 提供商包（native）
│   │   ├── __init__.py
│   │   └── chat.py         # Chat 模型适配
│   ├── openai/             # OpenAI 提供商包（native）
│   │   ├── __init__.py
│   │   └── chat.py         # Chat 模型适配
│   └── backends/
│       └── hf/             # 通用本地推理后端（HF）
│           ├── __init__.py
│           └── chat.py     # 通用 HF CausalLM 适配
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
- **配置驱动**：服务器配置从 `config/app_config.json` 读取，支持灵活部署

## 服务启动与配置

### 启动方式

项目支持两种启动方式：

#### 1. 直接启动（推荐）

```bash
python main.py
```

#### 2. 使用 uvicorn 命令

```bash
# 开发环境（支持热重载）
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 生产环境
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 服务器配置

服务器配置通过 `config/app_config.json` 中的 `server` 部分进行管理：

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": true,
    "description": "API服务器配置"
  }
}
```

**配置参数说明**：

- **host**: 服务器监听地址

  - `"0.0.0.0"`: 监听所有网络接口（推荐用于容器化部署）
  - `"127.0.0.1"`: 仅监听本地回环地址（仅本机访问）
  - `"localhost"`: 等同于 `127.0.0.1`

- **port**: 服务器端口号

  - 默认: `8000`
  - 生产环境建议使用: `8080`、`80`、`443` 等标准端口

- **reload**: 热重载开关
  - `true`: 开发环境，代码变更时自动重启服务
  - `false`: 生产环境，提高性能稳定性

### 热重载机制

项目实现了智能的热重载机制：

```python
# main.py 中的实现逻辑
if reload:
    # 使用字符串形式启用热重载
    uvicorn.run("main:app", host=host, port=port, reload=True)
else:
    # 直接传递应用对象
    uvicorn.run(app, host=host, port=port, reload=False)
```

**热重载特性**：

- ✅ **开发友好**: 代码修改后自动重启，无需手动操作
- ✅ **性能优化**: 生产环境关闭热重载，避免性能开销
- ✅ **配置驱动**: 通过配置文件控制，无需修改代码
- ✅ **错误处理**: 自动处理 uvicorn 的 reload 参数要求

### 环境部署建议

#### 开发环境

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 8000,
    "reload": true
  }
}
```

#### 生产环境

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8080,
    "reload": false
  }
}
```

#### Docker 部署

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": false
  }
}
```

### 启动日志示例

正常启动时的日志输出：

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [5328] using WatchFiles
Connected to: <socket.socket fd=692, family=2, type=1, proto=0, laddr=('127.0.0.1', 49742), raddr=('127.0.0.1', 49720)>.
2025-09-12 10:16:59,015 - watchfiles.main - INFO - 5 changes detected
```

**日志说明**：

- `Uvicorn running on http://0.0.0.0:8000`: 服务启动成功
- `Started reloader process`: 热重载进程启动
- `Connected to`: 数据库连接成功
- `changes detected`: 文件监控检测到变更

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

### 4. 日志系统

全局日志配置，支持控制台和文件输出：

```python
# 在代码中使用日志
from app.infra.logging import get_logger

logger = get_logger(__name__)
logger.info("这是一条日志信息")
logger.error("这是一条错误日志")
```

**日志配置**（在 `app_config.json` 中）：

```json
{
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": {
      "console": {
        "enabled": true,
        "level": "INFO"
      },
      "file": {
        "enabled": true,
        "level": "DEBUG",
        "filename": "logs/chocolate.log",
        "use_daily_rotation": true,
        "rotation_when": "midnight",
        "rotation_interval": 1,
        "backup_count": 30,
        "encoding": "utf-8"
      }
    }
  }
}
```

**参数说明**：

- **logging.level**：全局日志级别阈值（DEBUG/INFO/WARNING/ERROR/CRITICAL）。低于该级别的日志会被丢弃。
- **logging.format**：日志格式模板，占位符如`%(asctime)s`（时间）、`%(name)s`（日志器名）、`%(levelname)s`（级别）、`%(message)s`（消息）。
- **logging.handlers.console.enabled**：是否启用控制台输出。
- **logging.handlers.console.level**：控制台输出的最低级别（应不低于全局级别）。
- **logging.handlers.file.enabled**：是否启用文件输出。
- **logging.handlers.file.level**：文件输出的最低级别（通常设为 DEBUG 以便留存更多排障信息）。
- **logging.handlers.file.filename**：基础日志文件名（如`logs/chocolate.log`）。当启用按日期轮转时，系统会自动追加日期生成当日文件，如`logs/chocolate_2025-09-10.log`。
- **logging.handlers.file.use_daily_rotation**：是否启用“按日期轮转”。开启后每天午夜自动切换新文件，并按`backup_count`保留历史文件。
- **logging.handlers.file.rotation_when**：日期轮转触发点，默认`midnight`。可选值还包括`S`/`M`/`H`/`D`/`W0`~`W6`（由 TimedRotatingFileHandler 解析）。
- **logging.handlers.file.rotation_interval**：轮转间隔数值（与`rotation_when`配合使用）。例如按天轮转时写`1`表示每天生成新文件。
- **logging.handlers.file.backup_count**：历史日志保留数量（天/份）。超出后自动删除最旧文件。
- **logging.handlers.file.encoding**：日志文件编码（默认`utf-8`），避免中文乱码。
- （仅在未启用按日期轮转时可用）**logging.handlers.file.max_bytes**：按大小轮转的单文件大小上限（字节），与`backup_count`配合使用。
- （互斥说明）当`use_daily_rotation=true`时使用“按日期轮转”，忽略`max_bytes`；当`use_daily_rotation=false`时使用“按大小轮转”，此时可配置`max_bytes`与`backup_count`。

**特性**：

- **按日期轮转**：每天午夜自动创建新日志文件（如 `chocolate_2025-01-15.log`）
- **保留历史**：保留最近 30 天的日志文件
- **双重输出**：控制台和文件同时输出
- **第三方库控制**：自动设置第三方库日志级别为 WARNING
- **灵活配置**：支持按日期或按大小轮转两种模式

欢迎根据业务需要进行裁剪与扩展。
