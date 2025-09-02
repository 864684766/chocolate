# Chocolate: 可扩展的 LangChain Agent 工程

这是一个基于 **LangChain** 和 **FastAPI** 的智能对话系统项目，专为初学者设计。本项目帮你快速理解并实践三个核心技术：

## 📚 核心技术简介

### 1. LangChain 是什么？

**LangChain** 是一个用于构建语言模型应用的框架，它帮助开发者：

- 将大语言模型（如 ChatGPT、Google Gemini）集成到应用中
- 让 AI 能够调用外部工具（如搜索、计算器、API 接口）
- 管理对话记忆和上下文

**简单理解**：就像给 AI 助手提供了一个"工具箱"，让它不仅能聊天，还能帮你搜索信息、计算数学题、调用 API 等。

### 2. FastAPI 是什么？

**FastAPI** 是一个现代的 Python Web 框架，用于构建 API 接口：

- 自动生成 API 文档
- 类型检查和数据验证
- 高性能且易于使用

**简单理解**：就像搭建一个"服务台"，前端或其他程序可以通过 HTTP 请求来调用你的 AI 功能。

### 3. Agent 是什么？

**Agent** 是 AI 领域的概念，指能够：

- 理解用户需求
- 自主选择合适的工具
- 执行任务并返回结果

**简单理解**：就像一个聪明的助理，会根据你的问题自动判断需要用什么工具来解决。

## 🎯 项目特色

- **多模型支持**：轻松切换不同的大语言模型（Google Gemini、OpenAI GPT 等）
- **工具扩展**：内置搜索、HTTP 请求、计算器等工具，可按需添加更多
- **模块化 API**：按业务功能组织代码，便于维护和扩展
- **友好文档**：详细的注释和示例，适合学习和二次开发

## 🚀 快速开始（从零到可用）

1. 安装 Python 3.10+（Windows/Mac/Linux 均可）。
2. 克隆或下载本项目到本地。
3. 进入项目目录并创建虚拟环境（推荐）：

```
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate
```

4. 安装依赖：

```
pip install -r requirements.txt
```

5. 配置应用（编辑 config/app_config.json）：

```
# 关键位置：
# providers.google.api_key          # Google API Key
# providers.google.models.*         # 各模型的温度/别名/安全阈值/generation_config
# llm.default_provider, default_model, default_temperature, request_timeout
```

6. 启动 API 服务（默认 http://localhost:8000）：

```
python -m scripts.run_api
```

7. 使用命令行客户端调用（通过 API）：

- 单轮模式：

```
python main.py -q "你好，帮我写一首关于春天的五言绝句"
```

- 交互模式：

```
python main.py
```

- Demo（两段示例，依然通过 API 调用）：

```
python main.py --demo
```

8. 用浏览器打开自动文档（FastAPI 提供）：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

如果你遇到“连接失败”，请先确认第 6 步中的 API 服务已成功启动。

---

## 目录结构

```
chocolate/
  ├─ app/
  │  ├─ __init__.py
  │  ├─ config.py          # 配置中心（读取 config/app_config.json）
  │  ├─ llm.py             # LLM 统一入口（委托给工厂获取聊天模型）
  │  ├─ llm_adapters/      # LLM 适配器层（原 providers）
  │  │  ├─ __init__.py
  │  │  ├─ factory.py      # 工厂 + 注册表，按配置选择适配器
  │  │  └─ google.py       # Google Gemini 适配器示例
  │  ├─ tools/             # 工具层（search/http/calc 等）
  │  │  ├─ __init__.py
  │  │  ├─ calculator.py
  │  │  ├─ http.py
  │  │  └─ search.py
  │  ├─ tools.py           # 工具聚合（示例便捷导出）
  │  ├─ agent.py           # 构建完整的 ReAct Agent 执行器（create_react_agent + AgentExecutor）
  │  └─ api/               # FastAPI 模块化路由（类似 Koa2 的业务拆分）
  │     ├─ __init__.py     # create_app() / include_router() 总装配
  │     ├─ health.py       # 系统健康检查
  │     └─ agent.py        # Agent 相关端点（/agent/invoke）
  ├─ scripts/
  │  ├─ __init__.py
  │  ├─ run_agent.py       # CLI 启动：交互/单轮
  │  └─ run_api.py         # Web 启动：uvicorn app.api:app --reload
  ├─ main.py               # 演示入口（可选保留）
  ├─ requirements.txt
  └─ README.md
```

注：为兼容历史代码，`app/api.py` 仍导出 `app = create_app()`，启动脚本可继续使用 `app.api:app`。

## 配置系统

### 配置文件结构

项目使用基于 JSON 文件的配置管理，为后续接入 nacos 做准备。主要配置文件为 `config/app_config.json`，包含以下配置项：

#### 1. 应用基础配置 (`app`)

- `name`: 应用名称
- `version`: 应用版本
- `description`: 应用描述

#### 2. 服务器配置 (`server`)

- `host`: 服务器监听地址
- `port`: 服务器端口
- `reload`: 是否启用热重载

#### 3. 大语言模型配置 (`llm`)

- `default_provider`: 默认提供商
- `default_model`: 默认模型
- `default_temperature`: 默认温度参数
- `request_timeout`: 请求超时时间

#### 4. 缓存配置 (`cache`)

- `max_cache_size`: 最大缓存数量

#### 5. Agent 配置 (`agent`)

- `verbose`: 是否显示详细日志
- `max_iterations`: 最大迭代次数
- `max_execution_time`: 最大执行时间（秒）
- `handle_parsing_errors`: 是否处理解析错误
- `return_intermediate_steps`: 是否返回中间步骤

#### 6. 提供商配置 (`providers`)

以提供商为 key，包含各提供商的 API 密钥和模型配置：

- `api_key`: 提供商的 API 密钥
- `description`: 提供商描述
- `models`: 该提供商支持的模型列表，每个模型包含：
  - `description`: 模型描述
  - `max_tokens`: 最大 token 数
  - `temperature`: 温度参数

#### 7. 模型别名 (`aliases`)

每个模型可以配置多个别名，用于快捷访问：

- `aliases`: 字符串数组，包含该模型的别名列表
- 例如：`"aliases": ["gpt4"]` 表示可以通过 `ai_type="gpt4"` 来使用该模型

#### 8. 环境配置 (`environment`)

- `description`: 环境配置描述

#### 9. 工具配置 (`tools`)

- `available_tools`: 可用工具列表

#### 10. 提示词配置 (`prompts`)

- `react_template`: ReAct Agent 的提示词模板
- `final_output_formatter`: 最终输出格式化提示词模板

### 配置说明

配置文件采用 JSON 格式，所有配置项都集中在 `config/app_config.json` 中管理。主要特点：

1. **以提供商为中心**: 配置按提供商组织，每个提供商包含其 API 密钥和支持的模型列表
2. **模型配置独立**: 每个模型都有独立的配置参数（max_tokens、temperature 等）
3. **模型别名**: 每个模型可以配置多个别名，提供快捷访问方式
4. **提示词配置化**: 所有提示词模板都配置化，便于调整和优化

**重要**:

- 使用前请将配置文件中的 API 密钥占位符替换为实际的 API 密钥
- 配置文件不存在时会抛出错误，确保配置文件存在且格式正确

### 配置使用方法

```python
from app.config import get_config_manager, get_config

# 获取配置管理器
config_manager = get_config_manager()

# 获取特定配置项
server_config = config_manager.get_server_config()
agent_config = config_manager.get_agent_config()
cache_config = config_manager.get_cache_config()

# 或者使用便捷函数
all_config = get_config()
llm_config = get_config("llm")
```

### 配置系统重构说明

本项目已完成配置系统的重大重构，主要变更包括：

#### 1. 配置文件结构重新设计

- **以提供商为中心**: 配置按提供商组织，每个提供商包含其 API 密钥和支持的模型列表
- **支持多种模型**: 每个提供商可配置多个模型（如 gemini-2.5-flash、gemini-2.5-pro 等）
- **模型别名**: 每个模型可以配置多个别名，提供快捷访问方式

#### 2. 环境变量完全移除

- 删除 python-dotenv 依赖
- 移除所有环境变量相关代码
- 所有配置统一在 JSON 文件中管理

#### 3. 提示词配置化

- 将 ReAct 模板配置化，便于调整和优化
- 简化提示词结构，移除冗余的格式化步骤

#### 4. 错误处理优化

- 配置文件不存在时直接抛出错误，避免默认配置的冗余
- 提供清晰的错误信息，便于问题定位

### 接入 Nacos 的准备

当前配置系统已经为接入 nacos 做好了准备：

1. **配置结构标准化**: 所有配置项都统一在 JSON 文件中管理
2. **配置分层**: 按功能模块划分配置项，便于 nacos 中的命名空间管理
3. **配置获取接口**: 提供了统一的配置获取接口，便于替换为 nacos 客户端
4. **配置重载机制**: 支持配置热重载，便于 nacos 配置变更监听

接入 nacos 时的主要修改点：

1. 在`ConfigManager`中添加 nacos 客户端
2. 修改`_load_config`方法，从 nacos 获取配置
3. 添加 nacos 配置变更监听器
4. 在`reload_config`方法中触发 nacos 配置刷新

## 安装与运行

- 安装依赖：

```
pip install -r requirements.txt
```

- 启动 Web API（默认 http://localhost:8000/）：

```
python -m scripts.run_api
```

- 交互式 CLI：

```
python -m scripts.run_agent
```

## Web API 说明（含示例）

- 健康检查：GET /health
  - 响应：包含系统状态、可用 AI 类型、缓存信息等
- 缓存信息：GET /cache-info
  - 响应：显示当前缓存的模型和 Agent 链信息
- 清除缓存：POST /clear-cache

  - 响应：清除所有缓存并返回状态

- 调用 Agent：POST /agent/invoke

  - 请求体（JSON）：
    ```json
    {
      "input": "你好，请用一句话解释量子计算",
      "ai_type": "gpt4", // 可选：指定AI类型（gpt4, gpt35, gemini, claude）
      "provider": "google", // 可选：直接指定提供商
      "session_id": "可选字符串，用于区分会话（开启会话记忆）"
    }
    ```
  - 响应体（JSON）：
    ```json
    {
      "answer": "这是模型生成的回答文本"
    }
    ```

- 关于会话记忆：

  - 当提供相同的 session_id 时，服务端会在本次请求中携带此前的对话历史，使 Agent 在 ReAct 多步推理的基础上，能够基于上下文进行更合理的工具调用与回答。
  - 未提供 session_id 时，会使用临时访客会话（guest），不同请求之间不共享历史。

- ReAct Agent 能力：

  - Agent 会在一次请求内执行若干轮“思考-行动-观察”循环，自主决定是否调用工具（search_docs/http_get/calc），并最终给出答案。
  - 你可以在输入中直接给出纠错信息，或在同一 session_id 下连续发起请求以实现“反馈-修正-再回答”的闭环。

- curl 调用示例：

  ```bash
  curl -X POST http://localhost:8000/agent/invoke \
       -H "Content-Type: application/json" \
       -d '{"input": "帮我计算 2+3*4", "session_id": "demo-1"}'
  ```

- Python 代码调用示例：

  ```python
  import requests

  sid = "demo-1"
  resp1 = requests.post(
      "http://localhost:8000/agent/invoke",
      json={"input": "先抓取 https://example.com", "session_id": sid},
      timeout=60
  )
  print(resp1.json())

  # 在同一会话中继续追问，Agent 将记住上一轮操作与上下文
  resp2 = requests.post(
      "http://localhost:8000/agent/invoke",
      json={"input": "刚才页面里标题是什么？", "session_id": sid},
      timeout=60
  )
  print(resp2.json())
  ```

当前实现：

- 同步一次性返回（非流式）；支持会话记忆（内存型，可替换为 Redis/DB）。
- 可扩展方向：SSE/WebSocket 流式输出、会话持久化、鉴权与限流、统一日志与追踪。

## 🧪 运行测试

项目提供了完整的测试用例来验证核心功能，测试文件统一放置在 `tests/` 目录下。

### 前置要求

如需使用 pytest 框架运行测试，需额外安装测试依赖：

```bash
pip install pytest
```

### 测试运行方式

#### 方式一：使用 pytest（推荐）

```bash
# 运行所有测试
pytest tests/

# 运行特定测试文件
pytest tests/test_agent_capabilities.py
pytest tests/test_comprehensive.py

# 详细输出模式
pytest tests/ -v

# 显示测试覆盖的函数和执行过程
pytest tests/ -v -s
```

#### 方式二：直接运行测试文件

每个测试文件都支持独立运行，内置了简化的测试执行逻辑：

```bash
# 运行 Agent 能力测试
python tests/test_agent_capabilities.py

# 运行综合功能测试
python tests/test_comprehensive.py
```

### 现有测试文件说明

- **`tests/test_agent_capabilities.py`**：
  - 测试 ReAct Agent 执行器的核心功能
  - 验证会话记忆管理机制
  - 检查工具（计算器、搜索、HTTP）的基本可用性
- **`tests/test_comprehensive.py`**：
  - 综合验证项目重构后的稳定性
  - 测试工厂模式和模块导入
  - 验证向后兼容性

### 编写新测试文件的指南

当您需要为项目添加新的测试时，请遵循以下约定：

1. **文件命名**：使用 `test_*.py` 格式，放置在 `tests/` 目录下
2. **导入路径**：使用项目根目录的相对导入，如 `from app.agent import build_agent`
3. **测试类**：建议使用 `Test*` 命名的类来组织相关测试用例
4. **独立运行支持**：在文件末尾添加 `if __name__ == "__main__":` 块来支持直接运行

示例测试文件结构：

```python
"""
测试描述
"""
import pytest
from app.your_module import your_function

class TestYourFeature:
    """测试您的功能"""

    def test_basic_functionality(self):
        """测试基本功能"""
        result = your_function()
        assert result is not None

    def test_edge_cases(self):
        """测试边界情况"""
        # 您的测试逻辑
        pass

if __name__ == "__main__":
    # 支持直接运行的简化测试逻辑
    test_instance = TestYourFeature()
    test_instance.test_basic_functionality()
    print("✓ All tests passed!")
```

### 测试最佳实践

- 在提交代码前运行完整测试：`pytest tests/`
- 为新功能编写对应的测试用例
- 使用有意义的测试用例名称和注释
- 优先测试核心功能和边界情况

## 🔄 本次升级新增内容（v2.0 版本）

### 升级概览

本项目在原有基础上进行了重大升级，将轻量级的策略链提升为功能完整的 ReAct Agent 执行器，并引入会话记忆功能。同时进行了性能优化和架构改进。

### 🚀 性能优化亮点

#### 1. 智能缓存系统

- **模型实例缓存**：基于 `(ai_type, provider)` 的 LRU 缓存，避免重复创建
- **Agent 链缓存**：智能缓存 Agent 链实例，提升响应速度
- **自动内存管理**：LRU 淘汰策略，防止内存泄漏

#### 2. 配置管理优化

- **AI 类型映射**：支持 `ai_type` 参数自动映射到对应的提供商和模型
- **动态配置**：支持运行时添加新的 AI 类型映射
- **多配置实例**：解决了全局缓存导致的配置冲突问题

#### 3. 监控和管理接口

- **健康检查**：`/health` 接口显示系统状态和缓存信息
- **缓存管理**：`/cache-info` 和 `/clear-cache` 接口
- **实时监控**：缓存使用率、可用 AI 类型等统计信息

### 核心变化对比

| 特性       | 原版本             | 新版本                         |
| ---------- | ------------------ | ------------------------------ |
| Agent 类型 | 简化的关键词策略链 | 完整的 ReAct Agent 执行器      |
| 推理模式   | 单次工具调用       | 多步"思考-行动-观察"循环       |
| 会话记忆   | 无                 | 基于 session_id 的会话历史管理 |
| 错误处理   | 基础               | 内置最大迭代次数和超时保护     |
| 工具集成   | 硬编码             | LangChain 工具装饰器标准集成   |

### 新增核心组件详解

#### 1. ReAct Agent 执行器 (<mcfile name="agent.py" path="d:\test\chocolate\app\agent.py"></mcfile>)

**核心函数：** <mcsymbol name="build_agent" filename="agent.py" path="d:\test\chocolate\app\agent.py" startline="10" type="function"></mcsymbol>

```python
def build_agent() -> AgentExecutor:
    """构建 ReAct Agent 执行器，支持多步推理和工具调用"""
```

**关键特性：**

- 使用 `create_react_agent` 创建标准 ReAct 风格的推理代理
- 配置 `AgentExecutor` 支持最大迭代次数（10 次）和超时保护（120 秒）
- 集成三大工具：搜索、HTTP 请求、计算器
- 自动处理"思考-行动-观察"循环，直到得出最终答案

**工作原理：**

1. 接收用户输入
2. Agent 分析是否需要调用工具
3. 如需工具，执行工具并观察结果
4. 基于观察结果继续思考或给出最终答案
5. 重复 2-4 步，直到完成任务或达到限制

#### 2. 会话记忆管理 (<mcfile name="agent.py" path="d:\test\chocolate\app\api\agent.py"></mcfile>)

**会话历史函数：** <mcsymbol name="get_session_history" filename="agent.py" path="d:\test\chocolate\app\api\agent.py" startline="16" type="function"></mcsymbol>

```python
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """基于 session_id 获取或创建会话历史"""
```

**API 接口函数：** <mcsymbol name="agent_invoke" filename="agent.py" path="d:\test\chocolate\app\api\agent.py" startline="29" type="function"></mcsymbol>

```python
def agent_invoke(req: InvokeRequest):
    """带会话记忆的 Agent 调用接口"""
```

**实现机制：**

- 使用 `InMemoryChatMessageHistory` 在内存中维护每个会话的对话历史
- 通过 `RunnableWithMessageHistory` 包装 Agent，自动注入历史上下文
- 支持同一 session_id 下的多轮对话连续性

#### 3. 升级后的依赖

新增关键依赖项（已添加到 <mcfile name="requirements.txt" path="d:\test\chocolate\requirements.txt"></mcfile>）：

```
langchain>=0.1.0         # 核心 LangChain 库，提供 AgentExecutor
langchain-community>=0.0.20  # 社区工具和组件
```

### 向后兼容性说明

✅ **API 接口保持兼容**：`/agent/invoke` 接口签名不变
✅ **响应格式保持兼容**：仍返回 `{"answer": "..."}`
✅ **工具功能保持兼容**：search_docs、http_get、calc 工具行为不变
✅ **CLI 脚本保持兼容**：<mcfile name="run_agent.py" path="d:\test\chocolate\scripts\run_agent.py"></mcfile> 和 <mcfile name="main.py" path="d:\test\chocolate\main.py"></mcfile> 正常工作

⚠️ **新增可选功能**：session_id 为可选参数，不影响现有调用方式

## 📖 代码阅读指南（初学者友好版）

### 第一层：项目整体理解

**推荐阅读顺序**：先理解每个模块的作用，再深入具体实现。

#### 1️⃣ 入口点理解（3 个主要入口）

- **Web API 入口**：<mcfile name="run_api.py" path="d:\test\chocolate\scripts\run_api.py"></mcfile>

  - 作用：启动 HTTP 服务，提供 `/agent/invoke` 接口
  - 查看重点：如何启动 FastAPI 应用

- **交互式 CLI 入口**：<mcfile name="run_agent.py" path="d:\test\chocolate\scripts\run_agent.py"></mcfile>

  - 作用：直接与 Agent 交互，无需启动服务
  - 查看重点：<mcsymbol name="main" filename="run_agent.py" path="d:\test\chocolate\scripts\run_agent.py" startline="6" type="function"></mcsymbol> 函数的交互循环

- **命令行客户端**：<mcfile name="main.py" path="d:\test\chocolate\main.py"></mcfile>
  - 作用：通过 HTTP 调用 API 服务的客户端
  - 查看重点：如何构建请求和处理响应

#### 2️⃣ 配置系统理解

- **配置加载**：<mcfile name="config.py" path="d:\test\chocolate\app\config.py"></mcfile>
  - 作用：从 `config/app_config.json` 读取配置
  - 查看重点：`providers`/`llm`/`prompts` 等配置项
  - **关键提示**：不再使用环境变量，密钥与模型均在 JSON 中集中管理

### 第二层：核心功能深入

#### 3️⃣ AI 模型抽象层

- **统一接口**：<mcfile name="llm.py" path="d:\test\chocolate\app\llm.py"></mcfile>

  - 作用：对外提供统一的 `get_chat_model()` 函数
  - 查看重点：如何委托给工厂模式

- **工厂模式**：<mcfile name="factory.py" path="d:\test\chocolate\app\llm_adapters\factory.py"></mcfile>

  - 作用：根据配置动态选择不同的 AI 模型提供商
  - 查看重点：`LLMProviderFactory` 类的注册表机制

- **具体适配器**：<mcfile name="google.py" path="d:\test\chocolate\app\llm_adapters\google.py"></mcfile>
  - 作用：封装 Google Gemini 模型的具体调用
  - 查看重点：如何配置模型参数

#### 4️⃣ 工具系统（Agent 的"手脚"）

**阅读顺序**：先看单个工具，再看集成方式

- **计算器工具**：<mcfile name="calculator.py" path="d:\test\chocolate\app\tools\calculator.py"></mcfile>

  - 新手友好度：⭐⭐⭐⭐⭐（最简单）
  - 关键代码：`@tool` 装饰器的使用
  - 学习重点：如何将普通函数转换为 LangChain 工具

- **HTTP 工具**：<mcfile name="http.py" path="d:\test\chocolate\app\tools\http.py"></mcfile>

  - 新手友好度：⭐⭐⭐⭐
  - 学习重点：错误处理和超时机制

- **搜索工具**：<mcfile name="search.py" path="d:\test\chocolate\app\tools\search.py"></mcfile>

  - 新手友好度：⭐⭐⭐
  - 学习重点：如何与外部搜索 API 集成

- **工具集成**：<mcfile name="__init__.py" path="d:\test\chocolate\app\tools\__init__.py"></mcfile>
  - 作用：统一导出所有工具，便于 Agent 调用
  - 查看重点：导出列表的管理

#### 5️⃣ Agent 核心（大脑）

**⚠️ 建议有一定基础后再深入，这是最复杂的部分**

- **Agent 构建器**：<mcfile name="agent.py" path="d:\test\chocolate\app\agent.py"></mcfile>
  - 新手友好度：⭐⭐（需要理解 LangChain 概念）
  - 核心函数：<mcsymbol name="build_agent" filename="agent.py" path="d:\test\chocolate\app\agent.py" startline="10" type="function"></mcsymbol>
  - 学习重点：
    1. `create_react_agent` 如何创建推理代理
    2. `AgentExecutor` 如何控制执行流程
    3. 工具列表如何传递给 Agent
    4. 最大迭代次数和超时设置的作用

**ReAct 工作原理图解**：

```
用户输入 → Agent分析 → 需要工具？
                        ↓是
                    选择并调用工具
                        ↓
                    观察工具结果
                        ↓
                    继续思考 → 是否完成？
                        ↓否        ↓是
                    ←─────────   输出最终答案
```

### 第三层：API 和服务层

#### 6️⃣ FastAPI 应用结构

- **应用工厂**：<mcfile name="__init__.py" path="d:\test\chocolate\app\api\__init__.py"></mcfile>

  - 作用：创建 FastAPI 应用并注册路由
  - 学习重点：<mcsymbol name="create_app" filename="__init__.py" path="d:\test\chocolate\app\api\__init__.py" startline="10" type="function"></mcsymbol> 函数的模块化设计

- **健康检查**：<mcfile name="health.py" path="d:\test\chocolate\app\api\health.py"></mcfile>

  - 新手友好度：⭐⭐⭐⭐⭐（最简单的 API 示例）
  - 学习重点：FastAPI 路由的基本写法

- **Agent API**：<mcfile name="agent.py" path="d:\test\chocolate\app\api\agent.py"></mcfile>
  - 新手友好度：⭐⭐⭐
  - 核心功能：<mcsymbol name="agent_invoke" filename="agent.py" path="d:\test\chocolate\app\api\agent.py" startline="29" type="function"></mcsymbol>
  - 学习重点：
    1. `RunnableWithMessageHistory` 如何包装 Agent
    2. <mcsymbol name="get_session_history" filename="agent.py" path="d:\test\chocolate\app\api\agent.py" startline="16" type="function"></mcsymbol> 会话管理机制
    3. 请求/响应模型的定义（Pydantic）

### 学习建议和读代码技巧

#### 🎯 分层学习法

1. **第一周**：熟悉项目结构，运行各种入口点，理解整体流程
2. **第二周**：深入工具系统，尝试添加新工具
3. **第三周**：理解 Agent 工作原理，研究 ReAct 模式
4. **第四周**：学习 API 设计，尝试添加新接口

#### 🔍 调试技巧

1. **加日志**：在关键函数入口加 `print` 语句观察数据流
2. **单步测试**：使用 CLI 模式测试单个功能
3. **API 测试**：使用 curl 或 Postman 测试接口
4. **分离测试**：单独测试工具函数，再测试集成

#### 📚 扩展学习资源

- **LangChain 官方文档**：https://python.langchain.com/docs/
- **FastAPI 官方教程**：https://fastapi.tiangolo.com/tutorial/
- **ReAct 论文解读**：搜索 "ReAct Reasoning and Acting in Language Models"

#### ⚠️ 常见理解误区

1. **误区**：认为 Agent 是黑盒子
   **正确理解**：Agent 是按照 ReAct 模式执行的确定性流程

2. **误区**：工具调用是随机的
   **正确理解**：LLM 会根据问题内容智能选择合适的工具

3. **误区**：会话记忆很复杂
   **正确理解**：就是在内存中保存对话历史，通过 session_id 区分

## 代码导读（从上到下看一遍）

- app/config.py：加载 .env 配置（如 PROVIDER、GOOGLE_API_KEY 等）。
- app/llm_adapters/：放置不同模型提供商的适配器（如 Google/Gemini）。
- app/llm.py：对外只暴露 get_chat_model()，内部调用工厂返回已配置好的模型实例。
- app/tools/：可调用的工具函数（搜索、HTTP、计算器），在 Agent 需要时被调用。
- app/agent.py：已升级为通用 ReAct Agent 执行器（create_react_agent + AgentExecutor），多步推理并自主调用工具。
- app/api/：FastAPI 的模块化路由，/health 和 /agent/invoke 都在这里；已基于 session_id 引入会话历史（RunnableWithMessageHistory）。
- scripts/run_api.py：启动 Web 服务。
- scripts/run_agent.py：交互式 CLI（本地直连模型）。
- main.py：命令行客户端，通过 HTTP 调用 /agent/invoke。

## 动手扩展：新增一个“翻倍计算”工具（示例）

目标：新增一个工具，把用户输入的数字翻倍，并让 Agent 会在输入包含“翻倍”时调用它。

1. 新增文件 app/tools/double.py：

```python
# app/tools/double.py
from langchain_core.tools import tool

@tool
def double_num(x: str) -> str:
    """把数字翻倍，参数 x 是字符串形式的数字。"""
    try:
        n = float(x)
        return str(n * 2)
    except Exception:
        return "请输入数字，例如：翻倍 12"
```

2. 在 app/tools/**init**.py 中导出：

```python
# app/tools/__init__.py
from .calculator import calc
from .http import http_get
from .search import search_docs
from .double import double_num  # 新增导出
```

3. 修改 app/agent.py 的 maybe_use_tool：

```python
# app/agent.py
# ... existing code ...
from .tools import search_docs, http_get, calc, double_num
# ... existing code ...
        elif "翻倍" in user_input:
            expr = user_input.replace("翻倍", "").strip()
            ctx = double_num.invoke(expr)
# ... existing code ...
```

4. 重启 API 服务并测试：

```
python -m scripts.run_api
# 另开一个终端：
curl -X POST http://localhost:8000/agent/invoke -H "Content-Type: application/json" -d '{"input":"翻倍 21"}'
```

如果响应返回 42，说明你已经完成了一个“新增工具 + 接入 Agent”的闭环！

## 动手扩展：新增一个业务 API 路由（示例）

目标：新增 /math/sum 接口，接收两个数字并返回求和结果。

1. 新增文件 app/api/math.py：

```python
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class SumReq(BaseModel):
    a: float
    b: float

class SumResp(BaseModel):
    result: float

@router.post("/sum", response_model=SumResp)
def sum_api(req: SumReq):
    return SumResp(result=req.a + req.b)
```

2. 在 app/api/**init**.py 中注册：

```python
from .math import router as math_router
# ... existing code ...
app.include_router(math_router, prefix="/math", tags=["math"])  # 新增注册
```

3. 重启 API 并测试：

```
python -m scripts.run_api
curl -X POST http://localhost:8000/math/sum -H "Content-Type: application/json" -d '{"a": 1.5, "b": 2.3}'
```

如果返回 {"result": 3.8}，说明一切正常。

---

## 设计与模式

- 工厂 + 注册表（LLM 适配层）
  - 位置：app/llm_adapters/factory.py
  - 作用：按 PROVIDER 动态选择适配器，解耦业务与具体大模型实现。
- 工具装饰器（工具层）
  - 位置：app/tools/
  - 作用：以 LangChain 的 @tool 将函数暴露为可调用工具，便于扩展和测试。
- 策略式工具选择（Agent 层）
  - 位置：app/agent.py
  - 作用：基于关键词的轻量策略，必要时调用工具后再组织回答。
- 配置集中管理
  - 位置：app/config.py
  - 作用：从 .env 读取配置并校验关键参数（如 GOOGLE_API_KEY）。
- 智能缓存系统
  - 位置：app/llm_adapters/factory.py, app/core/agent_service.py
  - 作用：LRU 缓存模型实例和 Agent 链，提升性能并防止内存泄漏。
- 服务层抽象
  - 位置：app/core/agent_service.py
  - 作用：统一管理 Agent 链的创建和缓存，提供监控接口。

## 如何新增一个业务 API 模块（Koa2 风格）

1. 在 `app/api/` 下新增文件，如 `orders.py`，定义你的路由。
2. 在 `app/api/__init__.py` 内引入并 `include_router()` 完成注册。
3. 按需添加 prefix、tags 以形成清晰的业务边界。

示例：新增 `app/api/orders.py` 并在 `__init__.py` 里 include 后，即可通过 `/orders/...` 对外提供接口。

## 如何扩展大模型适配器（以 OpenAI 为例）

1. 新增文件 app/llm_adapters/openai.py：

```python
# app/llm_adapters/openai.py
from typing import Any
from langchain_openai import ChatOpenAI

class OpenAIProvider:
    @staticmethod
    def build_chat_model() -> Any:
        # 这里简单示例，实际可从 app.config 里读取模型名、温度等
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
```

2. 在 app/llm_adapters/factory.py 中注册：

```python
from .openai import OpenAIProvider  # 新增导入
# 在 LLMProviderFactory._providers = { ... } 中加入：
# "openai": OpenAIProvider.build_chat_model,
```

3. 在 `config/app_config.json` 中设置：

```
{
  "providers": {
    "openai": {
      "api_key": "你的OpenAIKey",
      "models": {
        "gpt-4o-mini": { "description": "示例", "max_tokens": 8192, "temperature": 0.7, "aliases": ["gpt4o"] }
      }
    }
  },
  "llm": { "default_provider": "openai", "default_model": "gpt-4o-mini" }
}
```

4. 安装依赖：

```
pip install langchain-openai
```

现在，Agent 就会自动使用 OpenAI 的模型进行回答。

## 常见问题（FAQ）

- 启动报错缺少密钥？

  - 检查 `config/app_config.json` 的 `providers.<name>.api_key` 是否已填写

- API 调不通/连接失败？

  - 先启动服务：`python -m scripts.run_api`
  - 确认端口未被占用（默认 8000），可换端口：编辑 scripts/run_api.py

- 返回 500 错误？

  - 查看控制台日志，很多是因为未配置密钥/无网络/Provider 配置错误
  - 如果是工具导致的错误（比如 http_get 抓取失败），请检查输入参数

- 如何切换模型提供商？

  - 方法 1：修改 `llm.default_provider`（如 openai、google）
  - 方法 2：在 API 调用时指定 `ai_type` 或 `provider` 参数
  - 方法 3：在 `providers` 中为不同模型配置不同 `api_key`

- 如何启用流式输出？

  - 目前接口是一次性返回。可以在 FastAPI 中新增 SSE/WebSocket 路由，并在 Agent 端改用流式生成。

- 安全提示
  - 请勿把 API Key 写死在代码中，统一通过 `config/app_config.json`（或企业密钥管理服务）管理
  - 如果要对外提供服务，请加入鉴权（Token/签名）与基本限流
  - 注意 CORS 设置，避免被任意前端调用

---

文档已合并（原 docs/DEVELOPMENT.md、docs/PATTERNS.md 内容已纳入本 README）。

## ChromaDB 向量数据库集成

### ChromaDBHelper 使用指南

`ChromaDBHelper` 是一个完整的 ChromaDB 数据库助手类，提供了增删改查等完整的数据库操作功能。

#### 特性

- ✅ 连接管理和资源控制
- ✅ 完整的 CRUD 操作
- ✅ 异常处理和错误管理
- ✅ 上下文管理器支持
- ✅ 类型提示和文档
- ✅ 向后兼容性

#### 快速开始

```python
from app.core.chroma.db_helper import ChromaDBHelper

# 创建助手实例
helper = ChromaDBHelper()

try:
    # 自动连接数据库
    client = helper.get_client()

    # 获取或创建集合
    collection = helper.get_collection("my_collection")

    # 执行数据库操作...

finally:
    # 断开连接
    helper.disconnect()
```

#### 使用上下文管理器（推荐）

```python
with ChromaDBHelper() as helper:
    # 自动管理连接生命周期
    collection = helper.get_collection("my_collection")
    # 执行操作...
# 自动断开连接
```

#### API 参考

##### 连接管理

- **`connect()`**: 建立数据库连接，返回 `chromadb.ClientAPI` 实例
- **`disconnect()`**: 断开数据库连接
- **`get_client()`**: 获取数据库客户端实例，如果未连接则自动连接
- **`is_connected()`**: 检查是否已连接到数据库

##### 集合管理

- **`get_collection(name, create_if_not_exists=True)`**: 获取或创建集合
- **`list_collections()`**: 列出所有集合名称
- **`delete_collection(name)`**: 删除集合
- **`modify(name, name=None, metadata=None)`**: 修改集合属性

##### 数据操作

- **`add(collection_name, documents=None, embeddings=None, metadatas=None, ids=None, **kwargs)`\*\*: 向集合中添加数据
- **`query(collection_name, query_texts=None, query_embeddings=None, n_results=10, where=None, where_document=None, include=None, **kwargs)`\*\*: 查询集合中的数据
- **`get(collection_name, ids=None, where=None, where_document=None, include=None, **kwargs)`\*\*: 获取集合中的数据
- **`update(collection_name, ids, embeddings=None, metadatas=None, documents=None, **kwargs)`\*\*: 更新集合中的数据
- **`delete(collection_name, ids=None, where=None, where_document=None)`**: 删除集合中的数据

##### 辅助操作

- **`count(collection_name)`**: 获取集合中的数据数量
- **`peek(collection_name, limit=10)`**: 预览集合中的数据

#### 类型提示说明

##### Literal 类型

`Literal` 是 Python 类型提示系统中的一个特殊类型，用于限制变量或参数只能接受特定的字面量值。它提供了比普通字符串类型更严格的类型检查。

```python
from typing import Literal

# 在我们的代码中，IncludeField 限制了 include 参数只能是这些值：
IncludeField = Literal["documents", "embeddings", "metadatas", "distances", "uris", "data"]

# 使用示例
def query(include: Optional[List[IncludeField]] = None):
    # include 只能是 None 或者包含上述值的列表
    pass

# 正确的用法
query(include=["documents", "metadatas"])  # ✅ 正确
query(include=["embeddings", "distances"]) # ✅ 正确

# 错误的用法
query(include=["invalid_field"])           # ❌ 类型错误！
```

#### 配置要求

确保你的 `config/app_config.json` 包含向量数据库配置：

```json
{
  "vector_database": {
    "host": "localhost",
    "port": 8000
  }
}
```

#### 错误处理

所有方法都会抛出 `DatabaseConnectionError` 异常，建议使用 try-catch 进行错误处理：

```python
try:
    helper.add(
        collection_name="documents",
        documents=["测试文档"],
        ids=["test1"]
    )
    print("数据添加成功")
except DatabaseConnectionError as e:
    print(f"数据库操作失败: {e}")
```

#### 最佳实践

1. **使用上下文管理器**: 推荐使用 `with` 语句自动管理连接生命周期
2. **异常处理**: 始终使用 try-catch 处理可能的异常
3. **资源管理**: 及时断开不需要的连接
4. **批量操作**: 对于大量数据，考虑批量添加而不是逐条添加
5. **缓存策略**: 合理使用集合缓存，避免重复创建集合对象
6. **返回值处理**: 注意 `add`、`update`、`delete` 方法没有返回值，`query` 和 `get` 方法返回 ChromaDB 结果对象
7. **类型安全**: 使用 `Literal` 类型确保 `include` 参数的值是有效的

#### 向后兼容

为了保持向后兼容性，仍然提供了以下便捷函数：

- `get_chroma_client()`: 获取 ChromaDB 客户端实例
- `connection_db()`: 获取数据库连接（已废弃，建议使用 ChromaDBHelper 类）

#### 示例代码

完整的示例代码请参考 `app/core/chroma/example_usage.py` 文件，其中包含了各种使用场景的详细示例。

### ChromaDB 拼写检查问题解决方案

#### 问题描述

在使用 ChromaDB 时，我们遇到了一个常见的拼写检查问题：

- **ChromaDB API 使用 `metadatas`**（复数形式）
- **PyCharm/IDE 拼写检查器认为 `metadatas` 是错误的**
- **如果改为 `metadata`（单数形式），会导致类型错误**

#### 为什么 ChromaDB 使用 `metadatas`？

ChromaDB 是一个向量数据库，它的 API 设计遵循以下原则：

1. **语义清晰性**: `metadatas` 表示多个元数据对象，而 `metadata` 表示单个元数据对象
2. **API 一致性**: 与 `documents`、`embeddings` 等复数形式保持一致
3. **官方标准**: 这是 ChromaDB 官方 API 的规范，不能随意更改

#### 解决方案

##### 1. 类型别名（推荐）

我们创建了类型别名来避免拼写检查问题：

```python
# 类型别名，用于避免拼写检查问题
MetadataList = List[Dict[str, Any]]  # 元数据列表类型

def add(
    self,
    collection_name: str,
    metadatas: Optional[MetadataList] = None,  # 使用类型别名
    # ... 其他参数
) -> None:
    pass
```

##### 2. 详细注释说明

在代码中添加详细的注释，说明为什么使用 `metadatas`：

```python
# 注意：ChromaDB API 使用 "metadatas"（复数形式），这是官方 API 的正确字段名
# 虽然拼写检查器可能认为这是错误的，但这是 ChromaDB 库的标准
IncludeField = Literal["documents", "embeddings", "metadatas", "distances", "uris", "data"]
```

#### 如何配置 IDE

##### PyCharm

1. **添加单词到字典**:

   - 右键点击 `metadatas`
   - 选择 "Spelling" → "Add to Dictionary"

2. **配置拼写检查器**:

   - File → Settings → Editor → Natural Languages → Spelling
   - 在 "Custom Dictionaries" 中添加技术术语

3. **忽略特定文件**:
   - 在项目设置中标记 `db_helper.py` 为技术文档

##### VS Code

1. **安装拼写检查扩展**:

   - Code Spell Checker
   - 配置自定义词典

2. **工作区设置**:
   ```json
   {
     "cSpell.words": ["metadatas", "chromadb", "vectorization"]
   }
   ```

#### 最佳实践

1. **保持 API 一致性**: 始终使用 ChromaDB 官方 API 的字段名
2. **类型安全**: 使用类型别名和 Literal 类型确保类型安全
3. **文档化**: 在代码中添加清晰的注释说明
4. **配置管理**: 使用配置文件来管理工具的行为

#### 总结

`metadatas` 是 ChromaDB 官方 API 的正确字段名，虽然拼写检查器可能认为这是错误的，但我们必须遵循官方规范。通过类型别名、详细注释和适当的配置，我们可以既保持代码的正确性，又避免拼写检查的干扰。

记住：**API 的正确性比拼写检查器的建议更重要！**

### 模型提供商与 LangChain 库对照表（学习向）

- 说明：不同提供商在 LangChain 里通常对应不同的 Python 包/类；若平台提供"OpenAI 兼容"端点，则可直接用 `langchain_openai.ChatOpenAI` 并设置 `base_url`。

- 常见对照：

  - Google Gemini → `langchain_google_genai.ChatGoogleGenerativeAI`（需配置 `safety_settings`、`max_output_tokens` 等专有项）
  - OpenAI 官方 → `langchain_openai.ChatOpenAI`
  - DeepSeek/Moonshot/SiliconFlow/Together/OpenRouter/Groq（OpenAI 兼容）→ `langchain_openai.ChatOpenAI` + `base_url`
  - Anthropic Claude → `langchain_anthropic.ChatAnthropic`
  - 智谱 GLM → `langchain_community.chat_models.ChatZhipuAI`
  - 通义千问 Qwen → `langchain_community.chat_models.ChatTongyi`

- 查询入口：
  - LangChain 官方 Integrations（按 Provider 搜索）：`https://python.langchain.com/docs/integrations/providers/`。
  - 具体到聊天模型（Chat Models）：`https://python.langchain.com/docs/integrations/chat/`。
  - OpenAI 兼容如何使用：参考 `langchain_openai` 文档并查看各平台"OpenAI-Compatible API"说明。

### LangSmith 配置与使用（本项目已内置）

- 我们在 `app/core/agent_service.py` 中读取 `config/app_config.json` 的 `observability.langsmith` 段并注入进程环境变量：

  - `enabled`: 是否开启追踪；
  - `api_key`: 你的 LangSmith API Key；
  - `project`: 项目名；
  - `endpoint`: LangSmith 服务地址（默认 `https://api.smith.langchain.com`）。

- 生效方式：当 `enabled=true` 时，代码会设置以下变量（无需你在系统层面额外配置）：

  - `LANGCHAIN_TRACING_V2=true`
  - `LANGCHAIN_API_KEY` 与 `LANGSMITH_API_KEY`
  - `LANGCHAIN_PROJECT` 与 `LANGSMITH_PROJECT`
  - `LANGCHAIN_ENDPOINT` 与 `LANGSMITH_ENDPOINT`

- 访问方式：调用任意接口后，前往 LangSmith 控制台（`https://smith.langchain.com`），在对应 Project 下查看 Run/Trace，即可看到链路、提示词、耗时、错误等信息。

- 常见问题：
  - 必须写 `LANGCHAIN_API_KEY` 吗？不必须。本项目会把 `app_config.json` 中的 `observability.langsmith.api_key` 注入到进程环境变量，等效于你手工设置。
  - 看不到 token 级别输出？很多 Provider（尤其是非流式或空返回）不会有 token 级别的分块；请展开最底层模型节点并查看 Metadata/Run 面板。
  - 错误排查建议：若模型节点没有任何候选/元数据，多为权限/配额/区域问题；若有 `safety_ratings` 或 `block_reason`，则是安全策略拦截。
