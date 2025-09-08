# Claude Code 优化方案实施指南

## 快速开始

本指南提供 Claude Code 优化方案的具体实施步骤，帮助开发团队快速上手并逐步实现系统优化。

## 前置条件

### 环境要求

- Python 3.9+
- 现有 Chocolate 项目环境
- 开发工具：IDE、Git、测试框架

### 依赖包

```bash
# 新增依赖
pip install asyncio-mqtt  # 异步消息队列
pip install redis         # 缓存系统
pip install psutil        # 系统监控
pip install ast-tools     # 代码分析
```

## 阶段一：基础架构重构

### 1.1 创建多 Agent 架构

#### 步骤 1：创建 Agent 基础结构

```bash
# 创建目录结构
mkdir -p app/core/agents
mkdir -p app/core/memory
mkdir -p app/core/steering
mkdir -p app/core/code_search
```

#### 步骤 2：实现基础 Agent 类

创建 `app/core/agents/base_agent.py`：

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import asyncio
from dataclasses import dataclass

@dataclass
class AgentMessage:
    """Agent 间通信消息"""
    sender: str
    receiver: str
    content: Any
    message_type: str
    timestamp: float

class BaseAgent(ABC):
    """基础 Agent 抽象类"""

    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.message_queue = asyncio.Queue()
        self.is_running = False

    @abstractmethod
    async def process_message(self, message: AgentMessage) -> Any:
        """处理接收到的消息"""
        pass

    async def send_message(self, receiver: str, content: Any, msg_type: str = "default"):
        """发送消息给其他 Agent"""
        message = AgentMessage(
            sender=self.agent_id,
            receiver=receiver,
            content=content,
            message_type=msg_type,
            timestamp=asyncio.get_event_loop().time()
        )
        # 这里需要实现消息路由逻辑
        pass

    async def start(self):
        """启动 Agent"""
        self.is_running = True
        while self.is_running:
            try:
                message = await asyncio.wait_for(
                    self.message_queue.get(), timeout=1.0
                )
                await self.process_message(message)
            except asyncio.TimeoutError:
                continue
```

#### 步骤 3：实现主调度 Agent

创建 `app/core/agents/main_agent.py`：

```python
from .base_agent import BaseAgent, AgentMessage
from typing import Dict, List, Any
import json

class MainAgent(BaseAgent):
    """主调度 Agent，负责任务分解和协调"""

    def __init__(self):
        super().__init__("main_agent", "coordinator")
        self.sub_agents = {}
        self.task_queue = []

    async def process_message(self, message: AgentMessage) -> Any:
        """处理用户请求和子 Agent 反馈"""
        if message.message_type == "user_request":
            return await self._handle_user_request(message.content)
        elif message.message_type == "task_result":
            return await self._handle_task_result(message.content)

    async def _handle_user_request(self, request: str) -> Dict[str, Any]:
        """处理用户请求，分解为子任务"""
        # 1. 分析请求类型
        request_type = self._analyze_request_type(request)

        # 2. 创建任务计划
        task_plan = self._create_task_plan(request, request_type)

        # 3. 分发任务给子 Agent
        results = await self._distribute_tasks(task_plan)

        # 4. 整合结果
        return await self._synthesize_results(results)

    def _analyze_request_type(self, request: str) -> str:
        """分析请求类型"""
        code_keywords = ['function', 'class', 'method', 'import', 'def', 'return']
        doc_keywords = ['document', 'file', 'text', 'content']

        if any(keyword in request.lower() for keyword in code_keywords):
            return "code_analysis"
        elif any(keyword in request.lower() for keyword in doc_keywords):
            return "document_search"
        else:
            return "general"

    def _create_task_plan(self, request: str, request_type: str) -> List[Dict[str, Any]]:
        """创建任务计划"""
        tasks = []

        if request_type == "code_analysis":
            tasks.append({
                "agent": "code_agent",
                "task": "analyze_code",
                "params": {"query": request}
            })
        elif request_type == "document_search":
            tasks.append({
                "agent": "rag_agent",
                "task": "search_documents",
                "params": {"query": request}
            })

        return tasks

    async def _distribute_tasks(self, task_plan: List[Dict[str, Any]]) -> List[Any]:
        """分发任务给子 Agent"""
        results = []
        for task in task_plan:
            agent_id = task["agent"]
            if agent_id in self.sub_agents:
                result = await self.sub_agents[agent_id].execute_task(task)
                results.append(result)
        return results
```

### 1.2 实现智能上下文管理

#### 步骤 1：创建记忆管理器

创建 `app/core/memory/memory_manager.py`：

```python
import json
import os
from typing import Dict, Any, Optional
from datetime import datetime
import hashlib

class MemoryManager:
    """智能记忆管理器"""

    def __init__(self, project_root: str):
        self.project_root = project_root
        self.memory_dir = os.path.join(project_root, ".chocolate", "memory")
        self._ensure_memory_dir()

        # 记忆层次
        self.project_memory = ProjectMemory(self.memory_dir)
        self.session_memory = SessionMemory(self.memory_dir)
        self.user_memory = UserMemory(self.memory_dir)

    def _ensure_memory_dir(self):
        """确保记忆目录存在"""
        os.makedirs(self.memory_dir, exist_ok=True)

    async def store_memory(self, content: str, memory_type: str,
                          session_id: str = None, user_id: str = None):
        """存储记忆"""
        if memory_type == "project":
            await self.project_memory.store(content)
        elif memory_type == "session":
            await self.session_memory.store(content, session_id)
        elif memory_type == "user":
            await self.user_memory.store(content, user_id)

    async def retrieve_memory(self, query: str, memory_type: str = "all") -> Dict[str, Any]:
        """检索记忆"""
        results = {}

        if memory_type in ["all", "project"]:
            results["project"] = await self.project_memory.search(query)

        if memory_type in ["all", "session"]:
            results["session"] = await self.session_memory.search(query)

        if memory_type in ["all", "user"]:
            results["user"] = await self.user_memory.search(query)

        return results

class ProjectMemory:
    """项目级记忆"""

    def __init__(self, memory_dir: str):
        self.memory_file = os.path.join(memory_dir, "project_memory.json")
        self.memories = self._load_memories()

    def _load_memories(self) -> Dict[str, Any]:
        """加载项目记忆"""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    async def store(self, content: str):
        """存储项目记忆"""
        memory_id = hashlib.md5(content.encode()).hexdigest()[:8]
        self.memories[memory_id] = {
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "type": "project"
        }
        await self._save_memories()

    async def search(self, query: str) -> List[Dict[str, Any]]:
        """搜索项目记忆"""
        results = []
        query_lower = query.lower()

        for memory_id, memory in self.memories.items():
            if query_lower in memory["content"].lower():
                results.append(memory)

        return results

    async def _save_memories(self):
        """保存记忆到文件"""
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.memories, f, ensure_ascii=False, indent=2)
```

## 阶段二：检索系统优化

### 2.1 实现直接代码搜索

#### 步骤 1：创建代码分析器

创建 `app/core/code_search/code_analyzer.py`：

```python
import ast
import os
from typing import Dict, List, Any, Set
from dataclasses import dataclass

@dataclass
class CodeElement:
    """代码元素"""
    name: str
    type: str  # function, class, method, variable
    file_path: str
    line_number: int
    docstring: str
    dependencies: List[str]
    signature: str

class CodeAnalyzer:
    """代码结构分析器"""

    def __init__(self, project_root: str):
        self.project_root = project_root
        self.code_index = {}
        self.dependency_graph = {}

    def analyze_project(self) -> Dict[str, Any]:
        """分析整个项目"""
        python_files = self._find_python_files()

        for file_path in python_files:
            self._analyze_file(file_path)

        return {
            "files": list(self.code_index.keys()),
            "total_elements": sum(len(elements) for elements in self.code_index.values()),
            "dependency_graph": self.dependency_graph
        }

    def _find_python_files(self) -> List[str]:
        """查找所有 Python 文件"""
        python_files = []
        for root, dirs, files in os.walk(self.project_root):
            # 跳过虚拟环境和缓存目录
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']

            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))

        return python_files

    def _analyze_file(self, file_path: str):
        """分析单个文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)
            elements = []

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    element = self._extract_function_info(node, file_path)
                    elements.append(element)
                elif isinstance(node, ast.ClassDef):
                    element = self._extract_class_info(node, file_path)
                    elements.append(element)

            self.code_index[file_path] = elements

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")

    def _extract_function_info(self, node: ast.FunctionDef, file_path: str) -> CodeElement:
        """提取函数信息"""
        dependencies = self._extract_dependencies(node)

        return CodeElement(
            name=node.name,
            type="function",
            file_path=file_path,
            line_number=node.lineno,
            docstring=ast.get_docstring(node) or "",
            dependencies=dependencies,
            signature=f"def {node.name}({', '.join(arg.arg for arg in node.args.args)})"
        )

    def _extract_dependencies(self, node: ast.AST) -> List[str]:
        """提取依赖关系"""
        dependencies = set()

        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                dependencies.add(child.id)
            elif isinstance(child, ast.Attribute):
                dependencies.add(child.attr)

        return list(dependencies)

    def search_code(self, query: str) -> List[CodeElement]:
        """搜索代码"""
        results = []
        query_lower = query.lower()

        for file_path, elements in self.code_index.items():
            for element in elements:
                if (query_lower in element.name.lower() or
                    query_lower in element.docstring.lower() or
                    query_lower in element.signature.lower()):
                    results.append(element)

        return results
```

### 2.2 实现混合搜索引擎

创建 `app/core/hybrid_search.py`：

```python
from typing import List, Dict, Any, Optional
from .code_search.code_analyzer import CodeAnalyzer
from ..rag.service.ingestion_helpers import search_documents

class HybridSearchEngine:
    """混合搜索引擎"""

    def __init__(self, project_root: str):
        self.code_analyzer = CodeAnalyzer(project_root)
        self.vector_search = None  # 现有的向量搜索

    async def search(self, query: str, search_type: str = "auto") -> Dict[str, Any]:
        """执行混合搜索"""
        if search_type == "auto":
            search_type = self._determine_search_type(query)

        results = {
            "query": query,
            "search_type": search_type,
            "results": []
        }

        if search_type == "code":
            code_results = self.code_analyzer.search_code(query)
            results["results"] = self._format_code_results(code_results)
        elif search_type == "document":
            doc_results = await search_documents(query)
            results["results"] = doc_results
        else:
            # 混合搜索
            code_results = self.code_analyzer.search_code(query)
            doc_results = await search_documents(query)

            results["results"] = {
                "code": self._format_code_results(code_results),
                "documents": doc_results
            }

        return results

    def _determine_search_type(self, query: str) -> str:
        """自动确定搜索类型"""
        code_keywords = ['function', 'class', 'method', 'import', 'def', 'return', 'async', 'await']
        doc_keywords = ['document', 'file', 'text', 'content', 'readme', 'api']

        query_lower = query.lower()

        code_score = sum(1 for keyword in code_keywords if keyword in query_lower)
        doc_score = sum(1 for keyword in doc_keywords if keyword in query_lower)

        if code_score > doc_score:
            return "code"
        elif doc_score > code_score:
            return "document"
        else:
            return "hybrid"

    def _format_code_results(self, code_results: List) -> List[Dict[str, Any]]:
        """格式化代码搜索结果"""
        formatted_results = []

        for element in code_results:
            formatted_results.append({
                "name": element.name,
                "type": element.type,
                "file_path": element.file_path,
                "line_number": element.line_number,
                "docstring": element.docstring,
                "signature": element.signature,
                "dependencies": element.dependencies
            })

        return formatted_results
```

## 阶段三：集成和测试

### 3.1 集成到现有系统

#### 步骤 1：修改 Agent 服务

更新 `app/core/agent_service.py`：

```python
# 在现有代码基础上添加
from .agents.main_agent import MainAgent
from .hybrid_search import HybridSearchEngine

class EnhancedAgentService(AgentService):
    """增强的 Agent 服务"""

    def __init__(self):
        super().__init__()
        self.main_agent = MainAgent()
        self.search_engine = HybridSearchEngine(".")

    async def process_request_enhanced(self, request: str, session_id: str) -> str:
        """使用增强的 Agent 处理请求"""
        # 1. 使用主 Agent 分析任务
        task_plan = await self.main_agent._handle_user_request(request)

        # 2. 执行混合搜索
        search_results = await self.search_engine.search(request)

        # 3. 整合结果并返回
        return await self._synthesize_response(task_plan, search_results)
```

#### 步骤 2：创建测试用例

创建 `tests/test_claude_code_optimization.py`：

```python
import pytest
import asyncio
from app.core.agents.main_agent import MainAgent
from app.core.hybrid_search import HybridSearchEngine

class TestClaudeCodeOptimization:
    """Claude Code 优化功能测试"""

    @pytest.fixture
    def main_agent(self):
        return MainAgent()

    @pytest.fixture
    def search_engine(self):
        return HybridSearchEngine(".")

    @pytest.mark.asyncio
    async def test_main_agent_request_handling(self, main_agent):
        """测试主 Agent 请求处理"""
        request = "查找所有认证相关的函数"
        result = await main_agent._handle_user_request(request)

        assert "task_plan" in result
        assert len(result["task_plan"]) > 0

    @pytest.mark.asyncio
    async def test_hybrid_search(self, search_engine):
        """测试混合搜索"""
        query = "用户认证"
        results = await search_engine.search(query)

        assert "query" in results
        assert "search_type" in results
        assert "results" in results

    def test_code_analyzer(self):
        """测试代码分析器"""
        from app.core.code_search.code_analyzer import CodeAnalyzer

        analyzer = CodeAnalyzer(".")
        project_info = analyzer.analyze_project()

        assert "files" in project_info
        assert "total_elements" in project_info
        assert project_info["total_elements"] > 0
```

### 3.2 性能监控

创建 `app/core/monitoring.py`：

```python
import time
import psutil
from typing import Dict, Any
from functools import wraps

class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self.metrics = {}

    def measure_time(self, func_name: str):
        """测量函数执行时间"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                result = await func(*args, **kwargs)
                end_time = time.time()

                execution_time = end_time - start_time
                self._record_metric(func_name, "execution_time", execution_time)

                return result

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()

                execution_time = end_time - start_time
                self._record_metric(func_name, "execution_time", execution_time)

                return result

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator

    def _record_metric(self, func_name: str, metric_name: str, value: float):
        """记录指标"""
        if func_name not in self.metrics:
            self.metrics[func_name] = {}

        if metric_name not in self.metrics[func_name]:
            self.metrics[func_name][metric_name] = []

        self.metrics[func_name][metric_name].append(value)

    def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        summary = {}

        for func_name, metrics in self.metrics.items():
            summary[func_name] = {}
            for metric_name, values in metrics.items():
                if values:
                    summary[func_name][metric_name] = {
                        "avg": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "count": len(values)
                    }

        return summary

# 全局监控实例
performance_monitor = PerformanceMonitor()
```

## 部署和运维

### 4.1 配置更新

更新 `config/app_config.json`：

```json
{
  "claude_code_optimization": {
    "enabled": true,
    "multi_agent": {
      "max_agents": 5,
      "message_queue_size": 1000
    },
    "memory": {
      "compression_threshold": 0.92,
      "max_memory_size": "100MB"
    },
    "search": {
      "hybrid_search_enabled": true,
      "code_search_weight": 0.7,
      "vector_search_weight": 0.3
    },
    "performance": {
      "monitoring_enabled": true,
      "metrics_retention_days": 30
    }
  }
}
```

### 4.2 启动脚本

创建 `scripts/start_enhanced.py`：

```python
#!/usr/bin/env python3
"""
启动增强版 Chocolate 系统
"""

import asyncio
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.agents.main_agent import MainAgent
from app.core.memory.memory_manager import MemoryManager
from app.core.hybrid_search import HybridSearchEngine
from app.core.monitoring import performance_monitor

async def main():
    """主启动函数"""
    print("🚀 启动 Claude Code 优化版 Chocolate 系统...")

    # 1. 初始化记忆管理器
    print("📝 初始化记忆管理器...")
    memory_manager = MemoryManager(".")

    # 2. 初始化混合搜索引擎
    print("🔍 初始化混合搜索引擎...")
    search_engine = HybridSearchEngine(".")

    # 3. 启动主 Agent
    print("🤖 启动主 Agent...")
    main_agent = MainAgent()
    await main_agent.start()

    # 4. 启动性能监控
    print("📊 启动性能监控...")
    system_metrics = performance_monitor.get_system_metrics()
    print(f"系统状态: CPU {system_metrics['cpu_percent']}%, 内存 {system_metrics['memory_percent']}%")

    print("✅ 系统启动完成！")

    # 保持运行
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 系统正在关闭...")

if __name__ == "__main__":
    asyncio.run(main())
```

## 验证和测试

### 5.1 功能验证

运行以下命令验证功能：

```bash
# 运行测试
python -m pytest tests/test_claude_code_optimization.py -v

# 启动增强版系统
python scripts/start_enhanced.py

# 性能基准测试
python -m pytest tests/performance/ -v --benchmark-only
```

### 5.2 性能对比

创建性能对比脚本 `scripts/performance_comparison.py`：

```python
import asyncio
import time
from app.core.agent_service import AgentService
from app.core.enhanced_agent_service import EnhancedAgentService

async def performance_comparison():
    """性能对比测试"""

    # 原始系统
    original_service = AgentService()

    # 增强系统
    enhanced_service = EnhancedAgentService()

    test_queries = [
        "查找所有认证相关的函数",
        "如何实现用户登录功能？",
        "项目的架构是什么？",
        "有哪些工具可以使用？"
    ]

    print("📊 性能对比测试")
    print("=" * 50)

    for query in test_queries:
        print(f"\n🔍 测试查询: {query}")

        # 测试原始系统
        start_time = time.time()
        original_result = await original_service.process_request(query, "test_session")
        original_time = time.time() - start_time

        # 测试增强系统
        start_time = time.time()
        enhanced_result = await enhanced_service.process_request_enhanced(query, "test_session")
        enhanced_time = time.time() - start_time

        # 计算性能提升
        improvement = ((original_time - enhanced_time) / original_time) * 100

        print(f"原始系统: {original_time:.3f}s")
        print(f"增强系统: {enhanced_time:.3f}s")
        print(f"性能提升: {improvement:.1f}%")

if __name__ == "__main__":
    asyncio.run(performance_comparison())
```

## 总结

本实施指南提供了 Claude Code 优化方案的详细实施步骤，包括：

1. **基础架构重构**：多 Agent 架构和智能上下文管理
2. **检索系统优化**：直接代码搜索和混合检索策略
3. **集成和测试**：与现有系统的集成和全面测试
4. **部署和运维**：配置更新和启动脚本
5. **验证和测试**：功能验证和性能对比

通过按照本指南逐步实施，您的 Chocolate 项目将获得显著的性能提升和用户体验改善。建议按照阶段逐步实施，确保每个阶段都经过充分测试后再进入下一阶段。
