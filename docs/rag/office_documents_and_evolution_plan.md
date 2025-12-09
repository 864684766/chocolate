# Office 文档支持与 RAG 演进路线图（2025 年底技术栈）

本文档基于 2025 年底最新技术发展，详细说明 PDF/Word/Excel 信息提取技术栈、RAG 完善标准，以及从 RAG 到 RAG+Reasoning 到 KAG 的完整演进路径。

## 一、PDF、Word、Excel 信息提取技术栈（2025 年底）

### 1.1 PDF 提取技术栈

#### 推荐方案（按优先级）

**1. PyMuPDF (Fitz) - 首选**

- **优势**：
  - 速度快、内存效率高
  - 支持复杂布局（多栏、表格、图片）
  - 可提取文本、图片、元数据、目录结构
  - 支持 PDF 表单和注释提取
- **适用场景**：生产环境、大规模处理、复杂 PDF 文档
- **安装**：`pip install pymupdf`
- **代码示例**：

```python
import fitz  # PyMuPDF

def extract_pdf_pymupdf(content: bytes) -> Dict[str, Any]:
    doc = fitz.open(stream=content, filetype="pdf")
    pages = []
    toc = doc.get_toc()

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        # 提取表格
        tables = page.find_tables()
        pages.append({
            "text": text,
            "page_number": page_num + 1,
            "tables": [table.extract() for table in tables]
        })

    return {"pages": pages, "toc": toc}
```

**2. pdfplumber - 表格提取专用**

- **优势**：
  - 表格提取准确率高
  - 保留表格结构（行列、单元格）
  - 支持复杂表格布局
- **适用场景**：财务报告、数据表格为主的 PDF
- **安装**：`pip install pdfplumber`
- **代码示例**：

```python
import pdfplumber

def extract_pdf_pdfplumber(content: bytes) -> Dict[str, Any]:
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        pages = []
        for page in pdf.pages:
            text = page.extract_text()
            tables = page.extract_tables()
            pages.append({
                "text": text,
                "tables": tables
            })
    return {"pages": pages}
```

**3. pypdf (原 PyPDF2) - 轻量级**

- **优势**：轻量、简单、基础功能完整
- **劣势**：复杂布局处理能力弱
- **适用场景**：简单 PDF、文本为主
- **安装**：`pip install pypdf`

#### 最佳实践建议

- **混合使用**：PyMuPDF 提取基础内容 + pdfplumber 提取表格
- **性能优化**：大文件使用多进程并行处理
- **错误处理**：加密 PDF、损坏 PDF 的容错机制

### 1.2 Word 文档提取技术栈

#### 推荐方案

**python-docx - 标准方案**

- **优势**：
  - 官方推荐，社区活跃
  - 支持段落、表格、标题、列表、图片元数据
  - 保留文档结构（标题层级、样式信息）
  - 支持.docx 格式（不支持.doc 旧格式）
- **安装**：`pip install python-docx`
- **代码示例**：

```python
from docx import Document
import io

def extract_word(content: bytes) -> Dict[str, Any]:
    doc = Document(io.BytesIO(content))

    paragraphs = []
    tables = []

    for para in doc.paragraphs:
        if para.text.strip():
            paragraphs.append({
                "text": para.text,
                "style": para.style.name,
                "level": getattr(para.style, 'level', 0)
            })

    for table in doc.tables:
        table_data = []
        for row in table.rows:
            row_data = [cell.text for cell in row.cells]
            table_data.append(row_data)
        tables.append(table_data)

    return {
        "paragraphs": paragraphs,
        "tables": tables,
        "core_properties": {
            "title": doc.core_properties.title,
            "author": doc.core_properties.author,
            "created": str(doc.core_properties.created)
        }
    }
```

**对于.doc 旧格式**：

- 使用 `python-docx2txt` 或 `antiword`（Linux）
- 或转换为.docx 后再处理

### 1.3 Excel 提取技术栈

#### 推荐方案

**1. openpyxl - 首选（.xlsx 格式）**

- **优势**：
  - 支持.xlsx 格式（Excel 2007+）
  - 保留公式、样式、图表元数据
  - 支持大文件处理（流式读取）
- **安装**：`pip install openpyxl`
- **代码示例**：

```python
from openpyxl import load_workbook
import io

def extract_excel_openpyxl(content: bytes) -> Dict[str, Any]:
    wb = load_workbook(io.BytesIO(content), data_only=True)

    sheets_data = []
    for sheet_name in wb.sheetnames:
        sheet = wb[sheet_name]
        rows = []
        for row in sheet.iter_rows(values_only=True):
            rows.append(list(row))
        sheets_data.append({
            "sheet_name": sheet_name,
            "data": rows,
            "max_row": sheet.max_row,
            "max_column": sheet.max_column
        })

    return {"sheets": sheets_data}
```

**2. pandas - 数据分析场景**

- **优势**：直接转为 DataFrame，便于数据分析
- **安装**：`pip install pandas openpyxl`
- **代码示例**：

```python
import pandas as pd
import io

def extract_excel_pandas(content: bytes) -> Dict[str, Any]:
    excel_file = pd.ExcelFile(io.BytesIO(content))

    sheets_data = {}
    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        sheets_data[sheet_name] = {
            "data": df.to_dict('records'),
            "shape": df.shape
        }

    return {"sheets": sheets_data}
```

**3. xlrd - 旧格式支持（.xls）**

- **安装**：`pip install xlrd`
- **注意**：xlrd 2.0+不再支持.xlsx，仅支持.xls

#### 最佳实践建议

- **格式检测**：根据文件扩展名和 Magic Number 选择提取器
- **大文件处理**：使用流式读取，避免内存溢出
- **表格序列化**：将表格转换为 Markdown 或结构化文本，便于 RAG 处理

## 二、完成提取器和分块器后是否算完善 RAG？

### 2.1 RAG 系统完善标准

完成 PDF/Word/Excel 的提取器和分块器后，**基础 RAG 功能已完善**，但还需要以下检查项：

#### ✅ 必须完成（基础 RAG）

1. **数据接入层（Ingestion）**

   - ✅ 支持多种文件格式上传
   - ✅ 文件类型检测和路由
   - ✅ 元数据提取

2. **数据处理层（Processing）**

   - ✅ 文本提取器（PlainTextExtractor）
   - ✅ PDF 提取器（新增）
   - ✅ Word 提取器（新增）
   - ✅ Excel 提取器（新增）
   - ✅ 图像/视频/音频提取器（已有）
   - ✅ 分块策略（Text/PDF/Word/Excel/Image/Video/Audio）
   - ✅ 质量评估器
   - ✅ 元数据管理

3. **向量化层（Vectorization）**

   - ✅ 向量编码（Embedder）
   - ✅ 向量存储（ChromaDB）
   - ✅ 关键词索引（Meilisearch）

4. **检索层（Retrieval）**

   - ✅ 向量检索（VectorRetriever）
   - ✅ 关键词检索（MeilisearchRetriever）
   - ✅ 混合检索（HybridSearcher）
   - ✅ 重排（CrossEncoderReranker）
   - ✅ 上下文构建（ContextBuilder）

5. **应用层（Application）**
   - ✅ RAG 查询接口
   - ✅ LLM 生成集成

#### ⚠️ 建议优化（提升 RAG 质量）

1. **检索优化**

   - 查询扩展（Query Expansion）
   - 查询重写（Query Rewriting）
   - 多轮对话上下文管理

2. **生成优化**

   - Prompt 工程优化
   - 引用和溯源
   - 答案验证机制

3. **性能优化**
   - 批量处理优化
   - 缓存机制
   - 异步处理

### 2.2 结论

**完成提取器和分块器后，基础 RAG 系统已完善，可以进入 RAG+Reasoning 阶段。**

但建议先进行以下验证：

- 测试各种文档类型的提取质量
- 验证分块策略的合理性
- 评估检索和生成效果
- 优化关键性能指标

## 三、从 RAG 到 RAG+Reasoning 到 KAG 的准备工作

### 3.1 RAG → RAG+Reasoning 准备工作

#### 技术栈准备

**1. LangChain/LangGraph 集成**

- **LangGraph**：用于构建多步骤推理流程
- **安装**：`pip install langgraph langchain langchain-community`
- **用途**：构建推理链、多 Agent 协作

**2. 推理框架选择**

**方案 A：Chain-of-Thought (CoT) 推理**

- **实现方式**：
  - 在 Prompt 中引导模型进行逐步推理
  - 使用 LangChain 的`Chain`组件构建推理链
- **适用场景**：简单推理任务、单步推理

**方案 B：Multi-Agent RAG (MA-RAG)**

- **架构**：
  - Planner Agent：规划查询步骤
  - Step Definer Agent：定义执行步骤
  - Extractor Agent：提取相关信息
  - QA Agent：生成最终答案
- **实现**：使用 LangGraph 构建 Agent 工作流
- **适用场景**：复杂查询、多步骤推理

**方案 C：RECON (REasoning with CONdensation)**

- **特点**：在推理循环中集成摘要模块，压缩证据
- **优势**：减少上下文长度，提高效率
- **实现**：需要训练摘要模型

#### 代码架构准备

**目录结构**：

```
app/rag/reasoning/
  __init__.py
  base.py                    # 推理引擎基类
  cot_reasoner.py            # Chain-of-Thought推理器
  multi_agent_reasoner.py    # 多Agent推理器
  query_expander.py          # 查询扩展器
  reasoning_retriever.py     # 推理增强检索器
  orchestrator.py            # 推理编排器
```

**关键组件**：

1. **推理引擎基类**

```python
class ReasoningEngine(ABC):
    @abstractmethod
    def reason(self, query: str, context: List[Dict]) -> Dict[str, Any]:
        """执行推理，返回推理步骤和结论"""
        pass
```

2. **查询扩展器**

```python
class QueryExpander:
    def expand(self, query: str) -> List[str]:
        """扩展查询，生成多个相关查询"""
        pass
```

3. **推理增强检索器**

```python
class ReasoningRetriever:
    def retrieve_with_reasoning(self, query: str) -> List[RetrievedItem]:
        """基于推理的检索，支持多跳推理"""
        pass
```

#### 配置准备

在`config/app_config.json`中添加：

```json
{
  "rag": {
    "reasoning": {
      "enabled": true,
      "method": "multi_agent", // "cot" | "multi_agent" | "recon"
      "max_reasoning_steps": 5,
      "query_expansion": {
        "enabled": true,
        "max_queries": 3
      }
    }
  }
}
```

#### 依赖安装

```bash
pip install langgraph langchain langchain-community
pip install langchain-openai  # 或其他LLM适配器
```

### 3.2 RAG+Reasoning → KAG 准备工作

#### 技术栈准备

**1. 知识图谱数据库**

**方案 A：Neo4j（推荐）**

- **优势**：
  - 成熟的图数据库
  - 丰富的 Cypher 查询语言
  - 与 LangChain 集成良好
  - 支持复杂图查询
- **安装**：

  ```bash
  # Docker方式
  docker run -p 7474:7474 -p 7687:7687 neo4j:latest

  # Python客户端
  pip install neo4j langchain-neo4j
  ```

**方案 B：ArangoDB**

- **优势**：多模型数据库（文档+图）
- **适用场景**：需要文档和图混合存储

**方案 C：NebulaGraph**

- **优势**：分布式图数据库，适合大规模场景

**2. 实体和关系提取**

**方案 A：使用 LLM 提取**

- **工具**：LangChain 的`GraphDocument`和`GraphRAG`
- **流程**：
  1. 从文档中提取实体和关系
  2. 构建知识图谱
  3. 存储到 Neo4j

**方案 B：使用 NER 模型**

- **工具**：spaCy、Stanford NER、BERT-based NER
- **适用场景**：结构化实体提取

**方案 C：混合方案（推荐）**

- LLM 提取复杂关系
- NER 模型提取基础实体
- 结合使用提高准确率

#### 代码架构准备

**目录结构**：

```
app/rag/knowledge_graph/
  __init__.py
  extractors/
    __init__.py
    entity_extractor.py        # 实体提取器
    relation_extractor.py      # 关系提取器
    llm_extractor.py           # LLM提取器
  builders/
    __init__.py
    graph_builder.py           # 图谱构建器
    neo4j_builder.py           # Neo4j构建器
  retrievers/
    __init__.py
    graph_retriever.py         # 图谱检索器
    multi_hop_retriever.py    # 多跳检索器
  schemas.py                   # 图谱数据模型
```

**关键组件**：

1. **实体提取器**

```python
class EntityExtractor:
    def extract(self, text: str) -> List[Entity]:
        """从文本中提取实体"""
        pass
```

2. **关系提取器**

```python
class RelationExtractor:
    def extract(self, text: str, entities: List[Entity]) -> List[Relation]:
        """提取实体间的关系"""
        pass
```

3. **图谱构建器**

```python
class GraphBuilder:
    def build_from_documents(self, chunks: List[ProcessedChunk]) -> KnowledgeGraph:
        """从文档块构建知识图谱"""
        pass
```

4. **图谱检索器**

```python
class GraphRetriever:
    def retrieve(self, query: str, max_hops: int = 2) -> List[GraphNode]:
        """基于图谱的多跳检索"""
        pass
```

#### 配置准备

在`config/app_config.json`中添加：

```json
{
  "rag": {
    "knowledge_graph": {
      "enabled": true,
      "database": {
        "type": "neo4j",
        "uri": "bolt://localhost:7687",
        "user": "neo4j",
        "password": "your_password"
      },
      "extraction": {
        "method": "llm", // "llm" | "ner" | "hybrid"
        "model": "gpt-4",
        "entities": ["PERSON", "ORG", "LOC", "PRODUCT"],
        "relations": ["WORKS_FOR", "LOCATED_IN", "PRODUCES"]
      },
      "retrieval": {
        "max_hops": 2,
        "max_nodes": 50
      }
    }
  }
}
```

#### 依赖安装

```bash
# Neo4j
pip install neo4j langchain-neo4j

# 实体提取（可选）
pip install spacy
python -m spacy download zh_core_web_sm  # 中文模型

# 图谱可视化（可选）
pip install networkx matplotlib
```

### 3.3 完整演进路径

#### 阶段 1：完善基础 RAG（当前阶段）

- ✅ 实现 PDF/Word/Excel 提取器
- ✅ 实现对应的分块策略
- ✅ 测试和优化提取质量
- **时间估算**：1-2 周

#### 阶段 2：RAG+Reasoning（2-3 周）

**Week 1：基础推理能力**

- 实现 CoT 推理引擎
- 集成到现有 RAG 流程
- 测试简单推理任务

**Week 2：多 Agent 系统**

- 实现 MA-RAG 架构
- 使用 LangGraph 构建工作流
- 测试复杂查询

**Week 3：优化和集成**

- 查询扩展和重写
- 性能优化
- 与现有系统集成

#### 阶段 3：KAG（3-4 周）

**Week 1：知识图谱基础**

- 搭建 Neo4j 环境
- 实现实体和关系提取
- 构建基础图谱

**Week 2：图谱检索**

- 实现图谱检索器
- 多跳推理查询
- 与向量检索融合

**Week 3：KAG 生成**

- 实现 KAG 生成器
- 图谱+向量混合检索
- 答案生成优化

**Week 4：优化和测试**

- 性能优化
- 准确性测试
- 系统集成

## 四、基于 2025 年底技术发展的关键要点

### 4.1 2025 年 RAG+Reasoning 趋势

1. **多 Agent 协作成为主流**

   - MA-RAG 框架广泛应用
   - Agent 间协作机制成熟
   - 动态工作流编排

2. **推理效率优化**

   - RECON 等压缩技术
   - 推理步骤优化
   - 上下文管理改进

3. **LangGraph 成为标准**
   - LangChain 官方推荐
   - 工作流编排标准
   - 社区支持完善

### 4.2 2025 年 KAG 趋势

1. **知识图谱+RAG 深度融合**

   - KG-R1 等强化学习框架
   - 多粒度索引（KET-RAG）
   - 可解释性增强（KG-SMILE）

2. **成本优化**

   - 多粒度索引降低构建成本
   - 高效检索算法
   - 增量更新机制

3. **认知启发方法**
   - CogGRAG 等认知启发框架
   - 人类认知过程模拟
   - 自我验证机制

### 4.3 技术选型建议

**推理框架**：

- **首选**：LangGraph + Multi-Agent RAG
- **备选**：CoT-RAG（简单场景）

**知识图谱**：

- **首选**：Neo4j + LangChain 集成
- **大规模**：NebulaGraph
- **混合需求**：ArangoDB

**实体提取**：

- **首选**：LLM 提取（准确率高）
- **备选**：Hybrid 方案（LLM + NER）

## 五、实施检查清单

### 5.1 RAG 完善检查清单

- [ ] PDF 提取器实现（PyMuPDF）
- [ ] Word 提取器实现（python-docx）
- [ ] Excel 提取器实现（openpyxl）
- [ ] 对应分块策略实现
- [ ] 工厂类注册新类型
- [ ] 配置文件更新
- [ ] 单元测试编写
- [ ] 集成测试验证
- [ ] 性能测试
- [ ] 文档更新

### 5.2 RAG+Reasoning 准备检查清单

- [ ] LangGraph 环境搭建
- [ ] 推理引擎基类设计
- [ ] CoT 推理器实现
- [ ] 多 Agent 架构设计
- [ ] 查询扩展器实现
- [ ] 推理增强检索器实现
- [ ] 配置项添加
- [ ] 与现有 RAG 集成
- [ ] 测试用例编写

### 5.3 KAG 准备检查清单

- [ ] Neo4j 环境搭建
- [ ] 实体提取器实现
- [ ] 关系提取器实现
- [ ] 图谱构建器实现
- [ ] 图谱检索器实现
- [ ] 多跳推理查询实现
- [ ] 与向量检索融合
- [ ] KAG 生成器实现
- [ ] 配置项添加
- [ ] 性能优化

## 六、参考资源

### 6.1 官方文档

- **LangGraph**：https://python.langchain.com/docs/langgraph
- **LangChain RAG**：https://python.langchain.com/docs/use_cases/question_answering
- **Neo4j + LangChain**：https://neo4j.com/developer-blog/knowledge-graph-rag-application/

### 6.2 学术论文（2025）

- **MA-RAG**：Multi-Agent Retrieval-Augmented Generation (arXiv:2505.20096)
- **KG-R1**：Efficient and Transferable Agentic Knowledge Graph RAG (arXiv:2509.26383)
- **KET-RAG**：Cost-Efficient Multi-Granular Indexing Framework (arXiv:2502.09304)
- **CogGRAG**：Human Cognition Inspired RAG (arXiv:2503.06567)

### 6.3 开源项目

- **LangChain RAG Agent Chatbot**：https://github.com/riolaf05/langchain-rag-agent-chatbot
- **Knowledge Graph RAG**：https://github.com/rathcoding/knowledge-graph-rag
- **Cognito-LangGraph-RAG**：https://github.com/junfanz1/Cognito-LangGraph-RAG

---

**最后更新**：2025 年 12 月
**维护者**：Chocolate 项目团队
