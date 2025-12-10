# RAG 系统总览

本章节提供系统级视角，说明目标、能力边界、分层架构与多语言/多媒体的可插拔设计。

## 目标与范围

- 支持中文优先，逐步扩展到多语言（zh/en/ja/...）。
- 支持多种数据喂养：手动上传、网页爬虫、API 接入、文件监控。
- 支持多媒体：文本、Markdown、PDF、Word、Excel、图片、音频/视频（经转写/字幕）。
  - Office 文档：支持 PDF、Word（.docx/.doc）、Excel（.xlsx/.xls/.csv）
  - 表格处理：使用自然语言连接符格式（"列名是值"），优化语义搜索
- 支持召回+重排双阶段检索，向量库使用 ChromaDB。
- 面向生产：监控、权限、版本、回滚、热更新。

## 分层架构（高层）

1. 数据接入层 ingestion
2. 数据处理层 processing
3. 向量化层 vectorization
4. 存储层 storage（ChromaDB）
5. 检索层 retrieval（召回+重排）
6. RAG 应用层 rag（与 LLM/Gemini 集成）
7. 管理接口层 management（监控/配置/权限）

每层均通过清晰接口解耦，采用策略/工厂/适配器/管道等模式实现热插拔与可扩展。

## 多语言与多媒体的可插拔

- 多语言：在 processing 层按语言拆分子包，如 processing/lang/zh, en, ja；统一暴露 `LanguageProcessor` 接口，运行时根据文档语言选择实现。
- 多媒体：在 ingestion 层与 processing 层分别提供 `MediaParser`/`Extractor` 接口，按类型（text/pdf/image/audio/video）放置到独立文件，按需加载。

详见各章节的接口与示例代码。
