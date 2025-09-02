# RAG 应用层（Gemini 集成）

- 与 Gemini 2.5-flash 集成：将重排后的上下文拼接到提示词，生成答复。
- 会话记忆：沿用现有 `RunnableWithMessageHistory` 方案。
- 多模态：可将图片/音频摘要文本作为额外上下文片段。

## API 形态

- `/rag/query`：查询接口
- `/rag/manage`：知识库管理

## 最佳实践

- 上下文片段附带来源/时间戳，便于可追溯与引用。
