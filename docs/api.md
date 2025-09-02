# Web API 文档

## /agent/invoke

- 请求体：{"input": string, "ai_type"?: string, "provider"?: string, "session_id"?: string}
- 响应体：{"answer": string}

## 健康与缓存

- GET /health
- GET /cache-info
- POST /clear-cache

示例 curl：

```bash
curl -X POST http://localhost:8000/agent/invoke \
 -H "Content-Type: application/json" \
 -d '{"input":"帮我计算 2+3*4", "session_id":"demo-1"}'
```
