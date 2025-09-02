# 常见问题（FAQ）

- 缺少密钥：检查 `config/app_config.json` 的 providers.\*.api_key
- API 失败：确认服务已启动、端口未冲突、网络可用
- 流式输出：当前为非流式，可扩展为 SSE/WebSocket
- 安全：不要在代码中硬编码密钥；启用鉴权与限流
