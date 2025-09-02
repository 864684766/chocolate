# 快速开始

1. 创建虚拟环境并安装依赖：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

2. 配置 `config/app_config.json`：填入 provider 的 api_key 和默认模型。

3. 启动 API：

```bash
python -m scripts.run_api
```

4. 运行 CLI：

```bash
python -m scripts.run_agent
```

5. 打开文档：

- Swagger: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
