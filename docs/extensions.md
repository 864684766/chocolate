# 扩展与示例

## 新增工具示例：double_num

参见根 README 的示例片段（app/tools/double.py 与 agent 集成）。

## 新增业务 API 示例：/math/sum

参见根 README 的示例片段（app/api/math.py 与注册）。

## 适配新模型提供商

在 `app/llm_adapters/` 新增 provider，并在 factory 注册。

## 本地开源模型接入（Transformers 后端）

本项目支持“同一提供商下同时存在云模型与本地模型”。通过在 `config/app_config.json` 的 `providers.<provider>.models.<model>` 节点配置：

- 统一字段：`model`（可为官方模型名或本地路径）
- 后端选择：`backend`（`native`|`transformers`）。`native` 走官方 SDK；`transformers` 走本地推理。
- 生成参数建议沉到模型节点：`temperature`、`max_new_tokens`、`enable_thinking`、`device_map`、`torch_dtype`。

### 参数含义（统一在 providers.<provider>.models.<model> 下配置）

- **model**: 模型标识。可为 HF 仓库名（如 `Qwen/Qwen3-0.6B`）或本地路径（如 `D:/models/Qwen/Qwen3-0.6B`）。
- **backend**: 推理后端。`native` 表示走官方云 SDK；`transformers` 表示走本地 HF 后端。
- **temperature**: 采样温度，范围通常 0.0–1.0。数值越大越发散、越有创造力；越小越保守稳定。建议 0.2–0.8 之间调优。
- **device_map**: 设备映射策略。常见：`"auto"`（自动切分/放置）、`"cpu"`、`"cuda"`、`"cuda:0"`（指定 GPU）。
- **torch_dtype**: 张量精度。常见：`"auto"`、`"float16"`、`"bfloat16"`、`"float32"`。半精度可降显存、提吞吐；`bfloat16` 在部分平台表现更稳。
- **enable_thinking**: 是否启用“思维模式”（仅对支持该能力的模型有效，如 Qwen 的思考链输出）。启用后模型会在内部或可解析的片段里输出“思考内容”，我们再解析出“最终答案”。如果模型不支持，该开关将被忽略。

提示：“生成参数建议沉到模型节点”指的是将这些与生成相关的默认值写在 `providers.<provider>.models.<model>` 的配置里，而不是放到全局或编排层。这样同一个提供商下的不同模型可以有不同默认值，避免全局冲突，便于按模型独立调优。

示例：

```json
{
  "providers": {
    "qwen": {
      "models": {
        "Qwen3-local": {
          "model": "D:/models/Qwen/Qwen3-0.6B",
          "backend": "transformers",
          "temperature": 0.7,
          "max_new_tokens": 1024,
          "enable_thinking": true,
          "device_map": "auto",
          "torch_dtype": "auto"
        }
      }
    },
    "google": {
      "api_key": "...",
      "models": {
        "gemini-2.5-pro": {
          "model": "gemini-2.5-pro",
          "backend": "native",
          "temperature": 0.7,
          "max_new_tokens": 2048,
          "generation_config": { "response_mime_type": "text/plain" },
          "safety": {
            "hate_speech": "BLOCK_NONE",
            "harassment": "BLOCK_NONE",
            "sex": "BLOCK_NONE",
            "danger": "BLOCK_NONE"
          }
        }
      }
    }
  }
}
```

### 运行路径

- API 编排层（`/retrieval/search`）默认执行“向量召回 → 重排 → 生成”。
- 生成阶段：根据请求中的 `provider` + `ai_type`（模型名/本地路径）
  - 读取对应模型节点的配置（含 backend 与生成参数）
  - 工厂按 backend 路由：`native` → 走 `openai`/`google` 等官方 SDK；`transformers` → 走本地 `TransformersProvider`。

### 约束与建议

- 所有历史的 `model_name` 字段已统一为 `model`。
- 若后续不再使用云模型，仅需将各模型 `backend` 置为 `transformers`，无需修改代码。
- Qwen 的思维模式解析（</think>）由本地适配器内部处理；若使用其他 HF 模型，默认直接返回生成文本。
