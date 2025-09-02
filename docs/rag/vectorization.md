# 向量化层（Vectorization）

负责将文本块转换为向量；并管理模型与缓存。

## 模型适配与多语言

- 适配器模式统一接口 `EmbeddingModelAdapter`：encode(texts)->List[List[float]]
- 推荐：
  - 中文：`BAAI/bge-large-zh-v1.5`（Bi-encoder，召回）
  - 多语言：`intfloat/multilingual-e5-large`、`bge-m3`
- 缓存：LRU/磁盘缓存，避免重复计算

## 目录

```
app/vectorization/
  models/
    bge_adapter.py
    e5_adapter.py
    model_manager.py         # 动态选择与热插拔
  generator.py               # 调度批量向量化
  cache.py                   # 本地/分布式缓存
```

## 参数建议

- 归一化：BGE 建议 `normalize_embeddings=True`
- 批量：根据显存/内存设置 batch_size（32~128）
