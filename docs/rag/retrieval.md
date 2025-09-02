# 检索层（Retrieval）

双阶段：向量召回 + 交叉编码器重排。

## 召回（Bi-encoder）

- 与向量化层相同的嵌入模型族，保证向量空间一致。
- 多语言建议：`bge-m3`/`e5-multilingual`。

## 重排（Cross-encoder / Reranker）

- 不同模型确实存在语言友好性差异：
  - 英文/多语：`cross-encoder/ms-marco-MiniLM-L-6-v2`、`mMiniLMv2-*`
  - 中文/多语：`BAAI/bge-reranker-large`、`bge-reranker-base`（支持中文，效果优）
- 选择原则：
  - 如果以中文为主，优先 bge-reranker 系列；
  - 多语场景，选 multilingual 版本或在路由层按语言分配不同 reranker。

## 语言路由（Reranker Router）

建议在检索层增加“语言 → 重排模型”的策略路由：

```python
# app/retrieval/reranker_router.py
from typing import List, Tuple

class Reranker:
    def score(self, query: str, docs: List[str]) -> List[float]:
        raise NotImplementedError

class BGEReranker(Reranker):
    ...  # 封装 BAAI/bge-reranker-*

class MiniLMReranker(Reranker):
    ...  # 封装 cross-encoder/ms-marco-*

class MultilingualReranker(Reranker):
    ...  # 可选：多语 reranker

class RerankerRouter:
    def __init__(self):
        self.zh = BGEReranker(model="BAAI/bge-reranker-large")
        self.en = MiniLMReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.multi = MultilingualReranker(model="bge-reranker-base")

    def select(self, lang: str) -> Reranker:
        if lang.startswith("zh"):
            return self.zh
        if lang.startswith("en"):
            return self.en
        return self.multi
```

在 `DualStageRetriever` 的重排阶段，先进行语言检测（或通过上游标注），再调用 `RerankerRouter.select(lang)` 选择合适模型进行重排。

## 目录

```
app/retrieval/
  retriever.py          # 召回（向量搜索）
  reranker.py           # 重排序（cross-encoder）
  reranker_router.py    # 语言→重排模型路由（新增）
  hybrid_search.py      # 关键词 + 向量混合检索
  context_builder.py    # 上下文拼装
```

## TopK 与阈值

- 典型：召回 top_k=30~50，重排截断为 top_k=5~10。
