# 去重策略详解（更新）

## 概述

在 Chocolate 项目的 RAG 系统中，存在多种去重方法，每种方法都有其特定的使用场景和优势。本文档在原有内容基础上，新增了“统一规范化 + 稳定 ID + 批内去重 + 库内过滤”的工程化流程说明。

### 新增的标准流程

- 文本规范化：`normalize_text_for_vector(text)`（NFKC、去控制字符、空白折叠、截断）
- 元数据规范化：`normalize_meta_for_vector(meta)`（白名单保留与缺省填充，图片/区域展开）
- 稳定 ID：`{doc_id or filename}:{chunk_index}:{sha1(norm_text)[:16]}`
- 批内去重：按稳定 ID 字典去重
- 库内过滤：`get(ids=...)` 过滤已存在 ID 后再 `add`
- 统计日志：`raw / batch_dedup / existed / written`

## 去重方法分类

### 1. **精确去重 (Exact Deduplication)**

#### 方法：`md5_of()`

```python
def md5_of(text: str) -> str:
    """计算文本的 MD5 哈希（用于精确去重）"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()
```

**使用场景：**

- 完全相同的文本内容
- 需要快速识别重复文档
- 存储空间优化

**优势：**

- 计算速度快
- 内存占用小
- 100% 准确识别完全重复

**局限性：**

- 只能识别完全相同的文本
- 对微小差异（如空格、标点）敏感

### 2. **近似去重 (Approximate Deduplication)**

#### 方法：`near_duplicate()`

```python
def near_duplicate(c1: str, c2: str, threshold: float = 0.95, embed_model: Optional[str] = None) -> bool:
    """近似重复检测（基于向量余弦相似度）"""
```

**使用场景：**

- 语义相似但字面不同的文本
- 翻译后的相同内容
- 重写或改写的文档

**优势：**

- 能识别语义相似的重复内容
- 支持多语言文本
- 可调节相似度阈值

**局限性：**

- 计算成本较高
- 依赖外部模型
- 可能误判语义相关但不同的内容

### 3. **综合去重 (Hybrid Deduplication)**

#### 方法：`dedup_captions()`

```python
def dedup_captions(
    captions: List[str],
    *,
    approx: bool = True,
    threshold: float = 0.95,
    embed_model: Optional[str] = None,
) -> List[str]:
    """去重（精确 + 近似）"""
```

**使用场景：**

- 图像描述生成后的去重
- 多模态内容处理
- 需要高质量去重的场景

**优势：**

- 结合精确和近似去重的优势
- 保持原有顺序
- 可配置去重策略

## 在项目中的具体应用

### 1. **图像描述生成流程**

在 `ImageVisionExtractor` 中，去重策略的应用流程如下：

```python
# 1) 生成英文候选
captions = self._generate_with_image2text(...)

# 2) 规则过滤（长度/黑名单/乱码/重复片段）
if captions:
    captions = filter_captions(
        captions,
        min_len=int(filter_config.get("min_length", 5)),
        max_len=int(filter_config.get("max_length", 120)),
        blacklist_keywords=list(filter_config.get("blacklist_keywords", [])),
        max_gibberish_ratio=float(filter_config.get("max_gibberish_ratio", 0.3)),
        forbid_repeat_ngram=int(filter_config.get("forbid_repeat_ngram", 3)),
    )

# 3) 近似去重（可选）
if captions:
    captions = dedup_captions(
        captions,
        approx=True,
        threshold=0.95,
        embed_model=embed_conf.get("model"),
    )

# 4) CLIP 重排
captions = self._rerank_with_clip(image, captions)
```

**为什么需要多层去重？**

1. **生成阶段去重**：AI 模型可能生成重复的描述
2. **规则过滤**：去除低质量内容
3. **语义去重**：识别语义相似的描述
4. **视觉重排**：确保描述与图像匹配

### 2. **RAG 处理管道**

在 `ProcessingPipeline` 中，去重策略确保：

- **数据质量**：避免重复内容污染知识库
- **存储效率**：减少向量数据库的存储压力
- **检索准确性**：避免重复内容影响检索结果

## 去重策略的技术细节

### 1. **MD5 哈希去重**

```python
def md5_of(text: str) -> str:
    """计算文本的 MD5 哈希（用于精确去重）"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()
```

**技术特点：**

- 使用 UTF-8 编码确保多语言支持
- MD5 哈希提供 128 位唯一标识
- 哈希冲突概率极低（约 2^-64）

**使用示例：**

```python
# 检测完全重复的文档
doc1 = "这是一个测试文档"
doc2 = "这是一个测试文档"
doc3 = "这是另一个文档"

hash1 = md5_of(doc1)  # 相同哈希
hash2 = md5_of(doc2)  # 相同哈希
hash3 = md5_of(doc3)  # 不同哈希

print(hash1 == hash2)  # True
print(hash1 == hash3)  # False
```

### 2. **语义相似度去重**

```python
def near_duplicate(c1: str, c2: str, threshold: float = 0.95, embed_model: Optional[str] = None) -> bool:
    """近似重复检测（基于向量余弦相似度）"""
    try:
        from sentence_transformers import SentenceTransformer, util
        model_name = embed_model or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        model = SentenceTransformer(model_name)
        emb = model.encode([c1, c2], convert_to_tensor=True, normalize_embeddings=True)
        sim = float(util.cos_sim(emb[0], emb[1]).item())
        return sim >= threshold
    except (ImportError, ModuleNotFoundError) as e:
        logger.info(f"sentence-transformers not available for near-dup: {e}")
    except Exception as e:
        logger.warning(f"near-dup computation failed: {e}")
    return False
```

**技术特点：**

- 使用多语言句向量模型
- 余弦相似度计算语义相似性
- 支持可调节的相似度阈值
- 优雅降级处理依赖缺失

**使用示例：**

```python
# 检测语义相似的文档
doc1 = "今天天气很好"
doc2 = "今日天气不错"
doc3 = "明天会下雨"

similar = near_duplicate(doc1, doc2, threshold=0.8)  # True
different = near_duplicate(doc1, doc3, threshold=0.8)  # False
```

### 3. **综合去重策略**

```python
def dedup_captions(
    captions: List[str],
    *,
    approx: bool = True,
    threshold: float = 0.95,
    embed_model: Optional[str] = None,
) -> List[str]:
    """去重（精确 + 近似）"""
    seen_hash: set = set()
    result: List[str] = []
    for c in captions:
        h = md5_of(c)
        if h in seen_hash:
            continue
        if approx and any(near_duplicate(c, r, threshold=threshold, embed_model=embed_model) for r in result):
            continue
        seen_hash.add(h)
        result.append(c)
    return result
```

**技术特点：**

- 先进行精确去重（MD5）
- 再进行近似去重（语义相似度）
- 保持原有顺序
- 支持可配置的去重策略

## 性能优化策略

### 1. **分层去重**

```python
# 第一层：快速精确去重
seen_hash: set = set()
for c in captions:
    h = md5_of(c)
    if h in seen_hash:
        continue  # 跳过完全重复

# 第二层：语义去重（仅对通过第一层的内容）
if approx and any(near_duplicate(c, r, threshold=threshold) for r in result):
    continue  # 跳过语义重复
```

**优势：**

- 减少语义相似度计算次数
- 提高整体处理速度
- 保持去重质量

### 2. **缓存优化**

```python
# 缓存模型实例，避免重复加载
class DeduplicationCache:
    def __init__(self):
        self._model_cache = {}

    def get_model(self, model_name: str):
        if model_name not in self._model_cache:
            self._model_cache[model_name] = SentenceTransformer(model_name)
        return self._model_cache[model_name]
```

### 3. **批量处理**

```python
# 批量计算相似度，提高效率
def batch_near_duplicate(texts: List[str], threshold: float = 0.95) -> List[bool]:
    """批量检测近似重复"""
    if len(texts) < 2:
        return [False] * len(texts)

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

    # 计算相似度矩阵
    similarity_matrix = util.cos_sim(embeddings, embeddings)

    # 标记重复项
    is_duplicate = [False] * len(texts)
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if similarity_matrix[i][j] >= threshold:
                is_duplicate[j] = True  # 标记后续出现的为重复

    return is_duplicate
```

## 配置和调优

### 1. **相似度阈值调优**

```python
# 不同场景的推荐阈值
THRESHOLDS = {
    "strict": 0.98,    # 严格去重，几乎完全相同
    "normal": 0.95,    # 标准去重，语义相似
    "loose": 0.90,     # 宽松去重，包含改写
    "very_loose": 0.85 # 极宽松去重，包含翻译
}
```

### 2. **模型选择**

```python
# 不同语言的推荐模型
EMBEDDING_MODELS = {
    "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "chinese": "sentence-transformers/distiluse-base-multilingual-cased",
    "english": "sentence-transformers/all-MiniLM-L6-v2",
    "code": "microsoft/codebert-base"  # 代码去重专用
}
```

### 3. **性能监控**

```python
class DeduplicationMetrics:
    """去重性能监控"""

    def __init__(self):
        self.exact_duplicates = 0
        self.approximate_duplicates = 0
        self.total_processed = 0
        self.processing_time = 0.0

    def record_duplicate(self, duplicate_type: str, processing_time: float):
        if duplicate_type == "exact":
            self.exact_duplicates += 1
        elif duplicate_type == "approximate":
            self.approximate_duplicates += 1
        self.total_processed += 1
        self.processing_time += processing_time

    def get_stats(self) -> Dict[str, Any]:
        return {
            "exact_duplicate_rate": self.exact_duplicates / max(1, self.total_processed),
            "approximate_duplicate_rate": self.approximate_duplicates / max(1, self.total_processed),
            "avg_processing_time": self.processing_time / max(1, self.total_processed),
            "total_processed": self.total_processed
        }
```

## 最佳实践

### 1. **选择合适的去重策略**

```python
def choose_deduplication_strategy(content_type: str, quality_requirement: str) -> Dict[str, Any]:
    """根据内容类型和质量要求选择去重策略"""

    strategies = {
        "image_captions": {
            "exact": True,
            "approximate": True,
            "threshold": 0.95,
            "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        },
        "document_text": {
            "exact": True,
            "approximate": True,
            "threshold": 0.90,
            "model": "sentence-transformers/distiluse-base-multilingual-cased"
        },
        "code_snippets": {
            "exact": True,
            "approximate": False,  # 代码通常不需要语义去重
            "threshold": 0.98,
            "model": None
        }
    }

    return strategies.get(content_type, strategies["document_text"])
```

### 2. **错误处理和降级**

```python
def robust_deduplication(captions: List[str], **kwargs) -> List[str]:
    """健壮的去重处理，包含错误处理和降级策略"""

    try:
        # 尝试完整去重流程
        return dedup_captions(captions, **kwargs)
    except ImportError:
        # 降级到仅精确去重
        logger.warning("Semantic deduplication not available, using exact deduplication only")
        return exact_deduplication_only(captions)
    except Exception as e:
        # 降级到基础去重
        logger.error(f"Deduplication failed: {e}, using basic deduplication")
        return basic_deduplication(captions)

def exact_deduplication_only(captions: List[str]) -> List[str]:
    """仅使用精确去重"""
    seen_hash: set = set()
    result: List[str] = []
    for c in captions:
        h = md5_of(c)
        if h not in seen_hash:
            seen_hash.add(h)
            result.append(c)
    return result

def basic_deduplication(captions: List[str]) -> List[str]:
    """基础去重：仅去除完全相同的字符串"""
    return list(dict.fromkeys(captions))  # 保持顺序的去重
```

### 3. **内存优化**

```python
def memory_efficient_deduplication(captions: List[str], batch_size: int = 1000) -> List[str]:
    """内存高效的去重处理"""
    result: List[str] = []
    seen_hash: set = set()

    # 分批处理，避免内存溢出
    for i in range(0, len(captions), batch_size):
        batch = captions[i:i + batch_size]

        # 处理当前批次
        for c in batch:
            h = md5_of(c)
            if h not in seen_hash:
                seen_hash.add(h)
                result.append(c)

        # 可选：清理内存
        if i % (batch_size * 10) == 0:
            import gc
            gc.collect()

    return result
```

## 总结

Chocolate 项目中的去重策略采用了多层次、多方法的综合方案：

1. **精确去重**：快速识别完全重复的内容
2. **语义去重**：识别语义相似的内容
3. **综合去重**：结合两种方法的优势
4. **性能优化**：通过缓存、批处理等方式提高效率
5. **错误处理**：提供降级策略确保系统稳定性

这种设计确保了 RAG 系统能够有效处理各种类型的重复内容，同时保持良好的性能和稳定性。通过合理的配置和调优，可以适应不同的业务场景和质量要求。
