# 性能优化指南

本文档整理了 Chocolate RAG 系统常见的性能问题及其对应的优化方案，帮助开发者在遇到性能瓶颈时快速定位和解决问题。

## 目录

- [查询性能优化](#查询性能优化)
- [内存优化](#内存优化)
- [并发性能优化](#并发性能优化)
- [数据量增长优化](#数据量增长优化)
- [索引更新优化](#索引更新优化)
- [特定场景优化](#特定场景优化)
- [优化决策流程](#优化决策流程)
- [性能监控与测量](#性能监控与测量)

## 查询性能优化

### 问题表现

- 用户查询响应时间 > 2 秒
- 大量并发查询时系统卡顿
- 特定查询模式性能差

### 解决方案

#### 1. 缓存热门查询

```python
from functools import lru_cache
import hashlib

class CachedRetriever:
    def __init__(self, retriever):
        self.retriever = retriever
        self.cache = {}

    @lru_cache(maxsize=1000)
    def cached_search(self, query_hash: str, query: str, top_k: int, score_threshold: float):
        """缓存查询结果"""
        return self.retriever.search(RetrievalQuery(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold
        ))

    def search(self, q: RetrievalQuery):
        # 生成查询哈希作为缓存键
        query_hash = hashlib.md5(f"{q.query}_{q.top_k}_{q.score_threshold}".encode()).hexdigest()
        return self.cached_search(query_hash, q.query, q.top_k, q.score_threshold)
```

#### 2. 异步查询

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncRetriever:
    def __init__(self, retriever):
        self.retriever = retriever
        self.executor = ThreadPoolExecutor(max_workers=10)

    async def async_search(self, q: RetrievalQuery):
        """异步执行检索"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.retriever.search,
            q
        )
```

#### 3. 预计算索引

```python
class PrecomputedBM25:
    def __init__(self):
        self.index = None
        self.last_update = None

    def should_rebuild_index(self):
        """检查是否需要重建索引"""
        # 检查数据是否有更新
        current_time = time.time()
        return (self.index is None or
                current_time - self.last_update > 3600)  # 1小时重建一次

    def build_index_if_needed(self):
        """按需重建索引"""
        if self.should_rebuild_index():
            self._build_index()
            self.last_update = time.time()
```

## 内存优化

### 问题表现

- 服务器内存使用率 > 80%
- 系统频繁 OOM（内存溢出）
- 大量文档时索引占用过多内存

### 解决方案

#### 1. 分片存储

```python
class ShardedBM25:
    def __init__(self, shard_size=10000):
        self.shards = []
        self.shard_size = shard_size
        self.current_shard = 0

    def add_document(self, doc_id: str, text: str):
        """添加文档到分片"""
        if len(self.shards) == 0 or len(self.shards[self.current_shard].documents) >= self.shard_size:
            self.shards.append(BM25Shard())
            self.current_shard = len(self.shards) - 1

        self.shards[self.current_shard].add_document(doc_id, text)

    def search(self, query: str, top_k: int = 10):
        """跨分片搜索"""
        all_results = []
        for shard in self.shards:
            results = shard.search(query, top_k)
            all_results.extend(results)

        # 合并并排序结果
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]

class BM25Shard:
    def __init__(self):
        self.documents = {}
        self.tokenized_docs = []
        self.bm25_index = None

    def add_document(self, doc_id: str, text: str):
        tokens = jieba.lcut(text)
        self.documents[doc_id] = text
        self.tokenized_docs.append(tokens)
        self.bm25_index = BM25Okapi(self.tokenized_docs)
```

#### 2. 磁盘缓存

```python
import pickle
import os

class DiskCachedBM25:
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        self.index_file = os.path.join(cache_dir, "bm25_index.pkl")
        self.metadata_file = os.path.join(cache_dir, "metadata.pkl")
        os.makedirs(cache_dir, exist_ok=True)

    def save_index(self, index, metadata):
        """保存索引到磁盘"""
        with open(self.index_file, 'wb') as f:
            pickle.dump(index, f)
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(metadata, f)

    def load_index(self):
        """从磁盘加载索引"""
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            with open(self.index_file, 'rb') as f:
                index = pickle.load(f)
            with open(self.metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            return index, metadata
        return None, None
```

#### 3. 压缩存储

```python
from collections import defaultdict
import gzip

class CompressedBM25:
    def __init__(self):
        self.doc_freq = defaultdict(int)  # 使用 defaultdict 减少内存
        self.term_freq = {}  # 稀疏存储，只存非零值

    def add_document(self, doc_id: str, text: str):
        tokens = jieba.lcut(text)
        term_freq = defaultdict(int)

        for token in tokens:
            term_freq[token] += 1

        # 只存储非零词频
        self.term_freq[doc_id] = dict(term_freq)

        # 更新文档频率
        for token in set(tokens):
            self.doc_freq[token] += 1

    def compress_data(self):
        """压缩存储数据"""
        import json
        data = {
            'doc_freq': dict(self.doc_freq),
            'term_freq': self.term_freq
        }
        json_str = json.dumps(data)
        return gzip.compress(json_str.encode())
```

## 并发性能优化

### 问题表现

- 多用户同时查询时响应变慢
- 系统吞吐量低
- 资源竞争导致性能下降

### 解决方案

#### 1. 连接池

```python
from concurrent.futures import ThreadPoolExecutor
import threading

class PooledRetriever:
    def __init__(self, retriever, max_workers=10):
        self.retriever = retriever
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.lock = threading.Lock()

    def search(self, q: RetrievalQuery):
        """线程池执行检索"""
        future = self.executor.submit(self.retriever.search, q)
        return future.result()
```

#### 2. 读写分离

```python
import threading
import time

class ReadWriteBM25:
    def __init__(self):
        self.read_index = None
        self.write_index = None
        self.lock = threading.RLock()
        self.last_sync = time.time()
        self.sync_interval = 60  # 60秒同步一次

    def search(self, query: str):
        """读操作，使用读索引"""
        with self.lock:
            if self.read_index is None:
                return []
            return self.read_index.get_scores(query)

    def add_document(self, doc_id: str, text: str):
        """写操作，使用写索引"""
        with self.lock:
            if self.write_index is None:
                self.write_index = BM25Okapi([])
            self.write_index.add_document(doc_id, text)

            # 定期同步到读索引
            if time.time() - self.last_sync > self.sync_interval:
                self._sync_to_read_index()

    def _sync_to_read_index(self):
        """同步写索引到读索引"""
        if self.write_index is not None:
            self.read_index = self.write_index
            self.last_sync = time.time()
```

#### 3. 负载均衡

```python
import random
import time

class LoadBalancedRetriever:
    def __init__(self, retrievers):
        self.retrievers = retrievers
        self.current_index = 0
        self.request_counts = [0] * len(retrievers)
        self.response_times = [[] for _ in range(len(retrievers))]

    def search(self, q: RetrievalQuery):
        """负载均衡选择检索器"""
        # 选择负载最轻的检索器
        retriever_index = self._select_retriever()
        retriever = self.retrievers[retriever_index]

        start_time = time.time()
        try:
            result = retriever.search(q)
            response_time = time.time() - start_time
            self.response_times[retriever_index].append(response_time)
            return result
        finally:
            self.request_counts[retriever_index] += 1

    def _select_retriever(self):
        """选择最优检索器"""
        # 基于请求数和响应时间选择
        scores = []
        for i, retriever in enumerate(self.retrievers):
            request_count = self.request_counts[i]
            avg_response_time = (sum(self.response_times[i]) / len(self.response_times[i])
                               if self.response_times[i] else 0)
            score = request_count + avg_response_time * 100
            scores.append(score)

        return scores.index(min(scores))
```

## 数据量增长优化

### 问题表现

- 索引构建时间线性增长
- 查询时间随数据量增长
- 内存占用持续增加

### 解决方案

#### 1. 增量更新

```python
class IncrementalBM25:
    def __init__(self):
        self.index = None
        self.documents = {}
        self.document_hashes = {}

    def add_document(self, doc_id: str, text: str):
        """增量添加文档"""
        text_hash = hashlib.md5(text.encode()).hexdigest()

        # 检查文档是否已存在且内容未变化
        if doc_id in self.document_hashes and self.document_hashes[doc_id] == text_hash:
            return  # 文档未变化，跳过

        # 如果文档已存在但内容变化，先删除
        if doc_id in self.documents:
            self.remove_document(doc_id)

        # 添加新文档
        self.documents[doc_id] = text
        self.document_hashes[doc_id] = text_hash

        # 重建索引（对于小规模数据，全量重建可能更快）
        self._rebuild_index()

    def remove_document(self, doc_id: str):
        """删除文档"""
        if doc_id in self.documents:
            del self.documents[doc_id]
            del self.document_hashes[doc_id]
            self._rebuild_index()

    def _rebuild_index(self):
        """重建索引"""
        if not self.documents:
            self.index = BM25Okapi([])
            return

        tokenized_docs = []
        for text in self.documents.values():
            tokens = jieba.lcut(text)
            tokenized_docs.append(tokens)

        self.index = BM25Okapi(tokenized_docs)
```

#### 2. 分布式索引

```python
import requests
from typing import List

class DistributedBM25:
    def __init__(self, nodes: List[str]):
        self.nodes = nodes
        self.current_node = 0

    def search(self, query: str, top_k: int = 10):
        """分布式搜索"""
        all_results = []

        # 向所有节点发送查询
        for node in self.nodes:
            try:
                response = requests.post(
                    f"{node}/search",
                    json={"query": query, "top_k": top_k},
                    timeout=5
                )
                if response.status_code == 200:
                    results = response.json()
                    all_results.extend(results)
            except Exception as e:
                print(f"节点 {node} 查询失败: {e}")

        # 合并并排序结果
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:top_k]

    def add_document(self, doc_id: str, text: str):
        """分布式添加文档"""
        # 选择节点（可以基于文档ID哈希）
        node_index = hash(doc_id) % len(self.nodes)
        node = self.nodes[node_index]

        try:
            response = requests.post(
                f"{node}/add_document",
                json={"doc_id": doc_id, "text": text},
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            print(f"添加文档到节点 {node} 失败: {e}")
            return False
```

#### 3. 分层索引

```python
import time
from collections import OrderedDict

class TieredBM25:
    def __init__(self, hot_threshold=1000, warm_threshold=10000):
        self.hot_index = BM25Okapi([])      # 内存中的热门文档
        self.warm_index = None              # 磁盘中的温文档
        self.cold_index = None              # 磁盘中的冷文档

        self.hot_documents = OrderedDict()
        self.warm_documents = {}
        self.cold_documents = {}

        self.hot_threshold = hot_threshold
        self.warm_threshold = warm_threshold

        self.access_counts = {}  # 文档访问计数
        self.last_access = {}    # 最后访问时间

    def search(self, query: str, top_k: int = 10):
        """分层搜索"""
        all_results = []

        # 搜索热门文档
        if self.hot_index:
            hot_results = self.hot_index.get_scores(query)
            for i, score in enumerate(hot_results):
                if score > 0:
                    doc_id = list(self.hot_documents.keys())[i]
                    all_results.append({
                        'doc_id': doc_id,
                        'score': score,
                        'tier': 'hot'
                    })

        # 搜索温文档（如果存在）
        if self.warm_index:
            warm_results = self.warm_index.get_scores(query)
            for i, score in enumerate(warm_results):
                if score > 0:
                    doc_id = list(self.warm_documents.keys())[i]
                    all_results.append({
                        'doc_id': doc_id,
                        'score': score,
                        'tier': 'warm'
                    })

        # 按分数排序
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:top_k]

    def add_document(self, doc_id: str, text: str):
        """添加文档到合适的分层"""
        # 新文档默认加入热门层
        self.hot_documents[doc_id] = text
        self.access_counts[doc_id] = 0
        self.last_access[doc_id] = time.time()

        # 重建热门索引
        self._rebuild_hot_index()

        # 检查是否需要重新分层
        self._rebalance_tiers()

    def _rebalance_tiers(self):
        """重新平衡分层"""
        current_time = time.time()

        # 将访问次数少的文档移到温层
        to_warm = []
        for doc_id, access_count in self.access_counts.items():
            if (access_count < 5 and
                current_time - self.last_access[doc_id] > 3600):  # 1小时未访问
                to_warm.append(doc_id)

        for doc_id in to_warm:
            if doc_id in self.hot_documents:
                self.warm_documents[doc_id] = self.hot_documents[doc_id]
                del self.hot_documents[doc_id]
                del self.access_counts[doc_id]
                del self.last_access[doc_id]

        if to_warm:
            self._rebuild_hot_index()
            self._rebuild_warm_index()
```

## 索引更新优化

### 问题表现

- 文档频繁增删改
- 索引重建成本高
- 更新操作阻塞查询

### 解决方案

#### 1. 批量更新

```python
class BatchBM25:
    def __init__(self, batch_size=100):
        self.batch_size = batch_size
        self.pending_updates = []
        self.lock = threading.Lock()

    def add_document(self, doc_id: str, text: str):
        """添加到批量更新队列"""
        with self.lock:
            self.pending_updates.append(('add', doc_id, text))

            # 达到批量大小时执行更新
            if len(self.pending_updates) >= self.batch_size:
                self._flush_updates()

    def remove_document(self, doc_id: str):
        """添加到批量删除队列"""
        with self.lock:
            self.pending_updates.append(('remove', doc_id, None))

            if len(self.pending_updates) >= self.batch_size:
                self._flush_updates()

    def _flush_updates(self):
        """批量执行更新"""
        if not self.pending_updates:
            return

        # 处理批量更新
        adds = []
        removes = []

        for op, doc_id, text in self.pending_updates:
            if op == 'add':
                adds.append((doc_id, text))
            elif op == 'remove':
                removes.append(doc_id)

        # 执行删除
        for doc_id in removes:
            if doc_id in self.documents:
                del self.documents[doc_id]

        # 执行添加
        for doc_id, text in adds:
            self.documents[doc_id] = text

        # 重建索引
        self._rebuild_index()

        # 清空待处理队列
        self.pending_updates.clear()
```

#### 2. 版本控制

```python
import time
from typing import Dict, Any

class VersionedBM25:
    def __init__(self):
        self.versions = {}
        self.current_version = 0
        self.version_metadata = {}

    def create_new_version(self, metadata: Dict[str, Any] = None):
        """创建新版本"""
        self.current_version += 1
        self.versions[self.current_version] = BM25Okapi([])
        self.version_metadata[self.current_version] = {
            'created_at': time.time(),
            'document_count': 0,
            'metadata': metadata or {}
        }
        return self.current_version

    def add_document_to_version(self, version: int, doc_id: str, text: str):
        """向指定版本添加文档"""
        if version not in self.versions:
            raise ValueError(f"版本 {version} 不存在")

        # 这里需要实现版本特定的文档管理
        # 简化示例：直接重建索引
        self._rebuild_version_index(version)
        self.version_metadata[version]['document_count'] += 1

    def search_version(self, version: int, query: str, top_k: int = 10):
        """搜索指定版本"""
        if version not in self.versions:
            return []

        return self.versions[version].get_scores(query)[:top_k]

    def _rebuild_version_index(self, version: int):
        """重建指定版本的索引"""
        # 实现版本特定的索引重建逻辑
        pass
```

## 特定场景优化

### 问题表现

- 特定查询模式性能差
- 某些文档类型检索效果不好
- 特定业务场景需要特殊处理

### 解决方案

#### 1. 查询优化

```python
import re
from typing import Set

class QueryOptimizer:
    def __init__(self):
        self.stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        self.synonyms = {
            '电脑': ['计算机', 'PC', '台式机'],
            '手机': ['移动电话', '智能手机', '电话'],
            '汽车': ['车辆', '轿车', '机动车']
        }

    def optimize_query(self, query: str) -> str:
        """优化查询"""
        # 1. 去除停用词
        tokens = jieba.lcut(query)
        filtered_tokens = [token for token in tokens if token not in self.stop_words]

        # 2. 同义词扩展
        expanded_tokens = []
        for token in filtered_tokens:
            expanded_tokens.append(token)
            if token in self.synonyms:
                expanded_tokens.extend(self.synonyms[token])

        # 3. 查询重写
        optimized_query = ' '.join(expanded_tokens)

        # 4. 去除重复词
        unique_tokens = list(set(expanded_tokens))
        return ' '.join(unique_tokens)

    def extract_keywords(self, query: str) -> Set[str]:
        """提取关键词"""
        tokens = jieba.lcut(query)
        # 过滤停用词和短词
        keywords = {token for token in tokens
                   if len(token) > 1 and token not in self.stop_words}
        return keywords
```

#### 2. 文档预处理

```python
import re
from typing import Dict, Any

class DocumentPreprocessor:
    def __init__(self):
        self.noise_patterns = [
            r'<[^>]+>',  # HTML标签
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',  # URL
            r'\d{4}-\d{2}-\d{2}',  # 日期
            r'\d+\.\d+',  # 版本号
        ]

    def preprocess_document(self, doc_id: str, text: str) -> Dict[str, Any]:
        """文档预处理"""
        # 1. 去除噪音
        cleaned_text = text
        for pattern in self.noise_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text)

        # 2. 关键词提取
        keywords = self._extract_keywords(cleaned_text)

        # 3. 文档分类
        doc_type = self._classify_document(cleaned_text)

        # 4. 质量评估
        quality_score = self._assess_quality(cleaned_text)

        return {
            'doc_id': doc_id,
            'cleaned_text': cleaned_text,
            'keywords': keywords,
            'doc_type': doc_type,
            'quality_score': quality_score,
            'original_length': len(text),
            'cleaned_length': len(cleaned_text)
        }

    def _extract_keywords(self, text: str) -> Set[str]:
        """提取关键词"""
        tokens = jieba.lcut(text)
        # 使用TF-IDF或TextRank提取关键词
        # 简化示例：返回高频词
        from collections import Counter
        word_freq = Counter(tokens)
        return {word for word, freq in word_freq.most_common(10) if len(word) > 1}

    def _classify_document(self, text: str) -> str:
        """文档分类"""
        # 基于关键词的简单分类
        if any(keyword in text for keyword in ['代码', '函数', '变量', '类']):
            return 'code'
        elif any(keyword in text for keyword in ['API', '接口', '请求', '响应']):
            return 'api'
        elif any(keyword in text for keyword in ['错误', '异常', '问题', 'bug']):
            return 'error'
        else:
            return 'general'

    def _assess_quality(self, text: str) -> float:
        """质量评估"""
        # 基于文本长度、完整性等评估质量
        if len(text) < 10:
            return 0.1
        elif len(text) < 100:
            return 0.5
        else:
            return 1.0
```

#### 3. 混合检索

```python
from typing import List, Dict, Any

class HybridRetriever:
    def __init__(self, vector_retriever, keyword_retriever):
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.query_optimizer = QueryOptimizer()

    def search(self, query: str, top_k: int = 10,
               vector_weight: float = 0.7, keyword_weight: float = 0.3) -> List[Dict[str, Any]]:
        """混合检索"""
        # 1. 查询优化
        optimized_query = self.query_optimizer.optimize_query(query)

        # 2. 向量检索
        vector_results = self.vector_retriever.search(
            RetrievalQuery(query=query, top_k=top_k)
        )

        # 3. 关键词检索
        keyword_results = self.keyword_retriever.search(
            RetrievalQuery(query=optimized_query, top_k=top_k)
        )

        # 4. 结果融合
        merged_results = self._merge_results(
            vector_results, keyword_results,
            vector_weight, keyword_weight, top_k
        )

        return merged_results

    def _merge_results(self, vector_results, keyword_results,
                      vector_weight: float, keyword_weight: float, top_k: int):
        """融合检索结果"""
        # 构建文档ID到结果的映射
        doc_scores = {}

        # 处理向量检索结果
        for item in vector_results.items:
            doc_scores[item.id] = {
                'vector_score': item.score * vector_weight,
                'keyword_score': 0,
                'text': item.text,
                'metadata': item.metadata
            }

        # 处理关键词检索结果
        for item in keyword_results.items:
            if item.id in doc_scores:
                doc_scores[item.id]['keyword_score'] = item.score * keyword_weight
            else:
                doc_scores[item.id] = {
                    'vector_score': 0,
                    'keyword_score': item.score * keyword_weight,
                    'text': item.text,
                    'metadata': item.metadata
                }

        # 计算综合分数并排序
        final_results = []
        for doc_id, scores in doc_scores.items():
            final_score = scores['vector_score'] + scores['keyword_score']
            final_results.append({
                'doc_id': doc_id,
                'score': final_score,
                'vector_score': scores['vector_score'],
                'keyword_score': scores['keyword_score'],
                'text': scores['text'],
                'metadata': scores['metadata']
            })

        # 按分数排序并返回TopK
        final_results.sort(key=lambda x: x['score'], reverse=True)
        return final_results[:top_k]
```

## 优化决策流程

### 性能问题诊断

```python
class PerformanceDiagnostic:
    def __init__(self):
        self.metrics = {
            'query_time': [],
            'memory_usage': [],
            'cpu_usage': [],
            'concurrent_requests': 0
        }

    def should_optimize(self, problem_type: str, current_performance: Dict[str, float],
                       requirements: Dict[str, float]) -> tuple[bool, str]:
        """判断是否需要优化"""

        if problem_type == 'query_slow':
            if current_performance['avg_query_time'] > requirements['max_query_time']:
                return True, "查询响应时间超过阈值，建议实施缓存或异步查询"

        elif problem_type == 'memory_high':
            if current_performance['memory_usage'] > requirements['max_memory']:
                return True, "内存使用率过高，建议实施分片存储或磁盘缓存"

        elif problem_type == 'concurrent_poor':
            if current_performance['throughput'] < requirements['min_throughput']:
                return True, "并发性能不足，建议实施连接池或负载均衡"

        elif problem_type == 'data_growth':
            if current_performance['index_build_time'] > requirements['max_build_time']:
                return True, "索引构建时间过长，建议实施增量更新或分布式索引"

        return False, "当前性能满足要求"

    def get_optimization_strategy(self, problem_type: str) -> str:
        """获取优化策略建议"""
        strategies = {
            'query_slow': [
                "1. 实施查询缓存（LRU Cache）",
                "2. 异步查询处理",
                "3. 预计算热门查询索引",
                "4. 查询优化和重写"
            ],
            'memory_high': [
                "1. 分片存储（Sharded Storage）",
                "2. 磁盘缓存（Disk Cache）",
                "3. 压缩存储（Compressed Storage）",
                "4. 分层索引（Tiered Index）"
            ],
            'concurrent_poor': [
                "1. 连接池（Connection Pool）",
                "2. 读写分离（Read-Write Split）",
                "3. 负载均衡（Load Balancing）",
                "4. 异步处理（Async Processing）"
            ],
            'data_growth': [
                "1. 增量更新（Incremental Update）",
                "2. 分布式索引（Distributed Index）",
                "3. 批量处理（Batch Processing）",
                "4. 版本控制（Version Control）"
            ]
        }

        return strategies.get(problem_type, ["请先明确具体问题类型"])
```

## 性能监控与测量

### 监控指标

```python
import time
import psutil
import threading
from typing import Dict, List
from collections import defaultdict, deque

class PerformanceMonitor:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.query_times = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        self.cpu_usage = deque(maxlen=window_size)
        self.request_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.lock = threading.Lock()

    def record_query_time(self, query_time: float):
        """记录查询时间"""
        with self.lock:
            self.query_times.append(query_time)

    def record_system_metrics(self):
        """记录系统指标"""
        with self.lock:
            self.memory_usage.append(psutil.virtual_memory().percent)
            self.cpu_usage.append(psutil.cpu_percent())

    def record_request(self, endpoint: str, success: bool = True):
        """记录请求"""
        with self.lock:
            self.request_counts[endpoint] += 1
            if not success:
                self.error_counts[endpoint] += 1

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        with self.lock:
            return {
                'avg_query_time': sum(self.query_times) / len(self.query_times) if self.query_times else 0,
                'max_query_time': max(self.query_times) if self.query_times else 0,
                'min_query_time': min(self.query_times) if self.query_times else 0,
                'avg_memory_usage': sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0,
                'avg_cpu_usage': sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0,
                'total_requests': sum(self.request_counts.values()),
                'error_rate': sum(self.error_counts.values()) / max(sum(self.request_counts.values()), 1),
                'request_counts': dict(self.request_counts),
                'error_counts': dict(self.error_counts)
            }

    def is_performance_degraded(self, thresholds: Dict[str, float]) -> bool:
        """检查性能是否下降"""
        summary = self.get_performance_summary()

        if summary['avg_query_time'] > thresholds.get('max_query_time', 2.0):
            return True
        if summary['avg_memory_usage'] > thresholds.get('max_memory', 80.0):
            return True
        if summary['avg_cpu_usage'] > thresholds.get('max_cpu', 80.0):
            return True
        if summary['error_rate'] > thresholds.get('max_error_rate', 0.05):
            return True

        return False
```

### 性能测试

```python
import asyncio
import aiohttp
import time
from typing import List, Dict, Any

class PerformanceTester:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.results = []

    async def test_concurrent_queries(self, queries: List[str], concurrency: int = 10):
        """测试并发查询性能"""
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(concurrency)

            async def single_query(query: str):
                async with semaphore:
                    start_time = time.time()
                    try:
                        async with session.post(
                            f"{self.base_url}/search",
                            json={"query": query, "top_k": 10}
                        ) as response:
                            await response.json()
                            end_time = time.time()
                            return {
                                'query': query,
                                'response_time': end_time - start_time,
                                'status': response.status,
                                'success': response.status == 200
                            }
                    except Exception as e:
                        end_time = time.time()
                        return {
                            'query': query,
                            'response_time': end_time - start_time,
                            'status': 0,
                            'success': False,
                            'error': str(e)
                        }

            tasks = [single_query(query) for query in queries]
            results = await asyncio.gather(*tasks)

            self.results.extend(results)
            return results

    def analyze_results(self) -> Dict[str, Any]:
        """分析测试结果"""
        if not self.results:
            return {}

        successful_results = [r for r in self.results if r['success']]
        failed_results = [r for r in self.results if not r['success']]

        response_times = [r['response_time'] for r in successful_results]

        return {
            'total_queries': len(self.results),
            'successful_queries': len(successful_results),
            'failed_queries': len(failed_results),
            'success_rate': len(successful_results) / len(self.results),
            'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
            'max_response_time': max(response_times) if response_times else 0,
            'min_response_time': min(response_times) if response_times else 0,
            'error_details': [r['error'] for r in failed_results if 'error' in r]
        }
```

## 总结

### 优化原则

1. **先测量，再优化**：用数据说话，不要凭感觉
2. **找到瓶颈**：80% 的性能问题来自 20% 的代码
3. **渐进式优化**：一次解决一个问题
4. **保持简单**：优化后的代码仍然要可读可维护
5. **问题驱动**：遇到问题再优化，不要过早优化

### 优化决策矩阵

| 问题类型 | 数据量     | 并发量   | 推荐方案            |
| -------- | ---------- | -------- | ------------------- |
| 查询慢   | 小(<1 万)  | 低(<10)  | 缓存 + 查询优化     |
| 查询慢   | 小(<1 万)  | 高(>100) | 异步查询 + 连接池   |
| 查询慢   | 大(>10 万) | 低(<10)  | 分片存储 + 预计算   |
| 查询慢   | 大(>10 万) | 高(>100) | 分布式 + 负载均衡   |
| 内存高   | 任意       | 任意     | 分片存储 + 磁盘缓存 |
| 并发差   | 任意       | 高(>100) | 连接池 + 读写分离   |

### 监控指标阈值

| 指标         | 正常范围 | 警告阈值 | 危险阈值 |
| ------------ | -------- | -------- | -------- |
| 查询响应时间 | < 1 秒   | 1-2 秒   | > 2 秒   |
| 内存使用率   | < 60%    | 60-80%   | > 80%    |
| CPU 使用率   | < 50%    | 50-80%   | > 80%    |
| 错误率       | < 1%     | 1-5%     | > 5%     |
| 并发请求数   | < 50     | 50-100   | > 100    |

记住：**优化是手段，不是目的**。当前这个简单的 BM25 实现就是最好的选择，因为它解决了问题、代码简单、性能够用、容易理解和维护。等真正遇到性能瓶颈时，再针对性优化也不迟！
