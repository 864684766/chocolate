import chromadb
from typing import Optional, Dict, Any, List, Literal
from ....config import get_config_manager
from ....infra.exceptions.exceptions import DatabaseConnectionError

# 定义允许的include字段类型
# Literal 类型限制 include 参数只能是这些特定的字符串值
# 注意：ChromaDB API 使用 "metadatas"（复数形式），这是官方 API 的正确字段名
# 虽然拼写检查器可能认为这是错误的，但这是 ChromaDB 库的标准
IncludeField = Literal["documents", "embeddings", "metadatas", "distances", "uris", "data"]

# 类型别名，用于避免拼写检查问题
MetadataList = List[Dict[str, Any]]  # 元数据列表类型


class ChromaDBHelper:
    """ChromaDB 数据库助手类
    
    负责管理 ChromaDB 连接、集合操作和向量数据库相关功能
    """
    
    def __init__(self, config_manager=None):
        """初始化 ChromaDB 助手
        
        Args:
            config_manager: 配置管理器实例，如果为 None 则使用默认配置
        """
        self.config_manager = config_manager or get_config_manager()
        self._client: Optional[chromadb.ClientAPI] = None
        self._collections_cache: Dict[str, chromadb.Collection] = {}
        
    def _get_connection_config(self) -> Dict[str, Any]:
        """获取数据库连接配置
        
        Returns:
            包含 host 和 port 的配置字典
            
        Raises:
            DatabaseConnectionError: 当配置不完整时抛出
        """
        config = self.config_manager.get_vector_database_config()
        host = config.get("host")
        port = config.get("port")
        
        if not host or not port:
            raise DatabaseConnectionError(
                "向量数据库配置不完整，缺少 host 或 port"
            )
            
        return {"host": host, "port": port}
    
    def connect(self) -> chromadb.ClientAPI:
        """建立数据库连接
        
        Returns:
            ChromaDB 客户端 API 实例
            
        Raises:
            DatabaseConnectionError: 连接失败时抛出
        """
        if self._client is not None:
            return self._client
            
        try:
            config = self._get_connection_config()
            self._client = chromadb.HttpClient(
                host=config["host"], 
                port=config["port"]
            )
            
            # 验证连接
            self._client.heartbeat()
            return self._client
            
        except Exception as e:
            raise DatabaseConnectionError(
                f"无法连接到 ChromaDB: {str(e)}"
            ) from e
    
    def disconnect(self) -> None:
        """断开数据库连接"""
        if self._client:
            self._client = None
        self._collections_cache.clear()
    
    def get_client(self) -> chromadb.ClientAPI:
        """获取数据库客户端实例，如果未连接则自动连接
        
        Returns:
            ChromaDB 客户端 API 实例
        """
        return self.connect()
    
    def get_collection(self, name: str, create_if_not_exists: bool = True) -> chromadb.Collection:
        """获取或创建集合
        
        Args:
            name: 集合名称
            create_if_not_exists: 如果集合不存在是否创建
            
        Returns:
            ChromaDB 集合实例
            
        Raises:
            DatabaseConnectionError: 操作失败时抛出
        """
        if name in self._collections_cache:
            return self._collections_cache[name]
            
        try:
            client = self.get_client()
            
            if create_if_not_exists:
                collection = client.get_or_create_collection(name=name)
            else:
                collection = client.get_collection(name=name)
                
            self._collections_cache[name] = collection
            return collection
            
        except Exception as e:
            raise DatabaseConnectionError(
                f"获取集合 '{name}' 失败: {str(e)}"
            ) from e
    
    def list_collections(self) -> List[str]:
        """列出所有集合名称
        
        Returns:
            集合名称列表
        """
        try:
            client = self.get_client()
            collections = client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            raise DatabaseConnectionError(
                f"列出集合失败: {str(e)}"
            ) from e
    
    def delete_collection(self, name: str) -> bool:
        """删除集合
        
        Args:
            name: 集合名称
            
        Returns:
            删除成功返回 True
            
        Raises:
            DatabaseConnectionError: 删除失败时抛出
        """
        try:
            client = self.get_client()
            client.delete_collection(name=name)
            
            # 从缓存中移除
            if name in self._collections_cache:
                del self._collections_cache[name]
                
            return True
            
        except Exception as e:
            raise DatabaseConnectionError(
                f"删除集合 '{name}' 失败: {str(e)}"
            ) from e
    
    def add(
        self,
        collection_name: str,
        documents: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[MetadataList] = None,  # ChromaDB API 使用复数形式
        ids: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """向集合中添加数据
        
        Args:
            collection_name: 集合名称
            documents: 文档列表
            embeddings: 向量嵌入列表
            metadatas: 元数据列表（ChromaDB API 使用复数形式）
            ids: ID列表
            **kwargs: 其他参数
            
        Raises:
            DatabaseConnectionError: 操作失败时抛出
        """
        try:
            collection = self.get_collection(collection_name)
            collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
                **kwargs
            )
        except Exception as e:
            raise DatabaseConnectionError(
                f"向集合 '{collection_name}' 添加数据失败: {str(e)}"
            ) from e
    
    def query(
        self,
        collection_name: str,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include: Optional[List[IncludeField]] = None,
        **kwargs
    ) -> chromadb.QueryResult:
        """查询集合中的数据
        
        Args:
            collection_name: 集合名称
            query_texts: 查询文本列表
            query_embeddings: 查询向量列表
            n_results: 返回结果数量，默认 10
            where: 元数据过滤条件
            where_document: 文档内容过滤条件
            include: 包含的字段列表，只能是 ["documents", "embeddings", "metadatas", "distances", "uris", "data"] 中的值
            **kwargs: 其他参数
            
        Returns:
            chromadb.QueryResult - 查询结果对象
            
        Raises:
            DatabaseConnectionError: 操作失败时抛出
        """
        try:
            collection = self.get_collection(collection_name, create_if_not_exists=False)
            result = collection.query(
                query_texts=query_texts,
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                where_document=where_document,
                include=include,
                **kwargs
            )
            return result
        except Exception as e:
            raise DatabaseConnectionError(
                f"查询集合 '{collection_name}' 失败: {str(e)}"
            ) from e
    
    def get(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include: Optional[List[IncludeField]] = None,
        **kwargs
    ) -> chromadb.GetResult:
        """获取集合中的数据
        
        Args:
            collection_name: 集合名称
            ids: ID列表
            where: 元数据过滤条件
            where_document: 文档内容过滤条件
            include: 包含的字段列表，只能是 ["documents", "embeddings", "metadatas", "distances", "uris", "data"] 中的值
            **kwargs: 其他参数
            
        Returns:
            chromadb.GetResult - 获取结果对象
            
        Raises:
            DatabaseConnectionError: 操作失败时抛出
        """
        try:
            collection = self.get_collection(collection_name, create_if_not_exists=False)
            result = collection.get(
                ids=ids,
                where=where,
                where_document=where_document,
                include=include,
                **kwargs
            )
            return result
        except Exception as e:
            raise DatabaseConnectionError(
                f"获取集合 '{collection_name}' 数据失败: {str(e)}"
            ) from e
    
    def update(
        self,
        collection_name: str,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[MetadataList] = None,  # ChromaDB API 使用复数形式
        documents: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """更新集合中的数据
        
        Args:
            collection_name: 集合名称
            ids: 要更新的ID列表
            embeddings: 新的向量嵌入列表
            metadatas: 新的元数据列表（ChromaDB API 使用复数形式）
            documents: 新的文档列表
            **kwargs: 其他参数
            
        Raises:
            DatabaseConnectionError: 操作失败时抛出
        """
        try:
            collection = self.get_collection(collection_name, create_if_not_exists=False)
            collection.update(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
                **kwargs
            )
        except Exception as e:
            raise DatabaseConnectionError(
                f"更新集合 '{collection_name}' 数据失败: {str(e)}"
            ) from e
    
    def delete(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> None:
        """删除集合中的数据
        
        Args:
            collection_name: 集合名称
            ids: 要删除的ID列表
            where: 元数据过滤条件
            where_document: 文档内容过滤条件
            
        Raises:
            DatabaseConnectionError: 操作失败时抛出
        """
        try:
            collection = self.get_collection(collection_name, create_if_not_exists=False)
            collection.delete(
                ids=ids,
                where=where,
                where_document=where_document
            )
        except Exception as e:
            raise DatabaseConnectionError(
                f"删除集合 '{collection_name}' 数据失败: {str(e)}"
            ) from e
    
    def modify(
        self,
        collection_name: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """修改集合属性
        
        Args:
            collection_name: 集合名称
            name: 新的集合名称
            metadata: 新的元数据
            
        Raises:
            DatabaseConnectionError: 操作失败时抛出
        """
        try:
            collection = self.get_collection(collection_name, create_if_not_exists=False)
            collection.modify(
                name=name,
                metadata=metadata
            )
            
            # 如果修改了名称，更新缓存
            if name and name != collection_name:
                if collection_name in self._collections_cache:
                    self._collections_cache[name] = self._collections_cache.pop(collection_name)
                    
        except Exception as e:
            raise DatabaseConnectionError(
                f"修改集合 '{collection_name}' 失败: {str(e)}"
            ) from e
    
    def count(self, collection_name: str) -> int:
        """获取集合中的数据数量
        
        Args:
            collection_name: 集合名称
            
        Returns:
            数据数量
            
        Raises:
            DatabaseConnectionError: 操作失败时抛出
        """
        try:
            collection = self.get_collection(collection_name, create_if_not_exists=False)
            return collection.count()
        except Exception as e:
            raise DatabaseConnectionError(
                f"获取集合 '{collection_name}' 数量失败: {str(e)}"
            ) from e
    
    def peek(self, collection_name: str, limit: int = 10) -> chromadb.GetResult:
        """预览集合中的数据
        
        Args:
            collection_name: 集合名称
            limit: 预览数量限制，默认 10
            
        Returns:
            chromadb.GetResult - 预览结果对象
            
        Raises:
            DatabaseConnectionError: 操作失败时抛出
        """
        try:
            collection = self.get_collection(collection_name, create_if_not_exists=False)
            return collection.peek(limit=limit)
        except Exception as e:
            raise DatabaseConnectionError(
                f"预览集合 '{collection_name}' 失败: {str(e)}"
            ) from e
    
    def is_connected(self) -> bool:
        """检查是否已连接到数据库
        
        Returns:
            已连接返回 True，否则返回 False
        """
        if self._client is None:
            return False
            
        try:
            self._client.heartbeat()
            return True
        except (ConnectionError, TimeoutError, OSError):
            # 只捕获具体连接相关异常，避免过于宽泛的异常处理
            return False
    
    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect()


# 便捷函数，保持向后兼容
def get_chroma_client() -> chromadb.ClientAPI:
    """获取 ChromaDB 客户端实例（便捷函数）
    
    Returns:
        ChromaDB 客户端 API 实例
        
    Note:
        建议使用 ChromaDBHelper 类来获得更好的资源管理
    """
    helper = ChromaDBHelper()
    return helper.get_client()


def connection_db() -> chromadb.ClientAPI:
    """获取数据库连接（向后兼容函数）
    
    Returns:
        ChromaDB 客户端 API 实例
        
    Note:
        此函数已废弃，建议使用 ChromaDBHelper 类
    """
    return get_chroma_client()