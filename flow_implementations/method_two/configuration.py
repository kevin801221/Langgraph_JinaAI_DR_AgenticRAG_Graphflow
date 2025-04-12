import os
from enum import Enum
from pydantic import BaseModel, Field
from typing import Any, Optional, Literal
from langchain_core.runnables import RunnableConfig

class JinaEmbeddingModel(Enum):
    """可用的 Jina embedding 模型。"""
    JINA_EMBEDDINGS_V3 = "jina-embeddings-v3"
    JINA_EMBEDDINGS_V2 = "jina-embeddings-v2"

class FlexibleConfiguration(BaseModel):
    """靈活輸入與 RAG 集成的配置類。"""
    
    # Jina API 設置
    jina_api_key: str = Field(
        default="",
        title="Jina API Key",
        description="Jina 服務的 API 密鑰"
    )
    
    # Reader 設置
    max_tokens_per_url: int = Field(
        default=10000,
        title="每個 URL 的最大 token 數",
        description="從每個 URL 提取的最大 token 數"
    )
    
    # Embedding 設置
    embedding_model: JinaEmbeddingModel = Field(
        default=JinaEmbeddingModel.JINA_EMBEDDINGS_V3,
        title="嵌入模型",
        description="用於創建嵌入的 Jina 模型"
    )
    
    # Qdrant 設置
    qdrant_url: str = Field(
        default="http://localhost:6333",
        title="Qdrant URL",
        description="Qdrant 向量數據庫的 URL"
    )
    qdrant_collection_name: str = Field(
        default="jina_research",
        title="Qdrant 集合名稱",
        description="Qdrant 中存儲嵌入的集合名稱"
    )
    
    # RAG 設置
    rag_top_k: int = Field(
        default=5,
        title="RAG Top K",
        description="檢索增強生成中檢索的文檔數量"
    )
    
    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "FlexibleConfiguration":
        """從 RunnableConfig 創建配置實例。"""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        
        # 從環境變量或配置中獲取原始值
        raw_values: dict[str, Any] = {
            name: os.environ.get(f"JINA_{name.upper()}", configurable.get(name))
            for name in cls.model_fields.keys()
        }
        
        # 過濾掉 None 值
        values = {k: v for k, v in raw_values.items() if v is not None}
        
        return cls(**values)
