import os
from enum import Enum
from pydantic import BaseModel, Field
from typing import Any, Optional, List
from langchain_core.runnables import RunnableConfig

class ToolType(Enum):
    """可用的工具類型。"""
    DEEP_SEARCH = "deep_search"
    URL_READER = "url_reader"
    EMBEDDING = "embedding"
    RAG_QUERY = "rag_query"
    SUMMARIZE = "summarize"

class AgentConfiguration(BaseModel):
    """Agentic RAG 智能助手的配置類。"""
    
    # Jina API 設置
    jina_api_key: str = Field(
        default="",
        title="Jina API Key",
        description="Jina 服務的 API 密鑰"
    )
    
    # 工具設置
    available_tools: List[ToolType] = Field(
        default_factory=lambda: [t for t in ToolType],
        title="可用工具",
        description="代理可以使用的工具列表"
    )
    
    # DeepSearch 設置
    reasoning_effort: str = Field(
        default="high",
        title="推理努力程度",
        description="Jina DeepSearch 的推理努力程度"
    )
    
    # Embedding 設置
    embedding_model: str = Field(
        default="jina-embeddings-v3",
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
    ) -> "AgentConfiguration":
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
