import operator
from dataclasses import dataclass, field
from typing_extensions import Annotated
from typing import List, Dict, Any, Optional, Literal

@dataclass(kw_only=True)
class FlexibleResearchState:
    """靈活輸入與 RAG 集成的狀態類。"""
    
    # 輸入
    input_type: Literal["url", "query"] = field(default=None)  # 輸入類型
    url_input: Optional[str] = field(default=None)  # URL 輸入
    query_input: Optional[str] = field(default=None)  # 查詢輸入
    
    # URL 處理結果
    url_content: Optional[str] = field(default=None)  # URL 內容
    processed_content: List[Dict[str, Any]] = field(default_factory=list)  # 結構化內容
    
    # 查詢處理結果
    rag_results: Optional[str] = field(default=None)  # RAG 結果
    retrieved_documents: List[Dict[str, Any]] = field(default_factory=list)  # 檢索的文檔
    
    # 向量嵌入
    vector_embeddings: List[Dict[str, Any]] = field(default_factory=list)  # 向量嵌入
    
    # 存儲結果
    storage_references: List[str] = field(default_factory=list)  # 存儲引用
    
    # 會話控制
    session_complete: bool = field(default=False)  # 會話是否完成

@dataclass(kw_only=True)
class FlexibleResearchStateInput:
    """輸入狀態類。"""
    input_type: Literal["url", "query"] = field(default=None)
    url_input: Optional[str] = field(default=None)
    query_input: Optional[str] = field(default=None)

@dataclass(kw_only=True)
class FlexibleResearchStateOutput:
    """輸出狀態類。"""
    rag_results: Optional[str] = field(default=None)
    url_content: Optional[str] = field(default=None)
    storage_references: List[str] = field(default_factory=list)
