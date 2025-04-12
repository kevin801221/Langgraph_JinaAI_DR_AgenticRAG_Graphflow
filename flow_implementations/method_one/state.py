import operator
from dataclasses import dataclass, field
from typing_extensions import Annotated
from typing import List, Dict, Any, Optional

@dataclass(kw_only=True)
class JinaResearchState:
    """狀態類，管理研究流程中的數據。"""
    
    # 輸入
    research_topic: str = field(default=None)  # 研究主題
    
    # DeepSearch 結果
    deep_research_report: str = field(default=None)  # 研究報告
    extracted_urls: Annotated[List[str], operator.add] = field(default_factory=list)  # URL 列表
    
    # Reader 結果
    url_contents: Dict[str, str] = field(default_factory=dict)  # URL 內容
    processed_content: List[Dict[str, Any]] = field(default_factory=list)  # 結構化內容
    
    # Embedding 結果
    vector_embeddings: List[Dict[str, Any]] = field(default_factory=list)  # 向量嵌入
    
    # 存儲結果
    storage_references: List[str] = field(default_factory=list)  # 存儲引用
    
    # 最終輸出
    final_summary: str = field(default=None)  # 最終摘要

@dataclass(kw_only=True)
class JinaResearchStateInput:
    """輸入狀態類。"""
    research_topic: str = field(default=None)

@dataclass(kw_only=True)
class JinaResearchStateOutput:
    """輸出狀態類。"""
    final_summary: str = field(default=None)
    storage_references: List[str] = field(default_factory=list)
