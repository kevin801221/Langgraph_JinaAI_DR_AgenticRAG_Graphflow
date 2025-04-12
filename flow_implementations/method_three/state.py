import operator
from dataclasses import dataclass, field
from typing_extensions import Annotated
from typing import List, Dict, Any, Optional

@dataclass(kw_only=True)
class AgentResearchState:
    """Agentic RAG 智能助手的狀態類。"""
    
    # 用戶輸入
    user_input: str = field(default=None)  # 用戶輸入
    
    # 意圖分析
    intent: str = field(default=None)  # 用戶意圖
    intent_details: Dict[str, Any] = field(default_factory=dict)  # 意圖詳情
    
    # 任務規劃
    task_plan: List[Dict[str, Any]] = field(default_factory=list)  # 任務計劃
    current_task_index: int = field(default=0)  # 當前任務索引
    
    # 工具選擇
    selected_tool: str = field(default=None)  # 選定的工具
    tool_parameters: Dict[str, Any] = field(default_factory=dict)  # 工具參數
    
    # 任務執行
    task_results: Annotated[List[Dict[str, Any]], operator.add] = field(default_factory=list)  # 任務結果
    current_task: Optional[Dict[str, Any]] = field(default=None)  # 當前任務
    
    # 反思
    reflection: str = field(default=None)  # 反思
    needs_more_research: bool = field(default=False)  # 是否需要更多研究
    
    # 最終響應
    final_response: str = field(default=None)  # 最終響應

@dataclass(kw_only=True)
class AgentResearchStateInput:
    """輸入狀態類。"""
    user_input: str = field(default=None)

@dataclass(kw_only=True)
class AgentResearchStateOutput:
    """輸出狀態類。"""
    final_response: str = field(default=None)
