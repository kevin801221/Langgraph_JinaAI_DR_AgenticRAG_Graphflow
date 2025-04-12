# 使用 LangGraph 和 Jina AI 構建智能研究助手：三種實現方案詳解

在當今數據爆炸的時代，研究人員、學者和知識工作者面臨著信息過載的挑戰。面對海量的文獻、網頁和資料，如何高效地收集、處理和利用這些信息成為一個亟待解決的問題。

本文將介紹如何使用 LangGraph 框架和 Jina AI 服務構建一個智能研究助手系統，該系統能夠自動化研究流程，處理和向量化內容，並提供智能檢索功能。我們將詳細介紹三種不同的實現方案，並通過代碼示例說明如何實現這些功能。

## 目錄

1. [技術背景介紹](#技術背景介紹)
2. [系統整體架構](#系統整體架構)
3. [方案一：自動研究流程](#方案一自動研究流程)
4. [方案二：靈活輸入與 RAG 集成](#方案二靈活輸入與-rag-集成)
5. [方案三：Agentic RAG 智能助手](#方案三agentic-rag-智能助手)
6. [三種方案的比較與選擇](#三種方案的比較與選擇)
7. [部署與擴展](#部署與擴展)
8. [結論與未來展望](#結論與未來展望)

## 技術背景介紹

在開始詳細介紹我們的系統之前，先簡單了解一下所使用的核心技術：

### LangGraph 框架

LangGraph 是一個基於 Python 的框架，專為構建複雜的 AI 應用流程而設計。它允許開發者將不同的處理節點組織為有向圖，實現數據的流動和狀態的管理。LangGraph 的主要優勢包括：

- **模組化設計**：將複雜任務分解為可管理的小節點
- **狀態管理**：在節點間傳遞和更新狀態
- **流程控制**：支持條件分支和循環
- **易於擴展**：靈活添加新節點和功能

### Jina AI 服務

Jina AI 提供了一系列強大的 AI 服務，用於深度研究、內容處理和向量化：

- **Jina DeepSearch**：進行深度網絡研究，生成綜合報告
- **Jina Reader**：提取和處理網頁內容
- **Jina Embeddings**：將內容轉換為向量表示

### Qdrant 向量數據庫

Qdrant 是一個專為向量相似性搜索設計的開源數據庫，提供高效的存儲和檢索功能，特別適合 AI 應用中的知識管理和檢索需求。

## 系統整體架構

無論選擇哪種實現方案，我們的系統都基於以下核心組件：

1. **LangGraph 框架**：用於構建處理流程
2. **Jina AI 服務**：提供核心 AI 功能
3. **Qdrant 向量數據庫**：存儲向量化知識
4. **狀態管理系統**：處理數據流動

接下來，我們將詳細介紹兩種不同的實現方案。

## 方案一：自動研究流程

方案一專注於自動化研究流程，從用戶提供的研究主題開始，自動進行深度研究、內容處理和向量化存儲。

### 流程圖

<!-- 在此處插入方案一的 LangGraph 流程圖，使用前面生成的 "方案一 LangGraph 流程圖" -->

### 核心節點解析

方案一包含 6 個主要節點，形成線性處理流程：

1. **perform_deep_research**：使用 Jina DeepSearch 進行深度研究
2. **extract_additional_urls**：從研究報告中提取額外 URL
3. **process_urls_with_reader**：處理 URL 內容
4. **create_embeddings**：創建向量嵌入
5. **store_embeddings**：存儲向量數據
6. **create_final_summary**：生成最終摘要

### 代碼實現

讓我們逐步實現這個系統的各個部分：

#### 1. 狀態管理類

首先，我們需要定義系統的狀態類，用於在節點間傳遞數據：

```python
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
```

#### 2. 配置管理

接下來，我們需要一個配置類來管理系統的各種參數：

```python
import os
from enum import Enum
from pydantic import BaseModel, Field
from typing import Any, Optional, Literal
from langchain_core.runnables import RunnableConfig

class JinaEmbeddingModel(Enum):
    """可用的 Jina embedding 模型。"""
    JINA_EMBEDDINGS_V3 = "jina-embeddings-v3"
    JINA_EMBEDDINGS_V2 = "jina-embeddings-v2"

class JinaConfiguration(BaseModel):
    """系統配置類。"""
    
    # Jina API 設置
    jina_api_key: str = Field(
        default="",
        title="Jina API Key",
        description="Jina 服務的 API 密鑰"
    )
    
    # DeepSearch 設置
    reasoning_effort: Literal["low", "medium", "high"] = Field(
        default="high",
        title="推理努力程度",
        description="Jina DeepSearch 的推理努力程度"
    )
    budget_tokens: int = Field(
        default=30,
        title="Token 預算",
        description="DeepSearch 的 token 預算"
    )
    max_attempts: int = Field(
        default=7,
        title="最大嘗試次數",
        description="DeepSearch 的最大嘗試次數"
    )
    max_returned_urls: int = Field(
        default=20,
        title="最大返回 URL 數",
        description="DeepSearch 返回的最大 URL 數量"
    )
    
    # 其他配置項...
    
    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "JinaConfiguration":
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
```

#### 3. 核心功能實現

現在，讓我們實現與 Jina 服務交互的核心功能：

```python
import json
import re
import httpx
from typing import List, Dict, Any, Optional
from langsmith import traceable

@traceable
def jina_deep_research(
    query: str, 
    api_key: str,
    reasoning_effort: str = "high",
    budget_tokens: int = 30,
    max_attempts: int = 7,
    max_returned_urls: int = 20
) -> Dict[str, Any]:
    """執行深度研究。"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # 準備學術源頭列表
    academic_sources = [
        "arxiv.org",
        "scholar.google.com",
        "ieee.org",
        # 更多學術源頭...
    ]
    
    payload = {
        "model": "jina-deepsearch-v1",
        "messages": [
            {
                "role": "user",
                "content": f"""請提供關於 '{query}' 的全面研究報告，包括：
                
                1. 最新和最有影響力的學術論文
                2. 主要研究趨勢和突破
                3. 最具影響力的研究者和機構
                4. 與其他領域的交叉點
                5. 實際應用和產業影響
                
                請優先引用高質量的學術資源，特別是發表在知名期刊和會議的論文。"""
            }
        ],
        "stream": False,
        "reasoning_effort": reasoning_effort,
        "budget_tokens": budget_tokens,
        "max_attempts": max_attempts,
        "no_direct_answer": False,
        "max_returned_urls": str(max_returned_urls),
        "response_format": {
            "type": "text"
        },
        "boost_hostnames": academic_sources
    }
    
    try:
        response = httpx.post(
            "https://deepsearch.jina.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120.0
        )
        response.raise_for_status()
        result = response.json()
        
        # 提取研究報告內容
        content = result["choices"][0]["message"]["content"]
        
        # 提取元數據
        metadata = {
            "model": result.get("model", "jina-deepsearch-v1"),
            "usage": result.get("usage", {}),
            "process_time": result.get("process_time", None)
        }
        
        # 獲取引用的 URL
        urls = []
        if "references" in result:
            urls = [ref.get("url") for ref in result["references"] if "url" in ref]
        
        return {
            "content": content,
            "metadata": metadata,
            "urls": urls
        }
        
    except Exception as e:
        print(f"Error in Jina DeepSearch: {str(e)}")
        return {
            "content": f"Error performing deep research: {str(e)}",
            "metadata": {},
            "urls": []
        }

# 更多核心功能函數...
```

#### 4. LangGraph 節點實現

接下來，實現 LangGraph 的各個處理節點：

```python
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

def perform_deep_research(state: JinaResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """執行深度研究節點。"""
    # 獲取配置
    configurable = JinaConfiguration.from_runnable_config(config)
    
    # 執行深度研究
    research_result = jina_deep_research(
        query=state.research_topic,
        api_key=configurable.jina_api_key,
        reasoning_effort=configurable.reasoning_effort,
        budget_tokens=configurable.budget_tokens,
        max_attempts=configurable.max_attempts,
        max_returned_urls=configurable.max_returned_urls
    )
    
    # 更新狀態
    return {
        "deep_research_report": research_result["content"],
        "extracted_urls": research_result.get("urls", [])
    }

def extract_additional_urls(state: JinaResearchState) -> Dict[str, Any]:
    """從研究報告中提取額外 URL 節點。"""
    # 提取額外 URL 的實現
    # ...
    
    return {"extracted_urls": additional_urls}

# 實現其他節點...
```

#### 5. 構建 LangGraph

最後，我們將所有節點組合為一個完整的 LangGraph：

```python
from langgraph.graph import START, END, StateGraph

def build_graph():
    """構建方案一的 LangGraph。"""
    
    # 創建圖形構建器
    builder = StateGraph(
        JinaResearchState, 
        input=JinaResearchStateInput, 
        output=JinaResearchStateOutput, 
        config_schema=JinaConfiguration
    )
    
    # 添加節點
    builder.add_node("perform_deep_research", perform_deep_research)
    builder.add_node("extract_additional_urls", extract_additional_urls)
    builder.add_node("process_urls_with_reader", process_urls_with_reader)
    builder.add_node("create_embeddings", create_embeddings)
    builder.add_node("store_embeddings", store_embeddings)
    builder.add_node("create_final_summary", create_final_summary)
    
    # 添加邊
    builder.add_edge(START, "perform_deep_research")
    builder.add_edge("perform_deep_research", "extract_additional_urls")
    builder.add_edge("extract_additional_urls", "process_urls_with_reader")
    builder.add_edge("process_urls_with_reader", "create_embeddings")
    builder.add_edge("create_embeddings", "store_embeddings")
    builder.add_edge("store_embeddings", "create_final_summary")
    builder.add_edge("create_final_summary", END)
    
    # 編譯圖形
    return builder.compile()

# 創建最終圖形
graph = build_graph()
```

### 使用方案一

使用方案一進行研究的示例代碼：

```python
# 設置環境
import os
os.environ["JINA_API_KEY"] = "your_jina_api_key"

# 進行研究
result = graph.invoke(
    {"research_topic": "量子計算在藥物發現中的應用"},
    config={"configurable": {"max_web_research_loops": 3}}
)

# 輸出結果
print("研究摘要：")
print(result.final_summary)
print("\n存儲引用：")
print(result.storage_references)
```

## 方案二：靈活輸入與 RAG 集成

方案二提供了一個更靈活的架構，允許用戶直接輸入 URL 或查詢，並整合了 RAG (Retrieval-Augmented Generation) 聊天機器人功能。

### 流程圖

<!-- 在此處插入方案二的 LangGraph 流程圖，使用前面生成的 "方案二 LangGraph 流程圖" -->

### 核心節點解析

方案二包含 7 個主要節點，形成分支和循環流程：

1. **route_input_type**：判斷輸入類型（URL 或查詢）
2. **process_urls_with_reader**：處理 URL 內容
3. **create_embeddings**：創建向量嵌入
4. **store_embeddings**：存儲向量數據
5. **process_query_with_rag**：使用 RAG 處理查詢
6. **route_completion**：決定流程完成方式
7. **create_final_summary**：生成最終摘要

### 代碼實現

讓我們逐步實現方案二的各個部分：

#### 1. 修改後的狀態類

對於方案二，我們需要一個更靈活的狀態類：

```python
@dataclass(kw_only=True)
class FlexibleResearchState:
    """靈活的研究狀態類。"""
    
    user_input: str = field(default=None)  # 用戶輸入（URL 或查詢）
    input_type: str = field(default=None)  # 輸入類型："url" 或 "query"
    urls: List[str] = field(default_factory=list)  # URL 列表
    url_contents: Dict[str, str] = field(default_factory=dict)  # URL 內容
    processed_content: List[Dict[str, Any]] = field(default_factory=list)  # 處理後的內容
    vector_embeddings: List[Dict[str, Any]] = field(default_factory=list)  # 向量嵌入
    storage_references: List[str] = field(default_factory=list)  # 存儲引用
    rag_response: str = field(default=None)  # RAG 回應
    final_summary: str = field(default=None)  # 最終摘要

@dataclass(kw_only=True)
class FlexibleResearchStateInput:
    """輸入狀態類。"""
    user_input: str = field(default=None)

@dataclass(kw_only=True)
class FlexibleResearchStateOutput:
    """輸出狀態類。"""
    rag_response: str = field(default=None)
    final_summary: str = field(default=None)
    storage_references: List[str] = field(default_factory=list)
```

#### 2. URL 檢測和路由函數

方案二需要路由函數來判斷輸入類型和流程方向：

```python
import re
from typing import Literal

def is_url(text: str) -> bool:
    """檢查文本是否為 URL。"""
    url_pattern = re.compile(
        r'^(?:http|ftp)s?://'  # http://, https://, ftp://, ftps://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # 域名
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
        r'(?::\d+)?'  # 可選端口
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return bool(url_pattern.match(text))

def route_input_type(state: FlexibleResearchState) -> Literal["process_urls", "process_query"]:
    """判斷輸入是 URL 還是查詢。"""
    # 更新輸入類型
    if is_url(state.user_input):
        state.input_type = "url"
        state.urls = [state.user_input]
        return "process_urls"
    else:
        state.input_type = "query"
        return "process_query"

def route_completion(state: FlexibleResearchState) -> Literal["end_session", "return_to_start"]:
    """判斷是結束會話還是返回起始節點。"""
    if state.input_type == "url":
        return "return_to_start"  # URL 處理完後返回起始節點
    else:
        return "end_session"  # 查詢處理完後結束會話
```

#### 3. RAG 處理節點

方案二需要一個處理查詢的 RAG 節點：

```python
from qdrant_client import QdrantClient

def process_query_with_rag(state: FlexibleResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """使用 RAG 處理查詢節點。"""
    # 獲取配置
    configurable = JinaConfiguration.from_runnable_config(config)
    
    # 連接 Qdrant
    client = QdrantClient(
        url=configurable.qdrant_url,
        api_key=configurable.qdrant_api_key
    )
    
    # 搜索相似向量
    search_results = client.search(
        collection_name=configurable.qdrant_collection,
        query_vector=get_query_embedding(state.user_input, configurable.jina_api_key),
        limit=5
    )
    
    # 提取相關內容
    contexts = []
    for result in search_results:
        if result.payload and "content" in result.payload:
            contexts.append(result.payload["content"])
    
    # 使用 LLM 生成回應
    response = generate_rag_response(state.user_input, contexts, configurable.jina_api_key)
    
    # 更新狀態
    return {"rag_response": response}

def get_query_embedding(query: str, api_key: str) -> List[float]:
    """獲取查詢的向量嵌入。"""
    # 使用 Jina Embeddings API 獲取查詢向量
    # 具體實現...
    
def generate_rag_response(query: str, contexts: List[str], api_key: str) -> str:
    """生成 RAG 回應。"""
    # 使用 Jina DeepSearch API 結合上下文生成回應
    # 具體實現...
```

#### 4. 構建方案二的 LangGraph

最後，構建方案二的 LangGraph：

```python
def build_flexible_graph():
    """構建方案二的 LangGraph。"""
    
    # 創建圖形構建器
    builder = StateGraph(
        FlexibleResearchState, 
        input=FlexibleResearchStateInput, 
        output=FlexibleResearchStateOutput, 
        config_schema=JinaConfiguration
    )
    
    # 添加節點
    builder.add_node("route_input_type", route_input_type)
    builder.add_node("process_urls_with_reader", process_urls_with_reader)
    builder.add_node("create_embeddings", create_embeddings)
    builder.add_node("store_embeddings", store_embeddings)
    builder.add_node("process_query_with_rag", process_query_with_rag)
    builder.add_node("route_completion", route_completion)
    builder.add_node("create_final_summary", create_final_summary)
    
    # 添加 URL 處理流程的邊
    builder.add_edge(START, "route_input_type")
    builder.add_conditional_edges(
        "route_input_type",
        route_input_type,
        {
            "process_urls": "process_urls_with_reader",
            "process_query": "process_query_with_rag"
        }
    )
    
    # URL 處理路徑
    builder.add_edge("process_urls_with_reader", "create_embeddings")
    builder.add_edge("create_embeddings", "store_embeddings")
    builder.add_edge("store_embeddings", "create_final_summary")
    builder.add_edge("create_final_summary", "route_completion")
    
    # 查詢處理路徑
    builder.add_edge("process_query_with_rag", "route_completion")
    
    # 完成路由
    builder.add_conditional_edges(
        "route_completion",
        route_completion,
        {
            "return_to_start": START,
            "end_session": END
        }
    )
    
    # 編譯圖形
    return builder.compile()

# 創建最終圖形
flexible_graph = build_flexible_graph()
```

### 使用方案二

使用方案二處理 URL 和查詢的示例代碼：

```python
# 設置環境
import os
os.environ["JINA_API_KEY"] = "your_jina_api_key"

# 處理 URL
url_result = flexible_graph.invoke(
    {"user_input": "https://arxiv.org/abs/2303.08774"},
    config={"configurable": {}}
)

# 處理查詢
query_result = flexible_graph.invoke(
    {"user_input": "量子計算如何用於藥物發現？"},
    config={"configurable": {}}
)

# 輸出結果
print("URL 處理摘要：")
print(url_result.final_summary)

print("\nRAG 回應：")
print(query_result.rag_response)
```

## 方案三：Agentic RAG 智能助手

方案三將 Agentic 技術與 RAG 相結合，創建一個更智能、更主動的研究助手。這種方案使系統不僅能被動回應用戶請求，還能主動思考、規劃和執行研究任務。

### 流程圖

<!-- 在此處插入方案三的 LangGraph 流程圖，需另外生成 "方案三 LangGraph 流程圖" -->

### 核心概念

Agentic RAG 將傳統的 RAG (檢索增強生成) 系統與智能代理 (Agent) 結合，具有以下關鍵特點：

1. **主動規劃**：根據用戶需求自動規劃研究步驟
2. **工具使用**：能夠選擇和使用不同工具完成任務
3. **反思與調整**：評估執行結果並調整後續計劃
4. **多輪互動**：支持複雜的多輪對話和任務執行

### 核心節點解析

方案三包含 9 個主要節點，形成一個智能循環系統：

1. **input_analyzer**：分析用戶輸入，確定意圖和需求
2. **task_planner**：規劃研究任務和步驟
3. **tool_selector**：選擇適當的工具執行任務
4. **url_processor**：處理 URL 內容
5. **query_executor**：執行搜索查詢
6. **research_synthesizer**：合成研究發現
7. **vector_store_manager**：管理向量存儲
8. **reflection_engine**：反思執行結果並決定下一步
9. **response_generator**：生成最終回應

### 代碼實現

讓我們逐步實現方案三的各個組件：

#### 1. 增強的狀態類

首先，我們需要一個更複雜的狀態類來支持智能代理的規劃和執行：

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple, Union

class TaskStatus(Enum):
    """任務狀態枚舉。"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class ToolType(Enum):
    """工具類型枚舉。"""
    URL_PROCESSOR = "url_processor"
    DEEP_SEARCH = "deep_search"
    READER = "reader"
    EMBEDDING = "embedding"
    RAG_QUERY = "rag_query"
    SUMMARIZER = "summarizer"

@dataclass
class Task:
    """任務類。"""
    id: str
    description: str
    tool: ToolType
    status: TaskStatus = TaskStatus.PENDING
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)

@dataclass(kw_only=True)
class AgentResearchState:
    """智能代理研究狀態類。"""
    # 用戶輸入和意圖
    user_input: str = field(default=None)
    user_intent: str = field(default=None)
    
    # 研究計劃和任務
    research_plan: List[str] = field(default_factory=list)
    tasks: Dict[str, Task] = field(default_factory=dict)
    current_task_id: Optional[str] = field(default=None)
    completed_task_ids: Set[str] = field(default_factory=set)
    
    # 研究數據
    urls: List[str] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)
    url_contents: Dict[str, str] = field(default_factory=dict)
    research_findings: List[Dict[str, Any]] = field(default_factory=list)
    
    # 向量存儲
    vector_embeddings: List[Dict[str, Any]] = field(default_factory=list)
    storage_references: List[str] = field(default_factory=list)
    
    # 反思和回應
    reflections: List[str] = field(default_factory=list)
    response: str = field(default=None)
    
    # 循環控制
    iteration_count: int = field(default=0)
    max_iterations: int = field(default=5)
    is_complete: bool = field(default=False)

@dataclass(kw_only=True)
class AgentResearchStateInput:
    """輸入狀態類。"""
    user_input: str = field(default=None)

@dataclass(kw_only=True)
class AgentResearchStateOutput:
    """輸出狀態類。"""
    response: str = field(default=None)
    research_findings: List[Dict[str, Any]] = field(default_factory=list)
    storage_references: List[str] = field(default_factory=list)
```

#### 2. 配置類擴展

擴展配置類以支持智能代理的設置：

```python
class AgentConfiguration(BaseModel):
    """智能代理配置類。"""
    
    # 基本配置項繼承自 JinaConfiguration
    
    # 代理特定配置
    max_iterations: int = Field(
        default=5,
        title="最大迭代次數",
        description="代理執行的最大迭代次數"
    )
    
    thinking_depth: Literal["shallow", "normal", "deep"] = Field(
        default="normal",
        title="思考深度",
        description="代理的思考深度和複雜度"
    )
    
    autonomous_mode: bool = Field(
        default=True,
        title="自主模式",
        description="是否允許代理自主執行而不需用戶確認"
    )
    
    allowed_tools: List[str] = Field(
        default=["url_processor", "deep_search", "reader", "embedding", "rag_query", "summarizer"],
        title="允許的工具",
        description="代理可以使用的工具列表"
    )
```

#### 3. 智能代理核心節點實現

現在，讓我們實現智能代理的各個核心節點：

```python
def input_analyzer(state: AgentResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """分析用戶輸入，識別意圖和需求。"""
    # 獲取配置
    configurable = AgentConfiguration.from_runnable_config(config)
    
    # 重置狀態
    state.iteration_count = 0
    state.is_complete = False
    state.research_plan = []
    state.tasks = {}
    state.completed_task_ids = set()
    
    # 使用 Jina AI 分析用戶意圖
    intent_analysis = analyze_user_intent(
        state.user_input, 
        configurable.jina_api_key
    )
    
    # 提取 URL（如果存在）
    urls = extract_urls_from_text(state.user_input)
    
    return {
        "user_intent": intent_analysis["intent"],
        "urls": urls,
        "iteration_count": 0,
        "is_complete": False
    }

def task_planner(state: AgentResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """規劃研究任務和步驟。"""
    # 獲取配置
    configurable = AgentConfiguration.from_runnable_config(config)
    
    # 根據用戶意圖和輸入生成研究計劃
    planning_result = generate_research_plan(
        user_input=state.user_input,
        user_intent=state.user_intent,
        urls=state.urls,
        api_key=configurable.jina_api_key,
        thinking_depth=configurable.thinking_depth
    )
    
    # 創建任務
    tasks = {}
    for i, step in enumerate(planning_result["steps"]):
        task_id = f"task_{i}"
        tasks[task_id] = Task(
            id=task_id,
            description=step["description"],
            tool=ToolType(step["tool"]),
            input_data=step.get("input_data", {})
        )
    
    # 設置任務依賴關係
    for i in range(1, len(tasks)):
        task_id = f"task_{i}"
        tasks[task_id].dependencies.append(f"task_{i-1}")
    
    # 確定第一個任務
    current_task_id = "task_0" if tasks else None
    
    return {
        "research_plan": planning_result["plan"],
        "tasks": tasks,
        "current_task_id": current_task_id
    }

def tool_selector(state: AgentResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """選擇合適的工具執行當前任務。"""
    # 檢查是否有當前任務
    if not state.current_task_id or state.current_task_id not in state.tasks:
        return {
            "is_complete": True,
            "response": "已完成所有研究任務。"
        }
    
    # 獲取當前任務
    current_task = state.tasks[state.current_task_id]
    
    # 檢查依賴任務是否已完成
    for dep_id in current_task.dependencies:
        if dep_id not in state.completed_task_ids:
            # 依賴任務未完成，等待
            return {}
    
    # 更新任務狀態
    current_task.status = TaskStatus.IN_PROGRESS
    
    # 根據工具類型返回下一個節點的名稱
    return {
        "current_task": current_task
    }

def task_executor(state: AgentResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """執行當前選定的任務。"""
    # 獲取當前任務
    if not hasattr(state, 'current_task') or not state.current_task:
        return {}
    
    current_task = state.current_task
    tool = current_task.tool
    
    # 獲取配置
    configurable = AgentConfiguration.from_runnable_config(config)
    
    # 根據工具類型執行任務
    if tool == ToolType.URL_PROCESSOR:
        # 處理 URL
        result = process_urls(
            urls=state.urls,
            api_key=configurable.jina_api_key
        )
        current_task.output_data = result
        state.url_contents = result["url_contents"]
        state.research_findings.extend(result["findings"])
        
    elif tool == ToolType.DEEP_SEARCH:
        # 執行深度搜索
        query = current_task.input_data.get("query", state.user_input)
        result = jina_deep_research(
            query=query,
            api_key=configurable.jina_api_key,
            reasoning_effort=configurable.reasoning_effort
        )
        current_task.output_data = result
        state.search_queries.append(query)
        state.research_findings.append({
            "type": "deep_search",
            "query": query,
            "content": result["content"]
        })
        # 如果返回了 URL，添加到列表中
        if result.get("urls"):
            state.urls.extend([url for url in result["urls"] if url not in state.urls])
    
    elif tool == ToolType.EMBEDDING:
        # 創建向量嵌入
        texts = []
        for finding in state.research_findings:
            if "content" in finding:
                texts.append(finding["content"])
        
        for url, content in state.url_contents.items():
            texts.append(content)
        
        if texts:
            result = jina_embedding(
                texts=texts,
                api_key=configurable.jina_api_key,
                model=configurable.embedding_model
            )
            current_task.output_data = result
            state.vector_embeddings = result.get("embeddings", [])
    
    # 標記任務為已完成
    current_task.status = TaskStatus.COMPLETED
    state.completed_task_ids.add(current_task.id)
    
    # 找出下一個可執行的任務
    next_task_id = None
    for task_id, task in state.tasks.items():
        if (task.status == TaskStatus.PENDING and 
            all(dep_id in state.completed_task_ids for dep_id in task.dependencies)):
            next_task_id = task_id
            break
    
    # 更新迭代計數
    state.iteration_count += 1
    
    return {
        "current_task_id": next_task_id,
        "iteration_count": state.iteration_count
    }

def reflection_engine(state: AgentResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """反思執行結果並決定下一步。"""
    # 獲取配置
    configurable = AgentConfiguration.from_runnable_config(config)
    
    # 檢查是否達到最大迭代次數
    if state.iteration_count >= configurable.max_iterations:
        return {
            "is_complete": True,
            "reflections": ["已達到最大迭代次數，完成研究。"]
        }
    
    # 檢查是否所有任務已完成
    all_completed = all(
        task.status == TaskStatus.COMPLETED for task in state.tasks.values()
    )
    
    if all_completed:
        # 生成反思
        reflection = reflect_on_research(
            user_input=state.user_input,
            research_findings=state.research_findings,
            api_key=configurable.jina_api_key
        )
        
        # 檢查是否需要進一步研究
        if reflection["needs_further_research"] and state.iteration_count < configurable.max_iterations:
            # 創建新的研究任務
            new_tasks = {}
            task_count = len(state.tasks)
            
            for i, new_step in enumerate(reflection["next_steps"]):
                task_id = f"task_{task_count + i}"
                new_tasks[task_id] = Task(
                    id=task_id,
                    description=new_step["description"],
                    tool=ToolType(new_step["tool"]),
                    input_data=new_step.get("input_data", {})
                )
            
            # 更新狀態
            state.tasks.update(new_tasks)
            state.current_task_id = list(new_tasks.keys())[0] if new_tasks else None
            state.reflections.append(reflection["reflection"])
            
            return {
                "tasks": state.tasks,
                "current_task_id": state.current_task_id,
                "reflections": state.reflections
            }
        else:
            # 完成研究
            return {
                "is_complete": True,
                "reflections": state.reflections + [reflection["reflection"]]
            }
    
    # 研究仍在進行中
    return {}

def response_generator(state: AgentResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """生成最終回應。"""
    # 獲取配置
    configurable = AgentConfiguration.from_runnable_config(config)
    
    # 生成綜合回應
    response = generate_comprehensive_response(
        user_input=state.user_input,
        research_findings=state.research_findings,
        reflections=state.reflections,
        api_key=configurable.jina_api_key
    )
    
    # 如果有向量嵌入，存儲到 Qdrant
    if state.vector_embeddings:
        # 準備存儲的元數據
        metadata = []
        for finding in state.research_findings:
            metadata.append({
                "type": finding.get("type", "unknown"),
                "source": finding.get("source", "unknown"),
                "query": finding.get("query", ""),
                "timestamp": datetime.now().isoformat()
            })
        
        # 存儲向量
        storage_result = store_in_qdrant(
            embeddings=state.vector_embeddings,
            metadata=metadata,
            qdrant_url=configurable.qdrant_url,
            collection_name=configurable.qdrant_collection,
            api_key=configurable.qdrant_api_key
        )
        
        # 更新存儲引用
        if storage_result["success"]:
            state.storage_references = [
                f"{configurable.qdrant_collection}:{i}" 
                for i in range(len(state.vector_embeddings))
            ]
    
    return {
        "response": response,
        "storage_references": state.storage_references
    }
```

#### 4. 輔助函數實現

實現智能代理所需的關鍵輔助函數：

```python
def analyze_user_intent(user_input: str, api_key: str) -> Dict[str, Any]:
    """分析用戶輸入的意圖。"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "jina-deepsearch-v1",
        "messages": [
            {
                "role": "system",
                "content": """你是一個專業的研究助手，負責分析用戶的研究需求。
                請分析用戶輸入並確定其研究意圖。返回以下內容：
                1. 主要意圖（查詢信息、處理URL、綜合研究等）
                2. 研究領域（如果適用）
                3. 關鍵問題或概念
                4. 搜索關鍵詞建議"""
            },
            {
                "role": "user",
                "content": f"請分析以下輸入的研究意圖:\n\n{user_input}"
            }
        ],
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = httpx.post(
            "https://deepsearch.jina.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30.0
        )
        response.raise_for_status()
        result = response.json()
        
        # 解析 JSON 回應
        intent_content = json.loads(result["choices"][0]["message"]["content"])
        
        return {
            "intent": intent_content.get("intent", "query"),
            "domain": intent_content.get("domain", "general"),
            "key_concepts": intent_content.get("key_concepts", []),
            "search_keywords": intent_content.get("search_keywords", [])
        }
    except Exception as e:
        print(f"分析用戶意圖時出錯: {str(e)}")
        return {"intent": "query"}  # 默認意圖

def generate_research_plan(
    user_input: str, 
    user_intent: str, 
    urls: List[str], 
    api_key: str,
    thinking_depth: str = "normal"
) -> Dict[str, Any]:
    """根據用戶輸入和意圖生成研究計劃。"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # 構建提示
    prompt = f"""作為一個專業研究助手，請為以下用戶需求制定研究計劃：

用戶輸入: {user_input}
識別的意圖: {user_intent}
提供的URL: {urls if urls else "無"}
思考深度: {thinking_depth}

請返回以下內容（JSON格式）：
1. 研究計劃的高層次描述
2. 詳細步驟，每個步驟包括：
   - 描述（具體任務）
   - 所需工具（url_processor, deep_search, reader, embedding, rag_query, summarizer）
   - 輸入數據（如果需要）

確保計劃是全面且系統化的，能夠解決用戶的研究需求。"""
    
    payload = {
        "model": "jina-deepsearch-v1",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = httpx.post(
            "https://deepsearch.jina.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30.0
        )
        response.raise_for_status()
        result = response.json()
        
        # 解析 JSON 回應
        plan_content = json.loads(result["choices"][0]["message"]["content"])
        
        return {
            "plan": plan_content.get("plan", ["研究計劃生成失敗"]),
            "steps": plan_content.get("steps", [])
        }
    except Exception as e:
        print(f"生成研究計劃時出錯: {str(e)}")
        return {
            "plan": ["研究計劃生成失敗"],
            "steps": []
        }

def reflect_on_research(
    user_input: str,
    research_findings: List[Dict[str, Any]],
    api_key: str
) -> Dict[str, Any]:
    """反思研究結果，確定是否需要進一步研究。"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # 準備研究發現摘要
    findings_summary = "\n\n".join([
        f"研究發現 {i+1} ({finding.get('type', 'unknown')}):\n{finding.get('content', '')[:500]}..."
        for i, finding in enumerate(research_findings)
    ])
    
    prompt = f"""作為專業研究助手，請反思目前的研究結果：

用戶需求: {user_input}

研究發現摘要:
{findings_summary}

請評估：
1. 現有研究是否充分解答了用戶需求
2. 是否存在知識缺口或未探索的方向
3. 是否需要進一步研究，如果需要，提出下一步研究步驟

請以JSON格式回答，包含以下字段：
- reflection: 反思總結
- needs_further_research: 布爾值
- next_steps: 如果需要進一步研究，列出下一步步驟（包括描述、工具和輸入數據）"""
    
    payload = {
        "model": "jina-deepsearch-v1",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = httpx.post(
            "https://deepsearch.jina.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30.0
        )
        response.raise_for_status()
        result = response.json()
        
        # 解析 JSON 回應
        reflection_content = json.loads(result["choices"][0]["message"]["content"])
        
        return {
            "reflection": reflection_content.get("reflection", "無法生成反思"),
            "needs_further_research": reflection_content.get("needs_further_research", False),
            "next_steps": reflection_content.get("next_steps", [])
        }
    except Exception as e:
        print(f"反思研究結果時出錯: {str(e)}")
        return {
            "reflection": "反思生成失敗",
            "needs_further_research": False,
            "next_steps": []
        }

def generate_comprehensive_response(
    user_input: str,
    research_findings: List[Dict[str, Any]],
    reflections: List[str],
    api_key: str
) -> str:
    """生成全面的回應。"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # 準備研究發現
    findings_text = "\n\n".join([
        f"研究發現 {i+1} ({finding.get('type', 'unknown')}):\n{finding.get('content', '')[:1000]}..."
        for i, finding in enumerate(research_findings)
    ])
    
    # 準備反思
    reflections_text = "\n".join([f"- {reflection}" for reflection in reflections])
    
    prompt = f"""作為專業研究助手，請根據以下研究結果為用戶提供全面回應：

用戶需求: {user_input}

研究發現:
{findings_text}

研究反思:
{reflections_text}

請提供一個結構化、全面且易於理解的回應，包括：
1. 對用戶問題的直接回答

## 部署與擴展

### 生產環境部署

將這些系統部署到生產環境需要考慮以下幾點：

1. **環境配置**：使用環境變量管理敏感信息
2. **錯誤處理**：實現全面的錯誤處理和重試機制
3. **監控與日誌**：設置監控和日誌系統
4. **擴展性**：實現水平擴展以處理高負載

示例 Docker 配置：

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安裝依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製代碼
COPY . .

# 設置環境變量
ENV JINA_API_KEY="your_jina_api_key"
ENV QDRANT_URL="http://qdrant:6333"

# 運行服務
CMD ["python", "main.py"]
```

### 擴展功能

我們的系統可以通過以下方式擴展：

1. **多語言支持**：使用多語言模型處理不同語言的內容
2. **自定義處理器**：允許用戶添加自定義內容處理器
3. **高級過濾**：根據文檔類型、日期和來源過濾內容
4. **協作功能**：支持多用戶協作研究

## 結論與未來展望

在本文中，我們詳細介紹了如何使用 LangGraph 和 Jina AI 服務構建智能研究助手系統的兩種方案：一種專注於自動化研究流程，另一種提供靈活的輸入方式並集成 RAG 功能。

這兩種方案各有優勢：方案一適合深度研究特定主題，方案二則更適合交互式問答和知識庫建設。開發者可以根據具體需求選擇合適的方案，或者結合兩種方案的優勢創建混合解決方案。

未來，我們計劃進一步擴展系統功能，包括多模態內容處理、知識圖譜集成和更高級的 Agent 能力。隨著 AI 技術的快速發展，這些系統將變得更加智能和高效，為研究工作提供更強大的支持。

希望本文對你構建自己的智能研究助手有所幫助！如果你有任何問題或建議，請在評論區留言。

---

## 參考資源

- [LangGraph 官方文檔](https://github.com/langchain-ai/langgraph)
- [Jina AI API 文檔](https://jina.ai/api)
- [Qdrant 文檔](https://qdrant.tech/documentation/)
- [向量搜索簡介](https://www.pinecone.io/learn/vector-search-basics/)