from typing import Dict, Any, List, Literal
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph

from method_two.state import FlexibleResearchState, FlexibleResearchStateInput, FlexibleResearchStateOutput
from method_two.configuration import FlexibleConfiguration
from method_two.utils import (
    process_url_with_reader,
    create_embeddings,
    store_embeddings_in_qdrant,
    process_query_with_rag
)

def route_input_type(state: FlexibleResearchState) -> Literal["process_urls_with_reader", "process_query_with_rag"]:
    """根據輸入類型路由到不同的處理節點。"""
    if state.input_type == "url":
        return "process_urls_with_reader"
    else:  # query
        return "process_query_with_rag"

def process_urls_with_reader_node(state: FlexibleResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """處理 URL 內容的節點。"""
    # 獲取配置
    configurable = FlexibleConfiguration.from_runnable_config(config)
    
    # 處理 URL
    if not state.url_input:
        return {
            "url_content": "No URL provided",
            "processed_content": []
        }
    
    result = process_url_with_reader(
        url=state.url_input,
        api_key=configurable.jina_api_key,
        max_tokens=configurable.max_tokens_per_url
    )
    
    processed_content = []
    if "content" in result and result["content"]:
        processed_content = [{
            "url": state.url_input,
            "content": result["content"],
            "title": result.get("title", ""),
            "source": "jina_reader"
        }]
    
    return {
        "url_content": result.get("content", "Failed to process URL"),
        "processed_content": processed_content
    }

def create_embeddings_node(state: FlexibleResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """創建向量嵌入的節點。"""
    # 獲取配置
    configurable = FlexibleConfiguration.from_runnable_config(config)
    
    # 創建嵌入
    result = create_embeddings(
        texts=state.processed_content,
        api_key=configurable.jina_api_key,
        model=configurable.embedding_model.value
    )
    
    return {"vector_embeddings": result.get("embeddings", [])}

def store_embeddings_node(state: FlexibleResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """存儲向量嵌入的節點。"""
    # 獲取配置
    configurable = FlexibleConfiguration.from_runnable_config(config)
    
    # 存儲嵌入
    storage_references = store_embeddings_in_qdrant(
        embeddings=state.vector_embeddings,
        collection_name=configurable.qdrant_collection_name,
        qdrant_url=configurable.qdrant_url
    )
    
    return {"storage_references": storage_references}

def process_query_with_rag_node(state: FlexibleResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """使用 RAG 處理查詢的節點。"""
    # 獲取配置
    configurable = FlexibleConfiguration.from_runnable_config(config)
    
    # 處理查詢
    if not state.query_input:
        return {
            "rag_results": "No query provided",
            "retrieved_documents": []
        }
    
    result = process_query_with_rag(
        query=state.query_input,
        api_key=configurable.jina_api_key,
        collection_name=configurable.qdrant_collection_name,
        qdrant_url=configurable.qdrant_url,
        top_k=configurable.rag_top_k,
        embedding_model=configurable.embedding_model.value
    )
    
    return {
        "rag_results": result.get("rag_results", "Failed to process query"),
        "retrieved_documents": result.get("retrieved_documents", [])
    }

def route_completion(state: FlexibleResearchState) -> Literal["end_session", "return_to_start"]:
    """判斷是結束會話還是返回起始節點。"""
    if state.input_type == "url":
        return "return_to_start"  # URL 處理完後返回起始節點
    else:
        return "end_session"  # 查詢處理完後結束會話

def build_graph():
    """構建方案二的 LangGraph。"""
    
    # 創建圖形構建器
    builder = StateGraph(
        FlexibleResearchState,
        input=FlexibleResearchStateInput,
        output=FlexibleResearchStateOutput,
        config_schema=FlexibleConfiguration
    )
    
    # 添加節點
    builder.add_node("route_input_type", route_input_type)
    builder.add_node("process_urls_with_reader", process_urls_with_reader_node)
    builder.add_node("create_embeddings", create_embeddings_node)
    builder.add_node("store_embeddings", store_embeddings_node)
    builder.add_node("process_query_with_rag", process_query_with_rag_node)
    builder.add_node("route_completion", route_completion)
    
    # 添加邊緣連接
    builder.add_edge(START, "route_input_type")
    
    # URL 處理路徑
    builder.add_conditional_edges(
        "route_input_type",
        lambda x: x.result,
        {
            "process_urls_with_reader": "process_urls_with_reader",
            "process_query_with_rag": "process_query_with_rag"
        }
    )
    
    builder.add_edge("process_urls_with_reader", "create_embeddings")
    builder.add_edge("create_embeddings", "store_embeddings")
    builder.add_edge("store_embeddings", "route_completion")
    
    # 查詢處理路徑
    builder.add_edge("process_query_with_rag", "route_completion")
    
    # 完成路由
    builder.add_conditional_edges(
        "route_completion",
        lambda x: x.result,
        {
            "end_session": END,
            "return_to_start": START
        }
    )
    
    # 編譯圖形
    return builder.compile()

# 創建圖形實例
graph = build_graph()
