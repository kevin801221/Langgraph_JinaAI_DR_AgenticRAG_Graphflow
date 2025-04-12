from typing import Dict, Any, List
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph

from method_one.state import JinaResearchState, JinaResearchStateInput, JinaResearchStateOutput
from method_one.configuration import JinaConfiguration
from method_one.utils import (
    perform_deep_research,
    extract_urls_from_report,
    process_url_with_reader,
    create_embeddings,
    store_embeddings_in_qdrant,
    create_summary_from_research
)

def perform_deep_research_node(state: JinaResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """執行深度研究的節點。"""
    # 獲取配置
    configurable = JinaConfiguration.from_runnable_config(config)
    
    # 執行深度研究
    result = perform_deep_research(
        research_topic=state.research_topic,
        api_key=configurable.jina_api_key,
        reasoning_effort=configurable.reasoning_effort,
        budget_tokens=configurable.budget_tokens,
        max_attempts=configurable.max_attempts,
        max_returned_urls=configurable.max_returned_urls
    )
    
    # 提取研究報告
    if "choices" in result and result["choices"]:
        deep_research_report = result["choices"][0]["message"]["content"]
    else:
        deep_research_report = f"Research failed: {result.get('error', 'Unknown error')}"
    
    return {"deep_research_report": deep_research_report}

def extract_additional_urls_node(state: JinaResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """從研究報告中提取額外 URL 的節點。"""
    # 從報告中提取 URL
    extracted_urls = extract_urls_from_report(state.deep_research_report)
    
    return {"extracted_urls": extracted_urls}

def process_urls_with_reader_node(state: JinaResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """處理 URL 內容的節點。"""
    # 獲取配置
    configurable = JinaConfiguration.from_runnable_config(config)
    
    # 處理每個 URL
    url_contents = {}
    processed_content = []
    
    for url in state.extracted_urls:
        result = process_url_with_reader(
            url=url,
            api_key=configurable.jina_api_key,
            max_tokens=configurable.max_tokens_per_url
        )
        
        if "content" in result:
            url_contents[url] = result["content"]
            processed_content.append({
                "url": url,
                "content": result["content"],
                "title": result.get("title", ""),
                "source": "jina_reader"
            })
    
    return {
        "url_contents": url_contents,
        "processed_content": processed_content
    }

def create_embeddings_node(state: JinaResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """創建向量嵌入的節點。"""
    # 獲取配置
    configurable = JinaConfiguration.from_runnable_config(config)
    
    # 創建嵌入
    result = create_embeddings(
        texts=state.processed_content,
        api_key=configurable.jina_api_key,
        model=configurable.embedding_model.value
    )
    
    return {"vector_embeddings": result.get("embeddings", [])}

def store_embeddings_node(state: JinaResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """存儲向量嵌入的節點。"""
    # 獲取配置
    configurable = JinaConfiguration.from_runnable_config(config)
    
    # 存儲嵌入
    storage_references = store_embeddings_in_qdrant(
        embeddings=state.vector_embeddings,
        collection_name=configurable.qdrant_collection_name,
        qdrant_url=configurable.qdrant_url
    )
    
    return {"storage_references": storage_references}

def create_final_summary_node(state: JinaResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """創建最終摘要的節點。"""
    # 獲取配置
    configurable = JinaConfiguration.from_runnable_config(config)
    
    # 創建摘要
    final_summary = create_summary_from_research(
        research_topic=state.research_topic,
        deep_research_report=state.deep_research_report,
        url_contents=state.url_contents,
        api_key=configurable.jina_api_key
    )
    
    return {"final_summary": final_summary}

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
    builder.add_node("perform_deep_research", perform_deep_research_node)
    builder.add_node("extract_additional_urls", extract_additional_urls_node)
    builder.add_node("process_urls_with_reader", process_urls_with_reader_node)
    builder.add_node("create_embeddings", create_embeddings_node)
    builder.add_node("store_embeddings", store_embeddings_node)
    builder.add_node("create_final_summary", create_final_summary_node)
    
    # 添加邊緣連接
    builder.add_edge(START, "perform_deep_research")
    builder.add_edge("perform_deep_research", "extract_additional_urls")
    builder.add_edge("extract_additional_urls", "process_urls_with_reader")
    builder.add_edge("process_urls_with_reader", "create_embeddings")
    builder.add_edge("create_embeddings", "store_embeddings")
    builder.add_edge("store_embeddings", "create_final_summary")
    builder.add_edge("create_final_summary", END)
    
    # 編譯圖形
    return builder.compile()

# 創建圖形實例
graph = build_graph()
