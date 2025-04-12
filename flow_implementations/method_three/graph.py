from typing import Dict, Any, List, Literal
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph

from method_three.state import AgentResearchState, AgentResearchStateInput, AgentResearchStateOutput
from method_three.configuration import AgentConfiguration, ToolType
from method_three.utils import (
    analyze_user_intent,
    create_task_plan,
    select_tool_for_task,
    execute_deep_search,
    execute_url_reader,
    execute_embedding,
    execute_rag_query,
    execute_summarize,
    reflect_on_results,
    generate_response
)

def input_analyzer(state: AgentResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """分析用戶輸入，識別意圖和需求。"""
    # 獲取配置
    configurable = AgentConfiguration.from_runnable_config(config)
    
    # 分析意圖
    intent_data = analyze_user_intent(
        user_input=state.user_input,
        api_key=configurable.jina_api_key
    )
    
    return {
        "intent": intent_data.get("intent", "unknown"),
        "intent_details": intent_data
    }

def task_planner(state: AgentResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """根據用戶意圖規劃任務。"""
    # 獲取配置
    configurable = AgentConfiguration.from_runnable_config(config)
    
    # 創建任務計劃
    task_plan = create_task_plan(
        intent_data=state.intent_details,
        api_key=configurable.jina_api_key
    )
    
    return {
        "task_plan": task_plan,
        "current_task_index": 0
    }

def tool_selector(state: AgentResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """為當前任務選擇合適的工具。"""
    # 獲取配置
    configurable = AgentConfiguration.from_runnable_config(config)
    
    # 獲取當前任務
    if state.current_task_index >= len(state.task_plan):
        return {
            "selected_tool": None,
            "tool_parameters": {}
        }
    
    current_task = state.task_plan[state.current_task_index]
    
    # 選擇工具
    tool_selection = select_tool_for_task(
        task=current_task,
        available_tools=[tool.value for tool in configurable.available_tools]
    )
    
    return {
        "selected_tool": tool_selection.get("selected_tool"),
        "tool_parameters": tool_selection.get("tool_parameters", {}),
        "current_task": current_task
    }

def task_executor(state: AgentResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """執行當前選定的任務。"""
    # 獲取當前任務
    if not hasattr(state, 'current_task') or not state.current_task:
        return {}
    
    # 獲取配置
    configurable = AgentConfiguration.from_runnable_config(config)
    
    # 根據選定的工具執行任務
    result = {}
    if state.selected_tool == ToolType.DEEP_SEARCH.value:
        result = execute_deep_search(
            parameters=state.tool_parameters,
            api_key=configurable.jina_api_key
        )
    elif state.selected_tool == ToolType.URL_READER.value:
        result = execute_url_reader(
            parameters=state.tool_parameters,
            api_key=configurable.jina_api_key
        )
    elif state.selected_tool == ToolType.EMBEDDING.value:
        result = execute_embedding(
            parameters=state.tool_parameters,
            api_key=configurable.jina_api_key
        )
    elif state.selected_tool == ToolType.RAG_QUERY.value:
        result = execute_rag_query(
            parameters=state.tool_parameters,
            api_key=configurable.jina_api_key,
            qdrant_url=configurable.qdrant_url,
            collection_name=configurable.qdrant_collection_name,
            top_k=configurable.rag_top_k
        )
    elif state.selected_tool == ToolType.SUMMARIZE.value:
        result = execute_summarize(
            parameters=state.tool_parameters,
            api_key=configurable.jina_api_key
        )
    
    # 更新任務索引
    current_task_index = state.current_task_index + 1
    
    return {
        "task_results": [result],
        "current_task_index": current_task_index
    }

def reflection_engine(state: AgentResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """反思執行結果並決定下一步。"""
    # 獲取配置
    configurable = AgentConfiguration.from_runnable_config(config)
    
    # 檢查是否完成所有任務
    if state.current_task_index >= len(state.task_plan):
        # 所有任務已完成，進行反思
        reflection_result = reflect_on_results(
            task_results=state.task_results,
            api_key=configurable.jina_api_key
        )
        
        return {
            "reflection": reflection_result.get("reflection", ""),
            "needs_more_research": reflection_result.get("needs_more_research", False)
        }
    else:
        # 還有任務未完成
        return {
            "needs_more_research": True
        }

def response_generator(state: AgentResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """生成最終響應。"""
    # 獲取配置
    configurable = AgentConfiguration.from_runnable_config(config)
    
    # 生成響應
    final_response = generate_response(
        task_results=state.task_results,
        user_input=state.user_input,
        api_key=configurable.jina_api_key
    )
    
    return {
        "final_response": final_response
    }

def route_reflection(state: AgentResearchState) -> Literal["task_selector", "response_generator"]:
    """根據反思結果決定下一步。"""
    if state.needs_more_research:
        return "tool_selector"
    else:
        return "response_generator"

def build_graph():
    """構建方案三的 LangGraph。"""
    
    # 創建圖形構建器
    builder = StateGraph(
        AgentResearchState,
        input=AgentResearchStateInput,
        output=AgentResearchStateOutput,
        config_schema=AgentConfiguration
    )
    
    # 添加節點
    builder.add_node("input_analyzer", input_analyzer)
    builder.add_node("task_planner", task_planner)
    builder.add_node("tool_selector", tool_selector)
    builder.add_node("task_executor", task_executor)
    builder.add_node("reflection_engine", reflection_engine)
    builder.add_node("response_generator", response_generator)
    
    # 添加邊緣連接
    builder.add_edge(START, "input_analyzer")
    builder.add_edge("input_analyzer", "task_planner")
    builder.add_edge("task_planner", "tool_selector")
    builder.add_edge("tool_selector", "task_executor")
    builder.add_edge("task_executor", "reflection_engine")
    
    # 反思路由
    builder.add_conditional_edges(
        "reflection_engine",
        route_reflection,
        {
            "tool_selector": "tool_selector",
            "response_generator": "response_generator"
        }
    )
    
    builder.add_edge("response_generator", END)
    
    # 編譯圖形
    return builder.compile()

# 創建圖形實例
graph = build_graph()
