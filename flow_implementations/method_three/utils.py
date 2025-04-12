import json
import httpx
from typing import List, Dict, Any, Optional

def analyze_user_intent(user_input: str, api_key: str) -> Dict[str, Any]:
    """分析用戶輸入的意圖。"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    prompt = f"""
    分析以下用戶輸入的意圖和需求：

    用戶輸入：{user_input}

    請識別：
    1. 主要意圖（research、question、url_processing 或 other）
    2. 如果是研究意圖，提取研究主題
    3. 如果是問題意圖，提取具體問題
    4. 如果是 URL 處理意圖，提取 URL
    5. 其他相關細節

    以 JSON 格式返回，包含以下字段：
    - intent: 主要意圖
    - research_topic: 研究主題（如果適用）
    - question: 具體問題（如果適用）
    - url: URL（如果適用）
    - details: 其他相關細節
    """
    
    payload = {
        "model": "jina-deepsearch-v1",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that analyzes user intent and extracts structured information."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = httpx.post(
            "https://api.jina.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        print(f"Intent analysis failed: {e}")
        return {
            "intent": "unknown",
            "error": str(e)
        }

def create_task_plan(intent_data: Dict[str, Any], api_key: str) -> List[Dict[str, Any]]:
    """根據意圖創建任務計劃。"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    intent = intent_data.get("intent", "unknown")
    
    prompt = f"""
    基於以下用戶意圖創建詳細的任務計劃：

    意圖數據：{json.dumps(intent_data, ensure_ascii=False)}

    請創建一個任務計劃，列出完成用戶請求所需的步驟。每個任務應包含：
    - task_id: 任務 ID（數字）
    - description: 任務描述
    - tool: 需要使用的工具（deep_search、url_reader、embedding、rag_query、summarize）
    - parameters: 工具所需的參數

    以 JSON 格式返回任務列表。
    """
    
    payload = {
        "model": "jina-deepsearch-v1",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that creates detailed task plans."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = httpx.post(
            "https://api.jina.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        return json.loads(content).get("tasks", [])
    except Exception as e:
        print(f"Task planning failed: {e}")
        return []

def select_tool_for_task(task: Dict[str, Any], available_tools: List[str]) -> Dict[str, Any]:
    """為任務選擇合適的工具。"""
    tool = task.get("tool", "")
    
    # 檢查工具是否可用
    if tool not in available_tools:
        return {
            "selected_tool": None,
            "error": f"Tool {tool} is not available"
        }
    
    return {
        "selected_tool": tool,
        "tool_parameters": task.get("parameters", {})
    }

def execute_deep_search(parameters: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """執行深度研究。"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    research_topic = parameters.get("research_topic", "")
    reasoning_effort = parameters.get("reasoning_effort", "high")
    
    payload = {
        "model": "jina-deepsearch-v1",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful research assistant."
            },
            {
                "role": "user",
                "content": f"Research this topic thoroughly: {research_topic}"
            }
        ],
        "reasoning_effort": reasoning_effort
    }
    
    try:
        response = httpx.post(
            "https://api.jina.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=300
        )
        response.raise_for_status()
        result = response.json()
        return {
            "success": True,
            "result": result["choices"][0]["message"]["content"],
            "task_type": "deep_search",
            "research_topic": research_topic
        }
    except Exception as e:
        print(f"Deep search failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "task_type": "deep_search",
            "research_topic": research_topic
        }

def execute_url_reader(parameters: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """執行 URL 閱讀。"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    url = parameters.get("url", "")
    max_tokens = parameters.get("max_tokens", 10000)
    
    payload = {
        "url": url,
        "max_tokens": max_tokens
    }
    
    try:
        response = httpx.post(
            "https://api.jina.ai/v1/reader",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return {
            "success": True,
            "result": result.get("content", ""),
            "title": result.get("title", ""),
            "task_type": "url_reader",
            "url": url
        }
    except Exception as e:
        print(f"URL reading failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "task_type": "url_reader",
            "url": url
        }

def execute_embedding(parameters: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """執行嵌入創建。"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    texts = parameters.get("texts", [])
    model = parameters.get("model", "jina-embeddings-v3")
    
    if not texts:
        return {
            "success": False,
            "error": "No texts provided for embedding",
            "task_type": "embedding"
        }
    
    payload = {
        "model": model,
        "input": texts
    }
    
    try:
        response = httpx.post(
            "https://api.jina.ai/v1/embeddings",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        
        embeddings = []
        for i, embedding_data in enumerate(result.get("data", [])):
            if i < len(texts):
                embeddings.append({
                    "text": texts[i],
                    "embedding": embedding_data.get("embedding", [])
                })
        
        return {
            "success": True,
            "embeddings": embeddings,
            "task_type": "embedding"
        }
    except Exception as e:
        print(f"Embedding creation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "task_type": "embedding"
        }

def execute_rag_query(parameters: Dict[str, Any], api_key: str, qdrant_url: str, collection_name: str, top_k: int = 5) -> Dict[str, Any]:
    """執行 RAG 查詢。"""
    query = parameters.get("query", "")
    
    if not query:
        return {
            "success": False,
            "error": "No query provided",
            "task_type": "rag_query"
        }
    
    try:
        # 為查詢創建嵌入
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "jina-embeddings-v3",
            "input": [query]
        }
        
        response = httpx.post(
            "https://api.jina.ai/v1/embeddings",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        
        query_embedding = result["data"][0]["embedding"]
        
        # 連接到 Qdrant
        from qdrant_client import QdrantClient
        
        client = QdrantClient(url=qdrant_url)
        
        # 搜索相似文檔
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        # 提取檢索到的文檔
        retrieved_documents = []
        for scored_point in search_result:
            retrieved_documents.append({
                "score": scored_point.score,
                "payload": scored_point.payload
            })
        
        # 使用 Jina DeepSearch 生成 RAG 結果
        if retrieved_documents:
            # 準備上下文
            context = ""
            for i, doc in enumerate(retrieved_documents):
                payload = doc["payload"]
                content = payload.get("content", "")
                source = payload.get("url", payload.get("source", "Unknown source"))
                context += f"Document {i+1} (Source: {source}):\n{content}\n\n"
            
            # 使用 Jina DeepSearch 生成回答
            payload = {
                "model": "jina-deepsearch-v1",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful research assistant. Use the provided context to answer the user's query. If the context doesn't contain relevant information, say so."
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuery: {query}\n\nPlease provide a comprehensive answer based on the context."
                    }
                ]
            }
            
            response = httpx.post(
                "https://api.jina.ai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            chat_result = response.json()
            
            rag_result = chat_result["choices"][0]["message"]["content"]
        else:
            rag_result = f"No relevant documents found for query: {query}"
        
        return {
            "success": True,
            "result": rag_result,
            "retrieved_documents": retrieved_documents,
            "task_type": "rag_query",
            "query": query
        }
    except Exception as e:
        print(f"RAG query failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "task_type": "rag_query",
            "query": query
        }

def execute_summarize(parameters: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """執行摘要生成。"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    content = parameters.get("content", "")
    
    if not content:
        return {
            "success": False,
            "error": "No content provided for summarization",
            "task_type": "summarize"
        }
    
    payload = {
        "model": "jina-deepsearch-v1",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that creates concise and comprehensive summaries."
            },
            {
                "role": "user",
                "content": f"Please summarize the following content:\n\n{content}"
            }
        ]
    }
    
    try:
        response = httpx.post(
            "https://api.jina.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return {
            "success": True,
            "result": result["choices"][0]["message"]["content"],
            "task_type": "summarize"
        }
    except Exception as e:
        print(f"Summarization failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "task_type": "summarize"
        }

def reflect_on_results(task_results: List[Dict[str, Any]], api_key: str) -> Dict[str, Any]:
    """反思任務結果並決定下一步。"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # 準備任務結果摘要
    results_summary = json.dumps(task_results, ensure_ascii=False, indent=2)
    
    prompt = f"""
    分析以下任務執行結果，並決定是否需要進一步研究：

    任務結果：
    {results_summary}

    請評估：
    1. 任務是否成功完成
    2. 是否獲得了足夠的信息
    3. 是否存在需要進一步研究的問題或領域
    4. 如果需要更多研究，請指出具體方向

    以 JSON 格式返回，包含以下字段：
    - reflection: 對結果的反思和評估
    - needs_more_research: 布爾值，表示是否需要更多研究
    - research_directions: 如果需要更多研究，列出研究方向
    """
    
    payload = {
        "model": "jina-deepsearch-v1",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that analyzes research results and provides reflections."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = httpx.post(
            "https://api.jina.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        print(f"Reflection failed: {e}")
        return {
            "reflection": f"Reflection failed: {e}",
            "needs_more_research": False
        }

def generate_response(task_results: List[Dict[str, Any]], user_input: str, api_key: str) -> str:
    """生成最終響應。"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # 準備任務結果摘要
    results_summary = json.dumps(task_results, ensure_ascii=False, indent=2)
    
    prompt = f"""
    基於以下任務執行結果，為用戶生成全面且信息豐富的響應：

    用戶輸入：
    {user_input}

    任務結果：
    {results_summary}

    請創建一個結構良好的響應，包括：
    1. 對用戶問題的直接回答
    2. 關鍵發現和見解
    3. 相關事實和數據
    4. 如果適用，包括來源引用

    使用 Markdown 格式，包括適當的標題、項目符號和強調。
    """
    
    payload = {
        "model": "jina-deepsearch-v1",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful research assistant that provides comprehensive and well-structured responses."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    try:
        response = httpx.post(
            "https://api.jina.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Response generation failed: {e}")
        return f"Response generation failed: {e}"
