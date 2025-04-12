import json
import httpx
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models

def process_url_with_reader(url: str, api_key: str, max_tokens: int = 10000) -> Dict[str, Any]:
    """使用 Jina Reader 處理 URL 內容。"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
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
        return response.json()
    except Exception as e:
        print(f"URL processing failed for {url}: {e}")
        return {"url": url, "error": str(e), "content": ""}

def create_embeddings(
    texts: List[Dict[str, Any]],
    api_key: str,
    model: str = "jina-embeddings-v3"
) -> Dict[str, Any]:
    """創建文本嵌入。"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # 準備輸入文本
    input_texts = []
    for item in texts:
        if "content" in item and item["content"]:
            # 如果有內容，添加到輸入文本
            input_texts.append(item["content"])
    
    if not input_texts:
        return {"error": "No valid content to embed", "embeddings": []}
    
    payload = {
        "model": model,
        "input": input_texts
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
        
        # 將嵌入與原始文本關聯
        embeddings_with_metadata = []
        for i, embedding_data in enumerate(result.get("data", [])):
            if i < len(texts):
                embeddings_with_metadata.append({
                    "embedding": embedding_data.get("embedding", []),
                    "metadata": texts[i]
                })
        
        return {"embeddings": embeddings_with_metadata}
    except Exception as e:
        print(f"Embedding creation failed: {e}")
        return {"error": str(e), "embeddings": []}

def store_embeddings_in_qdrant(
    embeddings: List[Dict[str, Any]],
    collection_name: str,
    qdrant_url: str = "http://localhost:6333"
) -> List[str]:
    """將嵌入存儲到 Qdrant 向量數據庫。"""
    try:
        # 連接到 Qdrant
        client = QdrantClient(url=qdrant_url)
        
        # 檢查集合是否存在，如果不存在則創建
        collections = client.get_collections().collections
        collection_exists = any(collection.name == collection_name for collection in collections)
        
        if not collection_exists:
            # 創建集合
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=len(embeddings[0]["embedding"]) if embeddings else 768,
                    distance=models.Distance.COSINE
                )
            )
        
        # 準備要上傳的點
        points = []
        for i, item in enumerate(embeddings):
            points.append(
                models.PointStruct(
                    id=i,
                    vector=item["embedding"],
                    payload=item["metadata"]
                )
            )
        
        # 上傳嵌入
        if points:
            operation_info = client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            # 返回存儲引用
            return [f"{collection_name}:{i}" for i in range(len(points))]
        else:
            return []
    except Exception as e:
        print(f"Qdrant storage failed: {e}")
        return []

def process_query_with_rag(
    query: str,
    api_key: str,
    collection_name: str,
    qdrant_url: str = "http://localhost:6333",
    top_k: int = 5,
    embedding_model: str = "jina-embeddings-v3"
) -> Dict[str, Any]:
    """使用 RAG 處理查詢。"""
    try:
        # 為查詢創建嵌入
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": embedding_model,
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
        client = QdrantClient(url=qdrant_url)
        
        # 檢查集合是否存在
        collections = client.get_collections().collections
        collection_exists = any(collection.name == collection_name for collection in collections)
        
        if not collection_exists:
            return {
                "error": f"Collection {collection_name} does not exist",
                "retrieved_documents": [],
                "rag_results": f"No documents found for query: {query}"
            }
        
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
                url = payload.get("url", "Unknown source")
                context += f"Document {i+1} (Source: {url}):\n{content}\n\n"
            
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
            
            rag_results = chat_result["choices"][0]["message"]["content"]
        else:
            rag_results = f"No relevant documents found for query: {query}"
        
        return {
            "retrieved_documents": retrieved_documents,
            "rag_results": rag_results
        }
    except Exception as e:
        print(f"RAG processing failed: {e}")
        return {
            "error": str(e),
            "retrieved_documents": [],
            "rag_results": f"Error processing query: {e}"
        }
