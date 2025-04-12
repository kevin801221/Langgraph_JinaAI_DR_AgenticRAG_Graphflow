import json
import re
import httpx
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

def perform_deep_research(
    research_topic: str,
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
        "reasoning_effort": reasoning_effort,
        "budget_tokens": budget_tokens,
        "max_attempts": max_attempts,
        "max_returned_urls": max_returned_urls
    }
    
    try:
        response = httpx.post(
            "https://api.jina.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=300  # 5分鐘超時
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Deep research failed: {e}")
        return {"error": str(e)}

def extract_urls_from_report(report: str) -> List[str]:
    """從研究報告中提取 URL。"""
    # 基本 URL 模式
    url_pattern = r'https?://[^\s()<>]+(?:\([\w\d]+\)|([^[:punct:]\s]|/))+'
    
    # 找到所有匹配項
    urls = re.findall(url_pattern, report)
    
    # 過濾並清理 URL
    cleaned_urls = []
    for url in urls:
        # 如果 URL 是元組（由於正則表達式捕獲組），取第一個元素
        if isinstance(url, tuple):
            url = url[0]
        
        # 確保 URL 有效
        parsed_url = urlparse(url)
        if parsed_url.scheme and parsed_url.netloc:
            cleaned_urls.append(url)
    
    # 去重
    return list(set(cleaned_urls))

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
        from qdrant_client import QdrantClient
        from qdrant_client.http import models
        
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

def create_summary_from_research(
    research_topic: str,
    deep_research_report: str,
    url_contents: Dict[str, str],
    api_key: str
) -> str:
    """從研究結果創建最終摘要。"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # 準備 URL 內容摘要
    url_summaries = []
    for url, content in url_contents.items():
        if content:
            url_summaries.append(f"URL: {url}\nContent Summary: {content[:500]}...")
    
    url_content_text = "\n\n".join(url_summaries)
    
    payload = {
        "model": "jina-deepsearch-v1",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful research assistant. Create a comprehensive summary of the research findings."
            },
            {
                "role": "user",
                "content": f"Research Topic: {research_topic}\n\nDeep Research Report:\n{deep_research_report}\n\nAdditional URL Content:\n{url_content_text}\n\nPlease create a comprehensive, well-structured summary of all the research findings. Include key insights, facts, and conclusions. Format the summary in Markdown with appropriate headings, bullet points, and sections."
            }
        ]
    }
    
    try:
        response = httpx.post(
            "https://api.jina.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Summary creation failed: {e}")
        return f"Summary creation failed: {e}"
