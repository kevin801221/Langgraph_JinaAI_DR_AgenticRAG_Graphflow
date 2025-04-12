#!/bin/bash

# 如果提供了參數，則使用該參數作為查詢內容
if [ $# -eq 1 ]; then
  QUERY="$1"
else
  # 否則提示用戶輸入查詢內容
  read -p "請輸入搜索主題: " QUERY
fi

# 設置API密鑰（可以根據需要更改）
API_KEY=""

# 執行API調用
curl https://deepsearch.jina.ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d @- <<EOFEOF
  {
    "model": "jina-deepsearch-v1",
    "messages": [
        {
            "role": "user",
            "content": "請提供關於 '$QUERY' 的全面且深入的研究報告，包括：

1. 該主題領域最新且最具影響力的學術論文（優先提供有DOI或arXiv ID的論文），包括：
   - 完整引用信息（標題、作者、期刊/會議、發表日期）
   - 核心發現和主要貢獻
   - 引用數據和影響力評估（如有）
   - 原始論文的直接鏈接

2. 該領域的主要研究趨勢、突破和未來發展方向

3. 該領域最具影響力的研究者、團隊和機構

4. 與其他相關領域的交叉點和整合可能性

5. 實際應用和產業影響

請優先引用高品質的學術資源，特別是arXiv、知名學術期刊和頂級會議發表的論文。每個重要論點都需要有來源引用支持。如有任何不確定的信息，請明確標註。"
        }
    ],
    "stream": false,
    "reasoning_effort": "high",
    "budget_tokens": 30,
    "max_attempts": 7,
    "no_direct_answer": false,
    "max_returned_urls": "20",
    "response_format": {
        "type": "text"
    },
    "boost_hostnames": [
        "arxiv.org",
        "scholar.google.com",
        "ieee.org",
        "acm.org",
        "nature.com",
        "science.org",
        "sciencedirect.com",
        "springer.com",
        "cell.com",
        "mdpi.com",
        "pnas.org",
        "acs.org",
        "researchgate.net",
        "semanticscholar.org"
    ]
  }
EOFEOF