curl -X POST 'https://api.jina.ai/v1/segment' \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer " \
  -d @- <<EOFEOF
  {
    "content": "\n  Jina AI: Your Search Foundation, Supercharged! 🚀\n  Ihrer Suchgrundlage, aufgeladen! 🚀\n  您的搜索底座，从此不同！🚀\n  検索ベース,もう二度と同じことはありません！🚀\n",
    "return_tokens": true,
    "return_chunks": true,
    "max_chunk_length": 1000
  }
EOFEOF

