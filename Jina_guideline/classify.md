curl https://api.jina.ai/v1/classify \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer " \
  -d @- <<EOFEOF
  {
    "model": "jina-clip-v2",
    "input": [
        {
            "text": "A sleek smartphone with a high-resolution display and multiple camera lenses"
        },
        {
            "text": "Fresh sushi rolls served on a wooden board with wasabi and ginger"
        },
        {
            "image": "https://picsum.photos/id/11/367/267"
        },
        {
            "image": "https://picsum.photos/id/22/367/267"
        },
        {
            "text": "Vibrant autumn leaves in a dense forest with sunlight filtering through"
        },
        {
            "image": "https://picsum.photos/id/8/367/267"
        }
    ],
    "labels": [
        "Technology and Gadgets",
        "Food and Dining",
        "Nature and Outdoors",
        "Urban and Architecture"
    ]
  }
EOFEOF
