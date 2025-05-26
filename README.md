# Sentiment Analysis API

A powerful sentiment analysis API that combines multiple models for accurate sentiment detection, including sarcasm detection capabilities.

## Features

- **Multi-model Sentiment Analysis**
  - VADER sentiment analysis
  - Transformer-based sentiment analysis
  - TextBlob sentiment analysis
  - Combined weighted scoring

- **Sarcasm Detection**
  - Rule-based sarcasm detection
  - Model-based sarcasm detection
  - Contextual analysis
  - Confidence scoring

## API Endpoints

- `POST /analyze`: Analyze single text
- `POST /analyze-batch`: Analyze multiple texts
- `GET /health`: Health check endpoint
- `GET /metrics`: Model performance metrics

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the API server:
```bash
uvicorn backend.api.api:app --host 0.0.0.0 --port 8000 --reload
```

2. Example API calls:

```python
import requests

# Single text analysis
response = requests.post(
    "http://localhost:8000/analyze",
    json={
        "text": "I love this product!",
        "include_stress": True,
        "include_keywords": True
    }
)
print(response.json())

# Batch analysis
response = requests.post(
    "http://localhost:8000/analyze-batch",
    json={
        "texts": [
            "I love this product!",
            "This is terrible.",
            "It's okay."
        ],
        "include_stress": True,
        "include_keywords": True
    }
)
print(response.json())
```

## API Response Format

### Single Text Analysis
```json
{
    "text": "input text",
    "processed_text": "preprocessed text",
    "sentiment": "Positive/Negative/Neutral",
    "vader_scores": {
        "neg": 0.0,
        "neu": 0.0,
        "pos": 0.0,
        "compound": 0.0
    },
    "transformer_score": 0.0,
    "textblob_scores": {
        "polarity": 0.0,
        "subjectivity": 0.0
    },
    "is_sarcastic": false,
    "sarcasm_score": 0.0,
    "final_score": 0.0,
    "adjusted_sentiment": null,
    "warning_message": null,
    "timestamp": "2024-03-26T10:53:02",
    "processing_time": "0.23s"
}
```

### Batch Analysis
```json
{
    "results": [
        // Array of single text analysis results
    ],
    "processing_time": "0.45s",
    "average_time_per_text": "0.15s"
}
```

## Project Structure

```
sentiment_analysis/
├── requirements.txt
└── backend/
    ├── api/
    │   └── api.py           # FastAPI endpoints
    └── core/
        └── enhanced_sentiment_analyzer.py  # Core sentiment analysis logic
```

## Performance

- Single text analysis: ~200-300ms
- Batch analysis: ~100ms per text
- Metrics endpoint: Cached for 5 minutes

## Error Handling

The API includes comprehensive error handling:
- Input validation
- Model loading errors
- Processing errors
- Rate limiting
- Detailed error messages

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
