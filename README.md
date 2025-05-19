# Enhanced Sentiment Analysis Tool

A comprehensive sentiment analysis tool that combines multiple models and techniques for accurate sentiment detection, including sarcasm detection and contextual understanding.

## Features

- **Multi-Model Sentiment Analysis**
  - VADER for rule-based sentiment analysis
  - DistilBERT transformer model for deep learning-based analysis
  - TextBlob for additional sentiment insights

- **Advanced Sarcasm Detection**
  - Model-based detection using fine-tuned T5 model
  - Rule-based detection using contextual patterns
  - Weighted scoring system for sarcasm confidence

- **Enhanced Text Preprocessing**
  - Lowercasing
  - Contraction expansion
  - Emoji/emoticon normalization
  - Character normalization
  - Spelling correction
  - URL and mention removal
  - Punctuation handling

- **Batch Processing**
  - Support for analyzing multiple texts
  - File input/output in CSV and JSON formats
  - Interactive CLI mode

- **Model Evaluation**
  - Performance evaluation on benchmark datasets
  - Confusion matrix visualization
  - Detailed metrics reporting

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd sentiment-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data:
```python
import nltk
nltk.download('vader_lexicon')
```

## Usage

### Command Line Interface

1. Analyze a single text:
```bash
python cli.py --text "I absolutely love waiting in traffic for two hours."
```

2. Analyze texts from a file:
```bash
python cli.py --file input.txt --format csv
```

3. Interactive mode:
```bash
python cli.py
```

4. Evaluate model performance:
```bash
python cli.py --evaluate --dataset sst2
```

### Python API

```python
from enhanced_sentiment_analyzer import EnhancedSentimentAnalyzer

# Initialize analyzer
analyzer = EnhancedSentimentAnalyzer()

# Analyze single text
result = analyzer.analyze_sentiment("I absolutely love waiting in traffic for two hours.")

# Analyze batch of texts
texts = [
    "Oh great, another Monday.",
    "The weather is beautiful today."
]
results = analyzer.analyze_batch(texts)

# Export results
analyzer.export_results(results, format='json')
```

## Output Format

The analysis results include:

- Original and processed text
- Sentiment classification (Positive/Negative/Neutral)
- VADER sentiment scores
- Transformer model score
- TextBlob sentiment scores
- Sarcasm detection results
- Warning messages for sarcasm
- Timestamp

Example JSON output:
```json
{
    "text": "I absolutely love waiting in traffic for two hours.",
    "processed_text": "i absolutely love waiting in traffic for two hours",
    "sentiment": "Negative",
    "vader_scores": {
        "neg": 0.0,
        "neu": 0.0,
        "pos": 1.0,
        "compound": 0.6369
    },
    "transformer_score": 0.95,
    "textblob_scores": {
        "polarity": 0.5,
        "subjectivity": 0.5
    },
    "is_sarcastic": true,
    "sarcasm_score": 0.85,
    "final_score": -0.8,
    "adjusted_sentiment": "Negative",
    "warning_message": "Strong sarcasm detected",
    "timestamp": "2024-02-19T12:34:56.789Z"
}
```

## Model Evaluation

The tool can evaluate its performance on benchmark datasets:

- SST-2 (Stanford Sentiment Treebank)
- IMDB
- Sentiment140

Evaluation results include:
- Classification metrics (precision, recall, F1-score)
- Confusion matrix visualization
- Detailed performance report

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
