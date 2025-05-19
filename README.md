# Enhanced Sentiment Analysis Tool

A comprehensive sentiment analysis tool that combines multiple approaches for accurate sentiment detection, including VADER, transformer models, and multilingual support.

## Features

- **Multiple Analysis Methods**:
  - VADER sentiment analysis with custom lexicon
  - DistilBERT transformer model
  - Hybrid approach combining both methods
  - Multilingual support (English, Spanish, French, German, Italian)

- **Advanced Text Processing**:
  - Contraction expansion
  - Emoji handling
  - Character normalization
  - Spelling correction
  - Punctuation removal
  - Sarcasm detection

- **Interactive Web Interface**:
  - Real-time sentiment analysis
  - Visual score representation
  - Multiple language support
  - Results export to CSV

## Requirements

- Python 3.x
- NLTK
- Transformers (Hugging Face)
- PyTorch
- Streamlit
- Other dependencies listed in requirements.txt

## Installation

1. Clone this repository or download the files
2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data:
```python
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run sentiment_analysis.py
```

2. Enter text in the text area
3. Select language (if not English)
4. Click "Analyze" to get results
5. View detailed scores and visualizations
6. Export results if needed

## Features in Detail

### Custom Lexicon
The tool extends VADER's lexicon with modern slang and domain-specific terms to improve accuracy.

### Text Preprocessing
- Expands contractions (e.g., "don't" → "do not")
- Converts emojis to text
- Normalizes repeated characters
- Corrects spelling errors
- Removes irrelevant punctuation

### Sentiment Analysis
- VADER scores (positive, negative, neutral, compound)
- Transformer model scores
- Combined sentiment score
- Sarcasm detection
- Multilingual support

### Visualization
- Interactive bar charts showing:
  - VADER scores
  - Transformer scores
  - Combined sentiment

## Project Structure

```
sentiment_analysis/
├── README.md
├── requirements.txt
└── sentiment_analysis.py
```

## Performance Considerations

- The transformer model provides more accurate and contextual sentiment analysis
- VADER is faster but less context-aware
- The hybrid approach balances speed and accuracy
- Sarcasm detection improves accuracy for ironic content
- Multilingual support enables analysis in multiple languages

## License

This project is open source and available under the MIT License.
