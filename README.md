# Sentiment Analysis Tool

A simple and interactive sentiment analysis tool that uses NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner) to analyze the sentiment of text input.

## Features

- Real-time sentiment analysis of user input
- Provides detailed sentiment scores:
  - Overall sentiment (Positive/Negative/Neutral)
  - Compound score (-1 to 1)
  - Positive, negative, and neutral component scores
- Interactive command-line interface
- Easy to use and understand

## Requirements

- Python 3.x
- NLTK library

## Installation

1. Clone this repository or download the files
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the script:
```bash
python sentiment_analysis.py
```

2. Enter your text when prompted
3. View the sentiment analysis results
4. Type 'quit' to exit the program

## Example Output

```
Enter text to analyze: I love this product! It's amazing!

Analysis Results:
Text: I love this product! It's amazing!
Sentiment: Positive
Compound Score: 0.862
Detailed Scores:
  - Positive: 0.741
  - Negative: 0.000
  - Neutral: 0.259
```

## How It Works

The tool uses NLTK's VADER sentiment analyzer, which is specifically attuned to sentiments expressed in social media. It provides:

- Sentiment Classification: Positive, Negative, or Neutral
- Compound Score: A normalized score between -1 (most negative) and +1 (most positive)
- Detailed Scores: Individual scores for positive, negative, and neutral components

## Project Structure

```
sentiment_analysis/
├── README.md
├── requirements.txt
└── sentiment_analysis.py
```

## License

This project is open source and available under the MIT License.
