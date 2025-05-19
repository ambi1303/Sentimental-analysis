import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download required NLTK data
nltk.download('vader_lexicon')

def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    
    # Get the compound score
    compound_score = sentiment_scores['compound']
    
    # Determine sentiment based on compound score
    if compound_score >= 0.05:
        sentiment = 'Positive'
    elif compound_score <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    return {
        'sentiment': sentiment,
        'compound_score': compound_score,
        'scores': sentiment_scores
    }

def main():
    print("\nWelcome to Sentiment Analysis!")
    print("Enter your text (type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        # Get user input
        text = input("\nEnter text to analyze: ").strip()
        
        # Check if user wants to quit
        if text.lower() == 'quit':
            print("\nThank you for using Sentiment Analysis!")
            break
            
        # Skip empty input
        if not text:
            print("Please enter some text to analyze.")
            continue
            
        # Analyze the text
        result = analyze_sentiment(text)
        
        # Print results
        print("\nAnalysis Results:")
        print(f"Text: {text}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Compound Score: {result['compound_score']:.3f}")
        print(f"Detailed Scores:")
        print(f"  - Positive: {result['scores']['pos']:.3f}")
        print(f"  - Negative: {result['scores']['neg']:.3f}")
        print(f"  - Neutral: {result['scores']['neu']:.3f}")

if __name__ == "__main__":
    main() 