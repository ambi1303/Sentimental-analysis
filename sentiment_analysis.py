import nltk
import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
import emoji
import contractions
import re
from textblob import TextBlob
import plotly.graph_objects as go
import streamlit as st

# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')

class EnhancedSentimentAnalyzer:
    def __init__(self):
        # Initialize VADER
        self.vader = SentimentIntensityAnalyzer()
        
        # Load custom lexicon
        self.load_custom_lexicon()

    def load_custom_lexicon(self):
        """Load and extend VADER lexicon with custom terms"""
        custom_lexicon = {
            'awesome': 2.0,
            'sucks': -2.0,
            'lit': 1.5,
            'fire': 1.5,
            'trash': -1.5,
            'dope': 1.5,
            'wack': -1.5,
            'sick': 1.5,
            'bomb': 1.5,
            'fail': -1.5
        }
        self.vader.lexicon.update(custom_lexicon)

    def preprocess_text(self, text):
        """Enhanced text preprocessing"""
        # Convert to lowercase
        text = text.lower()
        
        # Expand contractions
        text = contractions.fix(text)
        
        # Handle emojis
        text = emoji.demojize(text)
        
        # Normalize repeated characters
        text = re.sub(r'(.)\1+', r'\1\1', text)
        
        # Correct spelling
        text = str(TextBlob(text).correct())
        
        # Remove irrelevant punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text

    def analyze_sentiment(self, text):
        """Comprehensive sentiment analysis"""
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Get VADER scores
        vader_scores = self.vader.polarity_scores(processed_text)
        
        # Get TextBlob sentiment
        textblob_sentiment = TextBlob(processed_text).sentiment
        
        # Combine scores
        final_score = (vader_scores['compound'] + textblob_sentiment.polarity) / 2
        
        # Determine sentiment
        if final_score >= 0.05:
            sentiment = 'Positive'
        elif final_score <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        
        return {
            'text': text,
            'processed_text': processed_text,
            'sentiment': sentiment,
            'vader_scores': vader_scores,
            'textblob_scores': {
                'polarity': textblob_sentiment.polarity,
                'subjectivity': textblob_sentiment.subjectivity
            },
            'final_score': final_score
        }

def create_sentiment_visualization(results):
    """Create interactive visualization of sentiment scores"""
    fig = go.Figure()
    
    # Add VADER scores
    fig.add_trace(go.Bar(
        name='VADER',
        x=['Positive', 'Negative', 'Neutral'],
        y=[results['vader_scores']['pos'], 
           results['vader_scores']['neg'], 
           results['vader_scores']['neu']]
    ))
    
    # Add TextBlob scores
    fig.add_trace(go.Bar(
        name='TextBlob',
        x=['Polarity', 'Subjectivity'],
        y=[results['textblob_scores']['polarity'],
           results['textblob_scores']['subjectivity']]
    ))
    
    fig.update_layout(
        title='Sentiment Analysis Scores',
        barmode='group',
        yaxis_title='Score',
        showlegend=True
    )
    
    return fig

def main():
    st.title("Enhanced Sentiment Analysis")
    
    # Initialize analyzer
    analyzer = EnhancedSentimentAnalyzer()
    
    # Input text
    text = st.text_area("Enter text to analyze:", height=100)
    
    if st.button("Analyze"):
        if text:
            # Perform analysis
            results = analyzer.analyze_sentiment(text)
            
            # Display results
            st.subheader("Analysis Results")
            st.write(f"Sentiment: {results['sentiment']}")
            st.write(f"Final Score: {results['final_score']:.3f}")
            
            # Display detailed scores
            st.subheader("Detailed Scores")
            st.write("VADER Scores:", results['vader_scores'])
            st.write("TextBlob Scores:", results['textblob_scores'])
            
            # Create and display visualization
            fig = create_sentiment_visualization(results)
            st.plotly_chart(fig)
            
            # Export option
            if st.button("Export Results"):
                df = pd.DataFrame([results])
                df.to_csv('sentiment_results.csv', index=False)
                st.success("Results exported to sentiment_results.csv")
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main() 