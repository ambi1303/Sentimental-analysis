import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from models import SentimentAnalyzer
from preprocessing import TextPreprocessor

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
    
    # Add Transformer score
    fig.add_trace(go.Bar(
        name='Transformer',
        x=['Score'],
        y=[results['transformer_score']]
    ))
    
    # Add TextBlob scores
    fig.add_trace(go.Bar(
        name='TextBlob',
        x=['Polarity', 'Subjectivity'],
        y=[results['textblob_scores']['polarity'],
           results['textblob_scores']['subjectivity']]
    ))
    
    # Add sarcasm score if detected
    if results['is_sarcastic']:
        fig.add_trace(go.Bar(
            name='Sarcasm',
            x=['Confidence'],
            y=[results['sarcasm_score']]
        ))
    
    fig.update_layout(
        title='Sentiment Analysis Scores',
        barmode='group',
        yaxis_title='Score',
        showlegend=True
    )
    
    return fig

def display_sarcasm_warning(results):
    """Display appropriate warning for sarcasm detection"""
    if results['is_sarcastic']:
        if results['sarcasm_score'] > 0.8:
            st.error(f"⚠️ {results['warning_message']} (Confidence: {results['sarcasm_score']:.3f})")
        elif results['sarcasm_score'] > 0.6:
            st.warning(f"⚠️ {results['warning_message']} (Confidence: {results['sarcasm_score']:.3f})")
        else:
            st.info(f"ℹ️ {results['warning_message']} (Confidence: {results['sarcasm_score']:.3f})")

def main():
    st.title("Enhanced Sentiment Analysis")
    st.write("Analyze text sentiment with advanced features including sarcasm detection and contextual understanding.")
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    preprocessor = TextPreprocessor()
    
    # Input text
    text = st.text_area("Enter text to analyze:", height=100)
    
    # Preprocessing options
    st.subheader("Preprocessing Options")
    normalize_text = st.checkbox("Normalize text (expand contractions, handle emojis)", value=True)
    correct_spelling = st.checkbox("Correct spelling", value=True)
    
    if st.button("Analyze"):
        if text:
            # Preprocess text
            processed_text = text
            if normalize_text:
                processed_text = preprocessor.preprocess_text(text)
            
            # Perform analysis
            results = analyzer.analyze_sentiment(processed_text)
            
            # Display results
            st.subheader("Analysis Results")
            
            # Create columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                # Display sarcasm warning
                display_sarcasm_warning(results)
                
                if results['is_sarcastic']:
                    st.write(f"**Original Sentiment:** {results['sentiment']}")
                    st.write(f"**Adjusted Sentiment:** {results['adjusted_sentiment']}")
                else:
                    st.write(f"**Sentiment:** {results['sentiment']}")
                
                st.write(f"**Final Score:** {results['final_score']:.3f}")
            
            with col2:
                st.write("**Detailed Scores:**")
                st.write("VADER Scores:", results['vader_scores'])
                st.write(f"Transformer Score: {results['transformer_score']:.3f}")
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