import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import numpy as np
import re
import emoji
import contractions
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Union, Optional
import logging
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class EnhancedSentimentAnalyzer:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """Initialize the enhanced sentiment analyzer with multiple models."""
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('sentiment_analysis.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize VADER
        self.vader = SentimentIntensityAnalyzer()
        self.load_custom_lexicon()
        
        # Initialize transformer model
        self.transformer_model = None
        self.transformer_tokenizer = None
        self.load_transformer_model(model_name)
        
        # Initialize sarcasm detector
        self.sarcasm_detector = None
        self.load_sarcasm_detector()
        
        # Sarcasm detection thresholds
        self.sarcasm_thresholds = {
            'strong': 0.8,
            'moderate': 0.7,
            'possible': 0.6
        }
        
        # Enhanced sarcasm indicators with contextual weights
        self.sarcasm_indicators = {
            'positive_words': [
                'love', 'great', 'wonderful', 'amazing', 'perfect', 'best', 
                'excellent', 'fantastic', 'relaxing', 'enjoy', 'happy', 'glad'
            ],
            'negative_context': [
                'waiting', 'traffic', 'delay', 'problem', 'issue', 'bad', 
                'terrible', 'horrible', 'early', 'late', 'missed', 'failed',
                'monday', 'weekend', 'overtime', 'meeting', 'email'
            ],
            'sarcasm_phrases': {
                'oh great': 0.9,
                'yeah because': 0.9,
                'sure i love': 0.9,
                'just what i needed': 0.9,
                'of course': 0.8,
                'absolutely': 0.8,
                'totally': 0.8,
                'yeah right': 0.9,
                'sure thing': 0.8,
                'whatever': 0.7,
                'as if': 0.9,
                'like i care': 0.9,
                'how wonderful': 0.9,
                'sure': 0.7,
                'yeah': 0.7,
                'right': 0.7
            },
            'sarcasm_patterns': [
                (r'oh great.*', 0.9),
                (r'yeah.*because.*', 0.9),
                (r'sure.*love.*', 0.9),
                (r'just what.*needed.*', 0.9),
                (r'how.*wonderful.*', 0.9),
                (r'as if.*', 0.9),
                (r'like.*care.*', 0.9),
                (r'^sure.*', 0.7),
                (r'^yeah.*', 0.7),
                (r'^right.*', 0.7)
            ]
        }
        
        # Create output directory
        self.output_dir = Path('output')
        self.output_dir.mkdir(exist_ok=True)

    def load_custom_lexicon(self):
        """Load and extend VADER lexicon with custom terms."""
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
            'fail': -1.5,
            'sarcastic': -1.0,
            'ironic': -1.0,
            'yeah right': -1.5,
            'whatever': -1.0,
            'absolutely': 0.0,
            'totally': 0.0,
            'of course': 0.0,
            'relaxing': 1.0,
            'wonderful': 1.5,
            'perfect': 1.5,
            'sure': 0.0,
            'yeah': 0.0,
            'right': 0.0
        }
        self.vader.lexicon.update(custom_lexicon)
        self.logger.info("Custom lexicon loaded successfully")

    def load_transformer_model(self, model_name: str):
        """Load the transformer model for sentiment analysis."""
        try:
            self.transformer_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.transformer_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.logger.info(f"Transformer model {model_name} loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading transformer model: {e}")

    def load_sarcasm_detector(self):
        """Initialize sarcasm detection model."""
        try:
            model_name = "mrm8488/t5-base-finetuned-sarcasm-twitter"
            self.sarcasm_detector = pipeline(
                "text-classification",
                model=model_name,
                tokenizer=model_name
            )
            self.logger.info("Sarcasm detector loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading sarcasm detector: {e}")

    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing pipeline."""
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
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|\#\w+', '', text)
        
        # Remove irrelevant punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text

    def check_sarcasm_patterns(self, text: str) -> tuple:
        """Check for common sarcasm patterns using regex with weighted scores."""
        text_lower = text.lower()
        pattern_matches = []
        total_score = 0.0
        
        for pattern, weight in self.sarcasm_indicators['sarcasm_patterns']:
            if re.search(pattern, text_lower):
                pattern_matches.append(pattern)
                total_score += weight
        
        return len(pattern_matches) > 0, total_score / len(self.sarcasm_indicators['sarcasm_patterns'])

    def rule_based_sarcasm_detection(self, text: str) -> tuple:
        """Enhanced rule-based sarcasm detection using word patterns and context."""
        text_lower = text.lower()
        
        # Check for positive words in negative context
        positive_words_present = any(word in text_lower for word in self.sarcasm_indicators['positive_words'])
        negative_context_present = any(word in text_lower for word in self.sarcasm_indicators['negative_context'])
        
        # Check for sarcasm phrases with weights
        sarcasm_score = 0.0
        for phrase, weight in self.sarcasm_indicators['sarcasm_phrases'].items():
            if phrase in text_lower:
                sarcasm_score += weight
        
        # Check for sarcasm patterns
        pattern_match, pattern_score = self.check_sarcasm_patterns(text)
        
        # Calculate final sarcasm probability
        sarcasm_prob = 0.0
        
        if positive_words_present and negative_context_present:
            sarcasm_prob += 0.4
        
        if sarcasm_score > 0:
            sarcasm_prob += 0.3 * (sarcasm_score / len(self.sarcasm_indicators['sarcasm_phrases']))
        
        if pattern_match:
            sarcasm_prob += 0.3 * pattern_score
        
        return sarcasm_prob > self.sarcasm_thresholds['moderate'], sarcasm_prob

    def detect_sarcasm(self, text: str) -> tuple:
        """Enhanced sarcasm detection combining model and rule-based approaches."""
        # Get model-based sarcasm detection
        model_sarcasm = False
        model_score = 0.0
        
        if self.sarcasm_detector:
            try:
                result = self.sarcasm_detector(text)[0]
                model_sarcasm = result['label'] == 'LABEL_1'
                model_score = result['score']
            except Exception as e:
                self.logger.error(f"Error in model-based sarcasm detection: {e}")
        
        # Get rule-based sarcasm detection
        rule_sarcasm, rule_score = self.rule_based_sarcasm_detection(text)
        
        # Combine scores (weighted average)
        if model_score > 0:
            final_score = 0.6 * model_score + 0.4 * rule_score
        else:
            final_score = rule_score
        
        return final_score > self.sarcasm_thresholds['moderate'], final_score

    def get_transformer_sentiment(self, text: str) -> float:
        """Get sentiment score from transformer model."""
        if self.transformer_model:
            inputs = self.transformer_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True
            )
            with torch.no_grad():
                outputs = self.transformer_model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)[0]
            return scores[1].item()  # Positive sentiment score
        return 0.5  # Neutral if model not available

    def analyze_sentiment(self, text: str, preprocess: bool = True) -> Dict:
        """Comprehensive sentiment analysis with enhanced sarcasm detection."""
        # Preprocess text if requested
        processed_text = self.preprocess_text(text) if preprocess else text
        
        # Get VADER scores
        vader_scores = self.vader.polarity_scores(processed_text)
        
        # Get transformer scores
        transformer_score = self.get_transformer_sentiment(processed_text)
        
        # Detect sarcasm
        is_sarcastic, sarcasm_score = self.detect_sarcasm(processed_text)
        
        # Get TextBlob sentiment
        textblob_sentiment = TextBlob(processed_text).sentiment
        
        # Combine scores with weights
        vader_weight = 0.4
        transformer_weight = 0.4
        textblob_weight = 0.2
        
        final_score = (
            vader_weight * vader_scores['compound'] +
            transformer_weight * (transformer_score * 2 - 1) +  # Convert to [-1, 1] range
            textblob_weight * textblob_sentiment.polarity
        )
        
        # Check for neutral sentiment
        if vader_scores['neu'] > 0.85 and abs(textblob_sentiment.polarity) < 0.05:
            sentiment = 'Neutral'
            final_score = 0
        else:
            # Adjust score if sarcasm is detected
            if is_sarcastic:
                # Reverse the sentiment score
                final_score = -final_score
                # Adjust the magnitude based on sarcasm confidence
                final_score *= (1 + sarcasm_score)
            
            # Determine sentiment
            if final_score >= 0.05:
                sentiment = 'Positive'
            elif final_score <= -0.05:
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'
        
        # Generate warning message for sarcasm
        warning_message = None
        if is_sarcastic:
            if sarcasm_score > self.sarcasm_thresholds['strong']:
                warning_message = "Strong sarcasm detected"
            elif sarcasm_score > self.sarcasm_thresholds['moderate']:
                warning_message = "Moderate sarcasm detected"
            else:
                warning_message = "Possible sarcasm detected"
        
        results = {
            'text': text,
            'processed_text': processed_text if preprocess else None,
            'sentiment': sentiment,
            'vader_scores': vader_scores,
            'transformer_score': transformer_score,
            'textblob_scores': {
                'polarity': textblob_sentiment.polarity,
                'subjectivity': textblob_sentiment.subjectivity
            },
            'is_sarcastic': is_sarcastic,
            'sarcasm_score': sarcasm_score,
            'final_score': final_score,
            'adjusted_sentiment': sentiment if is_sarcastic else None,
            'warning_message': warning_message,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log the analysis
        self.logger.info(f"Analyzed text: {text[:50]}...")
        self.logger.info(f"Sentiment: {sentiment}, Sarcasm: {is_sarcastic}")
        
        return results

    def analyze_batch(self, texts: List[str], preprocess: bool = True) -> List[Dict]:
        """Analyze a batch of texts."""
        results = []
        for text in texts:
            result = self.analyze_sentiment(text, preprocess)
            results.append(result)
        return results

    def evaluate_model(self, dataset_name: str = "sst2") -> Dict:
        """Evaluate model performance on benchmark datasets."""
        try:
            # Load dataset
            dataset = load_dataset(dataset_name)
            
            # Prepare data
            texts = dataset['train']['sentence']
            labels = dataset['train']['label']
            
            # Analyze texts
            predictions = []
            for text in texts:
                result = self.analyze_sentiment(text)
                pred = 1 if result['sentiment'] == 'Positive' else 0
                predictions.append(pred)
            
            # Calculate metrics
            report = classification_report(labels, predictions, output_dict=True)
            conf_matrix = confusion_matrix(labels, predictions)
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(self.output_dir / 'confusion_matrix.png')
            plt.close()
            
            # Save evaluation results
            evaluation_results = {
                'dataset': dataset_name,
                'metrics': report,
                'confusion_matrix': conf_matrix.tolist(),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.output_dir / 'evaluation_results.json', 'w') as f:
                json.dump(evaluation_results, f, indent=4)
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error in model evaluation: {e}")
            return None

    def export_results(self, results: Union[Dict, List[Dict]], format: str = 'csv') -> str:
        """Export analysis results to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format.lower() == 'csv':
            if isinstance(results, dict):
                results = [results]
            df = pd.DataFrame(results)
            output_file = self.output_dir / f'sentiment_analysis_{timestamp}.csv'
            df.to_csv(output_file, index=False)
        elif format.lower() == 'json':
            output_file = self.output_dir / f'sentiment_analysis_{timestamp}.json'
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=4)
        else:
            raise ValueError("Unsupported format. Use 'csv' or 'json'.")
        
        self.logger.info(f"Results exported to {output_file}")
        return str(output_file)

def main():
    """Main function to demonstrate usage."""
    # Initialize analyzer
    analyzer = EnhancedSentimentAnalyzer()
    
    # Example texts
    texts = [
        "I absolutely love waiting in traffic for two hours.",
        "Oh great, another Monday. Just what I needed.",
        "Yeah, because waking up at 5 AM is so relaxing.",
        "The meeting is scheduled for 2 PM.",
        "The weather is cloudy today."
    ]
    
    # Analyze texts
    results = analyzer.analyze_batch(texts)
    
    # Export results
    analyzer.export_results(results, format='csv')
    analyzer.export_results(results, format='json')
    
    # Evaluate model
    evaluation_results = analyzer.evaluate_model()
    if evaluation_results:
        print("\nEvaluation Results:")
        print(json.dumps(evaluation_results['metrics'], indent=2))

if __name__ == "__main__":
    main() 