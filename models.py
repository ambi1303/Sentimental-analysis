import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import numpy as np
import re

class SentimentAnalyzer:
    def __init__(self):
        # Initialize VADER
        self.vader = SentimentIntensityAnalyzer()
        self.load_custom_lexicon()
        
        # Initialize transformer model
        self.transformer_model = None
        self.transformer_tokenizer = None
        self.load_transformer_model()
        
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
            'fail': -1.5,
            'sarcastic': -1.0,
            'ironic': -1.0,
            'yeah right': -1.5,
            'whatever': -1.0,
            'absolutely': 0.0,  # Neutral for sarcasm detection
            'totally': 0.0,     # Neutral for sarcasm detection
            'of course': 0.0,   # Neutral for sarcasm detection
            'relaxing': 1.0,    # Positive but often used sarcastically
            'wonderful': 1.5,   # Positive but often used sarcastically
            'perfect': 1.5,     # Positive but often used sarcastically
            'sure': 0.0,        # Neutral for sarcasm detection
            'yeah': 0.0,        # Neutral for sarcasm detection
            'right': 0.0        # Neutral for sarcasm detection
        }
        self.vader.lexicon.update(custom_lexicon)

    def load_transformer_model(self):
        """Load the DistilBERT model for sentiment analysis"""
        try:
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            self.transformer_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.transformer_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading transformer model: {e}")

    def load_sarcasm_detector(self):
        """Initialize sarcasm detection model"""
        try:
            # Using a model fine-tuned for sarcasm detection
            model_name = "mrm8488/t5-base-finetuned-sarcasm-twitter"
            self.sarcasm_detector = pipeline(
                "text-classification",
                model=model_name,
                tokenizer=model_name
            )
        except Exception as e:
            print(f"Error loading sarcasm detector: {e}")

    def check_sarcasm_patterns(self, text):
        """Check for common sarcasm patterns using regex with weighted scores"""
        text_lower = text.lower()
        pattern_matches = []
        total_score = 0.0
        
        for pattern, weight in self.sarcasm_indicators['sarcasm_patterns']:
            if re.search(pattern, text_lower):
                pattern_matches.append(pattern)
                total_score += weight
        
        return len(pattern_matches) > 0, total_score / len(self.sarcasm_indicators['sarcasm_patterns'])

    def rule_based_sarcasm_detection(self, text):
        """Enhanced rule-based sarcasm detection using word patterns and context"""
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

    def detect_sarcasm(self, text):
        """Enhanced sarcasm detection combining model and rule-based approaches"""
        # Get model-based sarcasm detection
        model_sarcasm = False
        model_score = 0.0
        
        if self.sarcasm_detector:
            try:
                result = self.sarcasm_detector(text)[0]
                model_sarcasm = result['label'] == 'LABEL_1'
                model_score = result['score']
            except Exception as e:
                print(f"Error in model-based sarcasm detection: {e}")
        
        # Get rule-based sarcasm detection
        rule_sarcasm, rule_score = self.rule_based_sarcasm_detection(text)
        
        # Combine scores (weighted average)
        if model_score > 0:
            final_score = 0.6 * model_score + 0.4 * rule_score
        else:
            final_score = rule_score
        
        return final_score > self.sarcasm_thresholds['moderate'], final_score

    def get_transformer_sentiment(self, text):
        """Get sentiment score from transformer model"""
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

    def analyze_sentiment(self, text):
        """Comprehensive sentiment analysis with enhanced sarcasm detection"""
        # Get VADER scores
        vader_scores = self.vader.polarity_scores(text)
        
        # Get transformer scores
        transformer_score = self.get_transformer_sentiment(text)
        
        # Detect sarcasm
        is_sarcastic, sarcasm_score = self.detect_sarcasm(text)
        
        # Get TextBlob sentiment
        textblob_sentiment = TextBlob(text).sentiment
        
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
        
        return {
            'text': text,
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
            'warning_message': warning_message
        }

class EnhancedSentimentAnalyzer:
    def __init__(self):
        # Initialize sarcasm detection threshold
        self.sarcasm_threshold = 0.6  # Lowered from 0.7 to catch more sarcastic cases
        
        # Enhanced sarcasm indicators
        self.positive_words = {
            'love', 'great', 'wonderful', 'amazing', 'fantastic', 'perfect',
            'excellent', 'brilliant', 'beautiful', 'happy', 'delighted', 'thrilled',
            'excited', 'fabulous', 'marvelous', 'splendid', 'terrific', 'outstanding'
        }
        
        self.negative_contexts = {
            'waiting', 'traffic', 'monday', 'waking up', 'early', 'meeting',
            'work', 'overtime', 'weekend', 'flat tire', 'slow', 'terrible',
            'surprise', 'problem', 'issue', 'delay', 'late', 'broken'
        }
        
        self.sarcasm_phrases = {
            'oh great', 'just what i needed', 'yeah right', 'sure thing',
            'of course', 'absolutely', 'definitely', 'totally', 'completely',
            'wonderful surprise', 'lovely', 'perfect timing', 'exactly what i wanted',
            'couldn\'t be better', 'just perfect', 'how nice', 'what a joy',
            'i\'m so happy', 'i love', 'i absolutely love', 'i really love',
            'i\'m thrilled', 'i\'m delighted', 'i\'m excited', 'i\'m overjoyed'
        }
        
        self.sarcasm_patterns = [
            r'(?i)(i|we|they|he|she|it)\s+(absolutely|totally|completely|really|truly)\s+(love|adore|enjoy|like)',
            r'(?i)(oh|ah|well)\s+(great|wonderful|perfect|lovely|fantastic)',
            r'(?i)(just|exactly)\s+(what|the)\s+(i|we|they|he|she|it)\s+(needed|wanted)',
            r'(?i)(couldn\'t|can\'t)\s+(be|get)\s+(any|more)\s+(better|worse)',
            r'(?i)(what|how)\s+(a|an)\s+(wonderful|lovely|perfect|great)\s+(surprise|day|time)',
            r'(?i)(i\'m|we\'re|they\'re|he\'s|she\'s|it\'s)\s+(so|really|absolutely|totally)\s+(happy|thrilled|delighted|excited)',
            r'(?i)(sure|yeah|right|of course|absolutely|definitely|totally)\s+(thing|why not|that\'s great)',
            r'(?i)(just|exactly)\s+(perfect|what i needed|what i wanted)',
            r'(?i)(i|we|they|he|she|it)\s+(just|really)\s+(love|adore|enjoy)\s+(to|when)',
            r'(?i)(isn\'t|aren\'t|wasn\'t|weren\'t)\s+(that|this)\s+(just|so|really)\s+(great|wonderful|perfect)'
        ]
        
        # Load models and initialize other components
        self._load_models()
        self._load_custom_lexicon()

    def check_sarcasm_patterns(self, text):
        """Check for sarcasm patterns using regex with weighted scores."""
        pattern_scores = []
        for pattern in self.sarcasm_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                # Calculate pattern score based on match length and position
                match_text = match.group(0)
                score = 0.7  # Base score for pattern match
                
                # Boost score if pattern contains strong sarcasm indicators
                if any(phrase in match_text for phrase in self.sarcasm_phrases):
                    score += 0.2
                
                # Boost score if pattern contains positive words in negative context
                if any(word in match_text for word in self.positive_words) and \
                   any(context in text.lower() for context in self.negative_contexts):
                    score += 0.1
                
                pattern_scores.append(score)
        
        return max(pattern_scores) if pattern_scores else 0.0

    def rule_based_sarcasm_detection(self, text):
        """Enhanced rule-based sarcasm detection with weighted scoring."""
        text_lower = text.lower()
        score = 0.0
        weights = {
            'positive_negative': 0.4,  # Weight for positive words in negative context
            'sarcasm_phrases': 0.3,    # Weight for known sarcasm phrases
            'patterns': 0.3            # Weight for pattern matching
        }
        
        # Check for positive words in negative contexts
        positive_in_negative = False
        for word in self.positive_words:
            if word in text_lower:
                for context in self.negative_contexts:
                    if context in text_lower:
                        positive_in_negative = True
                        break
                if positive_in_negative:
                    break
        
        if positive_in_negative:
            score += weights['positive_negative']
        
        # Check for known sarcasm phrases
        for phrase in self.sarcasm_phrases:
            if phrase in text_lower:
                score += weights['sarcasm_phrases']
                break
        
        # Check for sarcasm patterns
        pattern_score = self.check_sarcasm_patterns(text)
        score += pattern_score * weights['patterns']
        
        return min(score, 1.0)  # Cap the score at 1.0

    def detect_sarcasm(self, text):
        """Enhanced sarcasm detection combining model and rule-based approaches."""
        # Get model-based sarcasm score
        model_score = self._get_model_sarcasm_score(text)
        
        # Get rule-based sarcasm score
        rule_score = self.rule_based_sarcasm_detection(text)
        
        # Combine scores with adjusted weights
        combined_score = (model_score * 0.4) + (rule_score * 0.6)
        
        # Determine if text is sarcastic based on combined score
        is_sarcastic = combined_score >= self.sarcasm_threshold
        
        return {
            'is_sarcastic': is_sarcastic,
            'sarcasm_score': combined_score,
            'model_score': model_score,
            'rule_score': rule_score
        } 