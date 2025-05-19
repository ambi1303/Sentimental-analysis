import emoji
import contractions
import re
from textblob import TextBlob

class TextPreprocessor:
    @staticmethod
    def preprocess_text(text):
        """Enhanced text preprocessing pipeline"""
        # Convert to lowercase
        text = text.lower()
        
        # Expand contractions
        text = contractions.fix(text)
        
        # Handle emojis
        text = emoji.demojize(text)
        
        # Normalize repeated characters (e.g., "sooo" -> "so")
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

    @staticmethod
    def get_emoji_sentiment(emoji_text):
        """Get sentiment score for emoji text"""
        # Basic emoji sentiment mapping
        positive_emojis = ['😊', '😄', '😃', '😀', '😁', '😆', '😍', '🥰', '😘', '❤️', '👍', '🎉', '✨', '🌟']
        negative_emojis = ['😢', '😭', '😞', '😔', '😟', '😕', '😣', '😖', '😫', '😩', '😤', '😠', '😡', '👎']
        
        score = 0
        for emoji_char in emoji_text:
            if emoji_char in positive_emojis:
                score += 0.5
            elif emoji_char in negative_emojis:
                score -= 0.5
        
        return score

    @staticmethod
    def normalize_text(text):
        """Normalize text for better sentiment analysis"""
        # Replace common internet slang
        slang_dict = {
            'lol': 'laughing',
            'rofl': 'laughing',
            'lmao': 'laughing',
            'omg': 'oh my god',
            'wtf': 'what the',
            'idk': 'i do not know',
            'tbh': 'to be honest',
            'imo': 'in my opinion',
            'afaik': 'as far as i know',
            'fyi': 'for your information'
        }
        
        for slang, replacement in slang_dict.items():
            text = re.sub(r'\b' + slang + r'\b', replacement, text, flags=re.IGNORECASE)
        
        return text 