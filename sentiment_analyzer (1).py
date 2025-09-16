import torch
from transformers import pipeline
from keybert import KeyBERT

class SentimentAnalyzer:
    def __init__(self):
        print("Initializing SentimentAnalyzer...")
        try:
            # Using a DistilBERT model for sentiment analysis, which is a good balance of performance and speed.
            self.sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
            print("Sentiment pipeline initialized.")
        except Exception as e:
            print(f"Error initializing sentiment pipeline: {e}")
            self.sentiment_pipeline = None
            
        try:
            # Using KeyBERT for keyword extraction. It uses sentence-transformers models.
            self.keyword_model = KeyBERT()
            print("Keyword model initialized (KeyBERT).")
        except Exception as e:
            print(f"Error initializing keyword model: {e}")
            self.keyword_model = None

    def analyze_sentiment(self, text):
        if not self.sentiment_pipeline:
            return "ERROR", {"ERROR": 1.0}
        try:
            print(f"Analyzing sentiment for text: '{text}'")
            results = self.sentiment_pipeline(text)
            result = results[0]
            sentiment = result["label"]
            sentiment_scores = {sentiment: result["score"]}
            print(f"Sentiment analysis complete: {sentiment}, {sentiment_scores}")
            return sentiment, sentiment_scores
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return "ERROR", {"ERROR": 1.0}

    def extract_keywords(self, text):
        if not self.keyword_model:
            return []
        try:
            print(f"Extracting keywords for text: '{text}'")
            # KeyBERT returns a list of tuples (keyword, score)
            keywords_with_scores = self.keyword_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english')
            keywords = [keyword for keyword, score in keywords_with_scores]
            print(f"Keyword extraction complete: {keywords}")
            return keywords
        except Exception as e:
            print(f"Error extracting key phrases: {e}")
            return []

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()

    # Test sentiment analysis
    text1 = "I love this product! It's amazing and works perfectly."
    sentiment1, scores1 = analyzer.analyze_sentiment(text1)
    print(f"Text: '{text1}'")
    print(f"Sentiment: {sentiment1}, Scores: {scores1}")

    text2 = "This is a terrible experience. I'm very disappointed."
    sentiment2, scores2 = analyzer.analyze_sentiment(text2)
    print(f"Text: '{text2}'")
    print(f"Sentiment: {sentiment2}, Scores: {scores2}")

    text3 = "The weather is neither good nor bad today."
    sentiment3, scores3 = analyzer.analyze_sentiment(text3)
    print(f"Text: '{text3}'")
    print(f"Sentiment: {sentiment3}, Scores: {scores3}")

    # Test keyword extraction
    keywords1 = analyzer.extract_keywords(text1)
    print(f"Keywords for '{text1}': {keywords1}")

    keywords2 = analyzer.extract_keywords(text2)
    print(f"Keywords for '{text2}': {keywords2}")

