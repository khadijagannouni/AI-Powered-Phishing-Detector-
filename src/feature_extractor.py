import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp


URGENCY_KEYWORDS = [
    "urgent", "immediately", "verify", "suspended", "confirm", "account",
    "password", "security", "alert", "warning", "limited", "expires",
    "click", "login", "update", "validate", "unusual", "unauthorized",
]

#this class transforms processed text into numerical feature vectors

class FeatureExtractor:   

    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        self.tfidf_vector = None
        self.url_count: int = 0
        self.urgency_score: float = 0.0

#Fit the TF-IDF vectorizer on a training corpus

    def fit(self, corpus: list):
        self.vectorizer.fit(corpus)

    def transform(self, cleaned_text: str, raw_text: str = "") -> np.ndarray:
        
        tfidf = self.vectorizer.transform([cleaned_text])
        tfidf_dense = tfidf.toarray()

        urls = self.extract_urls(raw_text or cleaned_text)
        urgency = self.score_urgency(cleaned_text)

        extras = np.array([[len(urls), urgency]])
        return np.hstack([tfidf_dense, extras])
    
    #Fit on corpus and immediately transform all samples

    def fit_transform(self, corpus: list, raw_texts: list = None) -> np.ndarray:
        tfidf_matrix = self.vectorizer.fit_transform(corpus).toarray()
        extras = []
        for i, text in enumerate(corpus):
            raw = raw_texts[i] if raw_texts else text
            urls = self.extract_urls(raw)
            urgency = self.score_urgency(text)
            extras.append([len(urls), urgency])
        return np.hstack([tfidf_matrix, np.array(extras)])

    #  Helper methods                            


    #Returns a list of URLs found in the raw text
                          
    def extract_urls(self, text: str) -> list:
        pattern = r"(https?://\S+|www\.\S+)"
        return re.findall(pattern, text, re.IGNORECASE)
    
    #Returns a normalised urgency score [0, 1] based on keyword frequency.

    def score_urgency(self, text: str) -> float:
        tokens = text.lower().split()
        if not tokens:
            return 0.0
        hits = sum(1 for t in tokens if t in URGENCY_KEYWORDS)
        return min(hits / len(tokens) * 10, 1.0)   # scale & cap at 1