import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# this class cleans, tokenizes, and normalizes raw email text
class EmailProcessor:

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def preprocess(self, raw_text: str) -> str:
        text = raw_text.lower()
        text = re.sub(r"http\S+|www\S+", " urltoken ", text)  
        text = re.sub(r"[\w\.-]+@[\w\.-]+", " emailtoken ", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        tokens = self.tokenize(text)
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stop_words and len(t) > 1]
        return " ".join(tokens)

    def tokenize(self, text: str) -> list:
        return word_tokenize(text.lower())