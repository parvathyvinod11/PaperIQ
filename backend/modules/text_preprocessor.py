"""
Text Preprocessing Module
Tokenizes, cleans, and segments text for NLP tasks.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import List, Dict

# Ensure NLTK data is available
for pkg in ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger", "punkt_tab"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass


class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def remove_references_section(self, text: str) -> str:
        """Remove the references/bibliography section."""
        patterns = [
            r'\bREFERENCES\b[\s\S]*$',
            r'\bBIBLIOGRAPHY\b[\s\S]*$',
            r'\bWORKS CITED\b[\s\S]*$',
        ]
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match and match.start() > len(text) * 0.5:
                text = text[:match.start()]
                break
        return text

    def remove_noise(self, text: str) -> str:
        """Remove URLs, emails, and other noise."""
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\[[\d,\s]+\]', '', text)  # citation brackets like [1], [2,3]
        text = re.sub(r'\([\d]{4}\)', '', text)  # year in parens
        return text

    def sentence_tokenize(self, text: str) -> List[str]:
        """Split text into sentences."""
        return sent_tokenize(text)

    def word_tokenize_clean(self, text: str) -> List[str]:
        """Tokenize into words, remove stopwords, lemmatize."""
        tokens = word_tokenize(text.lower())
        tokens = [
            self.lemmatizer.lemmatize(t)
            for t in tokens
            if t.isalpha() and t not in self.stop_words and len(t) > 2
        ]
        return tokens

    def preprocess(self, text: str) -> Dict:
        """Full preprocessing pipeline."""
        text_no_refs = self.remove_references_section(text)
        text_clean = self.remove_noise(text_no_refs)

        sentences = self.sentence_tokenize(text_clean)
        tokens = self.word_tokenize_clean(text_clean)

        return {
            "processed_text": text_clean,
            "sentences": sentences,
            "tokens": tokens,
            "sentence_count": len(sentences),
            "token_count": len(tokens),
        }
