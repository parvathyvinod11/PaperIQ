"""
Topic Trend Analyzer
Analyzes topic frequency and trends using LDA topic modeling.
"""

from typing import Dict, List
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

for pkg in ["stopwords", "punkt", "punkt_tab"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass


class TrendAnalyzer:
    def __init__(self, n_topics: int = 6):
        self.n_topics = n_topics
        self.stop_words = set(stopwords.words("english"))

    def extract_topics_lda(self, texts: List[str], n_top_words: int = 8) -> List[Dict]:
        """Run LDA topic modeling on a list of texts."""
        if not texts:
            return []

        try:
            vectorizer = CountVectorizer(
                stop_words="english",
                max_df=0.95,
                min_df=1,
                max_features=1000,
                ngram_range=(1, 2),
            )
            dtm = vectorizer.fit_transform(texts)
            n_topics = min(self.n_topics, len(texts))

            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=20,
            )
            lda.fit(dtm)

            feature_names = vectorizer.get_feature_names_out()
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_word_indices = topic.argsort()[: -n_top_words - 1 : -1]
                top_words = [feature_names[i] for i in top_word_indices]
                topics.append({
                    "topic_id": topic_idx + 1,
                    "label": f"Topic {topic_idx + 1}",
                    "keywords": top_words,
                    "weight": round(float(topic.sum() / lda.components_.sum()), 4),
                })
            return topics
        except Exception as e:
            print(f"LDA failed: {e}")
            return []

    def extract_single_paper_topics(self, text: str) -> List[Dict]:
        """Extract topics from a single paper by chunking it."""
        words = text.split()
        chunk_size = 300
        chunks = [
            " ".join(words[i: i + chunk_size])
            for i in range(0, min(len(words), 2400), chunk_size)
        ]
        if not chunks:
            return []
        return self.extract_topics_lda(chunks, n_top_words=6)

    def keyword_frequency(self, text: str, top_n: int = 20) -> List[Dict]:
        """Compute keyword frequency in text."""
        tokens = word_tokenize(text.lower())
        freq: Dict[str, int] = {}
        for token in tokens:
            if token.isalpha() and token not in self.stop_words and len(token) > 3:
                freq[token] = freq.get(token, 0) + 1

        sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [{"keyword": kw, "frequency": cnt} for kw, cnt in sorted_freq]

    def year_topic_trends(self, papers: List[Dict]) -> Dict:
        """
        Analyze topic trends over time.
        papers: [{"year": int, "text": str, "title": str}]
        """
        if not papers:
            return {}

        # Group texts by year
        year_texts: Dict[int, List[str]] = {}
        for p in papers:
            year = p.get("year", 0)
            if year:
                year_texts.setdefault(year, []).append(p.get("text", ""))

        trend_data = {}
        for year, texts in sorted(year_texts.items()):
            combined = " ".join(texts)
            tokens = word_tokenize(combined.lower())
            freq: Dict[str, int] = {}
            for token in tokens:
                if token.isalpha() and token not in self.stop_words and len(token) > 3:
                    freq[token] = freq.get(token, 0) + 1
            top_5 = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]
            trend_data[str(year)] = [kw for kw, _ in top_5]

        return trend_data
