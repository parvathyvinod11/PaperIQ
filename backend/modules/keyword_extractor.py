"""
Keyword & Entity Extraction Module
Uses TF-IDF, RAKE, and KeyBERT for keyword extraction.
"""

import re
from typing import List, Dict, Tuple
from collections import Counter

import nltk
from nltk.corpus import stopwords
from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer
from keybert import KeyBERT

for pkg in ["stopwords", "punkt", "punkt_tab"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass


# Common research entities categories
ALGORITHM_PATTERNS = [
    r'\b(?:CNN|RNN|LSTM|GRU|BERT|GPT|ResNet|VGG|Transformer|SVM|Random Forest|XGBoost|'
    r'KNN|K-means|DBSCAN|PCA|GAN|VAE|YOLO|U-Net|EfficientNet|ViT)\b',
]
DATASET_PATTERNS = [
    r'\b(?:ImageNet|CIFAR|MNIST|COCO|Pascal VOC|SQuAD|GLUE|SuperGLUE|'
    r'WikiText|Penn Treebank|Yelp|Amazon Reviews|MS-COCO|KITTI|NuScenes)\b',
]
METRIC_PATTERNS = [
    r'\b(?:accuracy|precision|recall|F1[\s\-]score|AUC|ROC|BLEU|ROUGE|BERTScore|'
    r'mAP|IoU|PSNR|SSIM|perplexity|MAE|RMSE|MSE|R²)\b',
]


class KeywordExtractor:
    def __init__(self):
        self.stop_words = list(stopwords.words("english"))
        self._kw_model = None  # Lazy load

    @property
    def kw_model(self):
        if self._kw_model is None:
            self._kw_model = KeyBERT()
        return self._kw_model

    def extract_tfidf_keywords(self, texts: List[str], top_n: int = 20) -> List[Tuple[str, float]]:
        """Extract keywords using TF-IDF across multiple texts."""
        if not texts or all(len(t.strip()) == 0 for t in texts):
            return []
        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=500,
            )
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.sum(axis=0).A1
            keyword_scores = sorted(
                zip(feature_names, scores), key=lambda x: x[1], reverse=True
            )
            return keyword_scores[:top_n]
        except Exception as e:
            print(f"TF-IDF failed: {e}")
            return []

    def extract_rake_keywords(self, text: str, top_n: int = 15) -> List[Tuple[str, float]]:
        """Extract keywords using RAKE."""
        try:
            rake = Rake(stopwords=self.stop_words, min_length=1, max_length=4)
            rake.extract_keywords_from_text(text[:5000])
            phrases = rake.get_ranked_phrases_with_scores()
            return [(phrase, score) for score, phrase in phrases[:top_n]]
        except Exception as e:
            print(f"RAKE failed: {e}")
            return []

    def extract_keybert_keywords(self, text: str, top_n: int = 15) -> List[Tuple[str, float]]:
        """Extract keywords using KeyBERT (semantic)."""
        try:
            keywords = self.kw_model.extract_keywords(
                text[:3000],
                keyphrase_ngram_range=(1, 2),
                stop_words="english",
                top_n=top_n,
            )
            return keywords
        except Exception as e:
            print(f"KeyBERT failed: {e}")
            return []

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract research-specific entities using regex patterns."""
        entities = {
            "algorithms": [],
            "datasets": [],
            "metrics": [],
        }

        for pat in ALGORITHM_PATTERNS:
            matches = re.findall(pat, text, re.IGNORECASE)
            entities["algorithms"].extend(matches)

        for pat in DATASET_PATTERNS:
            matches = re.findall(pat, text, re.IGNORECASE)
            entities["datasets"].extend(matches)

        for pat in METRIC_PATTERNS:
            matches = re.findall(pat, text, re.IGNORECASE)
            entities["metrics"].extend(matches)

        # Deduplicate (case-insensitive)
        for key in entities:
            seen = set()
            deduped = []
            for item in entities[key]:
                if item.lower() not in seen:
                    seen.add(item.lower())
                    deduped.append(item)
            entities[key] = deduped

        return entities

    def extract_all(self, text: str) -> Dict:
        """Run all keyword extraction methods."""
        rake_kw = self.extract_rake_keywords(text)
        tfidf_kw = self.extract_tfidf_keywords([text])
        keybert_kw = self.extract_keybert_keywords(text)
        entities = self.extract_entities(text)

        # Merge and deduplicate top keywords
        all_kw: Dict[str, float] = {}
        for kw, score in rake_kw:
            all_kw[kw.lower()] = all_kw.get(kw.lower(), 0) + score * 0.3
        for kw, score in tfidf_kw:
            all_kw[kw.lower()] = all_kw.get(kw.lower(), 0) + score * 0.4
        for kw, score in keybert_kw:
            all_kw[kw.lower()] = all_kw.get(kw.lower(), 0) + score * 0.3

        merged = sorted(all_kw.items(), key=lambda x: x[1], reverse=True)[:25]

        return {
            "top_keywords": [{"keyword": kw, "score": round(sc, 4)} for kw, sc in merged],
            "rake_keywords": [{"keyword": kw, "score": round(sc, 4)} for kw, sc in rake_kw],
            "tfidf_keywords": [{"keyword": kw, "score": round(sc, 4)} for kw, sc in tfidf_kw],
            "keybert_keywords": [{"keyword": kw, "score": round(sc, 4)} for kw, sc in keybert_kw],
            "entities": entities,
        }
