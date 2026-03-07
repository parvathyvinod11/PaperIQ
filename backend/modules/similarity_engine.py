"""
Semantic Similarity Engine
Computes cosine similarity between papers using sentence-transformers.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class SimilarityEngine:
    def __init__(self):
        self._model = None  # Lazy load sentence-transformers
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2),
        )

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                print(f"SentenceTransformer not available: {e}")
        return self._model

    def _truncate(self, text: str, max_chars: int = 5000) -> str:
        return text[:max_chars]

    def compute_semantic_similarity(
        self, text1: str, text2: str
    ) -> float:
        """Compute semantic similarity using sentence-transformers."""
        model = self._get_model()
        if model is None:
            return self.compute_tfidf_similarity(text1, text2)
        try:
            emb1 = model.encode(self._truncate(text1), convert_to_numpy=True)
            emb2 = model.encode(self._truncate(text2), convert_to_numpy=True)
            sim = cosine_similarity([emb1], [emb2])[0][0]
            return round(float(sim), 4)
        except Exception:
            return self.compute_tfidf_similarity(text1, text2)

    def compute_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Fallback: TF-IDF cosine similarity."""
        try:
            vecs = self.tfidf_vectorizer.fit_transform([text1, text2])
            sim = cosine_similarity(vecs[0], vecs[1])[0][0]
            return round(float(sim), 4)
        except Exception:
            return 0.0

    def get_paper_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding vector for a paper."""
        model = self._get_model()
        if model is None:
            return None
        try:
            return model.encode(self._truncate(text), convert_to_numpy=True)
        except Exception:
            return None

    def batch_similarity(
        self, query_text: str, corpus_texts: List[str]
    ) -> List[float]:
        """Compute similarity between query and a corpus of texts."""
        model = self._get_model()
        if model is None:
            return [self.compute_tfidf_similarity(query_text, t) for t in corpus_texts]
        try:
            query_emb = model.encode(self._truncate(query_text), convert_to_numpy=True)
            corpus_embs = model.encode(
                [self._truncate(t) for t in corpus_texts], convert_to_numpy=True
            )
            sims = cosine_similarity([query_emb], corpus_embs)[0]
            return [round(float(s), 4) for s in sims]
        except Exception:
            return [self.compute_tfidf_similarity(query_text, t) for t in corpus_texts]

    def compare_papers(self, papers: List[Dict]) -> Dict:
        """
        Compare multiple papers pairwise.
        papers: [{"id": str, "text": str, "title": str}]
        """
        n = len(papers)
        if n < 2:
            return {"matrix": [], "pairs": []}

        texts = [p["text"] for p in papers]
        ids = [p.get("title", f"Paper {i+1}") for i, p in enumerate(papers)]

        # Build similarity matrix
        model = self._get_model()
        if model:
            try:
                embeddings = model.encode(
                    [self._truncate(t) for t in texts], convert_to_numpy=True
                )
                matrix = cosine_similarity(embeddings)
            except Exception:
                matrix = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        matrix[i][j] = self.compute_tfidf_similarity(texts[i], texts[j])
        else:
            matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    matrix[i][j] = self.compute_tfidf_similarity(texts[i], texts[j])

        # Build pair list
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append({
                    "paper1": ids[i],
                    "paper2": ids[j],
                    "similarity": round(float(matrix[i][j]), 4),
                })

        return {
            "labels": ids,
            "matrix": matrix.tolist(),
            "pairs": sorted(pairs, key=lambda x: x["similarity"], reverse=True),
        }
