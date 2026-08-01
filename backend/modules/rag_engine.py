"""
RAG Engine – PaperIQ
Implements true Retrieval-Augmented Generation for paper Q&A.

Pipeline:
  1. Chunk   – split paper text into overlapping passages
  2. Embed   – encode every chunk with all-MiniLM-L6-v2
  3. Retrieve – cosine-similarity search for the top-K chunks
               most relevant to the user's question
  4. Augment – inject retrieved chunks into the LLM prompt

This makes the system a genuine RAG architecture:
  Query → Embed → Vector Search → Retrieved Chunks → LLM Generate
"""

from __future__ import annotations

from typing import List, Dict, Optional, Tuple
import numpy as np


# ──────────────────────────────────────────────────────────
# Singleton RAG Engine
# ──────────────────────────────────────────────────────────

class RAGEngine:
    """
    Singleton embedding + retrieval engine built on Sentence-Transformers.

    Lazy-loads the model on first use so startup is instant.
    Shares the same model instance as SimilarityEngine (same checkpoint,
    Python caches the HuggingFace weights automatically).
    """

    _instance: Optional["RAGEngine"] = None

    def __new__(cls) -> "RAGEngine":
        if cls._instance is None:
            obj = super().__new__(cls)
            obj._model = None          # lazy
            cls._instance = obj
        return cls._instance

    # ── Model ──────────────────────────────────────────────

    def _get_model(self):
        """Lazy-load Sentence-Transformer (shared weights with SimilarityEngine)."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
                print("[RAGEngine] Loaded all-MiniLM-L6-v2")
            except Exception as e:
                print(f"[RAGEngine] SentenceTransformer unavailable: {e}")
        return self._model

    # ── Chunking ───────────────────────────────────────────

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 120,
        overlap: int = 25,
    ) -> List[str]:
        """
        Split *text* into word-level overlapping windows.

        Args:
            chunk_size: Words per chunk  (~90–120 words ≈ 1–2 paragraphs).
            overlap:    Words shared between consecutive chunks so context
                        doesn't get cut at boundaries.

        Returns:
            List of non-empty string chunks.
        """
        words = text.split()
        if not words:
            return []

        step = max(1, chunk_size - overlap)
        chunks: List[str] = []
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + chunk_size])
            if len(chunk.strip()) > 30:          # skip near-empty tail chunks
                chunks.append(chunk)
        return chunks

    # ── Text Extraction ────────────────────────────────────

    def extract_corpus(self, paper_context: dict) -> str:
        """
        Pull all meaningful text out of a paper_context dict.

        Combines section texts + summaries so retrieval covers both
        raw evidence and high-level interpretations.
        """
        parts: List[str] = []

        # Raw sections (abstract, introduction, methodology, results, …)
        sections = paper_context.get("sections", {})
        PRIORITY_ORDER = [
            "abstract", "introduction", "methodology", "methods",
            "results", "discussion", "conclusion", "related_work",
        ]
        # Add priority sections first so important content gets its own chunks
        for key in PRIORITY_ORDER:
            text = sections.get(key, "")
            if isinstance(text, str) and text.strip():
                parts.append(f"[{key.upper()}]\n{text.strip()}")

        # Any remaining sections not in the priority list
        for key, text in sections.items():
            if key in PRIORITY_ORDER or key == "full_text":
                continue
            if isinstance(text, str) and text.strip():
                parts.append(f"[{key.upper()}]\n{text.strip()}")

        # Generated summaries (act as compressed evidence)
        summaries = paper_context.get("summaries", {})
        for key, text in summaries.items():
            if isinstance(text, str) and text.strip():
                parts.append(f"[SUMMARY_{key.upper()}]\n{text.strip()}")

        return "\n\n".join(parts)

    # ── Indexing ───────────────────────────────────────────

    def build_index(
        self, paper_context: dict
    ) -> Tuple[List[str], Optional[np.ndarray]]:
        """
        Chunk the paper and compute dense embeddings for every chunk.

        Returns:
            (chunks, embeddings)  – embeddings is None if model unavailable.
        """
        corpus = self.extract_corpus(paper_context)
        if not corpus.strip():
            return [], None

        chunks = self.chunk_text(corpus)
        if not chunks:
            return [], None

        model = self._get_model()
        if model is None:
            return chunks, None

        try:
            embeddings = model.encode(
                chunks,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=32,
            )
            return chunks, embeddings
        except Exception as e:
            print(f"[RAGEngine] Embedding failed: {e}")
            return chunks, None

    # ── Retrieval ──────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        chunks: List[str],
        embeddings: np.ndarray,
        top_k: int = 5,
        min_score: float = 0.10,
    ) -> List[Dict]:
        """
        Semantic nearest-neighbour search.

        Args:
            query:      The user's natural-language question.
            chunks:     All passage strings from build_index().
            embeddings: Dense matrix (N × D) from build_index().
            top_k:      Maximum passages to return.
            min_score:  Cosine similarity floor – filters irrelevant chunks.

        Returns:
            List of {"chunk": str, "score": float} sorted by relevance desc.
        """
        model = self._get_model()
        if model is None or embeddings is None or not chunks:
            return []

        try:
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim

            query_emb = model.encode([query], convert_to_numpy=True)
            scores = cos_sim(query_emb, embeddings)[0]          # shape (N,)

            top_indices = scores.argsort()[::-1][:top_k]
            results = [
                {"chunk": chunks[i], "score": round(float(scores[i]), 4)}
                for i in top_indices
                if scores[i] >= min_score
            ]
            return results
        except Exception as e:
            print(f"[RAGEngine] Retrieval failed: {e}")
            return []

    # ── Convenience: one-shot retrieve ────────────────────

    def retrieve_for_query(
        self,
        query: str,
        paper_context: dict,
        top_k: int = 5,
    ) -> Tuple[List[Dict], int]:
        """
        Build index on-the-fly and return retrieved passages.

        Returns:
            (passages, total_chunks_indexed)
        """
        chunks, embeddings = self.build_index(paper_context)
        passages = self.retrieve(query, chunks, embeddings, top_k=top_k)
        return passages, len(chunks)
