"""
RAG Engine – PaperIQ
====================
Implements true Retrieval-Augmented Generation for paper Q&A.

Pipeline:
  1. Chunk    – split paper text into overlapping passages
  2. Embed    – encode every chunk with all-MiniLM-L6-v2
  3. Index    – store dense vectors in a FAISS IndexFlatIP (cosine via L2-norm)
               Indexes are persisted to disk by VectorStore and reused across
               restarts – the same paper is never re-embedded twice.
  4. Retrieve – FAISS nearest-neighbour search for the top-K chunks most
               relevant to the user's question
  5. Augment  – inject retrieved chunks into the LLM prompt

Query → Embed → VectorStore.search() → Retrieved Chunks → LLM Generate
"""

from __future__ import annotations

from typing import List, Dict, Optional, Tuple
import numpy as np


# ──────────────────────────────────────────────────────────
# Singleton RAG Engine
# ──────────────────────────────────────────────────────────

class RAGEngine:
    """
    Singleton embedding + retrieval engine backed by FAISS (via VectorStore).

    Lazy-loads the sentence-transformer model on first use.
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

    # ── Indexing (FAISS-backed) ────────────────────────────

    def build_index(
        self, paper_context: dict
    ) -> Tuple[List[str], Optional[np.ndarray]]:
        """
        Chunk the paper and compute dense embeddings for every chunk.

        On the first call for a given paper the chunks are embedded and
        persisted to disk via VectorStore.  Subsequent calls with the same
        paper return immediately (cache hit – no re-embedding).

        Returns:
            (chunks, embeddings) – embeddings may be None if the model is
            unavailable.  When the FAISS index is loaded from disk the
            embeddings array is set to None (FAISS holds the vectors itself).
        """
        from modules.vector_store import VectorStore

        vs = VectorStore()

        corpus = self.extract_corpus(paper_context)
        if not corpus.strip():
            return [], None

        paper_id = vs.paper_id_from_text(corpus)

        # ── Cache hit: index already on disk ──────────────
        if vs.has(paper_id):
            _, chunks = vs.load(paper_id)
            return chunks or [], None   # embeddings not needed – FAISS holds them

        # ── Cache miss: embed + build + persist ───────────
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
            vs.build(paper_id, chunks, embeddings)
            return chunks, embeddings
        except Exception as e:
            print(f"[RAGEngine] Embedding failed: {e}")
            return chunks, None

    # ── Retrieval (FAISS search) ───────────────────────────

    def retrieve(
        self,
        query: str,
        chunks: List[str],
        embeddings: Optional[np.ndarray],      # may be None when FAISS cache hit
        top_k: int = 5,
        min_score: float = 0.10,
        paper_context: Optional[dict] = None,  # needed to recover paper_id
    ) -> List[Dict]:
        """
        Semantic nearest-neighbour search via FAISS.

        If *embeddings* is None (FAISS cache hit) the method re-derives the
        paper_id from *paper_context* and calls VectorStore.search() directly.
        If *embeddings* is provided it encodes the query and calls FAISS too
        (vectors are already persisted).

        Args:
            query:         The user's natural-language question.
            chunks:        All passage strings from build_index().
            embeddings:    Dense matrix (N × D) – may be None on cache hit.
            top_k:         Maximum passages to return.
            min_score:     Cosine similarity floor.
            paper_context: Original paper dict – used to derive paper_id when
                           embeddings is None.

        Returns:
            List of {"chunk": str, "score": float} sorted by relevance desc.
        """
        from modules.vector_store import VectorStore

        vs = VectorStore()
        model = self._get_model()

        if model is None or not chunks:
            return []

        try:
            # Encode the query
            query_emb = model.encode([query], convert_to_numpy=True)  # (1, D)

            # Resolve paper_id
            if paper_context is not None:
                corpus = self.extract_corpus(paper_context)
                paper_id = vs.paper_id_from_text(corpus)
            elif embeddings is not None:
                # Derive paper_id from chunk text as a proxy
                corpus = " ".join(chunks)
                paper_id = vs.paper_id_from_text(corpus)
            else:
                return []

            # FAISS search
            distances, indices = vs.search(paper_id, query_emb[0], top_k=top_k)

            if distances is None or indices is None:
                # Fallback: sklearn cosine similarity (FAISS index not found)
                return self._fallback_retrieve(
                    query_emb, chunks, embeddings, top_k, min_score
                )

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(chunks):
                    continue
                score = float(dist)       # already cosine similarity (L2-normalised IP)
                if score >= min_score:
                    results.append({"chunk": chunks[idx], "score": round(score, 4)})

            return results

        except Exception as e:
            print(f"[RAGEngine] Retrieval failed: {e}")
            return []

    def _fallback_retrieve(
        self,
        query_emb: np.ndarray,
        chunks: List[str],
        embeddings: Optional[np.ndarray],
        top_k: int,
        min_score: float,
    ) -> List[Dict]:
        """Sklearn cosine-similarity fallback when FAISS index is unavailable."""
        if embeddings is None:
            return []
        try:
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim

            scores = cos_sim(query_emb, embeddings)[0]
            top_indices = scores.argsort()[::-1][:top_k]
            return [
                {"chunk": chunks[i], "score": round(float(scores[i]), 4)}
                for i in top_indices
                if scores[i] >= min_score
            ]
        except Exception as e:
            print(f"[RAGEngine] Fallback retrieval failed: {e}")
            return []

    # ── Convenience: one-shot retrieve ────────────────────

    def retrieve_for_query(
        self,
        query: str,
        paper_context: dict,
        top_k: int = 5,
    ) -> Tuple[List[Dict], int]:
        """
        Build (or load) index and return retrieved passages.

        Returns:
            (passages, total_chunks_indexed)
        """
        chunks, embeddings = self.build_index(paper_context)
        passages = self.retrieve(
            query,
            chunks,
            embeddings,
            top_k=top_k,
            paper_context=paper_context,
        )
        return passages, len(chunks)
