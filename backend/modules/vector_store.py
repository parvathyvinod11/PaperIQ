"""
VectorStore – PaperIQ
=====================
Wraps FAISS IndexFlatIP (cosine similarity via L2-normalised inner product)
with transparent on-disk persistence.

Each paper is identified by a SHA-256 hash of its full corpus text, so the
same paper is never re-embedded across restarts.

Storage layout (under VECTOR_STORE_PATH, default ./vector_store/):
    <paper_id>.faiss        – FAISS binary index
    <paper_id>.chunks.pkl   – pickled list[str] of chunk strings

Usage
-----
    vs = VectorStore()

    paper_id = vs.paper_id_from_text(full_corpus)

    if not vs.has(paper_id):
        vs.build(paper_id, chunks, embeddings)   # numpy float32 (N×D)

    index, chunks = vs.load(paper_id)
    D, I = vs.search(paper_id, query_emb, top_k=5)
"""

from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


class VectorStore:
    """
    Singleton FAISS-backed vector store with disk persistence.

    Indexes are keyed by SHA-256(corpus_text) so the same paper is never
    re-embedded across process restarts.
    """

    _instance: Optional["VectorStore"] = None

    def __new__(cls) -> "VectorStore":
        if cls._instance is None:
            obj = super().__new__(cls)
            obj._initialized = False
            cls._instance = obj
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        # Resolve storage directory from env (fallback: ./vector_store next to main.py)
        raw_path = os.getenv("VECTOR_STORE_PATH", "")
        if raw_path:
            self.store_dir = Path(raw_path)
        else:
            # Resolve relative to the backend directory so it always lands in
            # <project_root>/vector_store/ regardless of cwd.
            backend_dir = Path(__file__).resolve().parent.parent  # .../backend
            self.store_dir = backend_dir.parent / "vector_store"

        self.store_dir.mkdir(parents=True, exist_ok=True)
        print(f"[VectorStore] Storage: {self.store_dir}")

        # In-memory cache: paper_id → (faiss_index, chunks)
        self._cache: dict = {}

    # ── Helpers ────────────────────────────────────────────

    def _faiss_path(self, paper_id: str) -> Path:
        return self.store_dir / f"{paper_id}.faiss"

    def _chunks_path(self, paper_id: str) -> Path:
        return self.store_dir / f"{paper_id}.chunks.pkl"

    @staticmethod
    def paper_id_from_text(text: str) -> str:
        """Return a stable SHA-256 hex-digest for any corpus text."""
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

    # ── Existence check ────────────────────────────────────

    def has(self, paper_id: str) -> bool:
        """True if an index for *paper_id* exists (in-memory cache or disk)."""
        if paper_id in self._cache:
            return True
        return self._faiss_path(paper_id).exists() and self._chunks_path(paper_id).exists()

    # ── Build & persist ────────────────────────────────────

    def build(
        self,
        paper_id: str,
        chunks: List[str],
        embeddings: np.ndarray,
    ) -> None:
        """
        Create a FAISS IndexFlatIP from *embeddings*, persist to disk,
        and update the in-memory cache.

        Args:
            paper_id:   SHA-256 key for this paper.
            chunks:     List of passage strings (parallel to embeddings rows).
            embeddings: float32 array shape (N, D).
        """
        try:
            import faiss  # type: ignore
        except ImportError as e:
            print(f"[VectorStore] FAISS not available – skipping persist: {e}")
            return

        if embeddings is None or len(chunks) == 0:
            return

        # L2-normalise so inner product == cosine similarity
        vecs = embeddings.astype(np.float32).copy()
        faiss.normalize_L2(vecs)

        dim = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)

        # Persist
        faiss.write_index(index, str(self._faiss_path(paper_id)))
        with open(self._chunks_path(paper_id), "wb") as f:
            pickle.dump(chunks, f)

        # Cache
        self._cache[paper_id] = (index, chunks)
        print(f"[VectorStore] Saved index for {paper_id[:12]}… ({len(chunks)} chunks, dim={dim})")

    # ── Load from disk ─────────────────────────────────────

    def load(self, paper_id: str) -> Tuple[Optional[object], Optional[List[str]]]:
        """
        Return *(faiss_index, chunks)* for *paper_id*.

        Checks in-memory cache first, then disk.  Returns (None, None) if not found.
        """
        # In-memory cache hit
        if paper_id in self._cache:
            return self._cache[paper_id]

        # Disk hit
        if self._faiss_path(paper_id).exists() and self._chunks_path(paper_id).exists():
            try:
                import faiss  # type: ignore

                index = faiss.read_index(str(self._faiss_path(paper_id)))
                with open(self._chunks_path(paper_id), "rb") as f:
                    chunks = pickle.load(f)
                self._cache[paper_id] = (index, chunks)
                print(f"[VectorStore] Loaded index for {paper_id[:12]}… ({len(chunks)} chunks)")
                return index, chunks
            except Exception as e:
                print(f"[VectorStore] Load failed for {paper_id[:12]}…: {e}")

        return None, None

    # ── Search ─────────────────────────────────────────────

    def search(
        self,
        paper_id: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Run FAISS search for *query_embedding* against the stored index.

        Returns:
            (distances, indices) – both shape (1, top_k), or (None, None) if
            the index is not found.
        """
        try:
            import faiss  # type: ignore
        except ImportError:
            return None, None

        index, _ = self.load(paper_id)
        if index is None:
            return None, None

        q = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q)

        actual_k = min(top_k, index.ntotal)
        distances, indices = index.search(q, actual_k)
        return distances, indices

    # ── Metadata ───────────────────────────────────────────

    def info(self, paper_id: str) -> dict:
        """Return metadata about a stored index."""
        index, chunks = self.load(paper_id)
        if index is None:
            return {"paper_id": paper_id, "exists": False}
        return {
            "paper_id": paper_id,
            "exists": True,
            "total_chunks": len(chunks) if chunks else 0,
            "dim": index.d,
            "ntotal": index.ntotal,
        }

    def list_indexes(self) -> List[str]:
        """Return all paper_ids currently stored on disk."""
        return [p.stem for p in self.store_dir.glob("*.faiss")]
