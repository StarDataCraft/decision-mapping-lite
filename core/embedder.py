# core/embedder.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np

# Optional: sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# Fallback: TF-IDF
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    TfidfVectorizer = None


@dataclass
class EmbeddingConfig:
    # CPU-friendly open-source model
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


class HybridEmbedder:
    """
    Preferred: local sentence-transformers embeddings (no key).
    Fallback: TF-IDF if model can't be loaded (still no key, never crashes the app).
    """
    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self.backend = "sbert"
        self.model = None
        self.tfidf = None
        self._tfidf_docs = None

        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(cfg.model_name)
                return
            except Exception:
                # fall through to TF-IDF
                self.model = None

        self.backend = "tfidf"
        if TfidfVectorizer is None:
            raise RuntimeError(
                "Neither sentence-transformers nor sklearn(TF-IDF) is available. "
                "Add 'sentence-transformers' (and torch) or 'scikit-learn' to requirements.txt."
            )

    def _embed_sbert(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)

    def _fit_tfidf(self, docs: List[str]) -> None:
        # Fit once per doc set
        self.tfidf = TfidfVectorizer(max_features=5000)
        self._tfidf_docs = self.tfidf.fit_transform(docs)

    def top_k(self, query: str, docs: List[str], k: int = 5) -> List[int]:
        if not docs:
            return []

        if self.backend == "sbert":
            q = self._embed_sbert([query])[0]
            d = self._embed_sbert(docs)
            sims = (d @ q).astype(np.float32)  # normalized => dot = cosine
            idx = np.argsort(sims)[::-1][:k]
            return [int(i) for i in idx.tolist()]

        # TF-IDF fallback
        if self.tfidf is None or self._tfidf_docs is None:
            self._fit_tfidf(docs)
        qv = self.tfidf.transform([query])
        sims = (self._tfidf_docs @ qv.T).toarray().reshape(-1)
        idx = np.argsort(sims)[::-1][:k]
        return [int(i) for i in idx.tolist()]
