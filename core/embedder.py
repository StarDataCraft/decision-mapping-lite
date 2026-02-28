# core/embedder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None


@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize: bool = True


class HybridEmbedder:
    """
    Lightweight embedder:
    - Uses SentenceTransformer to embed documents and query
    - Returns cosine similarity ranking
    - Designed for CPU-friendly usage on Streamlit Cloud
    """

    def __init__(self, cfg: EmbeddingConfig):
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Add it to requirements.txt and redeploy."
            )
        self.cfg = cfg
        self.model = SentenceTransformer(cfg.model_name)

    def _embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, show_progress_bar=False)
        vecs = np.asarray(vecs, dtype=np.float32)
        if self.cfg.normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
            vecs = vecs / norms
        return vecs

    @staticmethod
    def _cosine_scores(q: np.ndarray, mat: np.ndarray) -> np.ndarray:
        # assume normalized -> cosine = dot
        return mat @ q

    def top_k(self, query: str, docs: List[str], k: int = 8) -> List[int]:
        idxs, _ = self.top_k_with_scores(query, docs, k=k)
        return idxs

    def top_k_with_scores(self, query: str, docs: List[str], k: int = 8) -> Tuple[List[int], List[float]]:
        """
        Returns:
          - indices of top k docs
          - similarity scores (cosine)
        """
        if not docs:
            return [], []
        qv = self._embed([query])[0]
        dv = self._embed(docs)
        scores = self._cosine_scores(qv, dv)  # shape [N]
        order = np.argsort(-scores)[: min(k, len(docs))]
        idxs = order.tolist()
        scs = [float(scores[i]) for i in idxs]
        return idxs, scs
