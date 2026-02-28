# core/embedder.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


@dataclass
class EmbeddingConfig:
    # 轻量、CPU 友好：适合 Streamlit Cloud
    # 你也可以换成更强但更重的，例如 "BAAI/bge-m3"（通常不建议云端CPU）
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


class LocalEmbedder:
    def __init__(self, cfg: EmbeddingConfig):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not available.")
        self.cfg = cfg
        self.model = SentenceTransformer(cfg.model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)

    def top_k(self, query: str, docs: List[str], k: int = 5) -> List[int]:
        q = self.embed([query])[0]
        d = self.embed(docs)
        sims = [float(np.dot(q, d[i])) for i in range(len(docs))]  # normalized => dot = cosine
        idx = np.argsort(sims)[::-1][:k]
        return [int(i) for i in idx.tolist()]
