# core/reranker.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import re


@dataclass
class RerankConfig:
    # weights: keep them simple and stable
    w_embed: float = 0.78
    w_lex: float = 0.22
    # small biases to avoid "Safeguards (none)"
    type_bias: Dict[str, float] = None

    def __post_init__(self):
        if self.type_bias is None:
            self.type_bias = {"safeguard": 0.06, "move": 0.02, "reframe": 0.0}


def _normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _char_ngrams(s: str, n: int = 3) -> List[str]:
    s = _normalize_text(s)
    s = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", s)  # keep alnum + CJK
    if len(s) < n:
        return [s] if s else []
    return [s[i : i + n] for i in range(0, len(s) - n + 1)]


def _lex_overlap_score(q: str, d: str) -> float:
    """
    Ultra-light lexical overlap:
    - character n-gram Jaccard (n=3)
    - Works for CN/EN without extra tokenizers
    """
    qg = set(_char_ngrams(q, 3))
    dg = set(_char_ngrams(d, 3))
    if not qg or not dg:
        return 0.0
    inter = len(qg & dg)
    union = len(qg | dg) + 1e-9
    return float(inter / union)


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    embed_scores: List[float],
    cfg: RerankConfig,
) -> List[Tuple[Dict[str, Any], float]]:
    """
    candidates: list of playbook items
    embed_scores: same length, cosine similarities from embedder
    returns: list of (item, final_score) sorted desc
    """
    assert len(candidates) == len(embed_scores)

    out = []
    for item, s_embed in zip(candidates, embed_scores):
        # build a compact doc text for lexical overlap
        doc_text = "\n".join(
            [
                item.get("title", ""),
                item.get("text", ""),
                " ".join(item.get("tags", []) or []),
                item.get("type", ""),
            ]
        )
        s_lex = _lex_overlap_score(query, doc_text)

        bias = cfg.type_bias.get(item.get("type", ""), 0.0)
        score = cfg.w_embed * float(s_embed) + cfg.w_lex * float(s_lex) + bias
        out.append((item, score))

    out.sort(key=lambda x: x[1], reverse=True)
    return out
