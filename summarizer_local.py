# summarizer_local.py
# Fast, dependency-light extractive summarizer for Streamlit Cloud.
# No Hugging Face / Transformers — uses scikit-learn TF-IDF to rank sentences.

from __future__ import annotations
from typing import List, Tuple
import re
import math

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def _clean_text(t: str) -> str:
    t = (t or "").replace("\r", " ").replace("\n", " ").strip()
    t = re.sub(r"\s+", " ", t)
    return t


def _split_sentences(text: str) -> List[str]:
    """Simple, fast sentence splitter (no NLTK download)."""
    text = _clean_text(text)
    # Split at ., !, ? followed by space or end; keep punctuation
    parts = re.split(r"(?<=[.!?])\s+", text)
    sents = [s.strip() for s in parts if s and len(s.strip()) > 2]
    # De-dup consecutive duplicates that come from wire copy
    deduped = []
    prev = None
    for s in sents:
        if s != prev:
            deduped.append(s)
        prev = s
    return deduped


def _post_filter_company(sents: List[str], company: str | None) -> List[str]:
    if not company:
        return sents
    key = company.lower()
    picked = [s for s in sents if key in s.lower()]
    return picked or sents  # if filtering removes everything, fall back to all


def _rank_sentences(sents: List[str], top_k: int, bias_terms: List[str] | None = None) -> List[Tuple[int, float]]:
    """
    Rank sentences with TF-IDF. Returns list of (index, score).
    bias_terms: optional keywords to boost (e.g., company name).
    """
    if not sents:
        return []

    # Short-circuit tiny inputs
    if len(sents) <= top_k:
        return [(i, 1.0) for i in range(len(sents))]

    # Vectorize sentences
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_df=0.9,
        min_df=1,
    )
    X = vec.fit_transform(sents)  # shape: (num_sents, vocab)

    # Sentence salience = L2 norm of TF-IDF row (proxy for info density)
    norms = np.sqrt((X.power(2)).sum(axis=1)).A1  # vector of length num_sents

    # Optional: boost sentences containing bias terms
    if bias_terms:
        boost = np.ones_like(norms)
        for i, s in enumerate(sents):
            low = s.lower()
            for term in bias_terms:
                if term in low:
                    boost[i] += 0.15  # small additive boost per term hit
        norms = norms * boost

    # Light position prior: earlier sentences slightly favored
    # (news articles often front-load important facts)
    n = len(sents)
    position_weight = np.array([1.0 + 0.15 * (1.0 - (i / max(1, n - 1))) for i in range(n)])
    scores = norms * position_weight

    # Get top_k by score; keep original order in the final list
    idx_sorted = np.argsort(-scores)[:top_k]
    idx_sorted = sorted(idx_sorted.tolist())  # preserve article flow
    return [(i, float(scores[i])) for i in idx_sorted]


def _to_bullets(selected: List[str], max_bullets: int) -> str:
    lines = []
    for s in selected[:max_bullets]:
        s = s.strip().rstrip(".")
        if len(s) > 300:  # keep bullets compact
            s = s[:297] + "…"
        lines.append(f"• {s}.")
    return "\n".join(lines) if lines else "No concise summary could be generated."


def summarize_chunks_local(
    chunks: List[str],
    bullets: int = 8,
    company: str | None = None,
) -> str:
    """
    HF-free extractive summarizer.
    - Cleans and splits text into sentences.
    - Ranks by TF-IDF salience (+ small position prior).
    - Biases toward sentences containing the company name if provided.
    - Returns bullet list.
    """
    if not chunks:
        return "No content available to summarize."

    # Concatenate a few chunks for speed (avoid huge inputs)
    joined = " ".join(_clean_text(c) for c in chunks[:6] if c and c.strip())
    if not joined:
        return "No content available to summarize."

    sents = _split_sentences(joined)
    sents = _post_filter_company(sents, company)

    # Target ~bullets*2 candidates to choose from (then we’ll cap to bullets)
    top_k = max(bullets * 2, bullets)
    bias = [company.lower()] if company else None

    try:
        ranked = _rank_sentences(sents, top_k=top_k, bias_terms=bias)
        picked = [sents[i] for i, _ in ranked]
    except Exception:
        # absolute fallback: first few sentences
        picked = sents[:top_k]

    return _to_bullets(picked, max_bullets=bullets)
