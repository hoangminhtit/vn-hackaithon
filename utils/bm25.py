import math
import os
import re
from collections import Counter
from typing import Dict, List


def _tokenize(text: str) -> List[str]:
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return [t for t in cleaned.split() if t]


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(os.getenv(name, str(default)))))
    except ValueError:
        return default


def _split_sentences(passage: str) -> List[str]:
    chunks = re.split(r"(?<=[.!?])\s+|(?<=\n)", passage)
    return [c.strip() for c in chunks if len(c.strip()) > 20]


def chunk_passage_words(
    passage: str,
    chunk_words: int | None = None,
    overlap_words: int | None = None,
) -> List[str]:
    """Sliding word windows for documents; query stays separate (BM25 over chunks)."""
    chunk_words = chunk_words if chunk_words is not None else _env_int("RAG_CHUNK_WORDS", 120, 40, 400)
    overlap_words = overlap_words if overlap_words is not None else _env_int("RAG_CHUNK_OVERLAP", 40, 25, 50)
    overlap_words = min(overlap_words, chunk_words - 1)
    words = passage.split()
    if len(words) <= chunk_words:
        return [passage.strip()] if passage.strip() else []

    step = max(1, chunk_words - overlap_words)
    chunks: List[str] = []
    for start in range(0, len(words), step):
        piece = " ".join(words[start : start + chunk_words]).strip()
        if len(piece) > 20:
            chunks.append(piece)
        if start + chunk_words >= len(words):
            break
    return chunks or [passage.strip()]


def _passage_units(passage: str) -> List[str]:
    mode = os.getenv("RAG_CHUNK_MODE", "overlap").strip().lower()
    if mode == "sentence":
        return _split_sentences(passage)
    return chunk_passage_words(passage)


def bm25_retrieve(passage: str, question: str, top_k: int = 12, max_chars: int = 5500) -> str:
    top_k = _env_int("RAG_BM25_TOP_K", top_k, 1, 30)
    max_chars = _env_int("RAG_BM25_MAX_CHARS", max_chars, 1000, 12000)
    sentences = _passage_units(passage)
    if not sentences:
        return passage[:max_chars]

    tokenized = [_tokenize(s) for s in sentences]
    query_tokens = _tokenize(question)
    if not query_tokens:
        return " ".join(sentences[:top_k])[:max_chars]

    n_docs = len(tokenized)
    avg_len = sum(len(t) for t in tokenized) / max(n_docs, 1)
    k1 = 1.5
    b = 0.75

    df: Dict[str, int] = {}
    for doc in tokenized:
        for token in set(doc):
            df[token] = df.get(token, 0) + 1

    idf = {
        token: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
        for token, freq in df.items()
    }

    scores: List[float] = []
    for doc in tokenized:
        freq = Counter(doc)
        dl = len(doc)
        score = 0.0
        for q in query_tokens:
            if q not in freq:
                continue
            tf = freq[q]
            denom = tf + k1 * (1 - b + b * dl / max(avg_len, 1e-6))
            score += idf.get(q, 0.0) * (tf * (k1 + 1) / max(denom, 1e-6))
        scores.append(score)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    top_indices.sort()
    context = " ".join(sentences[i] for i in top_indices)
    return context[:max_chars]
