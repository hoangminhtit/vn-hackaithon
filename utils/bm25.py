import math
import re
from collections import Counter
from typing import Dict, List, Optional

from utils.config import rag_bm25_max_chars, rag_bm25_top_k


def _tokenize(text: str) -> List[str]:
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return [t for t in cleaned.split() if t]


def _split_sentences(passage: str) -> List[str]:
    chunks = re.split(r"(?<=[.!?])\s+|(?<=\n)", passage)
    return [c.strip() for c in chunks if len(c.strip()) > 20]


def bm25_retrieve(passage: str, question: str, top_k: Optional[int] = None, max_chars: Optional[int] = None) -> str:
    top_k = top_k if top_k is not None else rag_bm25_top_k()
    max_chars = max_chars if max_chars is not None else rag_bm25_max_chars()
    sentences = _split_sentences(passage)
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
