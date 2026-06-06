import os


def env_int(name: str, default: int, minimum: int = 1, maximum: int | None = None) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def rag_max_context_chars() -> int:
    """Max chars of passage/context injected into the LLM user prompt."""
    return env_int("RAG_MAX_CONTEXT_CHARS", 12000, 1000, 32000)


def rag_full_passage_chars() -> int:
    """Passage shorter than this is sent in full; longer passages use BM25 retrieve."""
    return env_int("RAG_FULL_PASSAGE_CHARS", 12000, 1000, 32000)


def rag_bm25_max_chars() -> int:
    return env_int("RAG_BM25_MAX_CHARS", 10000, 1000, 32000)


def rag_bm25_top_k() -> int:
    return env_int("RAG_BM25_TOP_K", 12, 1, 50)
