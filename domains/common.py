import re
from typing import Dict, List, Set


def normalize_for_match(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> List[str]:
    return normalize_for_match(text).split()


def lexical_best_choice(question: str, choices: Dict[str, str], context: str = "") -> str:
    combined = f"{question} {context}"
    qn = normalize_for_match(combined)
    q_tokens = set(qn.split())
    if not q_tokens:
        return next(iter(choices.keys()))

    best_label = next(iter(choices.keys()))
    best_score = -1

    for label, text in choices.items():
        c_tokens = set(normalize_for_match(text).split())
        overlap = len(q_tokens & c_tokens)
        bonus = 1 if any(tok in qn for tok in c_tokens if len(tok) > 5) else 0
        score = overlap + bonus
        if score > best_score:
            best_score = score
            best_label = label

    return best_label


def ngram_best_choice(question: str, choices: Dict[str, str], context: str = "", n_range: tuple = (3, 5)) -> str:
    source_text = normalize_for_match(f"{context} {question}")
    source_tokens = source_text.split()

    source_ngrams: Set[str] = set()
    for n in range(n_range[0], n_range[1] + 1):
        for i in range(len(source_tokens) - n + 1):
            source_ngrams.add(" ".join(source_tokens[i:i + n]))

    if not source_ngrams:
        return next(iter(choices.keys()))

    best_label = next(iter(choices.keys()))
    best_score = 0

    for label, text in choices.items():
        c_tokens = normalize_for_match(text).split()
        score = 0
        for n in range(n_range[0], n_range[1] + 1):
            for i in range(len(c_tokens) - n + 1):
                ngram = " ".join(c_tokens[i:i + n])
                if ngram in source_ngrams:
                    score += n * n
        if score > best_score:
            best_score = score
            best_label = label

    return best_label
