import re
from typing import Dict


def normalize_for_match(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def lexical_best_choice(question: str, choices: Dict[str, str], context: str = "") -> str:
    qn = normalize_for_match(f"{question} {context}")
    q_tokens = set(qn.split())
    if not q_tokens:
        return next(iter(choices.keys()))

    best_label = next(iter(choices.keys()))
    best_score = -1
    for label, choice in choices.items():
        c_tokens = set(normalize_for_match(choice).split())
        overlap = len(q_tokens & c_tokens)
        bonus = 1 if any(tok in qn for tok in c_tokens if len(tok) > 5) else 0
        score = overlap + bonus
        if score > best_score:
            best_score = score
            best_label = label
    return best_label
