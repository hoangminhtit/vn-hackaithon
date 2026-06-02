from typing import Dict

from domains.common import lexical_best_choice, normalize_for_match


NEGATIVE_HINTS = ("sai", "không đúng", "không chính xác")
POSITIVE_HINTS = ("đúng", "chính xác")


def solve(processed: Dict) -> str:
    question = normalize_for_match(processed["question"])
    choices = processed["choices"]

    want_false = any(h in question for h in NEGATIVE_HINTS)
    want_true = any(h in question for h in POSITIVE_HINTS)

    if want_false or want_true:
        for label, text in choices.items():
            t = normalize_for_match(text)
            if want_false and any(h in t for h in NEGATIVE_HINTS):
                return label
            if want_true and any(h in t for h in POSITIVE_HINTS):
                return label

    return lexical_best_choice(processed["question"], choices)
