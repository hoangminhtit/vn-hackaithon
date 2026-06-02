from typing import Dict

from domains.common import lexical_best_choice, normalize_for_match


WANT_FALSE_HINTS = ("sai", "không đúng", "không chính xác", "nào sai", "ý sai")
WANT_TRUE_HINTS = ("đúng", "chính xác", "nào đúng", "ý đúng")


def solve(processed: Dict) -> str:
    question = normalize_for_match(processed["question"])
    choices = processed["choices"]

    want_false = any(h in question for h in WANT_FALSE_HINTS)
    want_true = any(h in question for h in WANT_TRUE_HINTS)

    if want_false or want_true:
        for label, text in choices.items():
            t = normalize_for_match(text)
            if want_false and any(h in t for h in WANT_FALSE_HINTS):
                return label
            if want_true and any(h in t for h in WANT_TRUE_HINTS):
                return label

    return lexical_best_choice(processed["question"], choices)
