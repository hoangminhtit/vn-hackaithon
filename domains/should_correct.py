from typing import Dict

from domains.common import lexical_best_choice
from prompts import question_polarity


def solve(processed: Dict) -> str:
    """Giải quyết câu hỏi thuộc should_correct domain bằng heuristic.

    Dựa vào prompts.question_polarity để xác định hướng phát biểu đúng/sai.
    """
    polarity = question_polarity(processed["question"])
    choices = processed["choices"]

    if polarity == "false":
        for label, text in choices.items():
            t = text.lower()
            if any(h in t for h in ("sai", "không đúng", "không chính xác", "không phải")):
                return label
    elif polarity == "true":
        for label, text in choices.items():
            t = text.lower()
            if any(h in t for h in ("đúng", "chính xác")) and "không" not in t:
                return label

    return lexical_best_choice(processed["question"], choices)
