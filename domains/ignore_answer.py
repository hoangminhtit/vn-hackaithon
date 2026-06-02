from typing import Dict

from domains.common import lexical_best_choice, normalize_for_match


NONE_HINTS = (
    "không thể cung cấp thông tin",
    "không có đáp án nào đúng",
    "tất cả đều sai",
    "none of the above",
)


def solve(processed: Dict) -> str:
    choices = processed["choices"]
    for label, text in choices.items():
        normalized = normalize_for_match(text)
        if any(h in normalized for h in NONE_HINTS):
            return label
    return lexical_best_choice(processed["question"], choices)
