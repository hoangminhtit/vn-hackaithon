from typing import Dict

from domains.common import lexical_best_choice


def solve(processed: Dict) -> str:
    return lexical_best_choice(processed["question"], processed["choices"])
