import re
from typing import Dict, Optional

from domains.common import lexical_best_choice, normalize_for_match


NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+")


def _extract_number(text: str) -> Optional[float]:
    match = NUMBER_RE.search(text.replace(",", "."))
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def _pick_closest_numeric_choice(choices: Dict[str, str], target: float) -> Optional[str]:
    best_label = None
    best_diff = float("inf")
    for label, text in choices.items():
        val = _extract_number(text)
        if val is None:
            continue
        diff = abs(val - target)
        if diff < best_diff:
            best_diff = diff
            best_label = label
    return best_label


def _solve_linear_decay(question: str, choices: Dict[str, str]) -> Optional[str]:
    qn = normalize_for_match(question)
    if "db dt" in qn and "-k b" in qn and "b0" in qn:
        for label, text in choices.items():
            tn = normalize_for_match(text)
            if "e" in tn and "-kt" in tn and "b0" in tn:
                return label
    return None


def _solve_midpoint_elasticity(question: str, choices: Dict[str, str]) -> Optional[str]:
    qn = normalize_for_match(question)
    if "co giãn" not in qn:
        return None

    nums = [float(n) for n in NUMBER_RE.findall(question.replace(",", "."))]
    if len(nums) < 4:
        return None
    p1, q1, p2, q2 = nums[0], nums[1], nums[2], nums[3]
    dq_over_q = (q2 - q1) / ((q1 + q2) / 2)
    dp_over_p = (p2 - p1) / ((p1 + p2) / 2)
    if dp_over_p == 0:
        return None
    elasticity = abs(dq_over_q / dp_over_p)
    return _pick_closest_numeric_choice(choices, elasticity)


def solve(processed: Dict) -> str:
    question = processed["question"]
    choices = processed["choices"]

    for solver in (_solve_linear_decay, _solve_midpoint_elasticity):
        result = solver(question, choices)
        if result:
            return result

    return lexical_best_choice(question, choices)
