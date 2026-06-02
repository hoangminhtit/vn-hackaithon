import re
from typing import Dict, Optional

from domains.common import lexical_best_choice, normalize_for_match


NUMBER_RE = re.compile(r"[-+]?\d*[.,]?\d+")


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
    mid_q = (q1 + q2) / 2
    mid_p = (p1 + p2) / 2
    if mid_p == 0 or mid_q == 0:
        return None
    dq_over_q = (q2 - q1) / mid_q
    dp_over_p = (p2 - p1) / mid_p
    if dp_over_p == 0:
        return None
    elasticity = abs(dq_over_q / dp_over_p)
    return _pick_closest_numeric_choice(choices, elasticity)


def _solve_gdp_deflator(question: str, choices: Dict[str, str]) -> Optional[str]:
    qn = question.lower()
    has_nominal = "danh nghĩa" in qn or "nominal" in qn
    has_real = "thực tế" in qn or "thực" in qn or "real" in qn
    has_gdp = "gdp" in qn
    if not (has_gdp and has_nominal and has_real):
        return None

    nums = [float(n) for n in NUMBER_RE.findall(question.replace(",", "."))]
    if len(nums) < 2:
        return None

    large_nums = [n for n in nums if n > 10]
    if len(large_nums) < 2:
        return None

    gdp_nominal = large_nums[0]
    gdp_real = large_nums[1]
    if gdp_real == 0:
        return None

    deflator = (gdp_nominal / gdp_real) * 100
    inflation = deflator - 100.0

    result = _pick_closest_numeric_choice(choices, inflation)
    if result:
        return result
    return _pick_closest_numeric_choice(choices, deflator)


def solve(processed: Dict) -> str:
    question = processed["question"]
    choices = processed["choices"]

    solvers = [
        _solve_linear_decay,
        _solve_midpoint_elasticity,
        _solve_gdp_deflator,
    ]

    for solver in solvers:
        result = solver(question, choices)
        if result:
            return result

    return lexical_best_choice(question, choices)
