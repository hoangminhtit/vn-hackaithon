import re
from typing import Dict, Optional

from domains.common import lexical_best_choice, normalize_for_match


NUMBER_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)*")


def _parse_vn_number(raw: str) -> Optional[float]:
    s = raw.strip()
    if not s:
        return None
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    elif re.fullmatch(r"\d{1,3}(?:\.\d{3})+", s):
        s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return None


def _extract_number(text: str) -> Optional[float]:
    match = NUMBER_RE.search(text)
    if not match:
        return None
    return _parse_vn_number(match.group())


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

    nums = [_parse_vn_number(n) for n in NUMBER_RE.findall(question)]
    nums = [n for n in nums if n is not None]
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

    nums = [_parse_vn_number(n) for n in NUMBER_RE.findall(question)]
    nums = [n for n in nums if n is not None]
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


def _eval_linear_eq(eq_str: str, P: float) -> Optional[float]:
    eq = eq_str.lower().replace(" ", "").replace("$", "")
    match = re.match(r"(-?\d+)\s*([-+])\s*(\d+)\*?p", eq)
    if match:
        const = float(match.group(1))
        sign = 1.0 if match.group(2) == "+" else -1.0
        coeff = float(match.group(3))
        return const + sign * coeff * P
    match2 = re.match(r"(-?\d+)\*?p\s*([-+])\s*(\d+)", eq)
    if match2:
        coeff = float(match2.group(1))
        sign = 1.0 if match2.group(2) == "+" else -1.0
        const = float(match2.group(3))
        return coeff * P + sign * const
    match_single_p = re.match(r"(-?\d+)\*?p", eq)
    if match_single_p:
        return float(match_single_p.group(1)) * P
    try:
        return float(eq)
    except ValueError:
        pass
    return None


def _solve_supply_demand(question: str, choices: Dict[str, str]) -> Optional[str]:
    qn = question.lower().replace(" ", "").replace("$", "")
    if "cầu" not in question.lower() or "cung" not in question.lower():
        return None
        
    qd_match = re.search(r"q_?d\s*=\s*([0-9p\+\-\*]+)", qn)
    qs_match = re.search(r"q_?s\s*=\s*([0-9p\+\-\*]+)", qn)
    if not qd_match or not qs_match:
        return None
        
    qd_eq = qd_match.group(1)
    qs_eq = qs_match.group(1)
    
    p_match = re.search(r"p\s*=\s*(\d+(?:\.\d+)?)", qn)
    if not p_match:
        return None
    P = float(p_match.group(1))
    
    Qd = _eval_linear_eq(qd_eq, P)
    Qs = _eval_linear_eq(qs_eq, P)
    if Qd is None or Qs is None:
        return None
        
    if "thiếu hụt" in question.lower():
        diff = max(0.0, Qd - Qs)
        return _pick_closest_numeric_choice(choices, diff)
    elif "dư thừa" in question.lower() or "thặng dư" in question.lower():
        diff = max(0.0, Qs - Qd)
        return _pick_closest_numeric_choice(choices, diff)
        
    return None


def _solve_money_multiplier(question: str, choices: Dict[str, str]) -> Optional[str]:
    qn = question.lower()
    if "số nhân tiền" not in qn and "số nhân của tiền" not in qn:
        return None
    pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", qn)
    if pct_match:
        r = float(pct_match.group(1)) / 100.0
    else:
        nums = [float(x) for x in re.findall(r"0\.\d+", qn)]
        if nums:
            r = nums[0]
        else:
            return None
    if r == 0:
        return None
    m = 1.0 / r
    return _pick_closest_numeric_choice(choices, m)


def _solve_effective_annual_rate(question: str, choices: Dict[str, str]) -> Optional[str]:
    qn = question.lower()
    if "lãi suất" not in qn or ("hiệu quả" not in qn and "thực tế" not in qn):
        return None
    pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", qn)
    if not pct_match:
        return None
    r = float(pct_match.group(1)) / 100.0
    
    n = None
    if "bán niên" in qn or "nửa năm" in qn:
        n = 2
    elif "hàng quý" in qn or "mỗi quý" in qn:
        n = 4
    elif "hàng tháng" in qn or "mỗi tháng" in qn:
        n = 12
    elif "hàng năm" in qn:
        n = 1
        
    if n is None:
        return None
        
    ear = (1.0 + r / n) ** n - 1.0
    return _pick_closest_numeric_choice(choices, ear * 100.0)


def _solve_inventory_turnover(question: str, choices: Dict[str, str]) -> Optional[str]:
    qn = question.lower()
    if "hàng tồn kho" not in qn or "chi phí" not in qn:
        return None
    numbers = [_parse_vn_number(n) for n in NUMBER_RE.findall(question)]
    numbers = [n for n in numbers if n is not None]
    if not numbers:
        return None
    cogs_candidates = [n for n in numbers if n >= 1000]
    if not cogs_candidates:
        return None
    cogs = cogs_candidates[0]
    
    pct_match = re.search(r"tăng[^0-9%]*(\d+(?:\.\d+)?)\s*%", qn)
    if pct_match:
        cogs *= (1.0 + float(pct_match.group(1)) / 100.0)
    else:
        pct_match_dec = re.search(r"giảm[^0-9%]*(\d+(?:\.\d+)?)\s*%", qn)
        if pct_match_dec:
            cogs *= (1.0 - float(pct_match_dec.group(1)) / 100.0)
            
    ratio = None
    ratio_match = re.search(r"luân chuyển hàng tồn kho là\s*(\d+(?:\.\d+)?)", qn)
    if not ratio_match:
        ratio_match = re.search(r"vòng quay hàng tồn kho là\s*(\d+(?:\.\d+)?)", qn)
    if not ratio_match:
        ratio_match = re.search(r"luân chuyển hàng tồn kho.*là\s*(\d+(?:\.\d+)?)", qn)
    if ratio_match:
        ratio = float(ratio_match.group(1))
    else:
        small_nums = [n for n in numbers if n < 100 and n != 10]
        if small_nums:
            ratio = small_nums[-1]
            
    if not ratio or ratio == 0:
        return None
        
    avg_inventory = cogs / ratio
    return _pick_closest_numeric_choice(choices, avg_inventory)


def _solve_cournot(question: str, choices: Dict[str, str]) -> Optional[str]:
    qn = question.lower()
    if "cournot" not in qn:
        return None
    if "hai doanh nghiệp" in qn or "2 doanh nghiệp" in qn:
        for label, text in choices.items():
            t = text.lower().replace(" ", "")
            if "a-c" in t and "3" in t and ("/" in t or "frac" in t):
                return label
    return None


_SPECIALIZED_SOLVERS = (
    _solve_linear_decay,
    _solve_midpoint_elasticity,
    _solve_gdp_deflator,
    _solve_supply_demand,
    _solve_money_multiplier,
    _solve_effective_annual_rate,
    _solve_inventory_turnover,
    _solve_cournot,
)


def solve_specialized(question: str, choices: Dict[str, str]) -> Optional[str]:
    for solver in _SPECIALIZED_SOLVERS:
        result = solver(question, choices)
        if result:
            return result
    return None


def solve(processed: Dict) -> str:
    question = processed["question"]
    choices = processed["choices"]

    result = solve_specialized(question, choices)
    if result:
        return result

    return lexical_best_choice(question, choices)
