import re
from typing import Dict


MATH_HINTS = [
    "phương trình",
    "xác suất",
    "tính toán",
    "bao nhiêu",
    "đạo hàm",
    "tích phân",
    "nồng độ",
    "hàm số",
    "giải phương trình",
    "bằng bao nhiêu",
    "giá trị",
    "tổng cộng",
    "trung bình",
    "tỷ lệ",
    "phần trăm",
    "lãi suất",
    "gdp",
    "lạm phát",
    "co giãn",
]

SHOULD_CORRECT_HINTS = [
    "đúng hay sai",
    "phát biểu nào sai",
    "phát biểu nào đúng",
    "khẳng định nào",
    "kiểm tra tính đúng",
    "chỉnh sửa",
    "nhận định nào",
    "câu nào đúng",
    "câu nào sai",
    "ý nào đúng",
    "ý nào sai",
    "mệnh đề nào",
]

IGNORE_HINTS = [
    "không có đáp án nào đúng",
    "không phương án nào đúng",
    "none of the above",
    "tất cả đều sai",
]


def route_question(processed: Dict) -> Dict:
    question = processed["question"].lower()
    passage = processed["passage"]
    has_long_passage = bool(passage and len(passage) > 300)

    if any(h in question for h in IGNORE_HINTS):
        return {"domain": "ignore_answer", "confidence": 0.9}

    if any(h in question for h in SHOULD_CORRECT_HINTS):
        return {"domain": "should_correct", "confidence": 0.85}

    has_math_expr = bool(re.search(r"[\d]+\s*[\+\-\*/=]", question))
    has_formula = bool(re.search(r"\$.*\$", processed["question"]))

    if has_long_passage:
        if has_math_expr or has_formula:
            return {"domain": "science", "confidence": 0.75}
        return {"domain": "rag", "confidence": 0.8}

    math_hits = sum(1 for h in MATH_HINTS if h in question)
    if math_hits >= 2 or has_math_expr or has_formula:
        return {"domain": "science", "confidence": 0.82}
    if math_hits == 1:
        return {"domain": "science", "confidence": 0.7}

    return {"domain": "multi_domain", "confidence": 0.6}


def apply_route_fallback(route_result: Dict, passage: str) -> str:
    domain = route_result["domain"]
    confidence = float(route_result.get("confidence", 0.0))

    if confidence < 0.5:
        if passage and len(passage) > 300:
            return "rag"
        return "multi_domain"
    return domain
