import re
from typing import Dict


MATH_HINTS = [
    "phương trình",
    "xác suất",
    "tính",
    "bao nhiêu",
    "đạo hàm",
    "tích phân",
    "nồng độ",
    "hàm số",
    "giải",
    "số",
]

SHOULD_CORRECT_HINTS = [
    "đúng hay sai",
    "phát biểu",
    "khẳng định",
    "kiểm tra tính đúng",
    "chỉnh sửa",
]

IGNORE_HINTS = [
    "không có đáp án nào đúng",
    "không phương án nào đúng",
    "none of the above",
]


def route_question(processed: Dict) -> Dict:
    question = processed["question"].lower()
    passage = processed["passage"]
    has_long_passage = bool(passage and len(passage) > 100)

    if any(h in question for h in IGNORE_HINTS):
        return {"domain": "ignore_answer", "confidence": 0.9}

    if any(h in question for h in SHOULD_CORRECT_HINTS):
        return {"domain": "should_correct", "confidence": 0.85}

    math_hits = sum(1 for h in MATH_HINTS if h in question)
    if math_hits >= 2 or re.search(r"[\d]+\s*[\+\-\*/=]", question):
        return {"domain": "math", "confidence": 0.82}

    if has_long_passage:
        return {"domain": "rag", "confidence": 0.8}

    return {"domain": "multi_domain", "confidence": 0.6}


def apply_route_fallback(route_result: Dict, passage: str) -> str:
    domain = route_result["domain"]
    confidence = float(route_result.get("confidence", 0.0))

    if passage and len(passage) > 100 and domain != "math":
        return "rag"
    if confidence < 0.6:
        return "multi_domain"
    return domain
