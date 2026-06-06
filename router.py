import re
from typing import Dict


MATH_STRONG_HINTS = [
    "phương trình",
    "đạo hàm",
    "tích phân",
    "nồng độ",
    "hàm số",
    "giải phương trình",
    "tính vận tốc",
    "tính gia tốc",
    "tính lực",
    "tính công",
    "tính nhiệt",
    "mol",
    "nguyên tử khối",
    "dung dịch",
    "trung hoà",
    "trung hòa",
    "điện trở",
    "tụ điện",
    "cuộn cảm",
    "tần số",
    "cournot",
    "eoq",
    "đẳng lượng",
    "số nhân tiền",
    "lượng đặt hàng tối ưu",
    "hàm sản xuất",
    "khấu hao",
    "phân rã",
    "độ phóng xạ",
    "hạt nhân",
]

MATH_WEAK_HINTS = [
    "xác suất",
    "tính toán",
    "bao nhiêu",
    "bằng bao nhiêu",
    "giá trị",
    "tổng cộng",
    "trung bình",
    "tỷ lệ",
    "phần trăm",
    "lãi suất",
    "co giãn",
    "khối lượng",
    "thể tích",
    "công suất",
    "hiệu suất",
    "điện áp",
]

THEORY_OVERRIDE_HINTS = [
    "là gì",
    "là khái niệm",
    "được hiểu là",
    "phản ánh",
    "có ý nghĩa",
    "mục đích",
    "vai trò",
    "chức năng",
    "đặc điểm",
    "bản chất",
    "dùng để",
    "nhằm",
    "viết tắt",
    "định nghĩa",
    "khái niệm",
    "biểu hiện",
    "phân loại",
    "xu hướng",
    "nguyên nhân",
    "kết quả",
    "ý nghĩa",
    "phát biểu",
    "nhận định",
    "khẳng định",
]

POLITICS_HINTS = [
    "hồ chí minh",
    "nguyễn ái quốc",
    "chính trị",
    "đảng cộng sản",
    "đảng lao động",
    "hiến pháp",
    "quốc hội",
    "nhà nước",
    "chủ nghĩa xã hội",
    "dân chủ",
    "cách mạng",
    "yêu nước",
    "dân tộc",
    "mác-lênin",
    "mác - lênin",
    "chống pháp",
    "tư sản",
    "vô sản",
    "giai cấp",
    "quyền lực",
    "hệ thống chính trị",
    "hội nghị hợp nhất",
    "đông dương",
    "nho giáo",
    "khổng tử",
    "bác hồ",
    "hành chính công",
]

SHOULD_CORRECT_HINTS = [
    "đúng hay sai",
    "phát biểu nào sai",
    "phát biểu nào đúng",
    "phát biểu nào sau đây",
    "khẳng định nào",
    "kiểm tra tính đúng",
    "nhận định nào",
    "câu nào đúng",
    "câu nào sai",
    "ý nào đúng",
    "ý nào sai",
    "mệnh đề nào",
    "điều nào",
    "ngoại trừ",
    "không đúng",
    "không chính xác",
    "là đúng",
    "là sai",
    "đáp án sai",
    "đáp án đúng",
    "chọn đáp án sai",
    "chọn phương án",
    "được hiểu là",
    "điểm khác biệt",
    "quan điểm của",
    "là quan điểm",
    "nguồn gốc nào",
    "bao gồm những",
    "hình thành từ",
    "chủ yếu là",
    "cốt lõi là",
]

POLICY_HINTS = [
    "nghị quyết",
    "quyết định số",
    "theo quy định",
    "theo luật",
    "luật số",
    "bộ luật",
    "mô hình hành chính",
    "sắp xếp",
    "sáp nhập",
    "đơn vị hành chính",
    "cấp hành chính",
]

HARMFUL_QUESTION_HINTS = [
    "làm thế nào để",
    "cách nào",
    "phương pháp nào",
    "cách hiệu quả nhất",
    "làm cách nào",
    "bằng cách nào",
]

HARMFUL_INTENT_HINTS = [
    "vi phạm",
    "lách",
    "trốn",
    "giả mạo",
    "làm giả",
    "phát tán tài liệu mật",
    "tham nhũng",
    "hối lộ",
    "phá hoại",
    "xúc phạm",
    "lật đổ",
    "chống phá",
    "chống lại",
    "suy yếu",
    "tránh bị phát hiện",
    "trái quy định",
    "trái pháp luật",
    "bất hợp pháp",
    "lợi dụng quyền lực",
    "tránh việc cung cấp thông tin",
]

REFUSE_CHOICE_HINTS = [
    "không thể cung cấp",
    "không thể hỗ trợ",
    "vi phạm pháp luật",
    "tôi không thể",
    "không hỗ trợ",
    "hành vi vi phạm",
    "không có phương án hợp lệ",
]

LEGITIMATE_TOPIC_HINTS = [
    "theo",
    "chủ tịch",
    "tư tưởng",
    "quan điểm",
    "lý luận",
    "nguyên tắc",
]

IGNORE_HINTS = [
    "không có đáp án nào đúng",
    "không phương án nào đúng",
    "none of the above",
    "tất cả đều sai",
]


SCIENCE_UNITS_RE = re.compile(
    r"\d+\s*(?:hz|khz|mhz|ghz|v|kv|mv|ma|a|ω|ohm|f|μf|nf|pf|"
    r"m/s|km/h|m/s²|kg|mg|g/mol|mol/l|ml|cm³|mm|cm|km|"
    r"j|kj|cal|kcal|w|kw|pa|kpa|atm|n|kn|°c|k)\b",
    re.IGNORECASE
)


def _has_numeric_data(question: str) -> bool:
    clean_q = re.sub(r"\d+/\d+/[A-ZĐa-z0-9\-]+", "", question)
    clean_q = re.sub(r"\d+/\d+/\d+", "", clean_q)
    clean_q = re.sub(r"\b(1[89]\d{2}|20[0-2]\d)\b", "", clean_q)
    
    numbers = re.findall(r"[-+]?\d+[.,]?\d*", clean_q)
    return len(numbers) >= 2 or ("%" in clean_q) or any(w in clean_q.lower() for w in ["đô la", "usd", "vnd", "đồng", "triệu", "tỷ"])


def _has_science_units(question: str) -> bool:
    return bool(SCIENCE_UNITS_RE.search(question))


CALC_INTENT_HINTS = [
    "tính ",
    "tính toán",
    "giải ",
    "tìm ",
    "bằng bao nhiêu",
    "bao nhiêu",
    "xác định giá trị",
    "điều gì xảy ra",
    "ảnh hưởng",
    "thay đổi",
    "tăng",
    "giảm",
]

QUANT_KEYWORDS = [
    "eoq",
    "lại kép",
    "lãi kép",
    "cournot",
    "hiệu suất",
    "sản lượng",
    "công suất",
    "trở kháng",
    "lực",
    "vận tốc",
    "gia tốc",
    "tần số",
    "chu kỳ",
    "phóng xạ",
    "phân rã",
    "áp suất",
    "thể tích",
    "mật độ",
    "lượng đặt hàng",
    "lãi suất",
]

QUANT_REASONING = [
    "tăng",
    "giảm",
    "thay đổi",
    "ảnh hưởng",
    "điều gì xảy ra",
    "gấp đôi",
    "gấp ba",
]


def _has_calculation_intent(question: str) -> bool:
    has_calc_word = any(h in question for h in CALC_INTENT_HINTS)
    has_number_indicator = bool(re.search(r"\d", question)) or any(w in question for w in ["gấp đôi", "gấp ba"])
    return has_calc_word and has_number_indicator


def _is_theory_question(question: str) -> bool:
    return any(h in question for h in THEORY_OVERRIDE_HINTS)


def route_question(processed: Dict) -> Dict:
    question = processed["question"].lower()
    passage = processed["passage"]
    choices = processed.get("choices", {})
    has_long_passage = bool(passage and (len(passage) > 200 or "được cung cấp" in question or "theo thông tin" in question))

    choices_text = " ".join(v.lower() for v in choices.values()) if choices else ""
    has_refuse_choice = any(kw in choices_text for kw in REFUSE_CHOICE_HINTS)

    if has_refuse_choice:
        has_harmful_pattern = any(h in question for h in HARMFUL_QUESTION_HINTS)
        has_harmful_intent = any(h in question for h in HARMFUL_INTENT_HINTS)
        if has_harmful_intent or (has_harmful_pattern and has_harmful_intent):
            return {"domain": "ignore_answer", "confidence": 0.95}
        if has_harmful_pattern and not any(kw in question for kw in LEGITIMATE_TOPIC_HINTS):
            return {"domain": "ignore_answer", "confidence": 0.90}

    if any(h in question for h in IGNORE_HINTS):
        return {"domain": "ignore_answer", "confidence": 0.9}

    clean_q = re.sub(r"\d+/\d+/[A-ZĐa-z0-9\-]+", "", question)
    clean_q = re.sub(r"\d+/\d+/\d+", "", clean_q)

    has_math_expr = bool(
        re.search(r"[\d]+\s*[\+\*/=]", clean_q)
        or re.search(r"[\d]+\s*-\s*[\d]", clean_q)
        and not re.search(r"(1[89]\d{2}|20[0-2]\d)\s*-\s*(1[89]\d{2}|20[0-2]\d|\d{2})", clean_q)
    )
    if re.search(r"(1[89]\d{2}|20[0-2]\d)\s*-\s*", clean_q) and not re.search(r"[\d]+\s*[\+\*/=]", clean_q):
        has_math_expr = False
    has_formula = bool(re.search(r"\$.*\$", processed["question"]))

    if has_long_passage:
        return {"domain": "rag", "confidence": 0.85}

    is_theory = _is_theory_question(question)
    is_policy = any(h in question for h in POLICY_HINTS)
    is_politics = any(h in question for h in POLITICS_HINTS)

    sc_hits = sum(1 for h in SHOULD_CORRECT_HINTS if h in question)
    wants_calc = _has_calculation_intent(question)
    is_quant_reasoning = any(w in question for w in QUANT_KEYWORDS) and any(w in question for w in QUANT_REASONING)

    # Politics/HCM questions should be routed to should_correct unless they contain math formulas/expressions
    if is_politics and not has_formula and not has_math_expr:
        return {"domain": "should_correct", "confidence": 0.88}

    if sc_hits > 0 and not has_math_expr and not has_formula and not wants_calc and not is_quant_reasoning:
        return {"domain": "should_correct", "confidence": 0.88 if is_theory else 0.85}

    has_numbers = _has_numeric_data(question)
    has_units = _has_science_units(processed["question"])
    strong_hits = sum(1 for h in MATH_STRONG_HINTS if h in question)
    weak_hits = sum(1 for h in MATH_WEAK_HINTS if h in question)

    # Avoid routing policy/politics theory questions to science
    if (is_policy or is_politics) and not has_formula and not has_math_expr:
        return {"domain": "multi_domain", "confidence": 0.80}

    if (wants_calc or is_quant_reasoning) and has_numbers and not is_policy and not is_politics:
        return {"domain": "science", "confidence": 0.92}

    if is_policy and not has_formula:
        return {"domain": "multi_domain", "confidence": 0.80}

    # LaTeX-only (has_formula) doesn't auto-route to science if it lacks calculation intent or strong math hints
    if has_formula:
        if wants_calc or is_quant_reasoning or strong_hits >= 1:
            return {"domain": "science", "confidence": 0.90}

    if strong_hits >= 1 or has_math_expr:
        return {"domain": "science", "confidence": 0.90}

    if has_units and has_numbers and not is_theory:
        return {"domain": "science", "confidence": 0.85}

    if weak_hits >= 1 and has_numbers and not is_theory and not is_policy and not is_politics:
        return {"domain": "science", "confidence": 0.82}

    if weak_hits >= 2 and has_numbers and not is_policy and not is_politics:
        return {"domain": "science", "confidence": 0.75}

    return {"domain": "multi_domain", "confidence": 0.6}


def apply_route_fallback(route_result: Dict, passage: str) -> str:
    domain = route_result["domain"]
    confidence = float(route_result.get("confidence", 0.0))

    if confidence < 0.5:
        if passage and len(passage) > 300:
            return "rag"
        return "multi_domain"
    return domain
