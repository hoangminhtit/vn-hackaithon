import re
import unicodedata
from typing import Dict, Optional


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

SCIENCE_SYMBOLIC_HINTS = [
    "biểu thức",
    "công thức",
    "ma trận",
    "phân phối",
    "chu kỳ",
    "vận tốc",
    "lưu lượng",
    "điện trở tương đương",
    "dẫn nhiệt",
    "con lắc",
    "poiseuille",
    "phương trình vi phân",
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

# ── Regex-based detectors ──────────────────────────────────────────────────

# Passage markers — dấu hiệu câu RAG. 
# Đối chiếu với các pattern trích xuất passage trong utils/preprocess.py.
_PASSAGE_MARKERS = ("đoạn thông tin:", "doan thong tin:", "[1] tiêu đề:", "[1] nội dung:")

# Pháp luật — bài viết điều luật
_LAW_ARTICLE_RE = re.compile(r"\b(?:điều|dieu)\s+\d+", flags=re.IGNORECASE)
_LAW_KEYWORD_RE = re.compile(
    r"\b(?:"
    r"bộ luật|bo luat|luật|luat|khoản|khoan|nghị định|nghi dinh|"
    r"thông tư|thong tu|hiến pháp|hien phap|tội|hình sự|hinh su|"
    r"dân sự|dan su|nghĩa vụ|nghia vu|trách nhiệm hình sự|"
    r"trach nhiem hinh su|xử phạt|xu phat|vi phạm|vi pham|"
    r"pháp luật|phap luat|hợp đồng|hop dong"
    r")\b",
    flags=re.IGNORECASE,
)

# Chính trị / lý luận Mác-Lênin — dùng Unicode folding để match cả có/không dấu
_POLYSCI_KEYWORD_RE = re.compile(
    r"\b(?:"
    r"ho chi minh|tu tuong|mac|lenin|angghen|"
    r"chu nghia|duy vat|bien chung|"
    r"kinh te chinh tri|gia tri thang du|tich luy tu ban|"
    r"chu nghia xa hoi|chu nghia cong san|"
    r"cach mang|giai cap|vo san|"
    r"dang cong san|dang ta|cuong linh|dai hoi dang|"
    r"phuong thuc san xuat|luc luong san xuat|quan he san xuat|"
    r"hinh thai kinh te|duy vat lich su|duy vat bien chung|"
    r"y thuc xa hoi|ton tai xa hoi"
    r")\b",
    flags=re.IGNORECASE,
)

# Alignment bait — câu hỏi harmful + đáp án từ chối
_HOW_TO_RE = re.compile(
    r"làm thế nào|cách nào|phương thức|bằng cách nào|làm sao|"
    r"cách hiệu quả|như thế nào|"
    r"hành động nào|hành vi nào|"
    r"thực hiện .{0,20}(?:nào|ra sao|thế nào)",
    flags=re.IGNORECASE,
)
_VIOLATION_RE = re.compile(
    r"vi phạm|trái pháp luật|trái quy định|phá hoại|tham nhũng|làm giả|"
    r"lợi dụng chức vụ|lợi dụng quyền|xuyên tạc|bịa đặt|"
    r"phát tán tài liệu mật|phát tán thông tin sai|"
    r"xúc phạm(?:\s+các)?\s+(?:biểu tượng|lãnh tụ)|"
    r"tạo ra văn bản trái|làm suy yếu|công chức.*tham nhũng|"
    r"phá hoại tinh thần",
    flags=re.IGNORECASE,
)
_REFUSAL_RE = re.compile(
    r"tôi không thể|không thể trả lời|không thể cung cấp",
    flags=re.IGNORECASE,
)


SCIENCE_UNITS_RE = re.compile(
    r"\d+\s*(?:hz|khz|mhz|ghz|v|kv|mv|ma|a|ω|ohm|f|μf|nf|pf|"
    r"m/s|km/h|m/s²|kg|mg|g/mol|mol/l|ml|cm³|mm|cm|km|"
    r"j|kj|cal|kcal|w|kw|pa|kpa|atm|n|kn|°c|k)\b",
    re.IGNORECASE
)

FORMULA_CHOICE_RE = re.compile(
    r"\\(?:frac|sqrt|begin|text|pi)|\^|_\{|[A-Za-z]\s*=|[A-Za-z]\([^)]*\)\s*=|\$",
    re.IGNORECASE,
)


def _fold_vietnamese(text: str) -> str:
    """Normalise Vietnamese text: remove diacritics + casefold.

    Allows regex patterns without diacritics to match accented Vietnamese.
    """
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(
        char for char in decomposed if unicodedata.category(char) != "Mn"
    ).casefold()


def _has_numeric_data(question: str) -> bool:
    clean_q = re.sub(r"\d+/\d+/[A-ZĐa-z0-9\-]+", "", question)
    clean_q = re.sub(r"\d+/\d+/\d+", "", clean_q)
    clean_q = re.sub(r"\b(1[89]\d{2}|20[0-2]\d)\b", "", clean_q)

    numbers = re.findall(r"[-+]?\d+[.,]?\d*", clean_q)
    return len(numbers) >= 2 or ("%" in clean_q) or any(w in clean_q.lower() for w in ["đô la", "usd", "vnd", "đồng", "triệu", "tỷ"])


def _has_science_units(question: str) -> bool:
    return bool(SCIENCE_UNITS_RE.search(question))


def _mostly_formula_choices(choices: list) -> bool:
    if not choices:
        return False
    hits = sum(1 for choice in choices if FORMULA_CHOICE_RE.search(str(choice)))
    return hits >= max(2, len(choices) // 2)


def is_law_question(question: str, choices: list) -> bool:
    """True nếu câu hỏi liên quan đến pháp luật VN và không có passage.

    """
    question_text = question.casefold()
    if any(marker in question_text for marker in _PASSAGE_MARKERS):
        return False

    text = "\n".join([question] + list(choices)).casefold()
    if _LAW_ARTICLE_RE.search(text):
        return True
    match = _LAW_KEYWORD_RE.search(text)
    if not match:
        return False
    # Tránh false positive: "quy luật" không phải pháp luật
    if match.group(0) in {"luật", "luat"} and text[
        max(0, match.start() - 4) : match.start()
    ] == "quy ":
        return False
    return True


def is_polysci_question(question: str, choices: list) -> bool:
    """True nếu câu hỏi về lý luận chính trị Mác-Lênin / tư tưởng HCM.

    Dùng Unicode folding để match cả dạng có/không dấu.
    """
    question_text = question.casefold()
    if any(marker in question_text for marker in _PASSAGE_MARKERS):
        return False

    text = _fold_vietnamese("\n".join([question] + list(choices)))
    return _POLYSCI_KEYWORD_RE.search(text) is not None


def detect_alignment_bait(question: str, choices: list) -> Optional[str]:
    """Trả về chữ cái của đáp án từ chối nếu câu hỏi là bẫy alignment.

    Chính xác hơn keyword list cũ.
    """
    ql = question.lower()
    if not _HOW_TO_RE.search(ql):
        return None
    if not _VIOLATION_RE.search(ql):
        return None
    for i, choice in enumerate(choices):
        if _REFUSAL_RE.search(choice.lower()):
            return chr(65 + i)
    return None


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
    choices_list = list(choices.values()) if choices else []
    original_question = processed["question"]

    has_long_passage = bool(passage and (len(passage) > 200 or "được cung cấp" in question or "theo thông tin" in question))

    # ── Layer 1: Ignore-answer ── regex alignment bait (chính xác hơn) ───────
    # Thử detect_alignment_bait trước — regex HOW_TO + VIOLATION + REFUSAL
    if detect_alignment_bait(original_question, choices_list) is not None:
        return {"domain": "ignore_answer", "confidence": 0.95}

    # Fallback keyword list cũ cho ignore_answer (bao phủ thêm edge cases)
    choices_text = " ".join(v.lower() for v in choices_list)
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
    has_formula = bool(re.search(r"\$.*\$", original_question))

    # ── Layer 2: RAG — câu có passage dài ────────────────────────────────────
    if has_long_passage:
        return {"domain": "rag", "confidence": 0.85}

    is_theory = _is_theory_question(question)
    is_policy = any(h in question for h in POLICY_HINTS)

    # ── Phát hiện chính trị/polysci: regex Unicode folding trước, fallback keyword list
    is_politics = is_polysci_question(original_question, choices_list)
    if not is_politics:
        is_politics = any(h in question for h in POLITICS_HINTS)

    sc_hits = sum(1 for h in SHOULD_CORRECT_HINTS if h in question)
    wants_calc = _has_calculation_intent(question)
    is_quant_reasoning = any(w in question for w in QUANT_KEYWORDS) and any(w in question for w in QUANT_REASONING)
    formula_choices = _mostly_formula_choices(choices_list)
    symbolic_science = has_formula and (
        formula_choices or any(h in question for h in SCIENCE_SYMBOLIC_HINTS)
    )

    # Politics/HCM questions → should_correct trừ khi có tính toán
    if is_politics and not has_formula and not has_math_expr:
        return {"domain": "should_correct", "confidence": 0.88}

    if sc_hits > 0 and not has_math_expr and not symbolic_science and not wants_calc and not is_quant_reasoning:
        return {"domain": "should_correct", "confidence": 0.88 if is_theory else 0.85}

    has_numbers = _has_numeric_data(question)
    has_units = _has_science_units(original_question)
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
        if symbolic_science and not is_policy and not is_politics:
            return {"domain": "science", "confidence": 0.88}
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
