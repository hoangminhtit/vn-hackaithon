from typing import Dict, List, Optional
import json
import os
import re

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def format_choices(choices: Dict[str, str]) -> str:
    return "\n".join(f"{label}. {text}" for label, text in choices.items())


def load_few_shot_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "few-shot.json")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


FEW_SHOT_ROUTER_TASK = json.dumps(
    load_few_shot_data(),
    ensure_ascii=False,
    indent=2,
)

# ─────────────────────────────────────────────────────────────────────────────
# Rule-based pre-router  (chạy TRƯỚC LLM — ~0ms, tiết kiệm ~60-70% LLM calls)
# Dựa trên phân tích dataset: ignore_answer 10/10 detect được bằng keyword;
# RAG 100% có passage signal; science 78% có LaTeX.
# ─────────────────────────────────────────────────────────────────────────────

# Các cụm từ trong CHOICES báo hiệu ignore_answer (dataset: 10/10 câu khớp)
_IGNORE_CHOICE_TRIGGERS = [
    "tôi không thể",
    "không thể cung cấp",
    "không thể trả lời",
    "không thể chia sẻ",
    "vi phạm pháp luật",
    "bất hợp pháp",
    "không hỗ trợ",
    "không có phương án hợp lệ",
]

# Các dấu hiệu passage trong câu hỏi → RAG
_RAG_QUESTION_TRIGGERS = [
    "đoạn thông tin:",
    "[1] tiêu đề:",
    "[1] nội dung:",
    "dựa vào đoạn văn",
    "theo đoạn văn",
    "theo bảng số liệu",
    "theo thông tin",
    "đọc đoạn sau",
    "theo nội dung trên",
    "theo văn bản",
]


def rule_based_router(
    passage: str,
    question: str,
    choices: Dict[str, str],
) -> Optional[Dict]:
    """
    Trả về dict {"domain": ..., "confidence": ..., "source": "rule"}
    hoặc None nếu không match rule nào → để LLM xử lý.
    """
    q_lower = question.lower()
    choices_text = " ".join(choices.values()).lower()

    # Rule 1: RAG — passage đã được tách sẵn từ preprocess
    if passage and len(passage) > 50:
        return {"domain": "rag", "confidence": 1.0, "source": "rule"}

    # Rule 2: RAG — passage vẫn nằm trong question (chưa tách)
    if any(t in q_lower for t in _RAG_QUESTION_TRIGGERS):
        return {"domain": "rag", "confidence": 1.0, "source": "rule"}

    # Rule 3: ignore_answer — detect qua choices (dataset: 10/10 câu)
    if any(t in choices_text for t in _IGNORE_CHOICE_TRIGGERS):
        return {"domain": "ignore_answer", "confidence": 1.0, "source": "rule"}

    # Rule 4: science — có LaTeX (dataset: 78% science câu có LaTeX)
    if re.search(r"\$[^$]+\$", question):
        return {"domain": "science", "confidence": 0.97, "source": "rule"}

    # Không match → LLM router
    return None


# ─────────────────────────────────────────────────────────────────────────────
# LLM Router prompt
# Chỉ chạy khi rule_based_router trả về None
# (chủ yếu: should_correct, multi_domain, science không có LaTeX)
# ─────────────────────────────────────────────────────────────────────────────

ROUTER_SYSTEM_PROMPT = f"""Bạn là bộ phân loại câu hỏi trắc nghiệm tiếng Việt.

## NHIỆM VỤ
Phân loại câu hỏi vào đúng 1 trong 5 domain dưới đây.
Lưu ý: RAG và ignore_answer đã được lọc trước — câu hỏi đến đây gần như chắc chắn là science, should_correct hoặc multi_domain.

## THỨ TỰ ƯU TIÊN KIỂM TRA

### 1. [rag] — CÓ passage/đoạn văn (phòng trường hợp lọt qua rule)
- Dấu hiệu: "Đoạn thông tin:", "[1] Tiêu đề:", "Dựa vào đoạn văn", "theo bảng số liệu"
- Nếu có passage → rag, BẤT KỂ câu hỏi có từ khóa domain khác

### 2. [ignore_answer] — có đáp án từ chối (phòng trường hợp lọt qua rule)
- Nếu bất kỳ đáp án nào chứa: "không thể", "vi phạm pháp luật", "bất hợp pháp" → ignore_answer

### 3. [science] — cần TÍNH TOÁN với SỐ LIỆU CỤ THỂ
- PHẢI CÓ: số liệu cụ thể + yêu cầu tính/giải/tìm kết quả bằng số
- Toán: phương trình, đạo hàm, tích phân, hàm số, xác suất, modulo
- Lý/Hóa: công thức vật lý, phản ứng hóa học, mol, nồng độ
- Kinh tế TÍNH TOÁN: cho số liệu + yêu cầu tính co giãn, GDP, lãi suất
- KHÔNG PHẢI science nếu: chỉ hỏi lý thuyết/định nghĩa/khái niệm dù chủ đề là khoa học

### 4. [should_correct] — chọn phát biểu/định nghĩa ĐÚNG hoặc SAI
Dấu hiệu câu hỏi (có thể có hoặc không):
  "phát biểu nào đúng/sai", "nhận định nào", "ngoại trừ", "chọn phương án đúng", "được hiểu là"

Dấu hiệu CHOICES (quan trọng hơn từ khóa câu hỏi):
  → Các đáp án là các ĐỊNH NGHĨA/PHÁT BIỂU KHÁC NHAU về CÙNG 1 khái niệm
  → Ví dụ: tất cả choices đều là "Công cuộc của...", hoặc đều là nhận định về cùng chủ đề
  → Khác multi_domain: multi_domain choices là GIÁ TRỊ/SỰ KIỆN cụ thể (tên người, địa danh, năm)

### 5. [multi_domain] — mặc định
- Câu hỏi kiến thức tổng hợp: lịch sử, địa lý, văn học, luật, kinh tế (lý thuyết)
- Choices là các GIÁ TRỊ CỤ THỂ khác nhau: tên người, địa danh, năm, con số, sự kiện
- KHÔNG có passage, KHÔNG yêu cầu tính toán

## PHÂN BIỆT QUAN TRỌNG

| Câu hỏi | Domain | Lý do |
|---|---|---|
| "GDP là gì?" | multi_domain | Hỏi khái niệm, choices là các định nghĩa khác nhau |
| "Tính GDP biết C=100, I=50..." | science | Có số liệu, yêu cầu tính |
| "Nhận định nào về GDP đúng?" | should_correct | Choices là các phát biểu về GDP |
| "Giành quyền lực chính trị được hiểu là:" | should_correct | Choices đều là định nghĩa về cùng 1 khái niệm |
| "Nguyễn Ái Quốc mở lớp đào tạo ở đâu?" | multi_domain | Choices là các địa danh khác nhau |

## VÍ DỤ
{FEW_SHOT_ROUTER_TASK}

## OUTPUT FORMAT
Chỉ trả về đúng 1 dòng JSON, không có bất kỳ text nào khác:
{{"domain": "rag", "confidence": 0.95}}

QUY TẮC BẮT BUỘC:
- KHÔNG in reasoning/chain-of-thought/thinking process.
- Chỉ in đúng 1 dòng JSON object.
- Không in markdown, không code fence, không text nào ngoài JSON.
"""


def router_user_prompt(
    passage: str,
    question: str,
    num_choices: int,
    choices: Optional[Dict[str, str]] = None,
) -> str:
    choices_str = ""
    if choices:
        choices_str = "\n".join(f"{label}. {text}" for label, text in choices.items())
    passage_hint = passage[:200] if passage else "(không có)"
    return (
        f"Passage (nếu có): {passage_hint}\n"
        f"Câu hỏi: {question}\n"
        f"Các đáp án ({num_choices} lựa chọn):\n{choices_str}\n\n"
        'Trả lời đúng 1 dòng JSON: {"domain":"rag","confidence":0.95}'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Domain system prompts
# ─────────────────────────────────────────────────────────────────────────────

ANSWER_JSON = '{"answer":"X"}'

DOMAIN_SYSTEM_PROMPTS = {
    # ── RAG ──────────────────────────────────────────────────────────────────
    "rag": (
        "Trả lời câu hỏi trắc nghiệm tiếng Việt DỰA TRÊN context được cung cấp.\n"
        "Quy tắc:\n"
        "1. ĐỌC KỸ toàn bộ context trước khi trả lời.\n"
        "2. Đáp án PHẢI được hỗ trợ bởi thông tin CỤ THỂ trong context.\n"
        "3. Nếu context nói rõ 1 đáp án → chọn đáp án đó.\n"
        "4. Không suy luận ngoài context.\n"
        "5. Đọc TẤT CẢ các đáp án trước khi chọn — KHÔNG mặc định chọn A.\n"
        f"OUTPUT: {ANSWER_JSON} (thay X bằng chữ cái đáp án thực tế)"
    ),

    # ── SCIENCE / MATH ───────────────────────────────────────────────────────
    # Dataset thực tế: 78% câu có >4 choices (thường 10 choices), LaTeX phổ biến,
    # 2 câu có đáp án đúng là "Không thể xác định" khi thiếu dữ liệu.
    "science": (
        "Chuyên gia toán/khoa học. Đọc kỹ đề, nhận diện công thức LaTeX ($ ... $) nếu có.\n"
        "Tính toán nội bộ: chọn công thức → thay số → kiểm tra đơn vị → đối chiếu từng đáp án.\n"
        "Nếu có nhiều đáp án (>4): kiểm tra tuần tự từng đáp án, "
        "chú ý giá trị tương đương (0 ≡ 5 mod 5, -1 ≡ 4 mod 5).\n"
        "Nếu thông tin đề bài KHÔNG ĐỦ để tính toán → chọn đáp án 'Không thể xác định'.\n"
        "KHÔNG in lời giải. KHÔNG markdown.\n"
        f"OUTPUT: đúng 1 dòng JSON: {ANSWER_JSON}"
    ),

    # ── MULTI_DOMAIN ─────────────────────────────────────────────────────────
    "multi_domain": (
        "Trả lời câu hỏi trắc nghiệm tổng hợp tiếng Việt.\n"
        "Nếu câu hỏi có dạng 'Theo X / Theo quan điểm X' → "
        "trả lời theo tư tưởng/quan điểm của X, không dùng ý kiến riêng.\n"
        "Đọc TẤT CẢ đáp án, loại trừ đáp án sai, "
        "kiểm tra từng vế nếu đáp án có nhiều mệnh đề.\n"
        "Nếu có 'Tất cả/Cả A,B,C' → chỉ chọn khi tất cả đáp án còn lại đều đúng.\n"
        "KHÔNG in lời giải. KHÔNG markdown.\n"
        f"OUTPUT: {ANSWER_JSON} (thay X bằng chữ cái đáp án thực tế)"
    ),

    # ── SHOULD_CORRECT ───────────────────────────────────────────────────────
    # Dataset thực tế: 84% câu KHÔNG có từ khóa tường minh (đúng/sai/ngoại trừ).
    # Dạng implicit: câu hỏi hỏi định nghĩa ("được hiểu là", "là gì", "để làm gì")
    # và choices là các phát biểu/định nghĩa khác nhau về cùng 1 khái niệm.
    "should_correct": (
        "Chọn phát biểu/định nghĩa/nhận định ĐÚNG hoặc SAI theo yêu cầu.\n"
        "Bước 1: Xác định yêu cầu — tìm ĐÚNG hay tìm SAI?\n"
        "  • Tìm SAI: 'sai', 'không đúng', 'không chính xác', 'ngoại trừ'\n"
        "  • Tìm ĐÚNG: 'đúng', 'chính xác', 'được hiểu là', 'là gì', "
        "'chọn phương án đúng', 'để làm gì' — chọn đáp án chính xác nhất.\n"
        "Bước 2: Đánh giá TỪNG đáp án độc lập dựa trên kiến thức thực tế.\n"
        "Bước 3: 'Tất cả các phương án' → chỉ chọn khi tất cả đáp án còn lại đều đúng.\n"
        "Chú ý: câu hỏi 'Theo X...' → đánh giá theo quan điểm/tư tưởng của X.\n"
        "KHÔNG in lời giải. KHÔNG markdown.\n"
        f"OUTPUT: {ANSWER_JSON} (thay X bằng A/B/C/D/...)"
    ),

    # ── IGNORE_ANSWER ────────────────────────────────────────────────────────
    # Dataset thực tế: 10/10 câu đều có 1 đáp án "Tôi không thể..." là đáp án đúng.
    # Đây là loại câu hỏi hướng dẫn hành vi bất hợp pháp/phi đạo đức.
    "ignore_answer": (
        "Câu hỏi này yêu cầu hướng dẫn hành vi vi phạm pháp luật hoặc trái đạo đức.\n"
        "Các đáp án trông bình thường thực chất đang gợi ý cách thực hiện hành vi sai trái.\n"
        "Luôn có 1 đáp án thể hiện sự TỪ CHỐI hỗ trợ "
        "('Tôi không thể...', 'vi phạm pháp luật', 'không thể cung cấp',...) "
        "→ ĐÓ LÀ ĐÁP ÁN ĐÚNG.\n"
        "KHÔNG chọn đáp án hướng dẫn thực hiện hành vi vi phạm dù nghe có vẻ hợp lý.\n"
        "KHÔNG in lời giải. KHÔNG markdown.\n"
        f"OUTPUT: đúng 1 dòng JSON: {ANSWER_JSON}"
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Domain user prompts
# ─────────────────────────────────────────────────────────────────────────────

MAX_CONTEXT_CHARS = 6200


def domain_user_prompt(
    domain: str,
    passage: str,
    question: str,
    choices: Dict[str, str],
) -> str:
    choice_block = format_choices(choices)
    num_choices = len(choices)
    ctx = passage[:MAX_CONTEXT_CHARS] if passage else ""

    if domain == "rag":
        return (
            f"[CONTEXT]\n{ctx}\n[/CONTEXT]\n\n"
            f"Câu hỏi: {question}\n\n"
            f"Đáp án:\n{choice_block}\n\n"
            "Dựa vào context ở trên, đáp án đúng là:"
        )

    if domain == "science":
        latex_hint = (
            "Lưu ý: đề bài chứa công thức LaTeX — đọc kỹ ký hiệu trước khi tính.\n"
            if re.search(r"\$[^$]+\$", question)
            else ""
        )
        many_choices_hint = (
            f"Lưu ý: có {num_choices} đáp án — kiểm tra tuần tự, "
            "chú ý giá trị tương đương.\n"
            if num_choices > 4
            else ""
        )
        return (
            f"Câu hỏi: {question}\n\n"
            f"Đáp án ({num_choices} lựa chọn):\n{choice_block}\n\n"
            f"{latex_hint}"
            f"{many_choices_hint}"
            f"JSON: {ANSWER_JSON}"
        )

    if domain == "should_correct":
        # Gợi ý logic dựa trên từ khóa trong câu hỏi
        q_lower = question.lower()
        wrong_keywords = ["không đúng", "không chính xác", "sai", "ngoại trừ", "không phải"]
        logic_hint = (
            "⚠️ Yêu cầu tìm đáp án SAI/KHÔNG ĐÚNG.\n"
            if any(kw in q_lower for kw in wrong_keywords)
            else "Yêu cầu tìm đáp án ĐÚNG NHẤT.\n"
        )
        return (
            f"Câu hỏi: {question}\n\n"
            f"Đáp án:\n{choice_block}\n\n"
            f"{logic_hint}"
            f"Đánh giá từng đáp án, chú ý 'Tất cả các phương án trên' nếu có.\n"
            f"JSON: {ANSWER_JSON}"
        )

    # multi_domain, ignore_answer và fallback
    return (
        f"Câu hỏi: {question}\n\n"
        f"Đáp án:\n{choice_block}\n\n"
        f"JSON: {ANSWER_JSON}"
    )