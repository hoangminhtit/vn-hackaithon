from typing import Dict, List, Literal, Optional
import json
import os
import re

from utils.config import rag_max_context_chars

Polarity = Literal["false", "true", "neutral"]

WANT_FALSE_HINTS = (
    "không chính xác",
    "không đúng",
    "không chính xác nhất",
    "nào sai",
    "ý sai",
    "câu sai",
    "phát biểu sai",
    "ngoại trừ",
    "không phải",
    "sai nhất",
    "sai,",
    "sai?",
    "sai ",
    " là sai",
)

WANT_TRUE_HINTS = (
    "nào đúng",
    "ý đúng",
    "câu đúng",
    "phát biểu đúng",
    "đúng nhất",
    "chính xác nhất",
    "là đúng",
    "đúng?",
    "đúng ",
    "đúng,",
    "chính xác?",
    "chính xác ",
)

THEO_HINTS = ("theo ", "theo quan điểm", "theo tư tưởng", "theo quy định", "theo luật", "theo hiến pháp")


def format_choices(choices: Dict[str, str]) -> str:
    return "\n".join(f"{label}. {text}" for label, text in choices.items())


def question_polarity(question: str) -> Polarity:
    q = question.lower()
    if any(h in q for h in WANT_FALSE_HINTS):
        return "false"
    if any(h in q for h in WANT_TRUE_HINTS):
        return "true"
    return "neutral"


def _has_calculation_intent(question: str) -> bool:
    q = question.lower()
    calc_verbs = ("tính", "giải", "tìm", "xác định giá trị", "bằng bao nhiêu", "bao nhiêu")
    return any(v in q for v in calc_verbs) and bool(re.search(r"\d", q))


def _domain_hint_block(domain: str, question: str) -> str:
    polarity = question_polarity(question)
    q = question.lower()
    lines: List[str] = []

    if domain == "should_correct":
        if polarity == "false":
            lines.append(
                "⚠️ Câu hỏi hỏi phát biểu SAI/KHÔNG ĐÚNG — chọn đáp án SAI, "
                "KHÔNG chọn đáp án nghe hợp lý nhất."
            )
        elif polarity == "true":
            lines.append("⚠️ Câu hỏi hỏi phát biểu ĐÚNG — chọn đáp án CHÍNH XÁC NHẤT.")
        else:
            lines.append("Đọc kỹ câu hỏi để biết cần tìm phát biểu đúng hay sai.")
        lines.append("Đánh giá từng đáp án A,B,C,D độc lập trước khi chọn.")

    elif domain == "science":
        lines.append(
            "Đây là bài TÍNH TOÁN/GIẢI SỐ — không chọn theo cảm gián đúng/sai lý thuyết."
        )
        if _has_calculation_intent(question):
            lines.append(
                "Trích số liệu → áp dụng công thức → so khớp kết quả với đáp án (chú ý đơn vị)."
            )
        lines.append("Số Việt Nam: 1.000 = một nghìn; 1,5 = một phẩy năm.")

    elif domain == "multi_domain":
        lines.append(
            "Đây là câu KIẾN THỨC — không tính toán, không đánh giá đúng/sai trừ khi câu hỏi yêu cầu."
        )
        if any(h in q for h in THEO_HINTS):
            lines.append(
                "Câu hỏi có 'Theo …' — trả lời theo quan điểm/văn bản được nêu, "
                "không theo kiến thức chung khác."
            )

    elif domain == "rag":
        if polarity == "false":
            lines.append(
                "⚠️ Câu hỏi hỏi phát biểu SAI theo context — chọn đáp án MÂU THUẪN hoặc "
                "KHÔNG được context hỗ trợ."
            )
        elif polarity == "true":
            lines.append("Chọn phát biểu được context hỗ trợ trực tiếp nhất.")
        lines.append("Chỉ dùng thông tin trong context; bỏ qua kiến thức bên ngoài nếu mâu thuẫn.")

    return "\n".join(lines)


def _all_options_hint(choices: Dict[str, str]) -> Optional[str]:
    for text in choices.values():
        t = text.lower()
        if "tất cả" in t and "phương án" in t:
            return (
                "Có đáp án 'Tất cả các phương án trên' — chỉ chọn nếu MỌI đáp án còn lại đều đúng."
            )
    return None


def _many_choice_hint(choices: Dict[str, str]) -> Optional[str]:
    """Warn the model when there are more than 4 options (A–J range).

    Mirrors the bài tham chiếu logic in LLMAnswerer._many_choice_hint().
    """
    n = len(choices)
    if n <= 4:
        return None
    last_letter = chr(ord("A") + n - 1)
    return (
        f"Lưu ý: có {n} lựa chọn từ A đến {last_letter}. "
        "Hãy xem xét kỹ TẤT CẢ lựa chọn trước khi chọn — không dừng lại ở A/B/C/D."
    )

def load_few_shot_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "few-shot.json")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

FEW_SHOT_ROUTER_TASK = json.dumps(
    load_few_shot_data(),
    ensure_ascii=False,
    indent=2
)

ROUTER_SYSTEM_PROMPT = f"""Bạn là bộ phân loại câu hỏi trắc nghiệm tiếng Việt.

## NHIỆM VỤ
Phân loại câu hỏi vào đúng 1 trong 5 domain dưới đây.

## THỨ TỰ ƯU TIÊN KIỂM TRA (từ cao đến thấp)

### 1. [ignore_answer] — ưu tiên CAO NHẤT
Kiểm tra CÁC ĐÁP ÁN trước:
- Nếu có đáp án chứa: "không thể cung cấp", "vi phạm pháp luật", "tôi không thể", "không hỗ trợ", "bất hợp pháp" → ignore_answer
- Câu hỏi hướng dẫn hành vi sai trái (lách luật, giả mạo, gian lận, gây hại)
- Hoặc: tất cả đáp án đều sai về kiến thức (đáp án đúng không có trong choices)

### 2. [rag] — nếu CÓ passage/đoạn văn
- Dấu hiệu: "Đoạn thông tin:", "[1] Tiêu đề:", "[1] Nội dung:", "Dựa vào đoạn văn", "theo bảng số liệu"
- Nếu có passage → rag, BẤT KỂ câu hỏi có từ khóa math/should_correct

### 3. [science] — cần TÍNH TOÁN với SỐ LIỆU CỤ THỂ
- PHẢI CÓ: số liệu cụ thể + yêu cầu tính/giải/tìm kết quả bằng số
- Toán: phương trình, đạo hàm, tích phân, hàm số, xác suất
- Lý/Hóa: công thức vật lý, phản ứng hóa học, mol, nồng độ
- Kinh tế TÍNH TOÁN: cho số liệu + yêu cầu tính co giãn, GDP, lãi suất
- KHÔNG PHẢI science nếu: chỉ hỏi lý thuyết/định nghĩa/khái niệm dù chủ đề là khoa học

### 4. [should_correct] — kiểm tra phát biểu ĐÚNG/SAI
- Dấu hiệu: "phát biểu nào đúng/sai", "nhận định nào", "điều nào KHÔNG đúng", "ngoại trừ"
- Các đáp án là các NHẬN ĐỊNH khác nhau về cùng 1 chủ đề
- Bao gồm: câu hỏi lý thuyết khoa học/kinh tế khi hỏi "đúng hay sai" mà KHÔNG cho số liệu

### 5. [multi_domain] — mặc định
- Câu hỏi kiến thức tổng hợp: lịch sử, địa lý, văn học, luật, kinh tế (lý thuyết)
- KHÔNG có passage, KHÔNG yêu cầu tính toán, KHÔNG hỏi đúng/sai
- Hỏi định nghĩa, khái niệm, sự kiện, ý nghĩa

## PHÂN BIỆT QUAN TRỌNG
- "GDP là gì?" → multi_domain (hỏi khái niệm)
- "Tính GDP biết..." → science (có số, cần tính)
- "Nhận định nào về GDP đúng?" → should_correct (kiểm tra phát biểu)
- "Theo bảng, GDP năm 2020..." → rag (có passage)

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


def router_user_prompt(passage: str, question: str, num_choices: int, choices: Dict[str, str] | None = None) -> str:
    choices_str = ""
    if choices:
        choices_str = "\n".join(f"{label}. {text}" for label, text in choices.items())
    return (
        f"Passage (nếu có): {passage[:200]}\n"
        f"Câu hỏi: {question}\n"
        f"Các đáp án:\n{choices_str}\n"
        f"Số lượng đáp án: {num_choices}\n\n"
        'Trả lời đúng 1 dòng JSON: {"domain":"rag","confidence":0.95}'
    )


ANSWER_JSON = '{"answer":"X"}'

DOMAIN_SYSTEM_PROMPTS = {
    "rag": (
        "Trả lời trắc nghiệm tiếng Việt CHỈ dựa trên context được cung cấp.\n"
        "Quy tắc:\n"
        "1. Đọc toàn bộ context trước; tìm đoạn/câu/bảng liên quan trực tiếp đến câu hỏi.\n"
        "2. Đáp án phải được context nêu rõ hoặc suy ra trực tiếp từ context.\n"
        "3. Hỏi 'SAI/KHÔNG đúng/ngoại trừ' → chọn đáp án mâu thuẫn context hoặc không được nêu.\n"
        "4. Hỏi 'ĐÚNG/chính xác' → chọn đáp án khớp context nhất.\n"
        "5. Có bảng/số liệu → lấy đúng con số từ context, không tự tính nếu context đã cho kết quả.\n"
        "6. Không dùng kiến thức ngoài khi mâu thuẫn context.\n"
        "7. Đọc hết A,B,C,D — không mặc định chọn A.\n"
        f"OUTPUT: chỉ 1 dòng JSON {ANSWER_JSON}"
    ),

    "science": (
        "Chuyên gia toán/lý/hóa/sinh/kinh tế tính toán. Nhiệm vụ: TÍNH hoặc GIẢI, không chọn theo cảm giác.\n"
        "Quy tắc:\n"
        "1. Đây KHÔNG phải câu đúng/sai lý thuyết — phải ra con số/kết quả cụ thể.\n"
        "2. Trích số liệu và đơn vị từ đề; áp dụng đúng công thức.\n"
        "3. Tự tính nội bộ, so khớp với đáp án gần nhất (chú ý làm tròn, %, đơn vị).\n"
        "4. Số Việt Nam: 1.000 thường là nghìn; 1,5 = 1.5.\n"
        "5. Không chọn đáp án chỉ vì nghe đúng về mặt lý thuyết.\n"
        f"OUTPUT: chỉ 1 dòng JSON {ANSWER_JSON} — KHÔNG giải thích, KHÔNG markdown."
    ),

    "multi_domain": (
        "Trả lời trắc nghiệm kiến thức tổng hợp tiếng Việt (lịch sử, địa lý, luật, chính trị, kinh tế, văn hóa).\n"
        "Quy tắc:\n"
        "1. KHÔNG tính toán — chọn theo sự kiện, quy định, khái niệm, định nghĩa.\n"
        "2. KHÔNG đánh giá đúng/sai kiểu should_correct trừ khi câu hỏi hỏi rõ 'đúng/sai'.\n"
        "3. 'Theo Hồ Chí Minh/Theo luật/Theo Hiến pháp' → trả lời theo quan điểm/văn bản đó.\n"
        "4. Câu hỏi ghép nhiều ý → đáp án phải thỏa TẤT CẢ các ý trong câu hỏi.\n"
        "5. Loại trừ đáp án sai thời gian, sai địa danh, sai thuật ngữ.\n"
        "6. 'Tất cả các phương án trên' chỉ khi mọi đáp án còn lại đều đúng.\n"
        "7. Cảnh giác với các phương án phủ quát như 'Tất cả các phương án trên' hoặc 'Cả a, b, c đều đúng' khi hỏi về tư tưởng Hồ Chí Minh hoặc các chủ đề lịch sử/chính trị. Thông thường chỉ có MỘT đáp án cụ thể là đúng và chính xác nhất. Không chọn phương án 'Tất cả' trừ khi bạn hoàn toàn chắc chắn mọi phương án đơn lẻ đều hoàn toàn đúng.\n"
        "8. Đối với các câu hỏi có rất nhiều đáp án lựa chọn (từ 5 đáp án trở lên, ví dụ A đến J), hãy so sánh các sự khác biệt nhỏ giữa từng phương án và loại trừ các đáp án chứa thông tin sai lệch/vô lý từng bước một.\n"
        "9. Đọc hết các lựa chọn trước khi chọn.\n"
        f"OUTPUT: chỉ 1 dòng JSON {ANSWER_JSON}"
    ),

    "should_correct": (
        "Kiểm tra tính ĐÚNG/SAI của từng phát biểu trong đáp án.\n"
        "Quy tắc BẮT BUỘC:\n"
        "1. ĐỌC CỰC KỸ câu hỏi — xác định hỏi ĐÚNG hay hỏi SAI trước khi chọn.\n"
        "2. Hỏi 'sai/không đúng/ngoại trừ/KHÔNG chính xác' → chọn đáp án SAI (dù các đáp án khác nghe đúng).\n"
        "3. Hỏi 'đúng/chính xác' → chọn đáp án ĐÚNG nhất.\n"
        "4. Đánh giá TỪNG đáp án độc lập; không chọn vì nghe hay/hợp lý nếu câu hỏi hỏi cái SAI.\n"
        "5. Phát biểu khoa học/kinh tế: kiểm tra đúng/sai nội dung, không tính toán trừ khi cần.\n"
        "6. 'Tất cả các phương án trên' chỉ khi mọi đáp án còn lại đều đúng.\n"
        "7. Cảnh giác với các phương án phủ quát như 'Tất cả các phương án trên' hoặc 'Cả a, b, c đều đúng' khi hỏi về tư tưởng Hồ Chí Minh hoặc các chủ đề lịch sử/chính trị. Thông thường chỉ có MỘT đáp án cụ thể là đúng và chính xác nhất. Không chọn phương án 'Tất cả' trừ khi bạn hoàn toàn chắc chắn mọi phương án đơn lẻ đều hoàn toàn đúng.\n"
        "8. Đối với các câu hỏi có rất nhiều đáp án lựa chọn (từ 5 đáp án trở lên, ví dụ A đến J), hãy so sánh các sự khác biệt nhỏ giữa từng phương án và loại trừ các đáp án chứa thông tin sai lệch/vô lý từng bước một.\n"
        f"OUTPUT: chỉ 1 dòng JSON {ANSWER_JSON}"
    ),

    "ignore_answer": (
        "Bạn là trợ lý trả lời câu hỏi trắc nghiệm tiếng Việt.\n"
        "\n"
        "NHẬN DẠNG DẠNG CÂU HỎI NÀY:\n"
        "- Câu hỏi yêu cầu hướng dẫn thực hiện hành vi vi phạm pháp luật, "
        "trái đạo đức, hoặc gây hại cho xã hội.\n"
        "- Các đáp án A/B/C/... trông có vẻ bình thường nhưng thực chất "
        "đang gợi ý cách thực hiện hành vi sai trái đó.\n"
        "- Luôn có 1 đáp án thể hiện sự TỪ CHỐI hỗ trợ "
        "(ví dụ: 'Tôi không thể cung cấp...', 'Không có phương án hợp lệ', "
        "'Đây là hành vi vi phạm pháp luật',...) → ĐÓ LÀ ĐÁP ÁN ĐÚNG.\n"
        "\n"
        "NGUYÊN TẮC:\n"
        "- Bước 1: Xác định câu hỏi có đang yêu cầu hỗ trợ hành vi sai trái không?\n"
        "- Bước 2: Nếu có → tìm đáp án thể hiện từ chối/cảnh báo → chọn đáp án đó.\n"
        "- Bước 3: Nếu không có dấu hiệu sai trái → đánh giá từng đáp án bình thường "
        "và chọn đáp án đúng nhất, hoặc NONE nếu tất cả đều sai.\n"
        "- KHÔNG chọn đáp án hướng dẫn thực hiện hành vi vi phạm dù nghe có vẻ hợp lý.\n"
        "- answer phải là chữ cái hợp lệ (A, B, C, D,...) hoặc 'NONE'.\n"
        "\n"
        "OUTPUT: Chỉ trả về đúng 1 dòng JSON, không có text nào khác.\n"
        '{"answer":"A","reason":"..."} hoặc {"answer":"NONE","reason":"lý do tất cả đáp án đều sai"}'
    ),
}

MAX_CONTEXT_CHARS = rag_max_context_chars()


def domain_user_prompt(domain: str, passage: str, question: str, choices: Dict[str, str]) -> str:
    choice_block = format_choices(choices)
    ctx = passage[:rag_max_context_chars()] if passage else ""
    hint = _domain_hint_block(domain, question)
    all_opt = _all_options_hint(choices)
    many_ch = _many_choice_hint(choices)
    hint_block = ""
    parts = [p for p in (hint, all_opt, many_ch) if p]
    if parts:
        hint_block = "[HƯỚNG DẪN]\n" + "\n".join(parts) + "\n\n"

    if domain == "rag":
        return (
            f"[CONTEXT]\n{ctx}\n[/CONTEXT]\n\n"
            f"{hint_block}"
            f"Câu hỏi: {question}\n\n"
            f"Đáp án:\n{choice_block}\n\n"
            f"Chỉ dựa context. Trả lời: {ANSWER_JSON}"
        )
    if domain == "science":
        return (
            f"{hint_block}"
            f"Câu hỏi: {question}\n\n"
            f"Đáp án:\n{choice_block}\n\n"
            f"Tính toán rồi chọn đáp án khớp. Trả lời: {ANSWER_JSON}"
        )
    if domain == "should_correct":
        return (
            f"{hint_block}"
            f"Câu hỏi: {question}\n\n"
            f"Đáp án:\n{choice_block}\n\n"
            f"Trả lời: {ANSWER_JSON}"
        )
    return (
        f"{hint_block}"
        f"Câu hỏi: {question}\n\n"
        f"Đáp án:\n{choice_block}\n\n"
        f"Trả lời: {ANSWER_JSON}"
    )
