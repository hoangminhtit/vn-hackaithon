from typing import Dict, List
import json
import os

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
        "Trả lời câu hỏi trắc nghiệm tiếng Việt DỰA TRÊN context.\n"
        "Quy tắc:\n"
        "1. ĐỌC KỸ toàn bộ context trước khi trả lời.\n"
        "2. Đáp án PHẢI được hỗ trợ bởi thông tin CỤ THỂ trong context.\n"
        "3. Nếu context nói rõ 1 đáp án → chọn đáp án đó.\n"
        "4. Không suy luận ngoài context.\n"
        "5. Đọc TẤT CẢ các đáp án A,B,C,D trước khi chọn — KHÔNG mặc định chọn A.\n"
        f"OUTPUT: {ANSWER_JSON} (thay X bằng A/B/C/D/...)"
    ),

    "science": (
        "Chuyên gia toán/khoa học. Tự tính toán nội bộ, đối chiếu kết quả với đáp án.\n"
        "KHÔNG in lời giải, KHÔNG markdown, KHÔNG giải thích.\n"
        f"OUTPUT: đúng 1 dòng JSON: {ANSWER_JSON}"
    ),

    "multi_domain": (
        "Trả lời câu hỏi trắc nghiệm tổng hợp tiếng Việt.\n"
        "Quy tắc:\n"
        "1. Xác định lĩnh vực: lịch sử, địa lý, luật, chính trị, kinh tế, CNTT...\n"
        "2. Đọc TẤT CẢ các đáp án trước khi chọn.\n"
        "3. Loại trừ đáp án rõ sai.\n"
        "4. Nếu câu hỏi hỏi 'là gì/ở đâu/khi nào' → chọn đáp án chính xác nhất.\n"
        "5. Nếu có đáp án 'Tất cả/Cả A,B,C' → kiểm tra mỗi đáp án kia, nếu tất cả đều đúng thì chọn nó.\n"
        "6. Câu hỏi 'Theo X, ...' → trả lời theo quan điểm/tư tưởng của X, không theo ý kiến riêng.\n"
        f"OUTPUT: {ANSWER_JSON} (thay X bằng A/B/C/D/...)"
    ),

    "should_correct": (
        "Chọn phát biểu/nhận định ĐÚNG hoặc SAI theo yêu cầu câu hỏi.\n"
        "Quy tắc:\n"
        "1. Hỏi 'sai/không đúng/ngoại trừ' → tìm đáp án SAI/không chính xác.\n"
        "2. Hỏi 'đúng/chính xác' → tìm đáp án ĐÚNG nhất.\n"
        "3. Đánh giá TỪNG đáp án độc lập dựa trên kiến thức thực tế.\n"
        "4. 'Tất cả các phương án trên' chỉ chọn khi mọi đáp án còn lại đều đúng.\n"
        f"OUTPUT: {ANSWER_JSON} (thay X bằng A/B/C/D/...)"
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

MAX_CONTEXT_CHARS = 6000


def domain_user_prompt(domain: str, passage: str, question: str, choices: Dict[str, str]) -> str:
    choice_block = format_choices(choices)
    ctx = passage[:MAX_CONTEXT_CHARS] if passage else ""
    if domain == "rag":
        return (
            f"[CONTEXT]\n{ctx}\n[/CONTEXT]\n\n"
            f"Câu hỏi: {question}\n\n"
            f"Đáp án:\n{choice_block}\n\n"
            "Dựa vào context ở trên, đáp án đúng là:"
        )
    if domain == "science":
        return (
            f"Câu hỏi: {question}\n\n"
            f"Đáp án:\n{choice_block}\n\n"
            f"JSON: {ANSWER_JSON}"
        )
    return (
        f"Câu hỏi: {question}\n\n"
        f"Đáp án:\n{choice_block}\n\n"
        "Đáp án đúng là:"
    )
