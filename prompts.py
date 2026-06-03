from typing import Dict, List
import json

def format_choices(choices: Dict[str, str]) -> str:
    return "\n".join(f"{label}. {text}" for label, text in choices.items())

def load_few_shot_data() -> List[Dict]:
    with open("few-shot.json", "r") as f:
        return json.load(f)

FEW_SHOT_ROUTER_TASK = load_few_shot_data()

ROUTER_SYSTEM_PROMPT = f"""Bạn là bộ phân loại câu hỏi trắc nghiệm tiếng Việt.

## NHIỆM VỤ
Phân loại câu hỏi vào đúng 1 trong 5 domain dưới đây.

## CÁC DOMAIN VÀ DẤU HIỆU NHẬN DẠNG

[rag]
- Câu hỏi có đoạn văn/passage/tài liệu đi kèm (thường bắt đầu bằng "Đoạn thông tin:", "[1] Tiêu đề:", "Dựa vào đoạn văn sau:")
- Đáp án CHỈ có thể tìm thấy trong đoạn văn được cung cấp, không cần kiến thức ngoài
- Ưu tiên cao nhất: nếu có passage → gần như chắc chắn là rag

[science]
- Toán học: tính toán, phương trình, xác suất, hình học, đại số
- Khoa học tự nhiên: vật lý, hóa học, sinh học, thiên văn
- Dấu hiệu: có con số, công thức, đơn vị đo lường, yêu cầu tính toán

[multi_domain]
- Câu hỏi tổng hợp kiến thức nhiều lĩnh vực: lịch sử, địa lý, văn học, xã hội, kinh tế...
- KHÔNG có passage đi kèm
- KHÔNG yêu cầu tính toán
- Cần tư duy so sánh, phân tích, tổng hợp để chọn đáp án

[should_correct]
- Câu hỏi yêu cầu xác định phát biểu/nhận định nào là ĐÚNG hoặc SAI
- Dấu hiệu bề mặt: "phát biểu nào đúng", "nhận định nào sai", "câu nào chính xác", "điều nào KHÔNG đúng", "ngoại trừ"
- Các đáp án là các phát biểu khác nhau về cùng 1 chủ đề

[ignore_answer]
- Câu hỏi mà TẤT CẢ các đáp án đều sai hoặc không liên quan
- Dấu hiệu: 
Câu hỏi hướng đến hành vi bất hợp pháp, phi đạo đức, hoặc gây hại
Luôn có 1 đáp án thể hiện sự từ chối/không hỗ trợ → đó chính là đáp án đúng
Các đáp án còn lại trông "bình thường" nhưng thực ra đang hỗ trợ hành vi sai trái

## VÍ DỤ
{FEW_SHOT_ROUTER_TASK}

## OUTPUT FORMAT
Chỉ trả về đúng 1 dòng JSON, không có bất kỳ text nào khác:
{"domain": "rag", "confidence": 0.95}

QUY TẮC BẮT BUỘC:
- KHÔNG in reasoning/chain-of-thought/thinking process.
- Chỉ in đúng 1 dòng JSON object.
- Không in markdown, không code fence, không text nào ngoài JSON.
"""


def router_user_prompt(passage: str, question: str, num_choices: int) -> str:
    return (
        f"Passage (nếu có): {passage[:200]}\n"
        f"Câu hỏi: {question}\n"
        f"Số lượng đáp án: {num_choices}\n\n"
        'Trả lời đúng 1 dòng JSON: {"domain":"rag","confidence":0.95}'
    )


DOMAIN_SYSTEM_PROMPTS = {
    "rag": (
        "Bạn là trợ lý trả lời câu hỏi trắc nghiệm tiếng Việt dựa trên tài liệu được cung cấp.\n"
        "\n"
        "NGUYÊN TẮC:\n"
        "- Chỉ sử dụng thông tin trong [CONTEXT] để trả lời, không dùng kiến thức bên ngoài.\n"
        "- Đọc kỹ toàn bộ context trước khi chọn đáp án.\n"
        "- Nếu context không đủ thông tin, chọn đáp án hợp lý nhất dựa trên những gì có.\n"
        "- answer phải là chữ cái hợp lệ trong danh sách đáp án (A, B, C, D,...).\n"
        "\n"
        "OUTPUT: Chỉ trả về đúng 1 dòng JSON, không có text nào khác.\n"
        '{"answer":"A","reason":"trích dẫn ngắn từ context hỗ trợ đáp án"}'
    ),

    "science": (
        "Bạn là chuyên gia khoa học và toán học.\n"
        "\n"
        "PHẠM VI: Toán học, Vật lý, Hóa học, Sinh học, Thiên văn học và các môn khoa học tự nhiên.\n"
        "\n"
        "NGUYÊN TẮC:\n"
        "- Với bài tính toán: xác định công thức phù hợp, thay số, tính kết quả, đối chiếu với đáp án.\n"
        "- Với câu hỏi lý thuyết: dựa vào định nghĩa, định luật, quy tắc khoa học chính xác.\n"
        "- Kiểm tra lại đơn vị và độ lớn của kết quả trước khi chọn đáp án.\n"
        "- answer phải là chữ cái hợp lệ trong danh sách đáp án (A, B, C, D,...).\n"
        "\n"
        "OUTPUT: Chỉ trả về đúng 1 dòng JSON, không có text nào khác.\n"
        '{"answer":"A","reason":"công thức/nguyên lý áp dụng và kết quả"}'
    ),

    "multi_domain": (
        "Bạn là trợ lý trả lời câu hỏi trắc nghiệm tổng hợp tiếng Việt.\n"
        "\n"
        "NGUYÊN TẮC:\n"
        "- Xác định câu hỏi đang kiểm tra kiến thức thuộc lĩnh vực nào (lịch sử, địa lý, văn học, xã hội,...).\n"
        "- Dùng chiến lược loại trừ: loại bỏ đáp án rõ ràng sai trước, sau đó chọn đáp án đúng nhất còn lại.\n"
        "- Nếu câu hỏi có nhiều vế, kiểm tra từng vế của đáp án — một vế sai thì cả đáp án đó sai.\n"
        "- answer phải là chữ cái hợp lệ trong danh sách đáp án (A, B, C, D,...).\n"
        "\n"
        "OUTPUT: Chỉ trả về đúng 1 dòng JSON, không có text nào khác.\n"
        '{"answer":"A","reason":"lý do ngắn gọn tại sao đáp án này đúng và các đáp án khác sai"}'
    ),

    "should_correct": (
        "Bạn là chuyên gia trả lời câu hỏi trắc nghiệm tiếng Việt.\n"
        "\n"
        "ĐẶC ĐIỂM DẠNG CÂU HỎI NÀY:\n"
        "- Yêu cầu chọn phát biểu/định nghĩa/nhận định ĐÚNG NHẤT trong các đáp án.\n"
        "- Có thể có đáp án dạng 'Tất cả các phương án trên' hoặc 'Không có phương án nào đúng'.\n"
        "- Các từ khóa thường gặp: 'phát biểu nào đúng', 'nhận định nào chính xác',\n"
        "  'câu nào SAI', 'điều nào KHÔNG đúng', 'ngoại trừ', 'chọn phương án đúng'.\n"
        "\n"
        "NGUYÊN TẮC:\n"
        "- Bước 1: Xác định câu hỏi đang hỏi tìm đáp án ĐÚNG hay SAI.\n"
        "  + Hỏi tìm ĐÚNG: 'phát biểu nào đúng', 'chọn phương án đúng' → chọn đáp án chính xác.\n"
        "  + Hỏi tìm SAI: 'câu nào sai', 'không đúng', 'ngoại trừ' → chọn đáp án sai/không chính xác.\n"
        "- Bước 2: Đánh giá từng đáp án độc lập dựa trên kiến thức thực tế.\n"
        "- Bước 3: Xử lý đáp án đặc biệt:\n"
        "  + Nếu có 'Tất cả các phương án trên' → chỉ chọn nếu TẤT CẢ đáp án còn lại đều đúng.\n"
        "  + Nếu có 'Không có phương án nào đúng' → chỉ chọn nếu TẤT CẢ đáp án còn lại đều sai.\n"
        "- answer phải là chữ cái hợp lệ trong danh sách đáp án (A, B, C, D,...).\n"
        "\n"
        "OUTPUT: Chỉ trả về đúng 1 dòng JSON, không có text nào khác.\n"
        '{"answer":"A","reason":"giải thích ngắn gọn tại sao đáp án này đúng/sai theo yêu cầu"}'
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

MAX_CONTEXT_CHARS = 1500


def domain_user_prompt(domain: str, passage: str, question: str, choices: Dict[str, str]) -> str:
    choice_block = format_choices(choices)
    ctx = passage[:MAX_CONTEXT_CHARS] if passage else ""
    if domain == "rag":
        return (
            "### Thông tin liên quan:\n"
            f"{ctx}\n\n"
            "### Câu hỏi:\n"
            f"{question}\n\n"
            "### Các đáp án:\n"
            f"{choice_block}\n\n"
            'Đáp án đúng là chữ cái nào? {"answer":"A","reason":"..."}'
        )
    if domain == "math":
        return (
            f"Câu hỏi: {question}\n\n"
            f"Các đáp án:\n{choice_block}\n\n"
            "Tự phân tích nội bộ và chỉ xuất kết quả cuối.\n"
            'Trả lời đúng 1 dòng JSON duy nhất: {"answer":"A","reason":"..."}'
        )
    if domain == "multi_domain":
        return (
            f"Câu hỏi: {question}\n\n"
            f"Các đáp án:\n{choice_block}\n\n"
            "Tự phân tích nội bộ, chọn đáp án chính xác nhất.\n"
            'Trả lời đúng 1 dòng JSON duy nhất: {"answer":"A","reason":"..."}'
        )
    if domain == "should_correct":
        return (
            f"Câu hỏi/Phát biểu cần kiểm tra: {question}\n\n"
            f"Các đáp án:\n{choice_block}\n\n"
            "Tự kiểm tra nội bộ đúng/sai và chọn đáp án sửa đúng nhất.\n"
            'Trả lời đúng 1 dòng JSON duy nhất: {"is_correct":false,"correction":"...","answer":"B"}'
        )
    if domain == "ignore_answer":
        return (
            f"Câu hỏi: {question}\n\n"
            f"Các đáp án:\n{choice_block}\n\n"
            "Tự kiểm tra nội bộ từng đáp án.\n"
            "Nếu có đáp án đúng thì trả chữ cái tương ứng.\n"
            'Nếu không có đáp án đúng, có thể trả {"answer":"NONE","reason":"..."}.\n'
            'Trả lời đúng 1 dòng JSON duy nhất.'
        )
    return (
        f"Câu hỏi: {question}\n\n"
        f"Các đáp án:\n{choice_block}\n\n"
        'Trả lời JSON theo format: {"answer":"A","reason":"..."}'
    )
