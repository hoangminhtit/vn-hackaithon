from typing import Dict


def format_choices(choices: Dict[str, str]) -> str:
    return "\n".join(f"{label}. {text}" for label, text in choices.items())


ROUTER_SYSTEM_PROMPT = """Bạn là bộ phân loại câu hỏi tiếng Việt. Chỉ trả về JSON, không giải thích.

Các domain:
- rag: câu hỏi dựa trên đoạn văn/passage được cung cấp
- math: toán học, tính toán, phương trình, xác suất, thống kê
- multi_domain: kết hợp nhiều lĩnh vực, cần tư duy tổng hợp
- should_correct: yêu cầu xác định phát biểu đúng hay sai
- ignore_answer: không có đáp án nào đúng trong các lựa chọn
"""


def router_user_prompt(passage: str, question: str, num_choices: int) -> str:
    return (
        f"Passage (nếu có): {passage[:200]}\n"
        f"Câu hỏi: {question}\n"
        f"Số lượng đáp án: {num_choices}\n\n"
        'Trả lời JSON: {"domain": "rag", "confidence": 0.95}'
    )


DOMAIN_SYSTEM_PROMPTS = {
    "rag": (
        "Bạn là trợ lý trả lời câu hỏi dựa trên tài liệu tiếng Việt.\n"
        "Chỉ sử dụng thông tin trong phần context được cung cấp.\n"
        "Chỉ trả về JSON hợp lệ, không giải thích ngoài JSON.\n"
        'Định dạng bắt buộc: {"answer":"A","reason":"...ngắn gọn..."}'
    ),
    "math": (
        "Bạn là chuyên gia toán học.\n"
        "Hãy suy luận từng bước, tự kiểm tra lại kết quả trước khi kết luận.\n"
        "Không cần trình bày dài dòng, nhưng phải chính xác.\n"
        "Chỉ trả về JSON hợp lệ.\n"
        'Định dạng bắt buộc: {"answer":"A","reason":"..."}'
    ),
    "multi_domain": (
        "Bạn là trợ lý phân tích câu hỏi tổng hợp đa lĩnh vực tiếng Việt.\n"
        "Hãy xác định trọng tâm câu hỏi, phân tích từng đáp án, loại trừ đáp án sai.\n"
        "Chỉ trả về JSON hợp lệ.\n"
        'Định dạng bắt buộc: {"answer":"A","reason":"..."}'
    ),
    "should_correct": (
        "Bạn là chuyên gia kiểm tra tính đúng/sai của thông tin tiếng Việt.\n"
        "Xác định phát biểu cần kiểm tra, đối chiếu và chọn đáp án chỉnh sửa đúng nhất.\n"
        "Chỉ trả về JSON hợp lệ.\n"
        'Định dạng khuyến nghị: {"is_correct":false,"correction":"...","answer":"B"}'
    ),
    "ignore_answer": (
        "Bạn là trợ lý phát hiện câu bẫy khi không có đáp án nào đúng.\n"
        "Đánh giá từng lựa chọn trước khi kết luận.\n"
        "Nếu không có đáp án đúng, ưu tiên chọn lựa chọn thể hiện 'không thể cung cấp thông tin' hoặc tương đương.\n"
        "Chỉ trả về JSON hợp lệ.\n"
        'Định dạng cho phép: {"answer":"A","reason":"..."} hoặc {"answer":"NONE","reason":"..."}'
    ),
}


def domain_user_prompt(domain: str, passage: str, question: str, choices: Dict[str, str]) -> str:
    choice_block = format_choices(choices)
    if domain == "rag":
        return (
            "### Thông tin liên quan:\n"
            f"{passage}\n\n"
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
            "Yêu cầu:\n"
            "1) Phân tích bước 1.\n"
            "2) Phân tích bước 2.\n"
            "3) Kiểm tra lại phép tính/kết luận.\n"
            'Trả lời JSON duy nhất: {"answer":"A","reason":"..."}'
        )
    if domain == "multi_domain":
        return (
            f"Câu hỏi: {question}\n\n"
            f"Các đáp án:\n{choice_block}\n\n"
            "Hãy làm theo các bước:\n"
            "1) Xác định lĩnh vực/khía cạnh chính của câu hỏi.\n"
            "2) Phân tích từng đáp án dựa trên kiến thức liên quan.\n"
            "3) Loại đáp án sai, chọn đáp án chính xác nhất.\n"
            'Trả lời JSON duy nhất: {"answer":"A","reason":"..."}'
        )
    if domain == "should_correct":
        return (
            f"Câu hỏi/Phát biểu cần kiểm tra: {question}\n\n"
            f"Các đáp án:\n{choice_block}\n\n"
            "Suy nghĩ theo trình tự:\n"
            "1) Phát biểu cần kiểm tra là gì?\n"
            "2) Phát biểu đúng hay sai? Vì sao?\n"
            "3) Nếu sai, đáp án nào sửa đúng nhất?\n"
            'Trả lời JSON duy nhất: {"is_correct":false,"correction":"...","answer":"B"}'
        )
    if domain == "ignore_answer":
        return (
            f"Câu hỏi: {question}\n\n"
            f"Các đáp án:\n{choice_block}\n\n"
            "Kiểm tra từng đáp án là đúng hay sai.\n"
            "Nếu có đáp án đúng thì chọn chữ cái tương ứng.\n"
            'Nếu không có đáp án đúng, có thể trả {"answer":"NONE","reason":"..."}.\n'
            'Trả lời JSON duy nhất: {"answer":"A","reason":"..."}'
        )
    return (
        f"Câu hỏi: {question}\n\n"
        f"Các đáp án:\n{choice_block}\n\n"
        'Trả lời JSON theo format: {"answer":"A","reason":"..."}'
    )
