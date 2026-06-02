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
        "Bạn là trợ lý trả lời câu hỏi dựa trên tài liệu tiếng Việt.\n"
        "Chỉ sử dụng thông tin trong phần context được cung cấp.\n"
        "KHÔNG in reasoning/chain-of-thought/thinking process.\n"
        "Chỉ trả về đúng 1 dòng JSON hợp lệ, không text nào ngoài JSON.\n"
        'Định dạng bắt buộc: {"answer":"A","reason":"...ngắn gọn..."}'
    ),
    "math": (
        "Bạn là chuyên gia toán học.\n"
        "Tự suy luận nội bộ để chọn đáp án chính xác.\n"
        "KHÔNG in reasoning/chain-of-thought/thinking process.\n"
        "Chỉ trả về đúng 1 dòng JSON hợp lệ.\n"
        'Định dạng bắt buộc: {"answer":"A","reason":"..."}'
    ),
    "multi_domain": (
        "Bạn là trợ lý phân tích câu hỏi tổng hợp đa lĩnh vực tiếng Việt.\n"
        "Tự phân tích nội bộ và chọn đáp án tốt nhất.\n"
        "KHÔNG in reasoning/chain-of-thought/thinking process.\n"
        "Chỉ trả về đúng 1 dòng JSON hợp lệ.\n"
        'Định dạng bắt buộc: {"answer":"A","reason":"..."}'
    ),
    "should_correct": (
        "Bạn là chuyên gia kiểm tra tính đúng/sai của thông tin tiếng Việt.\n"
        "Tự đối chiếu nội bộ và chọn đáp án chỉnh sửa đúng nhất.\n"
        "KHÔNG in reasoning/chain-of-thought/thinking process.\n"
        "Chỉ trả về đúng 1 dòng JSON hợp lệ.\n"
        'Định dạng khuyến nghị: {"is_correct":false,"correction":"...","answer":"B"}'
    ),
    "ignore_answer": (
        "Bạn là trợ lý phát hiện câu bẫy khi không có đáp án nào đúng.\n"
        "Tự đánh giá nội bộ từng lựa chọn trước khi kết luận.\n"
        "Nếu không có đáp án đúng, ưu tiên chọn lựa chọn thể hiện 'không thể cung cấp thông tin' hoặc tương đương.\n"
        "KHÔNG in reasoning/chain-of-thought/thinking process.\n"
        "Chỉ trả về đúng 1 dòng JSON hợp lệ.\n"
        'Định dạng cho phép: {"answer":"A","reason":"..."} hoặc {"answer":"NONE","reason":"..."}'
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
