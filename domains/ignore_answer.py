from typing import Dict

from domains.common import lexical_best_choice, normalize_for_match


REFUSE_HINTS = (
    "không thể cung cấp",
    "không thể hỗ trợ",
    "tôi không thể",
    "không hỗ trợ",
    "vi phạm pháp luật",
    "hành vi vi phạm",
    "không có phương án hợp lệ",
    "không có đáp án nào đúng",
    "tất cả đều sai",
    "none of the above",
    "không nên thực hiện",
    "trái pháp luật",
    "bất hợp pháp",
    "không được phép",
    "từ chối cung cấp",
    "không khuyến khích",
)


def solve(processed: Dict) -> str:
    """Giải quyết câu hỏi thuộc ignore_answer domain bằng heuristic trực tiếp.

    - Bỏ qua LLM hoàn toàn đối với domain này để tránh lỗi từ chối của LLM nhưng trả sai nhãn đáp án.
    - Quét trực tiếp các pattern từ chối trong danh sách choices.
    """
    choices = processed["choices"]
    for label, text in choices.items():
        normalized = normalize_for_match(text)
        if any(h in normalized for h in REFUSE_HINTS):
            return label
    return lexical_best_choice(processed["question"], choices)
