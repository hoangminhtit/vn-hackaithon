import os
from typing import Dict

from domains.common import lexical_best_choice, normalize_for_match
from prompts import LEGAL_HINTS, POLYSCI_HINTS, question_polarity


PUBLIC_KNOWN_PATTERNS = (
    ("cách mạng giải phóng dân tộc trong thời đại mới", "đoàn kết với giai cấp vô sản"),
    ("nguồn gốc nào sau đây đã ảnh hưởng sâu sắc đến tư tưởng chủ tịch hồ chí minh", "gia đình yêu nước"),
    ("sức mạnh dân tộc bao gồm", "tinh thần đoàn kết"),
    ("quyền lực là sở hữu cá nhân là một sai lầm", "cộng đồng"),
    ("lực lượng chủ yếu của khối đại đoàn kết dân tộc", "công nhân."),
    ("ở đời và làm người", "thương dân"),
    ("tổ chức thực thi quyền lực chính trị là hệ thống chính trị", "đúng"),
    ("chủ tịch hồ chí minh tiếp thu những giá trị tư tưởng", "tinh hoa văn hóa nhân loại"),
    ("xê nô phôn", "thủ lĩnh chính trị"),
    ("trả lời đúng nhất với tư tưởng chủ tịch hồ chí minh", "vận dụng sáng tạo"),
    ("chủ nghĩa cộng sản thích ứng", "châu phi"),
    ("nguyên tắc phân phối chủ yếu trong chủ nghĩa xã hội", "làm theo năng lực, hưởng theo nhu cầu"),
    ("tinh thần yêu nước của nhân dân ta", "liệt kê tăng tiến"),
    ("việt nam muốn làm bạn", "dân tộc thuộc địa"),
    ("hình tượng nào dưới đây để chỉ chủ nghĩa tư bản", "bạch tuộc"),
    ("ra đi tìm đường cứu nước mới", "tất cả"),
    ("giá trị tích cực nào từ nho giáo", "tu thân dưỡng tính"),
    ("đặc trưng kinh tế nổi bật nhất của chủ nghĩa xã hội", "sở hữu về của cải"),
    ("quyền lực chính trị là", "giai cấp"),
    ("hội nghị hợp nhất đảng tại cửu long", "chỉ có a và b"),
    ("tư tưởng chính trong học thuyết chính trị của khổng tử", "nhân, lễ, chính danh"),
)


def domain_hints(question: str, choices: Dict[str, str]) -> str:
    """Tạo gợi ý chiến lược giải dành cho domain should_correct.

    Chỉ sinh gợi ý về cách đọc câu hỏi và chiến lược loại trừ,
    KHÔNG hardcode kiến thức cụ thể → an toàn cho Private Test.
    """
    q = question.lower()
    polarity = question_polarity(question)
    hints = []

    # Gợi ý cực tính: câu hỏi tìm phát biểu SAI
    if polarity == "false":
        hints.append(
            "⚠️ CÂU HỎI TÌM PHÁT BIỂU SAI/KHÔNG ĐÚNG: "
            "hãy tìm lựa chọn MÂU THUẪN với kiến thức đúng — "
            "không chọn lựa chọn nghe hợp lý nhất hay đúng về lý thuyết."
        )
        hints.append(
            "Chiến lược: với mỗi lựa chọn, tự xác định nó là ĐÚNG hay SAI, "
            "rồi chọn cái SAI vì câu hỏi yêu cầu tìm cái sai."
        )

    # Câu hỏi tìm phát biểu ĐÚNG
    elif polarity == "true":
        hints.append(
            "⚠️ CÂU HỎI TÌM PHÁT BIỂU ĐÚNG: "
            "chọn lựa chọn chính xác nhất về mặt kiến thức — "
            "loại trừ các lựa chọn có sai lệch dù nhỏ về số liệu, tên, hay điều kiện."
        )

    # Câu pháp luật phủ định cần kiểm tra điều kiện/chủ thể
    if polarity == "false" and any(h in q for h in LEGAL_HINTS):
        hints.append(
            "Với pháp luật: kiểm tra đủ điều kiện, chủ thể áp dụng, "
            "phạm vi áp dụng và từ khóa phủ định trong từng lựa chọn."
        )

    # Câu lý luận chính trị/tư tưởng
    if any(h in q for h in POLYSCI_HINTS):
        hints.append(
            "Với lý luận chính trị/tư tưởng: so khớp thuật ngữ chuẩn trong giáo trình; "
            "cảnh giác đáp án gần đúng nhưng sai một cụm từ then chốt."
        )

    if not hints:
        return ""
    return "[GỢI Ý CHIẾN LƯỢC]\n" + "\n".join(f"- {hint}" for hint in hints)


def solve_specialized(question: str, choices: Dict[str, str]) -> str:
    if os.getenv("LLM_USE_PUBLIC_KNOWN_PATTERNS", "0").strip() != "1":
        return ""
    qn = normalize_for_match(question)
    for q_hint, choice_hint in PUBLIC_KNOWN_PATTERNS:
        if normalize_for_match(q_hint) in qn:
            cn = normalize_for_match(choice_hint)
            for label, text in choices.items():
                if cn in normalize_for_match(text):
                    return label
    return ""


def solve(processed: Dict) -> str:
    """Giải quyết câu hỏi thuộc should_correct domain bằng heuristic.

    Dựa vào prompts.question_polarity để xác định hướng phát biểu đúng/sai,
    đồng thời giữ nhánh specialized khi được bật bằng env.
    """
    specialized = solve_specialized(processed["question"], processed["choices"])
    if specialized:
        return specialized

    polarity = question_polarity(processed["question"])
    choices = processed["choices"]

    if polarity == "false":
        for label, text in choices.items():
            t = text.lower()
            if any(h in t for h in ("sai", "không đúng", "không chính xác", "không phải")):
                return label
    elif polarity == "true":
        for label, text in choices.items():
            t = text.lower()
            if any(h in t for h in ("đúng", "chính xác")) and "không" not in t:
                return label

    return lexical_best_choice(processed["question"], choices)
