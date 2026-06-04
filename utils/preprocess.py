import re
from typing import Dict, List


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def preprocess(item: Dict) -> Dict:
    question_raw = item.get("question", "")
    choices: List[str] = item.get("choices", [])

    labels = [chr(ord("A") + i) for i in range(len(choices))]
    choice_map = {label: text for label, text in zip(labels, choices)}

    passage = ""
    question_clean = question_raw

    passage_patterns = [
        r"(Đoạn thông tin:.*?)(?=Câu hỏi:)",
        r"(\[1\].*?)(?=Câu hỏi:)",
        r"(Nội dung:.*?)(?=Câu hỏi:)",
        r"(Dựa vào đoạn văn[^.]*\.)(?=Câu hỏi:)",
    ]

    for pattern in passage_patterns:
        match = re.search(pattern, question_raw, re.DOTALL | re.IGNORECASE)
        if match:
            passage = match.group(1).strip()
            passage = re.sub(r"^Đoạn thông tin:\s*", "", passage, flags=re.IGNORECASE).strip()
            passage = re.sub(r"^\[\d+\]\s*Tiêu đề:\s*[^\n]*\n?", "", passage, flags=re.IGNORECASE).strip()
            question_clean = question_raw[match.end() :].replace("Câu hỏi:", "").strip()
            break

    if not passage and "Câu hỏi:" in question_raw and len(question_raw) > 600:
        parts = question_raw.split("Câu hỏi:", maxsplit=1)
        if len(parts) == 2:
            passage = parts[0].strip()
            question_clean = parts[1].strip()

    return {
        "qid": item.get("qid", ""),
        "passage": passage,
        "question": _normalize_space(question_clean),
        "choices": choice_map,
        "num_choices": len(choices),
    }
