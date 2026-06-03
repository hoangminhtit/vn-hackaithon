import json
import re
from typing import Dict, List


def valid_labels(num_choices: int) -> List[str]:
    return [chr(ord("A") + i) for i in range(max(num_choices, 0))]


def extract_last_json_object_with_key(raw_output: str, required_key: str) -> Dict | None:
    clean = re.sub(r"```json|```", "", raw_output).strip()
    decoder = json.JSONDecoder()
    candidates: List[Dict] = []
    idx = 0
    while idx < len(clean):
        try:
            obj, end_idx = decoder.raw_decode(clean, idx)
            if isinstance(obj, dict) and required_key in obj:
                candidates.append(obj)
            idx = end_idx
        except json.JSONDecodeError:
            idx += 1
    return candidates[-1] if candidates else None


def parse_answer(raw_output: str, num_choices: int) -> str:
    labels = valid_labels(num_choices)
    if not labels:
        return "A"

    match = re.search(r'"answer"\s*:\s*"([A-Z])"', raw_output, re.IGNORECASE)
    if match:
        answer = match.group(1).upper()
        if answer in labels:
            return answer
    if match and match.group(1).upper() == "NONE":
        return "NONE"

    try:
        data = extract_last_json_object_with_key(raw_output, "answer")
        if data:
            answer = str(data.get("answer", "")).strip().upper()
            if answer in labels:
                return answer
    except Exception:
        pass

    # Look for strong answer patterns (search from end for recency)
    strong_patterns = [
        r'(?:đáp án đúng|đáp án|answer|chọn)\s*(?:là|:|\s)\s*\**\s*([A-Z])\b',
        r'(?:^|\n)\s*\**\s*([A-Z])\s*\**\s*$',  # standalone letter on line
    ]
    for pat in strong_patterns:
        matches = list(re.finditer(pat, raw_output, re.IGNORECASE | re.MULTILINE))
        if matches:
            ans = matches[-1].group(1).upper()
            if ans in labels:
                return ans

    # Last resort: find the LAST single capital letter in valid range
    all_caps = re.findall(r'\b([A-Z])\b', raw_output)
    for letter in reversed(all_caps):
        if letter in labels:
            return letter

    return labels[0]


def parse_route_output(raw_output: str) -> Dict:
    default = {"domain": "multi_domain", "confidence": 0.0}
    try:
        data = extract_last_json_object_with_key(raw_output, "domain")
        if not data:
            return default
        domain = str(data.get("domain", "multi_domain")).strip().lower()
        if domain == "math":
            domain = "science"
        confidence = float(data.get("confidence", 0.0))
        if domain not in {"rag", "science", "math", "multi_domain", "should_correct", "ignore_answer"}:
            domain = "multi_domain"
        confidence = max(0.0, min(1.0, confidence))
        return {"domain": domain, "confidence": confidence}
    except Exception:
        return default


def to_submission_row(result: Dict) -> Dict[str, str]:
    return {"qid": result["qid"], "answer": result["answer"]}
