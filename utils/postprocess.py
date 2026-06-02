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

    try:
        data = extract_last_json_object_with_key(raw_output, "answer")
        if data:
            answer = str(data.get("answer", "")).strip().upper()
            if answer in labels:
                return answer
    except Exception:
        pass

    for label in labels:
        if re.search(rf"\b{label}\b", raw_output):
            return label

    return labels[0]


def parse_route_output(raw_output: str) -> Dict:
    default = {"domain": "multi_domain", "confidence": 0.0}
    try:
        data = extract_last_json_object_with_key(raw_output, "domain")
        if not data:
            return default
        domain = str(data.get("domain", "multi_domain")).strip().lower()
        confidence = float(data.get("confidence", 0.0))
        if domain not in {"rag", "math", "multi_domain", "should_correct", "ignore_answer"}:
            domain = "multi_domain"
        confidence = max(0.0, min(1.0, confidence))
        return {"domain": domain, "confidence": confidence}
    except Exception:
        return default


def to_submission_row(result: Dict) -> Dict[str, str]:
    return {"qid": result["qid"], "answer": result["answer"]}
