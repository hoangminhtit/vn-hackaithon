import json
import re
from typing import Dict, List


def valid_labels(num_choices: int) -> List[str]:
    return [chr(ord("A") + i) for i in range(max(num_choices, 0))]


def parse_answer(raw_output: str, num_choices: int) -> str:
    labels = valid_labels(num_choices)
    if not labels:
        return "A"

    try:
        clean = re.sub(r"```json|```", "", raw_output).strip()
        json_matches = re.findall(r'\{[^{}]*"answer"[^{}]*\}', clean)
        if json_matches:
            data = json.loads(json_matches[-1])
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
        clean = re.sub(r"```json|```", "", raw_output).strip()
        json_matches = re.findall(r'\{[^{}]*"domain"[^{}]*\}', clean)
        if not json_matches:
            return default
        data = json.loads(json_matches[-1])
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
