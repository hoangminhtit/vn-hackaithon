import json
import re
from typing import Dict, List, Optional


def valid_labels(num_choices: int) -> List[str]:
    return [chr(ord("A") + i) for i in range(max(num_choices, 0))]


def extract_last_json_object_with_key(raw_output: str, required_key: str) -> Optional[Dict]:
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

    # ── Layer 1: "answer": "X" — works even on truncated JSON ──────────────
    match = re.search(r'"answer"\s*:\s*"([A-Z])"', raw_output, re.IGNORECASE)
    if match:
        answer = match.group(1).upper()
        if answer in labels:
            return answer
    # Handle NONE marker (ignore_answer domain)
    if re.search(r'"answer"\s*:\s*"NONE"', raw_output, re.IGNORECASE):
        return "NONE"

    # ── Layer 2: full JSON parse ─────────────────────────────────────────────
    try:
        data = extract_last_json_object_with_key(raw_output, "answer")
        if data:
            answer = str(data.get("answer", "")).strip().upper()
            if answer in labels:
                return answer
            if answer == "NONE":
                return "NONE"
    except Exception:
        pass

    # ── Layer 3: Vietnamese conclusion phrases (most → least specific) ───────
    vn_conclusion_patterns = [
        # "Đáp án cuối: A" — CoT pattern
        r'[Đđ]áp\s+án\s+cu[oô]i\s*[:\-]\s*\**\s*([A-Z])\b',
        # "Đáp án đúng là A" / "Đáp án là A"
        r'[Đđ]áp\s+án\s+(?:đúng\s+)?(?:là|:)\s*\**\s*([A-Z])\b',
        # "Chọn đáp án A" / "Chọn A"
        r'[Cc]h[oọ]n\s+(?:đáp\s+án\s+)?\**\s*([A-Z])\b',
        # "Câu trả lời là A" / "Câu trả lời: A"
        r'[Cc]âu\s+tr[aả]\s+l[oờ]i\s*(?:là|:)\s*\**\s*([A-Z])\b',
        # "Kết luận: A" / "Kết quả: A"
        r'[Kk]ết\s+(?:luận|quả)\s*[:\-]\s*\**\s*([A-Z])\b',
        # "Vậy đáp án là A"
        r'[Vv]ậy\s+(?:đáp\s+án|câu\s+trả\s+lời)\s*(?:là|:)\s*\**\s*([A-Z])\b',
        # English: "answer: A" / "Answer is A"
        r'[Aa]nswer\s*(?:is|:)\s*\**\s*([A-Z])\b',
        # Generic: "đáp án/answer" + colon/space + letter
        r'(?:đáp\s+án|answer|chọn)\s*[:\-\s]\s*\**\s*([A-Z])\b',
    ]
    for pat in vn_conclusion_patterns:
        matches = list(re.finditer(pat, raw_output, re.IGNORECASE))
        if matches:
            ans = matches[-1].group(1).upper()
            if ans in labels:
                return ans

    # ── Layer 4: standalone letter on its own line (A / **A** / (A)) ─────────
    standalone_patterns = [
        r'(?:^|\n)\s*\**\s*([A-Z])\s*\**\s*(?:\n|$)',   # plain or bolded
        r'(?:^|\n)\s*\(([A-Z])\)\s*(?:\n|$)',            # parenthesized (A)
    ]
    for pat in standalone_patterns:
        matches = list(re.finditer(pat, raw_output, re.MULTILINE))
        if matches:
            ans = matches[-1].group(1).upper()
            if ans in labels:
                return ans

    # ── Layer 5: last single capital letter in the valid range ───────────────
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
