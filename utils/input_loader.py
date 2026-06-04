import csv
import json
import os
import re
from typing import Dict, List


CHOICE_LABELS = [chr(ord("A") + i) for i in range(26)]


def _parse_choices_field(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(c) for c in parsed]
        except json.JSONDecodeError:
            pass
    if "|" in raw:
        return [p.strip() for p in raw.split("|") if p.strip()]
    return [raw]


def _choices_from_row(row: Dict[str, str]) -> List[str]:
    if "choices" in row and row["choices"]:
        return _parse_choices_field(row["choices"])

    choices: List[str] = []
    for label in CHOICE_LABELS:
        if label in row and str(row[label]).strip():
            choices.append(str(row[label]).strip())

    for key in sorted(row.keys()):
        if re.match(r"^choice[_\s]?([a-z])$", key, re.IGNORECASE) and str(row[key]).strip():
            choices.append(str(row[key]).strip())

    return choices


def read_csv_items(path: str) -> List[Dict]:
    items: List[Dict] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "qid" not in reader.fieldnames or "question" not in reader.fieldnames:
            raise ValueError(f"CSV must have columns qid, question (and choices or A,B,C,...): {path}")
        for row in reader:
            choices = _choices_from_row(row)
            if not choices:
                raise ValueError(f"Row {row.get('qid', '?')} has no choices in {path}")
            item: Dict = {
                "qid": str(row["qid"]).strip(),
                "question": str(row["question"]),
                "choices": choices,
            }
            if "answer" in row and str(row["answer"]).strip():
                item["answer"] = str(row["answer"]).strip().upper()
            items.append(item)
    return items


def read_json_items(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of {qid, question, choices}.")
    return data


def read_items(path: str) -> List[Dict]:
    lower = path.lower()
    if lower.endswith(".csv"):
        return read_csv_items(path)
    if lower.endswith(".json"):
        return read_json_items(path)
    raise ValueError(f"Unsupported input format (use .csv or .json): {path}")


def resolve_competition_input(data_dir: str = "/data") -> str:
    """BTC: ưu tiên public_test.csv, không có thì private_test.csv trong /data."""
    public_path = os.path.join(data_dir, "public_test.csv")
    private_path = os.path.join(data_dir, "private_test.csv")
    if os.path.isfile(public_path):
        return public_path
    if os.path.isfile(private_path):
        return private_path
    raise FileNotFoundError(
        f"No input in {data_dir}. Expected public_test.csv or private_test.csv"
    )
