import argparse
import csv
import json
import os
from typing import List, Dict

from pipeline import run_pipeline
from utils.llm import LLMClient


def load_dotenv_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def read_input(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of {qid, question, choices}.")
    return data


def write_output(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "answer"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"qid": row["qid"], "answer": row["answer"]})


def main() -> None:
    load_dotenv_file(".env")

    parser = argparse.ArgumentParser(description="Run MCQ pipeline on public/private JSON test.")
    parser.add_argument(
        "--input",
        default="data/public-test_1780368312.json",
        help="Path to input JSON file.",
    )
    parser.add_argument(
        "--output",
        default="output/pred.csv",
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(os.cpu_count() or 1, 8),
        help="Number of worker threads.",
    )
    parser.add_argument(
        "--mode",
        choices=["heuristic", "llm", "auto"],
        default="auto",
        help="Pipeline mode. auto=use LLM if env is configured, else heuristic.",
    )
    args = parser.parse_args()

    items = read_input(args.input)
    llm_client = LLMClient.from_env()
    if args.mode == "heuristic":
        llm_client = None
    if args.mode == "llm" and llm_client is None:
        raise ValueError(
            "LLM mode requires either (LLM_API_URL + LLM_MODEL) "
            "or (HF_TOKEN + HF_MODEL_ID/LLM_MODEL) in environment or .env."
        )

    results = run_pipeline(items, max_workers=args.workers, llm_client=llm_client)
    write_output(args.output, results)
    mode_used = "llm" if llm_client else "heuristic"
    print(f"Wrote {len(results)} predictions to {args.output} (mode={mode_used})")


if __name__ == "__main__":
    main()
