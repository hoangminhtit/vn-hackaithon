import argparse
import csv
import json
import os
from typing import Dict, List, Optional

from pipeline import run_pipeline
from utils.input_loader import read_items, resolve_competition_input
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
    return read_items(path)


def write_output(path: str, rows: List[Dict]) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "answer"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"qid": row["qid"], "answer": row["answer"]})


def write_jsonl(path: str, rows: List[Dict]) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve_input_path(cli_input: Optional[str]) -> str:
    if cli_input:
        return cli_input
    if os.getenv("COMPETITION", "").strip() == "1":
        return resolve_competition_input(os.getenv("DATA_DIR", "/data"))
    return "data/public-test_1780368312.json"


def _resolve_output_path(cli_output: Optional[str]) -> str:
    if cli_output:
        return cli_output
    if os.getenv("COMPETITION", "").strip() == "1":
        return os.path.join(os.getenv("OUTPUT_DIR", "/output"), "pred.csv")
    return "output/pred.csv"


def main() -> None:
    load_dotenv_file(".env")

    parser = argparse.ArgumentParser(
        description="Run MCQ pipeline. Chạy local: truyền --input JSON. Docker BTC: COMPETITION=1."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Input .json hoặc .csv. Mặc định local: data/public-test_1780368312.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV (qid, answer). Mặc định local: output/pred.csv",
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
    parser.add_argument(
        "--trace-output",
        default="",
        help="Optional JSONL path to save per-question LLM trace/reasoning output.",
    )
    parser.add_argument(
        "--wrong-output",
        default="",
        help="Optional JSONL path to save wrong/fallback questions for review.",
    )
    args = parser.parse_args()

    input_path = _resolve_input_path(args.input)
    output_path = _resolve_output_path(args.output)

    items = read_input(input_path)
    llm_client = None if args.mode == "heuristic" else LLMClient.from_env()
    if args.mode == "llm" and llm_client is None:
        raise ValueError(
            "LLM mode requires local model id via HF_MODEL_ID or LLM_MODEL "
            "(e.g. Qwen/Qwen3.5-4B) in environment or .env."
        )
    if llm_client is not None and args.workers != 1:
        print(f"LLM mode: overriding workers={args.workers} -> 1 for faster/stable local inference.")
        args.workers = 1

    results = run_pipeline(items, max_workers=args.workers, llm_client=llm_client)
    write_output(output_path, results)
    if args.trace_output:
        trace_rows = [
            {
                "qid": r["qid"],
                "domain": r.get("domain", ""),
                "answer": r["answer"],
                "route_fallback": bool(r.get("route_fallback", False)),
                "answer_fallback": bool(r.get("llm_fallback", False)),
                "raw_route": r.get("llm_raw_route", ""),
                "raw_answer": r.get("llm_raw_answer", ""),
            }
            for r in results
        ]
        write_jsonl(args.trace_output, trace_rows)
        print(f"Wrote trace log to {args.trace_output}")
    if args.wrong_output:
        wrong_rows = []
        for src, r in zip(items, results):
            is_wrong = bool(r.get("is_wrong", False))
            if is_wrong:
                wrong_rows.append(
                    {
                        "qid": r["qid"],
                        "question": src.get("question", ""),
                        "choices": src.get("choices", []),
                        "pred_answer": r["answer"],
                        "gold_answer": r.get("gold_answer", ""),
                        "domain": r.get("domain", ""),
                        "route_fallback": bool(r.get("route_fallback", False)),
                        "answer_fallback": bool(r.get("llm_fallback", False)),
                        "raw_answer": r.get("llm_raw_answer", ""),
                    }
                )
        write_jsonl(args.wrong_output, wrong_rows)
        print(f"Wrote {len(wrong_rows)} review rows to {args.wrong_output}")
    mode_used = "llm" if llm_client else "heuristic"
    print(f"Wrote {len(results)} predictions to {output_path} (mode={mode_used})")


if __name__ == "__main__":
    main()
