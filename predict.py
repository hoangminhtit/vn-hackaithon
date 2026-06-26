"""
predict.py — Entry-point chính theo yêu cầu BTC HackAIthon.

Luồng:
  1. Đọc /code/private_test.json hoặc /code/private_test.csv
     (hỗ trợ cả 2 định dạng; CSV được chuyển sang list dict chuẩn)
  2. Chạy pipeline theo vòng lặp for tuần tự, đo thời gian từng câu
  3. Ghi /code/submission.csv       (qid, answer)
  4. Ghi /code/submission_time.csv  (qid, answer, time)

Chạy local (dev):
  python predict.py [--input path/to/test.json] [--mode llm|heuristic]
"""

import argparse
import csv
import json
import os
import time
import re
from typing import Dict, List, Optional

from pipeline import process_question
from utils.input_loader import read_items
from utils.llm import LLMClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _resolve_input(cli_input: Optional[str]) -> str:
    """Tự động quét tìm file dữ liệu đầu vào (.json hoặc .csv) trong môi trường Docker/Kaggle/Local."""
    if cli_input and os.path.isfile(cli_input):
        return cli_input

    # Thư mục tìm kiếm: ưu tiên /data trước vì BTC thường mount dữ liệu vào đây
    search_dirs = ["/data", "/code", "data", "."]
    
    # Ưu tiên các file có chứa từ khóa gợi ý về tập dữ liệu test
    patterns = [r"test", r"public", r"private", r"val"]
    
    candidates = []
    for s_dir in search_dirs:
        if not os.path.isdir(s_dir):
            continue
        try:
            # Sắp xếp tên file để đảm bảo tính nhất quán
            for f in sorted(os.listdir(s_dir)):
                f_lower = f.lower()
                # Chỉ xử lý file .json hoặc .csv
                if not (f_lower.endswith(".json") or f_lower.endswith(".csv")):
                    continue
                # Tránh các file kết quả sinh ra và file config dự án
                if "submission" in f_lower or f_lower.startswith(".") or f_lower == "requirements.txt" or f_lower == "few-shot.json":
                    continue
                
                full_path = os.path.join(s_dir, f)
                if os.path.isfile(full_path):
                    candidates.append(full_path)
        except Exception:
            pass

    # 1. Ưu tiên 1: File nằm trong /data hoặc /code có chứa từ khóa quan trọng (test, public, private)
    for pattern in patterns:
        for c in candidates:
            if re.search(pattern, os.path.basename(c).lower()):
                print(f"[AUTO-INPUT] Phát hiện file test phù hợp nhất: {c}")
                return c

    # 2. Ưu tiên 2: Bất kỳ file .json hoặc .csv nào trong /data
    for c in candidates:
        if c.startswith("/data"):
            print(f"[AUTO-INPUT] Chọn file bất kỳ trong thư mục chấm thi /data: {c}")
            return c

    # 3. Ưu tiên 3: Chọn file đầu tiên tìm thấy được
    if candidates:
        print(f"[AUTO-INPUT] Chọn file tìm thấy đầu tiên: {candidates[0]}")
        return candidates[0]

    raise FileNotFoundError(
        "Không tìm thấy bất kỳ file dữ liệu đầu vào (.json hoặc .csv) nào trong các thư mục quét (/data, /code, .)."
    )


def _resolve_output_dir(cli_out: Optional[str]) -> str:
    """Thư mục xuất kết quả: /code khi chạy trong Docker BTC, thư mục hiện tại khi dev."""
    if cli_out:
        return cli_out
    if os.path.isdir("/code"):
        return "/code"
    return "."


def write_submission(out_dir: str, rows: List[Dict]) -> str:
    path = os.path.join(out_dir, "submission.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "answer"])
        writer.writeheader()
        for r in rows:
            writer.writerow({"qid": r["qid"], "answer": r["answer"]})
    return path


def write_submission_time(out_dir: str, rows: List[Dict]) -> str:
    path = os.path.join(out_dir, "submission_time.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "answer", "time"])
        writer.writeheader()
        for r in rows:
            writer.writerow({"qid": r["qid"], "answer": r["answer"], "time": r["time"]})
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv_file(".env")

    parser = argparse.ArgumentParser(
        description="Predict MCQ answers (BTC entry-point). "
                    "Reads private_test.json/csv, writes submission.csv + submission_time.csv."
    )
    parser.add_argument("--input", default=None, help="Input .json hoặc .csv (auto-detect nếu không truyền).")
    parser.add_argument("--output-dir", default=None, help="Thư mục lưu submission files (mặc định: /code khi Docker, '.' khi local).")
    parser.add_argument(
        "--mode",
        choices=["heuristic", "llm", "auto"],
        default=os.getenv("PIPELINE_MODE", "llm"),
        help="Pipeline mode. Mặc định lấy từ env PIPELINE_MODE.",
    )
    args = parser.parse_args()

    input_path = _resolve_input(args.input)
    out_dir = _resolve_output_dir(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"==> Input  : {input_path}")
    print(f"==> Out dir: {out_dir}")
    print(f"==> Mode   : {args.mode}")

    # Đọc file đầu vào (hỗ trợ cả .json và .csv)
    test = read_items(input_path)
    print(f"==> Loaded {len(test)} items")

    # Khởi tạo LLM client
    llm_client: Optional[LLMClient] = None
    if args.mode != "heuristic":
        llm_client = LLMClient.from_env()
    if args.mode == "llm" and llm_client is None:
        raise ValueError(
            "LLM mode yêu cầu cấu hình model (HF_MODEL_ID hoặc GGUF_FILE) trong .env."
        )

    mode_used = "llm" if llm_client else "heuristic"
    print(f"==> Pipeline mode used: {mode_used}")

    # ── Inference loop (tuần tự, đo thời gian từng câu) ──────────────────────
    results: List[Dict] = []

    for item in test:
        t0 = time.time()
        res = process_question(item, llm_client)
        elapsed = time.time() - t0

        results.append({
            "qid": res["qid"],
            "answer": res["answer"],
            "time": f"{elapsed:.4f}",
        })

    # ── Ghi kết quả ───────────────────────────────────────────────────────────
    sub_path = write_submission(out_dir, results)
    sub_time_path = write_submission_time(out_dir, results)

    print(f"==> Wrote {len(results)} predictions to {sub_path}")
    print(f"==> Wrote timing info to {sub_time_path}")


if __name__ == "__main__":
    main()
