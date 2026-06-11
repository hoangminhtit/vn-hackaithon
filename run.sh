#!/usr/bin/env bash
# Chạy pipeline MCQ → output/pred.csv (không eval).
# Usage:
#   ./run.sh                    # LLM, input/output mặc định
#   ./run.sh heuristic          # không dùng model
#   ./run.sh llm data/test.json output/pred.csv
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Chạy thử local — không dùng chế độ nộp Docker
unset COMPETITION

usage() {
  cat <<'EOF'
Usage:
  ./run.sh [llm|heuristic|auto] [input.json] [output.csv]

Ví dụ:
  ./run.sh
  ./run.sh heuristic
  ./run.sh llm data/public-test_1780368312.json output/pred.csv

Tải model GGUF (lần đầu):
  python download_model.py
  python download_model.py --token hf_xxxxx

Biến môi trường (tùy chọn):
  INPUT, OUTPUT  — ghi đè đường dẫn mặc định
  HF_TOKEN       — HuggingFace access token

Cấu hình: file .env ở thư mục gốc (run.py tự load).
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MODE="${1:-llm}"
INPUT="${2:-${INPUT:-data/public-test_1780368312.json}}"
OUTPUT="${3:-${OUTPUT:-output/pred.csv}}"

if [[ "$MODE" != "llm" && "$MODE" != "heuristic" && "$MODE" != "auto" ]]; then
  echo "Mode không hợp lệ: $MODE (chọn llm | heuristic | auto)"
  usage
  exit 1
fi

if [[ ! -f "$INPUT" ]]; then
  echo "Không tìm thấy input: $INPUT"
  echo "Đặt file JSON vào data/ hoặc truyền đường dẫn thứ 2."
  exit 1
fi

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/.venv/bin/activate"
fi

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON=python
fi

mkdir -p "$(dirname "$OUTPUT")"

echo "==> mode=$MODE input=$INPUT output=$OUTPUT"
exec "$PYTHON" run.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --mode "$MODE" \
  --workers 1
