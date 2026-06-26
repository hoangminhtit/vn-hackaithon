#!/usr/bin/env bash
# Entry-point phụ (local/Kaggle): gọi predict.py với path tuỳ chỉnh.
# Entry-point chính BTC vẫn là inference.sh ở root (CMD trong Dockerfile).
set -euo pipefail

cd /code

if [[ ! -f /code/.env && "${AUTO_COPY_ENV:-1}" != "0" && -f /code/.env.example ]]; then
  cp /code/.env.example /code/.env
  echo "==> created /code/.env from /code/.env.example"
fi

if [[ -f /code/.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /code/.env
  set +a
fi

MODE="${PIPELINE_MODE:-llm}"
INPUT="${INPUT:-/code/private_test.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/code}"

mkdir -p "$OUTPUT_DIR"

echo "==> Competition entry-point (scripts/docker_entry.sh)"
echo "    INPUT=$INPUT OUTPUT_DIR=$OUTPUT_DIR MODE=$MODE"

exec python3 predict.py \
  --input "$INPUT" \
  --output-dir "$OUTPUT_DIR" \
  --mode "$MODE"
