#!/usr/bin/env bash
# Entry-point BTC: /data/public_test.csv hoặc /data/private_test.csv → /output/pred.csv
set -euo pipefail

cd /app

export COMPETITION=1
export DATA_DIR="${DATA_DIR:-/data}"
export OUTPUT_DIR="${OUTPUT_DIR:-/output}"

if [[ ! -f /app/.env && "${AUTO_COPY_ENV:-1}" != "0" && -f /app/.env.example ]]; then
  cp /app/.env.example /app/.env
  echo "==> created /app/.env from /app/.env.example"
fi

if [[ -f /app/.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /app/.env
  set +a
fi

MODE="${PIPELINE_MODE:-llm}"
mkdir -p "$OUTPUT_DIR"

echo "==> Competition entry-point"
echo "    DATA_DIR=$DATA_DIR OUTPUT_DIR=$OUTPUT_DIR MODE=$MODE"

exec python3 run.py \
  --mode "$MODE" \
  --workers 1
