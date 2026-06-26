#!/usr/bin/env bash
# Entry-point BTC: /code/private_test.json (hoặc .csv) → /code/submission.csv + /code/submission_time.csv
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

echo "==> Competition entry-point"
echo "    PIPELINE_MODE=${PIPELINE_MODE:-llm}"

exec python3 predict.py
