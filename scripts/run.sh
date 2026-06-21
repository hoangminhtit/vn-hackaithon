#!/usr/bin/env bash
# Run the MCQ pipeline from any working directory.
# Examples:
#   bash scripts/run.sh
#   bash scripts/run.sh heuristic
#   bash scripts/run.sh llm data/public-test_1780368312.json output/pred.csv
#   bash scripts/run.sh kaggle
#   INPUT=/path/test.json OUTPUT=/path/pred.csv MODE=llm WORKERS=1 bash scripts/run.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run.sh [llm|heuristic|auto] [input.json|input.csv] [output.csv] [workers]
  bash scripts/run.sh kaggle [input.json|input.csv] [output.csv]

Environment overrides:
  PROJECT_ROOT    Project directory. Defaults to the parent of scripts/.
  MODE            llm | heuristic | auto. Positional mode has priority.
  INPUT           Input JSON/CSV path.
  OUTPUT          Output CSV path.
  WORKERS         Worker count. LLM mode should use 1.
  PYTHON          Python executable. Defaults to python3, then python.

Kaggle:
  bash scripts/run.sh kaggle
  KAGGLE_INPUT=/kaggle/input/.../public-test_1780368312.json bash scripts/run.sh kaggle

The Kaggle preset searches common /kaggle/input locations when INPUT/KAGGLE_INPUT is not set.
EOF
}

find_kaggle_input() {
  local explicit="${KAGGLE_INPUT:-}"
  if [[ -n "$explicit" && -f "$explicit" ]]; then
    printf '%s\n' "$explicit"
    return 0
  fi

  local known="/kaggle/input/datasets/binbinkin/vn-hackaithon/public-test_1780368312.json"
  if [[ -f "$known" ]]; then
    printf '%s\n' "$known"
    return 0
  fi

  local found=""
  if [[ -d /kaggle/input ]]; then
    found="$(find /kaggle/input -type f \( -name 'public-test_*.json' -o -name 'public_test*.csv' -o -name 'private_test*.csv' \) | head -n 1 || true)"
  fi
  if [[ -n "$found" ]]; then
    printf '%s\n' "$found"
    return 0
  fi

  return 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

PRESET="${1:-}"
if [[ "$PRESET" == "kaggle" ]]; then
  shift
  MODE="${MODE:-llm}"
  INPUT="${1:-${INPUT:-${KAGGLE_INPUT:-}}}"
  OUTPUT="${2:-${OUTPUT:-output/pred.csv}}"
  WORKERS="${3:-${WORKERS:-1}}"
  if [[ -z "$INPUT" ]]; then
    INPUT="$(find_kaggle_input || true)"
  fi
else
  MODE="${1:-${MODE:-${PIPELINE_MODE:-llm}}}"
  INPUT="${2:-${INPUT:-data/public-test_1780368312.json}}"
  OUTPUT="${3:-${OUTPUT:-output/pred.csv}}"
  WORKERS="${4:-${WORKERS:-1}}"
fi

case "$MODE" in
  llm|heuristic|auto) ;;
  *)
    echo "Invalid mode: $MODE (choose llm | heuristic | auto)" >&2
    usage
    exit 1
    ;;
esac

cd "$ROOT"

# Local/Kaggle direct runs should not use the Docker competition input resolver.
if [[ "${RUN_COMPETITION:-0}" != "1" ]]; then
  unset COMPETITION
fi

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/.venv/bin/activate"
fi

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON=python
fi

if [[ -z "$INPUT" || ! -f "$INPUT" ]]; then
  echo "Input not found: ${INPUT:-<empty>}" >&2
  echo "Pass an input path, set INPUT=..., or use KAGGLE_INPUT=... with the kaggle preset." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"

echo "==> root=$ROOT"
echo "==> mode=$MODE workers=$WORKERS"
echo "==> input=$INPUT"
echo "==> output=$OUTPUT"

exec "$PYTHON" run.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --mode "$MODE" \
  --workers "$WORKERS"
