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
  PYTHON          Python executable. Defaults to project venv, py -3, python3, then python.
  ENV_FILE        Env file path. Defaults to .env.
  ENV_EXAMPLE_FILE
                  Template env file path. Defaults to .env.example.
  AUTO_COPY_ENV   Copy ENV_EXAMPLE_FILE to ENV_FILE when missing. Defaults to 1.
  LLM_AUTO_INSTALL_LLAMA_CPP
                  Install llama-cpp-python automatically for llm/auto mode. Defaults to 1.
  LLAMA_CPP_FORCE_REINSTALL
                  Uninstall/reinstall llama-cpp-python before running. Defaults to 0.
  LLAMA_CPP_SPEC  Package spec. Defaults to llama-cpp-python>=0.3.0.
  LLAMA_CPP_EXTRA_INDEX_URL
                  Optional wheel index URL. Defaults to CUDA 12.5 wheels on Kaggle/Linux,
                  and PyPI-only on Windows local.

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

ensure_env_file() {
  local env_file="${ENV_FILE:-.env}"
  local example_file="${ENV_EXAMPLE_FILE:-.env.example}"

  if [[ ! -f "$env_file" && "${AUTO_COPY_ENV:-1}" != "0" && -f "$example_file" ]]; then
    cp "$example_file" "$env_file"
    echo "==> created $env_file from $example_file"
  fi

  if [[ ! -f "$env_file" ]]; then
    return 0
  fi

  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ -z "$line" || "$line" == \#* || "$line" != *=* ]] && continue
    local key="${line%%=*}"
    local value="${line#*=}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    if [[ ! -v "$key" ]]; then
      if [[ "$value" == \"*\" && "$value" == *\" ]]; then
        value="${value:1:${#value}-2}"
      elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
        value="${value:1:${#value}-2}"
      fi
      export "$key=$value"
    fi
  done < "$env_file"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

cd "$ROOT"
ensure_env_file

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

# Local/Kaggle direct runs should not use the Docker competition input resolver.
if [[ "${RUN_COMPETITION:-0}" != "1" ]]; then
  unset COMPETITION
fi

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/.venv/bin/activate"
elif [[ -f "$ROOT/.venv/Scripts/activate" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/.venv/Scripts/activate"
fi

python_works() {
  "$@" --version >/dev/null 2>&1
}

ensure_llama_cpp() {
  if [[ "$MODE" == "heuristic" || "${LLM_AUTO_INSTALL_LLAMA_CPP:-1}" == "0" ]]; then
    echo "==> skip llama-cpp-python auto install"
    return 0
  fi

  local spec="${LLAMA_CPP_SPEC:-llama-cpp-python>=0.3.0}"
  local force="${LLAMA_CPP_FORCE_REINSTALL:-0}"
  local uname_s
  uname_s="$(uname -s 2>/dev/null || true)"
  local default_extra_index=""
  case "$uname_s" in
    MINGW*|MSYS*|CYGWIN*) default_extra_index="" ;;
    *) default_extra_index="https://abetlen.github.io/llama-cpp-python/whl/cu125" ;;
  esac
  if [[ -n "${KAGGLE_KERNEL_RUN_TYPE:-}${KAGGLE_URL_BASE:-}" ]]; then
    default_extra_index="https://abetlen.github.io/llama-cpp-python/whl/cu125"
  fi
  local extra_index="${LLAMA_CPP_EXTRA_INDEX_URL:-$default_extra_index}"

  echo "==> python=$("${PYTHON_CMD[@]}" -c 'import sys; print(sys.executable)')"

  if [[ "$force" != "1" ]] && "${PYTHON_CMD[@]}" -c "import llama_cpp.llama_cpp" >/dev/null 2>&1; then
    echo "==> llama-cpp-python native backend is usable"
    return 0
  fi

  echo "==> installing llama-cpp-python ($spec)"
  "${PYTHON_CMD[@]}" -m pip uninstall -y llama-cpp-python
  install_cmd=(
    "${PYTHON_CMD[@]}" -m pip install
    --no-cache-dir
    --force-reinstall
    "$spec"
  )
  if [[ -n "$extra_index" && "$extra_index" != "0" && "$extra_index" != "none" ]]; then
    install_cmd+=(--extra-index-url "$extra_index")
  fi
  "${install_cmd[@]}"

  if ! "${PYTHON_CMD[@]}" -c "import llama_cpp.llama_cpp" >/dev/null 2>&1; then
    echo "llama-cpp-python was installed, but its native backend still cannot be loaded." >&2
    echo "Try forcing CPU mode locally with LLAMA_N_GPU_LAYERS=0, or reinstall with a wheel matching your local CUDA/Python." >&2
    exit 1
  fi
}

PYTHON_CMD=()
if [[ -n "${PYTHON:-}" ]]; then
  # Allow values like PYTHON="py -3" when running from Git Bash on Windows.
  # shellcheck disable=SC2206
  PYTHON_CMD=($PYTHON)
  if ! python_works "${PYTHON_CMD[@]}"; then
    echo "Configured PYTHON does not run: $PYTHON" >&2
    exit 1
  fi
elif [[ -x "$ROOT/.venv/Scripts/python.exe" ]]; then
  PYTHON_CMD=("$ROOT/.venv/Scripts/python.exe")
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON_CMD=("$ROOT/.venv/bin/python")
elif python_works py -3; then
  PYTHON_CMD=(py -3)
elif python_works python3; then
  PYTHON_CMD=(python3)
elif python_works python; then
  PYTHON_CMD=(python)
else
  echo "No usable Python interpreter found." >&2
  echo "On Windows, create and activate a venv with:" >&2
  echo "  py -3 -m venv .venv" >&2
  echo "  source .venv/Scripts/activate" >&2
  echo "  python -m pip install -r requirements.txt" >&2
  exit 1
fi

if [[ -z "$INPUT" || ! -f "$INPUT" ]]; then
  echo "Input not found: ${INPUT:-<empty>}" >&2
  echo "Pass an input path, set INPUT=..., or use KAGGLE_INPUT=... with the kaggle preset." >&2
  exit 1
fi

ensure_llama_cpp

mkdir -p "$(dirname "$OUTPUT")"

echo "==> root=$ROOT"
echo "==> mode=$MODE workers=$WORKERS"
echo "==> input=$INPUT"
echo "==> output=$OUTPUT"

exec "${PYTHON_CMD[@]}" run.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --mode "$MODE" \
  --workers "$WORKERS"
