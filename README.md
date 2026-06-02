# VSDS 2026 - MCQ Pipeline

Pipeline tra loi trac nghiem tieng Viet da domain, toi uu cho CPU va co che do LLM-backed + heuristic fallback.

## 1) Yeu cau

- Python 3.9+
- Input JSON dang list:
  - `qid` (string)
  - `question` (string)
  - `choices` (list[string])

## 2) Cau truc chinh

- `run.py`: entrypoint
- `pipeline.py`: orchestration
- `router.py`: heuristic router
- `prompts.py`: router/domain prompts cho LLM
- `utils/preprocess.py`: tach passage/question, map labels dong
- `utils/bm25.py`: BM25-like retrieval cho RAG
- `utils/postprocess.py`: parse JSON output + fallback
- `utils/llm.py`: OpenAI-compatible LLM client
- `domains/*.py`: domain solvers (heuristic fallback)

## 3) Cach chay

### A. Heuristic mode (khong can LLM)

```bash
python run.py --input "public-test_1780368312.json" --output "output/pred.csv" --mode heuristic
```

### B. Auto mode (uu tien LLM neu co env)

```bash
python run.py --input "public-test_1780368312.json" --output "output/pred.csv" --mode auto
```

### D. Dung model Qwen local bang transformers

Model link ban gui: [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)

Code hien tai load model local qua `transformers` (khong goi API/chat-completions).

Dat env:

```bash
export HF_MODEL_ID="Qwen/Qwen3.5-4B"
```

Hoac dung `LLM_MODEL` thay cho `HF_MODEL_ID`:

```bash
export LLM_MODEL="Qwen/Qwen3.5-4B"
```

Neu can gioi han token sinh:

```bash
export LLM_MAX_NEW_TOKENS="256"
```

Mac dinh model se duoc tai/cache vao thu muc `model/` trong project.
Neu muon doi thu muc, dat:

```bash
export HF_LOCAL_DIR="model"
```

Tao venv (khuyen nghi):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install thu vien:

```bash
pip install torch accelerate sentencepiece
pip install --upgrade transformers
```

Neu gap loi:
- `model type qwen3_5 ... Transformers does not recognize this architecture`

thi cai ban moi nhat tu source:

```bash
pip install --upgrade "git+https://github.com/huggingface/transformers.git"
```

Sau do chay:

```bash
python run.py --input "data/public-test_1780368312.json" --output "output/pred.csv" --mode llm
```

Log reasoning/raw output cho 1 cau (vi du `test_0001`):

```bash
export TRACE_LLM="1"
export TRACE_QID="test_0001"
python run.py --input "data/public-test_1780368312.json" --output "output/pred.csv" --mode llm
```

Xuat file review chi tiet:

```bash
python run.py \
  --input "data/public-test_1780368312.json" \
  --output "output/pred.csv" \
  --mode llm \
  --trace-output "output/llm_trace.jsonl" \
  --wrong-output "output/llm_wrong.jsonl"
```

Ghi chu `--wrong-output`:
- Neu input co `answer` (ground truth), file se la cac cau du doan sai.
- Neu input khong co `answer`, file se la cac cau bi fallback (de uu tien review prompt/parser).

## 4) Tham so CLI

- `--input`: duong dan file JSON input
- `--output`: duong dan CSV output
- `--workers`: so luong worker threads (mac dinh min(cpu_count, 8))
- `--mode`: `heuristic | llm | auto`
- `--trace-output`: file JSONL de luu route/answer raw cua LLM
- `--wrong-output`: file JSONL de luu cac cau can review (sai/fallback)

## 5) Output

File CSV:

```csv
qid,answer
test_0001,A
test_0002,C
...
```

## 6) Luu y

- `auto` mode se tu dong fallback sang heuristic neu chua cau hinh LLM.
- Trong `llm` mode, neu output LLM sai format, pipeline van co parse/fallback de dam bao tra ve nhan hop le.
- Nhan dap an la dong theo so luong choices (`A..Z`).
