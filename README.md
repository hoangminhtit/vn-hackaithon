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

### D. Dung truc tiep HuggingFace (Qwen model)

Model link ban gui: [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)

Pipeline da ho tro auto map sang HuggingFace Inference Router neu co `HF_TOKEN`.

Dat env:

```bash
export HF_TOKEN="hf_xxx_your_token"
export HF_MODEL_ID="Qwen/Qwen3.5-4B"
```

Hoac dung `LLM_MODEL` thay cho `HF_MODEL_ID`:

```bash
export HF_TOKEN="hf_xxx_your_token"
export LLM_MODEL="Qwen/Qwen3.5-4B"
```

Sau do chay:

```bash
python run.py --input "data/public-test_1780368312.json" --output "output/pred.csv" --mode llm
```

Ghi chu:
- Neu ban tu host model bang vLLM/SGLang local, co the dat:
  - `LLM_API_URL=http://localhost:8000/v1/chat/completions`
  - `LLM_MODEL=Qwen/Qwen3.5-4B`
  - `LLM_API_KEY=EMPTY`

## 4) Tham so CLI

- `--input`: duong dan file JSON input
- `--output`: duong dan CSV output
- `--workers`: so luong worker threads (mac dinh min(cpu_count, 8))
- `--mode`: `heuristic | llm | auto`

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
