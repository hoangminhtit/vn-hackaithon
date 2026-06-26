# VN Hackathon MCQ Solver

Submission cho BTC HackAIthon Bảng C Innovator.

**Entry-point:** `predict.py` → đọc `/code/private_test.json` → sinh `submission.csv` + `submission_time.csv` vào `/code/`.

## Docker Hub

Image nộp lên Docker Hub: https://hub.docker.com/r/nguyenvanhung777/vn-hackathon-mcq

```text
nguyenvanhung777/vn-hackathon-mcq:latest
```

---

## Pipeline Flow

```
private_test.json
      │
      ▼
 predict.py  ──── read_items() ────► [list câu hỏi]
      │
      │  for item in test:          ← vòng lặp tuần tự theo yêu cầu BTC
      │    t0 = time.time()
      ▼
 pipeline.process_question(item)
      │
      ├─► router.route_question()   ← phân loại domain (rag / math / multi_domain / ...)
      │         │
      │         ├─► domain = "rag"         → BM25 retrieve + RAG evidence LLM
      │         ├─► domain = "math"        → PoT (Program of Thought, chạy Python sandbox)
      │         ├─► domain = "should_correct" → CoT reasoning
      │         └─► domain = "multi_domain"   → CoT + Answer Verifier
      │
      ├─► llm_client.chat()         ← Qwen3.5-4B-Q4_K_M.gguf via llama-cpp-python
      │
      └─► parse_answer()            ← trích xuất A/B/C/D từ raw output
      │
      ▼
    elapsed = time.time() - t0
      │
      ▼
submission.csv          (qid, answer)
submission_time.csv     (qid, answer, time)
```

---

## Data Processing

- **Input:** file JSON (`private_test.json`) với cấu trúc `[{"id": "test_0001", "question": "...", "choices": {"A": "...", ...}, "context": "..."}]`
- **Tiền xử lý:** `utils/preprocess.py` — chuẩn hoá Unicode, loại bỏ ký tự đặc biệt, tách câu hỏi và lựa chọn
- **BM25 Retrieval:** `utils/bm25.py` — tự implement (không dùng thư viện ngoài), tách câu, tính TF-IDF để lấy đoạn context liên quan nhất
- **Output:** `submission.csv` + `submission_time.csv`

---

## Resource Initialization

Model GGUF được **bake sẵn vào Docker image** khi build — BTC không cần mount hay tải thêm:

```dockerfile
COPY model/ /code/model/
# model/Qwen3.5-4B-Q4_K_M.gguf phải có trước khi docker build
```

Nếu cần tải model thủ công (trước khi build):

```bash
python download_model.py
# hoặc
huggingface-cli download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf --local-dir model/
```

---

## Reproduce

### Cách 1: Chạy local bằng `scripts/run.sh`

```bash
# Mặc định: llm mode, input = data/public-test_1780368312.json, output vào output/
bash scripts/run.sh

# Chỉ định rõ input và output dir
bash scripts/run.sh llm data/public-test_1780368312.json output/

# Private test
bash scripts/run.sh llm data/private_test.json output/
```

Output sinh ra tại:
```
output/submission.csv          # qid,answer
output/submission_time.csv     # qid,answer,time
```

### Cách 2: Chạy bằng Docker (giống cách BTC chấm)

```bash
# Đảm bảo private_test.json nằm ở data/private_test.json
docker run --rm --gpus all \
  -v "$(pwd)/data/private_test.json:/code/private_test.json:ro" \
  -v "$(pwd)/output:/code" \
  nguyenvanhung777/vn-hackathon-mcq:latest
```

Kiểm tra output:

```bash
head output/submission.csv
head output/submission_time.csv
```

Output kỳ vọng:

```csv
# submission.csv
qid,answer
test_0001,A
test_0002,C

# submission_time.csv
qid,answer,time
test_0001,A,1.2345
test_0002,C,0.9871
```

---

## Cấu Hình

Các biến môi trường chính (xem `.env.example`):

```env
HF_MODEL_ID=unsloth/Qwen3.5-4B-GGUF
GGUF_FILE=Qwen3.5-4B-Q4_K_M.gguf
HF_LOCAL_DIR=model

LLM_MAX_NEW_TOKENS=16
LLM_USE_POT_SCIENCE=1        # PoT cho câu toán/khoa học
LLM_USE_COT_SHOULD_CORRECT=1 # CoT cho domain should_correct
LLM_USE_RAG_EVIDENCE=1       # RAG evidence retrieval
LLAMA_N_CTX=4096
# LLAMA_N_GPU_LAYERS=-1      # bật để full GPU offload
```

---

## Cấu Trúc Repo

```text
Dockerfile                    ← base nvidia/cuda:12.2.0-devel-ubuntu20.04
inference.sh                  ← CMD entry-point BTC (gọi predict.py)
predict.py                    ← main entry-point: đọc JSON, chạy pipeline, ghi 2 CSV
pipeline.py                   ← orchestrator pipeline
router.py                     ← phân loại domain câu hỏi
prompts.py                    ← prompt templates
domains/                      ← domain-specific logic (rag, math, multi_domain, ...)
utils/
  llm.py                      ← LLMClient (llama-cpp-python wrapper)
  bm25.py                     ← BM25 retrieval tự implement
  reasoning.py                ← PoT / CoT / RAG evidence / Answer Verifier
  input_loader.py             ← đọc JSON/CSV
  postprocess.py              ← parse A/B/C/D từ raw output
few-shot.json                 ← few-shot examples
download_model.py             ← tải GGUF từ HuggingFace
.env.example                  ← template cấu hình
requirements.txt              ← huggingface_hub, sentencepiece, sympy
scripts/run.sh                ← helper script chạy local
```
