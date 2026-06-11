# VN Hackathon MCQ Solver

Pipeline trắc nghiệm tiếng Việt: **route theo domain** → LLM local (Qwen GGUF via llama.cpp) + **fallback heuristic** khi parse/lỗi.

| Domain | Xử lý chính |
|--------|-------------|
| `rag` | BM25 trên passage + LLM (passage ngắn dùng full text) |
| `science` | LLM toán/khoa học + heuristic số (co giãn, GDP, …) |
| `multi_domain` | LLM tổng hợp |
| `should_correct` | LLM kiểm tra đúng/sai |
| `ignore_answer` | Heuristic (không gọi LLM) |

Thuyết minh chi tiết: [`PHUONG_PHAP.md`](PHUONG_PHAP.md)

---

## Chạy thử trên máy

### 1. Cài đặt (một lần)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`requirements.txt` không cài `llama-cpp-python` trực tiếp vì Linux/Kaggle dễ kéo nhầm CPU-only wheel. Trên Kaggle, ưu tiên cài CUDA prebuilt wheel thay vì build source:

```python
import torch

cuda = torch.version.cuda or ""
wheel_tags = {
    "11.8": "cu118",
    "12.1": "cu121",
    "12.2": "cu122",
    "12.3": "cu123",
    "12.4": "cu124",
    "12.5": "cu125",
    "13.0": "cu130",
    "13.2": "cu132",
}
major_minor = ".".join(cuda.split(".")[:2])
tag = wheel_tags.get(major_minor)
if tag is None and major_minor.startswith("12."):
    tag = "cu125"
if tag is None:
    tag = "cu125"
print("CUDA:", cuda, "llama-cpp wheel:", tag)
```

```bash
pip uninstall -y llama-cpp-python
pip install --no-cache-dir --force-reinstall "llama-cpp-python>=0.3.0" \
  --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/cu125"
```

Nếu cell Python ở trên in ra tag khác `cu125`, thay phần cuối URL bằng tag đó, ví dụ `cu124`. Với Kaggle CUDA `12.8`, dùng `cu125`. Sau khi chạy lệnh trên trong Kaggle notebook, restart kernel/runtime rồi chạy lại project. Khi load model, log phải có `llama-cpp-python CUDA backend xác nhận` hoặc dòng llama.cpp kiểu `offloaded ... layers to GPU`.

> **Lưu ý macOS:** Trên Apple Silicon, `llama-cpp-python` tự dùng Metal (GPU) khi cài từ pip. Trên Linux CUDA phải build với `-DGGML_CUDA=on` như trên.

### 2. Dữ liệu

Đặt public test JSON vào `data/`:

```text
data/public-test_1780368312.json
```

### 3. Chạy

```bash
chmod +x run.sh   # một lần
./run.sh
```

Kết quả: `output/pred.csv` (cột `qid`, `answer`).

Lần chạy đầu tiên sẽ **tự tải file GGUF** (~2.7 GB cho Q4_K_M) từ HuggingFace vào `model/`.

Gọi trực tiếp:

```bash
python3 run.py \
  --input data/public-test_1780368312.json \
  --output output/pred.csv \
  --mode llm \
  --workers 1
```

Tuỳ chọn:

```bash
./run.sh heuristic
./run.sh llm data/public-test_1780368312.json output/pred.csv
./run.sh --help
```

**Lưu ý:** Không `export COMPETITION=1` khi chạy local — biến đó chỉ dành cho container nộp BTC. `run.sh` đã `unset COMPETITION` sẵn.

---

## Cấu hình `.env`

```env
HF_MODEL_ID=unsloth/Qwen3.5-4B-GGUF
GGUF_FILE=Qwen3.5-4B-Q4_K_M.gguf
HF_LOCAL_DIR=model
LLM_MAX_NEW_TOKENS=16
LLM_ANSWER_MAX_TOKENS=16
LLM_USE_LLM_ROUTE=0
```

| Biến | Ý nghĩa |
|------|---------|
| `HF_MODEL_ID` | Repo GGUF trên HuggingFace |
| `GGUF_FILE` | Tên file `.gguf` cụ thể trong repo |
| `HF_LOCAL_DIR` | Cache local (mặc định `model/`) |
| `LLM_MAX_NEW_TOKENS` | Trần token sinh |
| `LLM_ANSWER_MAX_TOKENS` | Token bước trả lời |
| `LLM_USE_LLM_ROUTE` | `0` = route heuristic (nhanh), `1` = LLM route |
| `LLAMA_N_GPU_LAYERS` | `-1` = all GPU, `0` = CPU only |
| `LLAMA_N_CTX` | Context window (mặc định `4096`) |

LLM dùng **greedy** (`temperature=0`), deterministic.

Debug (tùy chọn): `TRACE_LLM=1`, `DEBUG_LLM=1`, `TRACE_QID=test_0001`

---

## Chế độ `--mode`

| Mode | Mô tả |
|------|--------|
| `heuristic` | Không load model |
| `llm` | Bắt buộc `HF_MODEL_ID` + `GGUF_FILE` |
| `auto` | Có model trong env → LLM, không → heuristic |

## Định dạng I/O

**Local dev — JSON:**

```json
[{"qid": "test_0001", "question": "...", "choices": ["A text", "B text"]}]
```

**Output (local & BTC):**

```csv
qid,answer
test_0001,A
```

## CLI `run.py`

| Tham số | Mặc định (local) |
|---------|------------------|
| `--input` | `data/public-test_1780368312.json` |
| `--output` | `output/pred.csv` |
| `--mode` | `auto` |
| `--workers` | ≤8 (LLM tự ép `1`) |
| `--trace-output` | (optional) JSONL debug |
| `--wrong-output` | (optional) JSONL câu fallback/sai |

---

## Nộp BTC (Docker)

### Checklist ban tổ chức

| Yêu cầu | Repo |
|--------|------|
| Image trên Docker Hub | Team `docker push` |
| Entry-point đọc `/data/public_test.csv` hoặc `private_test.csv` | `docker_entry.sh` |
| Ghi `/output/pred.csv` (`qid`, `answer`) | `run.py` |
| Source + reproduce | README + lệnh dưới |
| Thuyết minh | `PHUONG_PHAP.md` |

### Chuẩn bị trước `docker build`

1. Có `.env` ở root (copy từ `.env.example`) — được copy vào image lúc build.
2. Thư mục `model/` **đã có file `.gguf`** (chạy `./run.sh` một lần nếu còn trống):

```bash
./run.sh
ls model/*.gguf   # phải có file GGUF
```

Image bake sẵn GGUF model tại `/app/model` — BTC **không** cần mount `model/` hay tải HuggingFace.

### Build, push, chạy thử

```bash
docker build -t YOUR_USER/vn-hackathon-mcq:latest .
docker push YOUR_USER/vn-hackathon-mcq:latest

docker run --rm \
  -v "$(pwd)/data:/data:ro" \
  -v "$(pwd)/output:/output" \
  YOUR_USER/vn-hackathon-mcq:latest

head output/pred.csv
```

Trong container: `COMPETITION=1`, `source /app/.env`, đọc CSV trong `/data`, ghi `/output/pred.csv`.

**Input CSV BTC** — cột bắt buộc `qid`, `question`, và một trong:

- `A`, `B`, `C`, `D`, …
- hoặc `choices` (JSON array hoặc chuỗi phân tách `|`)

Heuristic không LLM khi chấm (nếu cần): `docker run -e PIPELINE_MODE=heuristic ...`

### Đồng bộ config khi nộp

- Chạy thử: chỉnh `.env`
- Nộp image: cùng giá trị trong `.env` (build copy vào image) và/hoặc `ENV` trong `Dockerfile`
- Sau khi đổi config → **build lại** image

---

## Cấu trúc repo

```text
├── run.sh / run.py          # Chạy local
├── docker_entry.sh          # Entry-point BTC
├── Dockerfile
├── pipeline.py / router.py / prompts.py
├── domains/                 # rag, math, multi_domain, …
├── utils/                   # preprocess, bm25, llm (llama.cpp), input_loader
├── data/                    # JSON test (local, gitignore)
├── model/                   # GGUF cache (gitignore, bake vào image)
├── output/                  # pred.csv
└── PHUONG_PHAP.md
```

---

## Lỗi thường gặp

| Lỗi | Cách xử lý |
|-----|------------|
| `No input in /data` | Đang `COMPETITION=1` thiếu CSV — local dùng `./run.sh` hoặc `--input` JSON |
| `LLM mode requires local model id` | Thêm `HF_MODEL_ID` vào `.env` hoặc `./run.sh heuristic` |
| `model/ không có file .gguf` khi `docker build` | Chạy `./run.sh` trước để tải GGUF |
| `llama_cpp` build lỗi | Cài `cmake` và `build-essential` (Linux) hoặc Xcode CLI tools (macOS) |
| JSON answer bị cắt | Tăng `LLM_ANSWER_MAX_TOKENS` (vd. 64) trong `.env` |

---

## Tài liệu thêm

- [`report.md`](report.md) — implementation (EN)
- [`pipeline_report.md`](pipeline_report.md) — thiết kế pipeline
