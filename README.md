# VN Hackathon MCQ Solver

Pipeline giải câu hỏi trắc nghiệm tiếng Việt với 2 lớp xử lý:
- Heuristic solver theo domain (`rag`, `math`, `multi_domain`, `should_correct`, `ignore_answer`)
- Local LLM qua `transformers` (mặc định dùng `Qwen/Qwen3.5-4B`)

Mục tiêu: chạy local ổn định, có fallback rõ ràng, có trace để debug.

## 1) Yêu cầu môi trường

- Python `3.10+` (khuyến nghị dùng `venv`)
- macOS/Linux/Windows đều chạy được
- Nếu dùng LLM mode: cài `torch` + `transformers` mới

## 2) Cài đặt nhanh

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install torch accelerate sentencepiece
pip install --upgrade transformers
```

Nếu gặp lỗi `qwen3_5 ... Transformers does not recognize this architecture`:

```bash
pip install --upgrade "git+https://github.com/huggingface/transformers.git"
```

## 3) Cấu hình `.env`

Tạo file `.env` ở root (hoặc copy từ `.env.example`):

```env
HF_MODEL_ID=Qwen/Qwen3.5-4B
HF_LOCAL_DIR=model
LLM_MAX_NEW_TOKENS=32
LLM_ANSWER_MAX_TOKENS=32
LLM_USE_LLM_ROUTE=0
```

Ý nghĩa biến môi trường:
- `HF_MODEL_ID` / `LLM_MODEL`: model id dùng cho local transformers
- `HF_LOCAL_DIR`: thư mục cache model local (mặc định `model/`)
- `LLM_MAX_NEW_TOKENS`: trần token sinh tối đa phía client
- `LLM_ANSWER_MAX_TOKENS`: token cho bước answer trong pipeline
- `LLM_USE_LLM_ROUTE`: `0` = route bằng heuristic (nhanh, mặc định), `1` = gọi LLM route

Lưu ý:
- `HF_TOKEN` **không dùng** trong flow local-only hiện tại (nếu set vẫn bị ignore trong code)
- Lần đầu chạy có thể tải model từ HF về `HF_LOCAL_DIR`; các lần sau đọc từ cache local

## 4) Chế độ chạy

Pipeline hỗ trợ `--mode`:
- `heuristic`: không dùng LLM
- `llm`: bắt buộc có `HF_MODEL_ID` hoặc `LLM_MODEL`
- `auto`: nếu có model env -> chạy LLM, không có -> fallback heuristic

## 5) Lệnh chạy chuẩn

### Heuristic mode

```bash
python run.py --input "data/public-test_1780368312.json" --output "output/pred.csv" --mode heuristic
```

### LLM mode (local transformers)

```bash
source .venv/bin/activate
export HF_MODEL_ID="Qwen/Qwen3.5-4B"
export HF_LOCAL_DIR="model"
export LLM_MAX_NEW_TOKENS=32
export LLM_ANSWER_MAX_TOKENS=32
export LLM_USE_LLM_ROUTE=0
unset TRACE_LLM
unset DEBUG_LLM

python run.py --input "data/public-test_1780368312.json" --output "output/pred.csv" --mode llm --workers 1
```

Ghi chú:
- Khi có LLM client, `run.py` sẽ tự ép `--workers=1` để tránh nghẽn local inference
- `utils/llm.py` đã bật `enable_thinking=False` để giảm latency output JSON

### Auto mode

```bash
python run.py --input "data/public-test_1780368312.json" --output "output/pred.csv" --mode auto
```

## 6) Trace và file review

### In trace ra terminal

```bash
export TRACE_LLM=1
python run.py --input "data/public-test_1780368312.json" --output "output/pred.csv" --mode llm --workers 1
```

Chỉ trace 1 câu:

```bash
export TRACE_LLM=1
export TRACE_QID="test_0001"
python run.py --input "data/public-test_1780368312.json" --output "output/pred.csv" --mode llm --workers 1
```

### Xuất trace JSONL và danh sách câu cần review

```bash
python run.py \
  --input "data/public-test_1780368312.json" \
  --output "output/pred.csv" \
  --mode llm \
  --workers 1 \
  --trace-output "output/llm_trace.jsonl" \
  --wrong-output "output/llm_wrong.jsonl"
```

- `--trace-output`: lưu route raw, answer raw, fallback flags theo từng `qid`
- `--wrong-output`: lưu các câu `is_wrong=true` (nếu input có gold `answer`), hoặc các câu fallback để review

## 7) Định dạng dữ liệu

### Input JSON

`run.py` yêu cầu input là `list` object:
- `qid`: string
- `question`: string
- `choices`: danh sách lựa chọn (preprocess sẽ map thành label A/B/C...)
- tùy chọn `answer`: gold label để chấm và xuất `wrong-output`

Ví dụ:

```json
[
  {
    "qid": "test_0001",
    "question": "Câu hỏi ...",
    "choices": ["Đáp án 1", "Đáp án 2", "Đáp án 3", "Đáp án 4"]
  }
]
```

### Output CSV

```csv
qid,answer
test_0001,A
test_0002,C
```

## 8) Tham số CLI

- `--input` (default: `data/public-test_1780368312.json`)
- `--output` (default: `output/pred.csv`)
- `--workers` (default: `min(cpu_count, 8)`, nhưng LLM mode bị override về `1`)
- `--mode` (`heuristic` | `llm` | `auto`)
- `--trace-output` (optional JSONL path)
- `--wrong-output` (optional JSONL path)

## 9) Các lỗi thường gặp

### `LLM mode requires local model id ...`

Thiếu `HF_MODEL_ID` / `LLM_MODEL`. Cách xử lý:
- export biến env trước khi chạy, hoặc
- thêm vào `.env` ở root

### `qwen3_5 ... does not recognize this architecture`

`transformers` quá cũ:

```bash
pip install --upgrade "git+https://github.com/huggingface/transformers.git"
```

### Chạy LLM chậm

Checklist:
- Dùng `LLM_MAX_NEW_TOKENS=32`, `LLM_ANSWER_MAX_TOKENS=32`
- Giữ `LLM_USE_LLM_ROUTE=0`
- Tắt `TRACE_LLM` và `DEBUG_LLM` khi chạy full
- Chạy `--workers 1`
- Đảm bảo đã bật `venv` đúng (tránh dùng nhầm system Python)
