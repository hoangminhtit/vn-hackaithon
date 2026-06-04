# VN Hackathon MCQ Solver

Pipeline giải câu hỏi trắc nghiệm tiếng Việt: route theo domain → LLM local (Qwen) + fallback heuristic.

---

## Chạy thử trên máy (khuyến nghị)

### 1. Cài đặt (một lần)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # chỉnh HF_MODEL_ID nếu cần
```

### 2. Đặt file test

Copy file JSON public test vào `data/`:

```
data/public-test_1780368312.json
```

### 3. Chạy

```bash
chmod +x run.sh   # một lần
./run.sh
```

Hoặc gọi trực tiếp:

```bash
source .venv/bin/activate
python3 run.py \
  --input data/public-test_1780368312.json \
  --output output/pred.csv \
  --mode llm \
  --workers 1
```

Kết quả: `output/pred.csv` (2 cột `qid`, `answer`).

### Tuỳ chọn

```bash
./run.sh heuristic                                    # không load model
./run.sh llm data/public-test_1780368312.json output/pred.csv
./run.sh --help
```

**Lưu ý:** Không set `COMPETITION=1` khi chạy thử local. Biến đó chỉ dùng trong Docker lúc nộp.

---

## Cấu hình `.env`

Copy `.env.example` → `.env`:

```env
HF_MODEL_ID=Qwen/Qwen3.5-4B
HF_LOCAL_DIR=model
LLM_MAX_NEW_TOKENS=32
LLM_ANSWER_MAX_TOKENS=64
LLM_USE_LLM_ROUTE=0
```

Tham số RAG / sampling: xem [`.env.example`](.env.example).

## Chế độ pipeline

| `--mode` | Mô tả |
|----------|--------|
| `heuristic` | Không load LLM |
| `llm` | Bắt buộc `HF_MODEL_ID` |
| `auto` | Có model → LLM, không → heuristic |

## Định dạng dữ liệu

### Input JSON (chạy thử)

```json
[
  {"qid": "test_0001", "question": "...", "choices": ["...", "..."]}
]
```

### Output CSV

```csv
qid,answer
test_0001,A
```

---

## Nộp BTC (Docker — chỉ khi submit)

Checklist yêu cầu ban tổ chức:

| Yêu cầu | Trạng thái |
|--------|------------|
| Docker Hub | ⚠️ `docker push` image của team |
| Entry-point đọc `/data/public_test.csv` hoặc `private_test.csv` | ✅ `docker_entry.sh` |
| Ghi `/output/pred.csv` (`qid`, `answer`) | ✅ |
| GitHub + reproduce | ✅ repo + lệnh dưới |
| Thuyết minh phương pháp | ✅ [`PHUONG_PHAP.md`](PHUONG_PHAP.md) |

### Trước khi build (bắt buộc)

Thư mục `model/` phải **đã có weights** (image bake sẵn, BTC không cần tải HuggingFace):

```bash
# Tải model 1 lần trên máy dev (nếu model/ còn trống)
source .venv/bin/activate
./run.sh    # hoặc chạy llm 1 câu — HF tải vào model/

ls model/   # phải thấy file cache (không rỗng)
```

Image sẽ **rất nặng** (~vài GB tùy model). `docker push` có thể lâu.

### Build & chạy thử giống BTC

```bash
docker build -t YOUR_USER/vn-hackathon-mcq:latest .
docker push YOUR_USER/vn-hackathon-mcq:latest

docker run --rm \
  -v "$(pwd)/data:/data:ro" \
  -v "$(pwd)/output:/output" \
  YOUR_USER/vn-hackathon-mcq:latest

head output/pred.csv
```

Không cần mount `model/` — weights đã nằm trong image tại `/app/model`.

Trong container: `.env` (nếu có lúc build) + `docker_entry.sh` `source .env`. Đọc CSV tại `/data`, ghi `/output/pred.csv`.

Input CSV BTC — cột `qid`, `question`, và `A,B,C,D` hoặc cột `choices` (JSON / phân tách `|`).

Chạy local vẫn dùng `.env`; chỉnh tham số dev ở đó. Khi đổi config nộp BTC, cập nhật `Dockerfile` (hoặc `.env.example` cho đồng bộ) rồi build lại image.

---

## Lỗi thường gặp

**`No input in /data`** — đang chạy với `COMPETITION=1` nhưng thiếu CSV. Chạy thử local dùng `./run.sh` hoặc `--input data/....json`; không export `COMPETITION=1`.

**`LLM mode requires local model id`** — thêm `HF_MODEL_ID` vào `.env`, hoặc `./run.sh heuristic`.

**`qwen3_5 ... does not recognize`** — `pip install --upgrade "git+https://github.com/huggingface/transformers.git"`

**JSON answer bị cắt** — tăng `LLM_ANSWER_MAX_TOKENS=64`.

---

## Tài liệu thêm

- [`PHUONG_PHAP.md`](PHUONG_PHAP.md) — phương pháp
- [`report.md`](report.md) — implementation (EN)
- [`pipeline_report.md`](pipeline_report.md) — thiết kế chi tiết
