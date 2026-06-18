# VN Hackathon MCQ Solver

Submission cho BTC dưới dạng Docker container. Container tự đọc dữ liệu trong `/data` và ghi kết quả ra `/output/pred.csv`.

## Docker Hub

Image nộp lên Docker Hub: https://hub.docker.com/repository/docker/hoangminhtit/vn-hackathon-mcq/general

```text
hoangminhtit/vn-hackathon-mcq:latest
```

## Entry-Point Contract

Container được thiết kế đúng theo yêu cầu đầu ra của BTC:

- Input được mount vào `/data`.
- Entrypoint tự tìm `/data/public_test.csv`; nếu không có thì dùng `/data/private_test.csv`.
- Output được ghi vào `/output/pred.csv`.
- File output có đúng hai cột:

```csv
qid,answer
```

Trong đó `answer` là một trong `A/B/C/D` hoặc các nhãn lựa chọn hợp lệ nếu đề có nhiều hơn 4 lựa chọn.

Input CSV cần có các cột bắt buộc:

```text
qid,question
```

Và một trong hai cách biểu diễn lựa chọn:

```text
A,B,C,D,...
```

hoặc:

```text
choices
```

Trong đó `choices` là JSON array hoặc chuỗi phân tách bằng `|`.

## Reproduce

Có 2 cách chạy lại kết quả: chạy bằng script trong repo hoặc chạy bằng Docker container.

### Cách 1: Chạy bằng `scripts/`

Chuẩn bị input CSV:

```text
data/
  public_test.csv
output/
```

Chạy pipeline bằng script:

```bash
bash scripts/run.sh llm data/public_test.csv output/pred.csv
```

Nếu dùng private test:

```bash
bash scripts/run.sh llm data/private_test.csv output/pred.csv
```

Kiểm tra kết quả:

```bash
head output/pred.csv
```

Output được ghi tại `output/pred.csv` với hai cột `qid,answer`.

### Cách 2: Chạy bằng Docker

Chuẩn bị thư mục local:

```text
data/
  public_test.csv
output/
```

Chạy container giống cách BTC chấm:

```bash
docker run --rm \
  -v "$(pwd)/data:/data:ro" \
  -v "$(pwd)/output:/output" \
  YOUR_DOCKERHUB_USERNAME/vn-hackathon-mcq:latest
```

Nếu dùng private test:

```text
data/
  private_test.csv
output/
```

Lệnh chạy giữ nguyên:

```bash
docker run --rm \
  -v "$(pwd)/data:/data:ro" \
  -v "$(pwd)/output:/output" \
  YOUR_DOCKERHUB_USERNAME/vn-hackathon-mcq:latest
```

Kiểm tra kết quả:

```bash
head output/pred.csv
```

Output kỳ vọng:

```csv
qid,answer
test_0001,A
test_0002,C
```

<!-- ## Build Và Push Image

Trước khi build, cần có:

- `.env` ở root repo.
- Thư mục `model/` chứa file GGUF `Qwen3.5-4B-Q4_K_M.gguf`.

Nếu chưa có model, có thể tải bằng:

```bash
python download_model.py
```

Build image:

```bash
docker build -t YOUR_DOCKERHUB_USERNAME/vn-hackathon-mcq:latest .
```

Mặc định image build CUDA cho các kiến trúc `75;80;86;89` (T4/Ampere/Ada). Nếu môi trường chấm dùng GPU khác, có thể override:

```bash
docker build \
  --build-arg CUDA_ARCHITECTURES="75;80;86;89" \
  -t YOUR_DOCKERHUB_USERNAME/vn-hackathon-mcq:latest .
```

Push lên Docker Hub:

```bash
docker push YOUR_DOCKERHUB_USERNAME/vn-hackathon-mcq:latest
``` -->

## Cấu Hình Đã Dùng

Các biến chính trong `.env.example`:

```env
HF_MODEL_ID=unsloth/Qwen3.5-4B-GGUF
GGUF_FILE=Qwen3.5-4B-Q4_K_M.gguf
HF_LOCAL_DIR=model

LLM_MAX_NEW_TOKENS=16
LLM_ANSWER_MAX_TOKENS=16
LLM_USE_LLM_ROUTE=0

LLM_USE_POT_SCIENCE=1
LLM_POT_MAX_TOKENS=512
LLM_POT_RETRIES=1
LLM_POT_TIMEOUT=2.0

LLM_USE_COT_SHOULD_CORRECT=1
LLM_USE_COT_MULTI=1
LLM_COT_MAX_TOKENS=384

LLM_USE_ANSWER_VERIFIER=1
LLM_VERIFY_MULTI=0
LLM_VERIFY_MULTI_MANY_CHOICES=0
LLM_VERIFY_MAX_TOKENS=320

RAG_MAX_CONTEXT_CHARS=12000
RAG_FULL_PASSAGE_CHARS=12000
RAG_BM25_MAX_CHARS=10000
RAG_BM25_TOP_K=12

LLM_USE_RAG_EVIDENCE=1
LLM_RAG_EVIDENCE_MAX_TOKENS=512

LLAMA_N_CTX=4096
# LLAMA_N_GPU_LAYERS=-1
```

Trong Docker image, `HF_LOCAL_DIR` được set về `/app/model` và model GGUF được bake sẵn vào image, nên BTC không cần mount model hoặc tải model khi chấm.

## Repo

Các file chính:

```text
Dockerfile
scripts/docker_entry.sh
run.py
pipeline.py
router.py
prompts.py
domains/
utils/
download_model.py
.env.example
```

Entrypoint Docker là `scripts/docker_entry.sh`, chạy `run.py` với `COMPETITION=1`, `DATA_DIR=/data`, `OUTPUT_DIR=/output`, và `PIPELINE_MODE=llm`.
