FROM python:3.10-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code + .env (nếu có lúc build)
COPY docker_entry.sh run.py pipeline.py router.py prompts.py few-shot.json ./
COPY domains ./domains
COPY utils ./utils
COPY .env* ./

# Bake weights — bắt buộc có thư mục model/ trước khi docker build (tải bằng ./run.sh lần đầu)
COPY model/ /app/model/
RUN test -n "$(ls -A /app/model 2>/dev/null)" || \
    (echo "ERROR: model/ trống. Chạy pipeline local 1 lần để tải Qwen vào model/ rồi build lại." && exit 1)

# Cấu hình pipeline (tương đương .env.example — không có secret, dùng khi BTC chạy image)
ENV HF_MODEL_ID=Qwen/Qwen3.5-4B
ENV HF_LOCAL_DIR=/app/model
ENV LLM_MAX_NEW_TOKENS=32
ENV LLM_ANSWER_MAX_TOKENS=32
ENV LLM_USE_LLM_ROUTE=0

# Entry-point BTC
ENV COMPETITION=1
ENV PIPELINE_MODE=llm
ENV DATA_DIR=/data
ENV OUTPUT_DIR=/output

RUN chmod +x /app/docker_entry.sh

ENTRYPOINT ["/app/docker_entry.sh"]
