FROM python:3.10-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code + .env (nếu có lúc build)
COPY docker_entry.sh run.py download_model.py pipeline.py router.py prompts.py few-shot.json ./
COPY domains ./domains
COPY utils ./utils
COPY .env* ./

# Bake GGUF weights — bắt buộc có file .gguf trong model/ trước khi docker build
COPY model/ /app/model/
RUN test -n "$(find /app/model -name '*.gguf' 2>/dev/null | head -1)" || \
    (echo "ERROR: model/ không có file .gguf. Chạy pipeline local 1 lần để tải GGUF rồi build lại." && exit 1)

# Cấu hình pipeline — GGUF mode
ENV HF_MODEL_ID=unsloth/Qwen3.5-4B-GGUF
ENV GGUF_FILE=Qwen3.5-4B-Q4_K_M.gguf
ENV HF_LOCAL_DIR=/app/model
ENV LLM_MAX_NEW_TOKENS=16
ENV LLM_ANSWER_MAX_TOKENS=16
ENV LLM_USE_LLM_ROUTE=0
ENV LLAMA_N_GPU_LAYERS=0
ENV LLAMA_N_CTX=4096

# Entry-point BTC
ENV COMPETITION=1
ENV PIPELINE_MODE=llm
ENV DATA_DIR=/data
ENV OUTPUT_DIR=/output

RUN chmod +x /app/docker_entry.sh

ENTRYPOINT ["/app/docker_entry.sh"]
