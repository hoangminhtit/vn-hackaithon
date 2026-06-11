# ── CUDA 12.1 runtime + cuDNN 8 (compat với Tesla T4 trên Kaggle) ──────────────
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

WORKDIR /app

# Cài Python 3.10 + build tools cần thiết để compile llama-cpp với CUDA
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    git build-essential cmake ninja-build \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Cài các dependency KHÔNG phải llama-cpp trước
COPY requirements.txt .
RUN grep -v 'llama.cpp' requirements.txt | pip install --no-cache-dir -r /dev/stdin

# ── Build llama-cpp-python TỪ SOURCE với CUDA support ───────────────────────
# CMAKE_ARGS kích hoạt CUDA backend, CUDA_DOCKER_ARCH=all để tương thích mọi GPU
RUN CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=all" \
    FORCE_CMAKE=1 \
    pip install --no-cache-dir --verbose \
    llama-cpp-python>=0.3.0 --no-binary llama-cpp-python

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
# LLAMA_N_GPU_LAYERS: để trống → Python tự auto-detect CUDA/CPU.
# Override thủ công: docker run -e LLAMA_N_GPU_LAYERS=-1 ...  (GPU full)
#                               -e LLAMA_N_GPU_LAYERS=0  ...  (CPU only)
ENV LLAMA_N_GPU_LAYERS=
ENV LLAMA_N_CTX=4096

# Entry-point BTC
ENV COMPETITION=1
ENV PIPELINE_MODE=llm
ENV DATA_DIR=/data
ENV OUTPUT_DIR=/output

RUN chmod +x /app/docker_entry.sh

ENTRYPOINT ["/app/docker_entry.sh"]
