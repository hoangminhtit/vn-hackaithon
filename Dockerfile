# ── CUDA 12.2 devel — bắt buộc theo yêu cầu BTC (CUDA 12.2) ────────────────────
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh

WORKDIR /code

# Cài Python 3.10 + build tools cần thiết để compile llama-cpp với CUDA.
# Ubuntu 22.04 có sẵn Python 3.10, tránh phụ thuộc deadsnakes PPA/GPG key.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip \
    git build-essential cmake ninja-build \
    curl \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Cài các dependency KHÔNG phải llama-cpp trước
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Build llama-cpp-python TỪ SOURCE với CUDA support ───────────────────────
# CMAKE_ARGS kích hoạt CUDA backend. Dùng danh sách arch cụ thể để tránh lỗi compute_ rỗng.
ARG CUDA_ARCHITECTURES="75;80;86;89;90"
RUN CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" \
    FORCE_CMAKE=1 \
    pip install --no-cache-dir --verbose \
    "llama-cpp-python>=0.3.0" --no-binary llama-cpp-python

# Code + .env (nếu có lúc build)
COPY inference.sh /code/inference.sh
COPY predict.py download_model.py pipeline.py router.py prompts.py few-shot.json ./
COPY domains ./domains
COPY utils ./utils
COPY .env* ./

# Bake GGUF weights — bắt buộc có file .gguf trong model/ trước khi docker build
COPY model/ /code/model/
RUN test -n "$(find /code/model -name '*.gguf' 2>/dev/null | head -1)" || \
    (echo "ERROR: model/ không có file .gguf. Chạy pipeline local 1 lần để tải GGUF rồi build lại." && exit 1)

# Cấu hình pipeline — GGUF mode
ENV HF_MODEL_ID=unsloth/Qwen3.5-4B-GGUF
ENV GGUF_FILE=Qwen3.5-4B-Q4_K_M.gguf
ENV HF_LOCAL_DIR=/code/model
ENV LLM_MAX_NEW_TOKENS=16
ENV LLM_ANSWER_MAX_TOKENS=16
ENV LLM_USE_LLM_ROUTE=0
ENV LLM_USE_POT_SCIENCE=1
ENV LLM_POT_MAX_TOKENS=512
ENV LLM_POT_RETRIES=1
ENV LLM_POT_TIMEOUT=2.0
ENV LLM_USE_COT_SHOULD_CORRECT=1
ENV LLM_USE_COT_MULTI=1
ENV LLM_COT_MAX_TOKENS=384
ENV LLM_USE_ANSWER_VERIFIER=1
ENV LLM_VERIFY_MULTI=0
ENV LLM_VERIFY_MULTI_MANY_CHOICES=0
ENV LLM_VERIFY_MAX_TOKENS=320
ENV LLM_USE_RAG_EVIDENCE=1
ENV LLM_RAG_EVIDENCE_MAX_TOKENS=512
ENV RAG_MAX_CONTEXT_CHARS=12000
ENV RAG_FULL_PASSAGE_CHARS=12000
ENV RAG_BM25_MAX_CHARS=10000
ENV RAG_BM25_TOP_K=12
# LLAMA_N_GPU_LAYERS: để trống → Python tự auto-detect CUDA/CPU.
# Override thủ công: docker run -e LLAMA_N_GPU_LAYERS=-1 ...  (GPU full)
#                               -e LLAMA_N_GPU_LAYERS=0  ...  (CPU only)
ENV LLAMA_N_GPU_LAYERS=-1
ENV LLAMA_N_CTX=4096

# Entry-point BTC
ENV COMPETITION=1
ENV PIPELINE_MODE=llm

RUN sed -i 's/\r$//' /code/inference.sh /code/.env* 2>/dev/null || true && \
    chmod +x /code/inference.sh

CMD ["bash", "inference.sh"]
