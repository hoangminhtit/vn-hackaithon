import os
import platform
import threading
from typing import Optional

from huggingface_hub import hf_hub_download

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Default GGUF file name — override via GGUF_FILE env var.
_DEFAULT_GGUF_FILE = "Qwen3.5-4B-Q4_K_M.gguf"


def _running_on_kaggle() -> bool:
    return bool(os.getenv("KAGGLE_KERNEL_RUN_TYPE") or os.getenv("KAGGLE_URL_BASE"))


def _cuda_install_hint() -> str:
    if _running_on_kaggle():
        return (
            "       Fix tren Kaggle notebook:\n"
            "         !pip uninstall -y llama-cpp-python\n"
            "         # Chon cu124/cu125 theo torch.version.cuda; cu125 thuong chay duoc tren driver moi.\n"
            "         !pip install --no-cache-dir --force-reinstall 'llama-cpp-python>=0.3.0' "
            "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125\n"
            "       Sau do restart kernel/runtime de Python load lai shared library CUDA."
        )
    return (
        "       Fix tren Linux CUDA:\n"
        "         pip uninstall -y llama-cpp-python\n"
        "         CMAKE_ARGS='-DGGML_CUDA=on' FORCE_CMAKE=1 "
        "pip install --no-cache-dir --force-reinstall --no-binary llama-cpp-python "
        "'llama-cpp-python>=0.3.0'"
    )


def _auto_detect_gpu_layers() -> int:
    """Auto-detect the best n_gpu_layers for the current hardware.

    Priority:
      1. LLAMA_N_GPU_LAYERS env var  — manual override (skip auto-detect)
      2. CUDA GPU available (via torch) → -1 (offload all layers to GPU)
      3. CUDA GPU available (via nvidia-smi fallback, khi không có torch) → -1
      4. Apple Metal (MPS) available → -1 (offload all layers to GPU)
      5. CPU only                    →  0 (no GPU offload)
    """
    # 1. Manual override via env var
    env_val = os.getenv("LLAMA_N_GPU_LAYERS", "").strip()
    if env_val:
        try:
            n = int(env_val)
            print(f"[LLM] LLAMA_N_GPU_LAYERS env override → n_gpu_layers={n}")
            return n
        except ValueError:
            print(f"[LLM] Warning: LLAMA_N_GPU_LAYERS='{env_val}' không hợp lệ, bỏ qua.")

    # 2. Try CUDA via torch (cách nhanh nhất và đáng tin nhất)
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            print(f"[LLM] ✅ CUDA GPU detected: {gpu_name} ({total_mb} MB VRAM) → n_gpu_layers=-1 (full GPU offload)")
            return -1
        else:
            print("[LLM] torch installed nhưng CUDA không khả dụng (CUDA not available).")
    except ImportError:
        pass  # torch chưa cài → thử fallback nvidia-smi

    # 3. Fallback CUDA check qua nvidia-smi (khi torch không có hoặc không thấy GPU)
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_info = result.stdout.strip().splitlines()[0].strip()
            print(f"[LLM] ✅ CUDA GPU detected via nvidia-smi: {gpu_info} → n_gpu_layers=-1 (full GPU offload)")
            return -1
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass  # nvidia-smi không có → không phải CUDA machine

    # 4. Try Apple Metal (MPS) — check via llama_cpp directly
    try:
        import llama_cpp  # type: ignore
        if getattr(llama_cpp, "GGML_USE_METAL", False) or (
            getattr(llama_cpp, "_LIB", None) and hasattr(llama_cpp._LIB, "llama_supports_gpu_offload")
        ):
            print("[LLM] ✅ Apple Metal detected → n_gpu_layers=-1 (full GPU offload)")
            return -1
    except Exception:
        pass

    # 4b. Fallback Metal check via platform + torch MPS
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["python3", "-c", "import torch; print(torch.backends.mps.is_available())"],
                capture_output=True, text=True, timeout=5
            )
            if result.stdout.strip() == "True":
                print("[LLM] ✅ Apple Metal (MPS) detected → n_gpu_layers=-1 (full GPU offload)")
                return -1
        except Exception:
            pass

    # 5. CPU fallback
    print("[LLM] ⚠️  Không tìm thấy GPU (CUDA/Metal) → n_gpu_layers=0 (CPU only)")
    return 0


def _verify_cuda_backend(n_gpu_layers: int) -> None:
    """Kiểm tra llama-cpp-python có CUDA backend không khi yêu cầu GPU offload.

    Root cause: dù CUDA GPU có mặt và n_gpu_layers=-1,
    nếu llama-cpp-python là CPU-only wheel thì GPU offload bị ignore hoàn toàn.

    Trong llama-cpp-python >= 0.3.x, internal C lib nằm ở:
        llama_cpp.llama_cpp._lib   (module llama_cpp.llama_cpp là ctypes binding)
    """
    if n_gpu_layers == 0:
        return  # CPU mode, không cần kiểm tra

    try:
        # llama-cpp-python 0.3.x: ctypes binding nằm ở sub-module llama_cpp.llama_cpp
        import llama_cpp.llama_cpp as _lc  # type: ignore

        lib = getattr(_lc, "_lib", None)
        if lib is None:
            print(
                "[LLM] ⚠️  Không tìm thấy _lib trong llama_cpp.llama_cpp — "
                "không thể xác nhận CUDA backend trước khi load model."
            )
            return

        if hasattr(lib, "llama_supports_gpu_offload"):
            supports = bool(lib.llama_supports_gpu_offload())
            if supports:
                print("[LLM] ✅ llama-cpp-python CUDA backend xác nhận (llama_supports_gpu_offload=True)")
            else:
                raise RuntimeError(
                    "[LLM] ❌ llama-cpp-python được cài CPU-ONLY!\n"
                    "       GPU offload sẽ bị ignore dù n_gpu_layers=-1.\n"
                    f"{_cuda_install_hint()}"
                )
        else:
            # llama_supports_gpu_offload không có — thử kiểm tra GGML_USE_CUDA flag
            ggml_cuda = getattr(lib, "GGML_USE_CUDA", None) or getattr(_lc, "GGML_USE_CUDA", None)
            if ggml_cuda is not None:
                print(f"[LLM] ✅ GGML_USE_CUDA={ggml_cuda} — CUDA backend có mặt")
            else:
                print(
                    "[LLM] ⚠️  Không thể xác nhận CUDA backend (API không tìm thấy). "
                    "Sẽ verify lại sau khi model load xong."
                )
    except ImportError:
        print("[LLM] ⚠️  Không import được llama_cpp.llama_cpp — bỏ qua pre-load check.")


def _verify_gpu_offload_post_load(model: "Llama", n_gpu_layers_requested: int) -> None:  # type: ignore[name-defined]
    """Verify sau khi Llama() init xong: đọc metadata thực tế để xác nhận GPU layers.

    Đây là cách chắc chắn nhất — llama.cpp log 'offloaded X/Y layers to GPU'
    khi verbose=True, nhưng ta cũng có thể đọc qua C binding trực tiếp.
    """
    if n_gpu_layers_requested == 0:
        return

    try:
        # llama-cpp-python 0.3.x: model._model là LlamaModel, model._ctx là LlamaContext
        # n_gpu_layers thực tế có thể đọc qua model.model_params
        params = getattr(model, "model_params", None)
        if params is not None:
            actual_layers = getattr(params, "n_gpu_layers", None)
            if actual_layers is not None:
                if actual_layers != 0:
                    print(f"[LLM] ✅ GPU offload confirmed: model_params.n_gpu_layers={actual_layers}")
                else:
                    print(
                        "[LLM] ❌ model_params.n_gpu_layers=0 — GPU offload KHÔNG hoạt động!\n"
                        "       llama-cpp-python có thể vẫn là CPU-only build."
                    )
                return

        # Fallback: thử đọc số layers qua C API nếu có
        import llama_cpp.llama_cpp as _lc  # type: ignore
        lib = getattr(_lc, "_lib", None)
        if lib and hasattr(lib, "llama_model_n_gpu_layers"):
            underlying = getattr(model, "_model", None)
            if underlying is not None:
                ptr = getattr(underlying, "model", None)
                if ptr is not None:
                    n_actual = lib.llama_model_n_gpu_layers(ptr)
                    if n_actual > 0:
                        print(f"[LLM] ✅ GPU offload confirmed: {n_actual} layers trên GPU")
                    else:
                        print("[LLM] ❌ llama_model_n_gpu_layers=0 — model đang chạy CPU!")
                    return

        # Nếu không có API nào — in hướng dẫn debug thủ công
        print(
            "[LLM] ℹ️  Không thể đọc GPU layer count qua API.\n"
            "       Để verify thủ công: set DEBUG_LLM=1 và tìm dòng 'offloaded X/Y layers to GPU' trong log."
        )
    except Exception as e:
        print(f"[LLM] ⚠️  Post-load GPU verify gặp lỗi (không nghiêm trọng): {e}")



class LLMClient:
    def __init__(self, model_id: str, max_new_tokens: int = 256, gguf_file: Optional[str] = None):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._lock = threading.Lock()

        # Resolve local cache directory.
        cache_dir_env = os.getenv("HF_LOCAL_DIR", "").strip()
        self.cache_dir = (
            os.path.abspath(cache_dir_env)
            if cache_dir_env
            else os.path.join(PROJECT_ROOT, "model")
        )
        os.makedirs(self.cache_dir, exist_ok=True)

        # Determine the GGUF filename to use.
        self.gguf_file = gguf_file or os.getenv("GGUF_FILE", "").strip() or _DEFAULT_GGUF_FILE

        # Resolve path to the .gguf file — download if needed.
        model_path = self._resolve_gguf_path()

        # Determine context size from env (default 4096 — enough for MCQ).
        try:
            n_ctx = int(os.getenv("LLAMA_N_CTX", "4096"))
        except ValueError:
            n_ctx = 4096

        # Auto-detect GPU and set n_gpu_layers accordingly.
        # LLAMA_N_GPU_LAYERS env var still works as a manual override.
        n_gpu_layers = _auto_detect_gpu_layers()

        # ❌ Kiểm tra ngay: nếu yêu cầu GPU offload mà llama-cpp-python là CPU-only,
        # báo lỗi rõ ràng thay vì im lặng chạy CPU
        _verify_cuda_backend(n_gpu_layers)

        # Lazy import so that llama_cpp is only needed when actually using LLM.
        from llama_cpp import Llama  # type: ignore

        # verbose=True bắt buộc khi GPU offload để thấy 'offloaded X/Y layers' trong log
        # Nếu DEBUG_LLM=1 hoặc GPU requested → bật verbose để capture GPU offload info
        debug_verbose = os.getenv("DEBUG_LLM", "").strip() == "1"
        load_verbose = debug_verbose or (n_gpu_layers != 0)

        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=load_verbose,
        )

        # Verify thực tế GPU layers sau khi load
        _verify_gpu_offload_post_load(self.model, n_gpu_layers)

        if not debug_verbose and load_verbose:
            # Đã bật verbose chỉ để verify GPU — log thêm hint
            print("[LLM] ℹ️  Set DEBUG_LLM=1 để xem full llama.cpp log khi cần debug.")

    def _resolve_gguf_path(self) -> str:
        """Return absolute path to the GGUF model file, downloading from HF if needed."""
        # 1. Check if the file already exists in the cache directory.
        local_path = os.path.join(self.cache_dir, self.gguf_file)
        if os.path.isfile(local_path):
            return local_path

        # 2. Try downloading from HuggingFace Hub.
        token = os.getenv("HF_TOKEN", "").strip() or None
        print(f"[LLM] Downloading {self.model_id}/{self.gguf_file} → {self.cache_dir} ...")
        downloaded = hf_hub_download(
            repo_id=self.model_id,
            filename=self.gguf_file,
            local_dir=self.cache_dir,
            token=token,
        )
        return downloaded

    @classmethod
    def from_env(cls) -> Optional["LLMClient"]:
        model = os.getenv("LLM_MODEL", "").strip() or os.getenv("HF_MODEL_ID", "").strip()
        if not model:
            return None
        try:
            max_new_tokens = int(os.getenv("LLM_MAX_NEW_TOKENS", "256"))
        except ValueError:
            max_new_tokens = 256
        return cls(model_id=model, max_new_tokens=max_new_tokens)

    def chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 256, enable_thinking: bool = False) -> str:
        """Send a chat-completion request and return the assistant's text reply."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        max_output_tokens = min(max_tokens, self.max_new_tokens)

        with self._lock:
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_output_tokens,
                temperature=0.0,  # greedy — deterministic
            )

        text: str = response["choices"][0]["message"]["content"] or ""
        text = text.strip()

        # Strip <think>…</think> blocks if thinking is disabled.
        if not enable_thinking and "</think>" in text:
            text = text.split("</think>", 1)[1].strip()

        return text
