import os
import threading
from typing import Optional

from huggingface_hub import hf_hub_download

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Default GGUF file name — override via GGUF_FILE env var.
_DEFAULT_GGUF_FILE = "Qwen3.5-4B-Q4_K_M.gguf"


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
    import platform
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

    Đây là root cause của lỗi: dù CUDA GPU có mặt và n_gpu_layers=-1,
    nếu llama-cpp-python được cài bằng 'pip install llama-cpp-python' thông thường
    thì nó là CPU-only prebuilt wheel và GPU offload sẽ bị ignore hoàn toàn.
    """
    if n_gpu_layers == 0:
        return  # CPU mode, không cần kiểm tra

    try:
        import llama_cpp  # type: ignore

        # llama-cpp >= 0.2.x expose thông tin build qua llama_cpp.__version__ và internal flags
        # Cách tin cậy nhất: gọi llama_supports_gpu_offload() qua binding
        lib = getattr(llama_cpp, "_lib", None) or getattr(llama_cpp, "_LIB", None)
        if lib is not None and hasattr(lib, "llama_supports_gpu_offload"):
            supports = lib.llama_supports_gpu_offload()
            if not supports:
                raise RuntimeError(
                    "[LLM] ❌ llama-cpp-python ĐƯỢC CÀI NHƯ CPU-ONLY! \n"
                    "       GPU offload sẽ bị ignore dù n_gpu_layers=-1.\n"
                    "       Fix: rebuild llama-cpp-python với CUDA:\n"
                    "         CMAKE_ARGS='-DGGML_CUDA=on' FORCE_CMAKE=1 "
                    "pip install llama-cpp-python --no-binary llama-cpp-python"
                )
            else:
                print("[LLM] ✅ llama-cpp-python CUDA backend xác nhận (llama_supports_gpu_offload=True)")
            return

        # Fallback: thử load model với 1 layer và kiểm tra offload metadata
        # (phương pháp này tốn thời gian hơn, chỉ dùng khi không có API ở trên)
        print(
            "[LLM] ⚠️  Không thể xác nhận CUDA backend qua API. "
            "Nếu model chạy chậm bất thường, kiểm tra llama-cpp-python build flags."
        )
    except ImportError:
        pass  # llama_cpp chưa load tới đây, bỏ qua



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

        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=os.getenv("DEBUG_LLM", "").strip() == "1",
        )

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
