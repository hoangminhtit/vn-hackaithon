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
      2. CUDA GPU available          → -1 (offload all layers to GPU)
      3. Apple Metal (MPS) available → -1 (offload all layers to GPU)
      4. CPU only                    →  0 (no GPU offload)
    """
    # 1. Manual override via env var
    env_val = os.getenv("LLAMA_N_GPU_LAYERS", "").strip()
    if env_val:
        try:
            return int(env_val)
        except ValueError:
            pass

    # 2. Try CUDA
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[LLM] CUDA GPU detected: {gpu_name} → n_gpu_layers=-1")
            return -1
    except ImportError:
        pass

    # 3. Try Apple Metal (MPS) — check via llama_cpp directly
    try:
        import llama_cpp  # type: ignore
        # llama_cpp built with Metal support exposes GGML_USE_METAL
        if getattr(llama_cpp, "GGML_USE_METAL", False) or getattr(llama_cpp, "_LIB", None) and hasattr(llama_cpp._LIB, "llama_supports_gpu_offload"):
            print("[LLM] Apple Metal detected → n_gpu_layers=-1")
            return -1
    except Exception:
        pass

    # 3b. Fallback Metal check via platform
    import platform
    import subprocess
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["python3", "-c", "import torch; print(torch.backends.mps.is_available())"],
                capture_output=True, text=True, timeout=5
            )
            if result.stdout.strip() == "True":
                print("[LLM] Apple Metal (MPS) detected → n_gpu_layers=-1")
                return -1
        except Exception:
            pass

    # 4. CPU fallback
    print("[LLM] No GPU detected → n_gpu_layers=0 (CPU only)")
    return 0


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
