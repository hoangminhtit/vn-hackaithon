import os
import threading
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class LLMClient:
    def __init__(self, model_id: str, max_new_tokens: int = 256):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._lock = threading.Lock()
        # Default cache directory inside project so model files are local and explicit.
        cache_dir_env = os.getenv("HF_LOCAL_DIR", "").strip()
        self.cache_dir = (
            os.path.abspath(cache_dir_env)
            if cache_dir_env
            else os.path.join(PROJECT_ROOT, "model")
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        # Keep remote code loading explicit and constrained by default.
        allow_remote_code = self.model_id.startswith("Qwen/")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=self.cache_dir,
            trust_remote_code=allow_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=allow_remote_code,
        )

    def _inference_device(self) -> torch.device:
        model_device = getattr(self.model, "device", None)
        if isinstance(model_device, torch.device) and model_device.type != "meta":
            return model_device
        for param in self.model.parameters():
            if param.device.type != "meta":
                return param.device
        return torch.device("cpu")

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

    def chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 256) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        except Exception:
            fallback_prompt = f"{system_prompt}\n\n{user_prompt}"
            encoded = self.tokenizer(fallback_prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        device = self._inference_device()
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        max_output_tokens = min(max_tokens, self.max_new_tokens)
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            eos_id = getattr(self.model.config, "eos_token_id", None)
        if isinstance(eos_id, list):
            eos_id = eos_id[0] if eos_id else None
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id
        with self._lock:
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_output_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
        generated_ids = output_ids[0][input_ids.shape[1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
