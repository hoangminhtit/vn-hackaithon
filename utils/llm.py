import os
import threading
import warnings
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
        
        # Select device and precision based on hardware availability
        if torch.cuda.is_available():
            device_target = None
            device_map = "auto"
            dtype = torch.float16
        elif torch.backends.mps.is_available():
            device_target = "mps"
            device_map = None
            dtype = torch.float16
        else:
            device_target = "cpu"
            device_map = None
            dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=self.cache_dir,
            trust_remote_code=allow_remote_code,
        )

        load_kwargs = dict(
            cache_dir=self.cache_dir,
            torch_dtype=dtype,
            trust_remote_code=allow_remote_code,
            low_cpu_mem_usage=True,
        )
        if device_map:
            load_kwargs["device_map"] = device_map
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        if device_target is not None:
            self.model = self.model.to(device_target)

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
        if os.getenv("HF_TOKEN", "").strip():
            warnings.warn(
                "HF_TOKEN is ignored in local-transformers mode. "
                "Use HF_MODEL_ID/LLM_MODEL and local cache only.",
                stacklevel=2,
            )
        model = os.getenv("LLM_MODEL", "").strip() or os.getenv("HF_MODEL_ID", "").strip()
        if not model:
            return None
        try:
            max_new_tokens = int(os.getenv("LLM_MAX_NEW_TOKENS", "256"))
        except ValueError:
            max_new_tokens = 256
        return cls(model_id=model, max_new_tokens=max_new_tokens)

    def chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 256, enable_thinking: bool = False) -> str:
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
                enable_thinking=enable_thinking,
            )
        except TypeError:
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
            eos_ids = eos_id
        else:
            eos_ids = [eos_id] if eos_id is not None else []

        if not enable_thinking:
            think_end_ids = self.tokenizer.encode("</think>", add_special_tokens=False)
            if think_end_ids:
                eos_ids.append(think_end_ids[-1])
        eos_ids = list(set(i for i in eos_ids if i is not None))

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else (eos_ids[0] if eos_ids else 0)
        with self._lock:
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_output_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=eos_ids if eos_ids else None,
            )
        generated_ids = output_ids[0][input_ids.shape[1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        return text
