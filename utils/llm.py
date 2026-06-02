import json
import os
import urllib.error
import urllib.request
from typing import Optional


class LLMClient:
    def __init__(self, api_url: str, model: str, api_key: str = "", timeout: int = 60):
        self.api_url = api_url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> Optional["LLMClient"]:
        api_url = os.getenv("LLM_API_URL", "").strip()
        model = os.getenv("LLM_MODEL", "").strip() or os.getenv("HF_MODEL_ID", "").strip()
        api_key = os.getenv("LLM_API_KEY", "").strip()

        # HuggingFace Inference Providers router (OpenAI-compatible).
        # If user only provides HF token/model, we auto-wire this endpoint.
        hf_token = os.getenv("HF_TOKEN", "").strip()
        if not api_url and hf_token:
            api_url = "https://router.huggingface.co/v1/chat/completions"
            api_key = hf_token if not api_key else api_key

        if not api_url or not model:
            return None
        return cls(api_url=api_url, model=model, api_key=api_key)

    def chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 256) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.api_url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("LLM response does not match chat-completions schema") from exc
