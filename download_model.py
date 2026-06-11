#!/usr/bin/env python3
"""Download GGUF model file from HuggingFace Hub.

Usage:
    python download_model.py                          # dùng env vars / defaults
    python download_model.py --token hf_xxxxx         # truyền token trực tiếp
    HF_TOKEN=hf_xxxxx python download_model.py        # token qua env

Env vars:
    HF_TOKEN          — HuggingFace access token (nếu repo cần auth)
    HF_MODEL_ID       — repo id (mặc định: unsloth/Qwen3.5-4B-GGUF)
    GGUF_FILE         — tên file GGUF (mặc định: Qwen3.5-4B-Q4_K_M.gguf)
    HF_LOCAL_DIR      — thư mục lưu (mặc định: model/)
"""
import argparse
import os
import sys

from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GGUF model from HuggingFace Hub")
    parser.add_argument("--repo", default=None, help="HF repo id (default: $HF_MODEL_ID or unsloth/Qwen3.5-4B-GGUF)")
    parser.add_argument("--file", default=None, help="GGUF filename (default: $GGUF_FILE or Qwen3.5-4B-Q4_K_M.gguf)")
    parser.add_argument("--dir", default=None, help="Local directory (default: $HF_LOCAL_DIR or model/)")
    parser.add_argument("--token", default=None, help="HF access token (default: $HF_TOKEN)")
    args = parser.parse_args()

    repo = args.repo or os.getenv("HF_MODEL_ID", "").strip() or "unsloth/Qwen3.5-4B-GGUF"
    filename = args.file or os.getenv("GGUF_FILE", "").strip() or "Qwen3.5-4B-Q4_K_M.gguf"
    local_dir = args.dir or os.getenv("HF_LOCAL_DIR", "").strip() or "model"
    token = args.token or os.getenv("HF_TOKEN", "").strip() or None

    local_dir = os.path.abspath(local_dir)
    os.makedirs(local_dir, exist_ok=True)

    local_path = os.path.join(local_dir, filename)
    if os.path.isfile(local_path):
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"✅ Đã có sẵn: {local_path} ({size_mb:.0f} MB)")
        return

    print(f"⬇️  Tải {repo}/{filename}")
    print(f"   → {local_dir}/")
    if token:
        print(f"   🔑 Token: {token[:8]}...{token[-4:]}")
    else:
        print(f"   ⚠️  Không có token (public repo OK)")

    downloaded = hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=local_dir,
        token=token,
    )
    size_mb = os.path.getsize(downloaded) / (1024 * 1024)
    print(f"✅ Tải xong: {downloaded} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
