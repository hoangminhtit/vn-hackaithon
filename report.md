# Implementation Report

## Scope

This implementation upgrades the original heuristic pipeline into an LLM-backed pipeline while preserving robust heuristic fallbacks.

Input format:
- JSON list of objects: `{qid, question, choices}`

Output format:
- CSV with schema: `qid,answer`

## What Was Implemented

### 1) Pre-processing
- File: `utils/preprocess.py`
- Dynamic answer label mapping for arbitrary choice counts: `A, B, C, ...`
- Passage vs question splitting from embedded `question` content
- Regex patterns for common Vietnamese structures (`Đoạn thông tin`, `[1]`, `Nội dung`)

### 2) Retrieval for RAG branch (CPU-friendly)
- File: `utils/bm25.py`
- Pure Python BM25-like scoring
- Sentence splitting + query overlap scoring
- Top-k context extraction with output length cap

### 3) Router
- Files: `router.py`, `prompts.py`
- Two-level routing:
  1. LLM router prompt (`domain`, `confidence`)
  2. Heuristic fallback router if LLM fails or output is invalid
- Fallback policy from design:
  - Long passage favors `rag` (except `math`)
  - Low confidence falls back to `multi_domain`

### 4) Domain handlers
- Files:
  - `domains/rag.py`
  - `domains/math.py`
  - `domains/multi_domain.py`
  - `domains/should_correct.py`
  - `domains/ignore_answer.py`
- LLM-backed answering by domain-specific prompts (defined in `prompts.py`)
- Heuristic fallbacks remain active for all domains when:
  - LLM API is unavailable
  - LLM output cannot be parsed
  - Runtime errors occur

### 5) Post-processing / parsing
- File: `utils/postprocess.py`
- `parse_answer()` parses JSON answer and validates against dynamic labels
- Fallback to label regex detection, then first valid label
- `parse_route_output()` parses JSON routing output safely

### 6) LLM client (OpenAI-compatible endpoint)
- File: `utils/llm.py`
- Environment-driven configuration:
  - `LLM_API_URL`
  - `LLM_MODEL`
  - `LLM_API_KEY` (optional)
- Calls chat completion style API and returns raw assistant text

### 7) Pipeline orchestration
- Files: `pipeline.py`, `run.py`
- Threaded batch processing for throughput
- Runtime modes:
  - `heuristic`: force heuristic-only
  - `llm`: force LLM mode
  - `auto`: use LLM when env is configured, else heuristic

## Files Added/Updated

Added:
- `utils/llm.py`
- `prompts.py`
- `report.md`
- `README.md`

Updated:
- `pipeline.py`
- `run.py`
- `utils/postprocess.py`

Existing modules retained and used:
- `router.py`
- `utils/preprocess.py`
- `utils/bm25.py`
- `domains/*`

## Notes

- The code is designed so you can swap models/endpoints without changing pipeline logic.
- If the LLM output format is imperfect, the system still returns valid labels through fallback parsing + heuristic domain solvers.
