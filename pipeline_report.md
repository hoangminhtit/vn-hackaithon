# Pipeline Thiết Kế Hệ Thống Trả Lời Trắc Nghiệm Đa Lĩnh Vực (Tiếng Việt)

---

## 📌 Tổng quan bài toán

- **Input:** Câu hỏi trắc nghiệm tiếng Việt, mỗi câu có thể kèm đoạn văn dài (passage nhúng trực tiếp trong `question`)
- **Output:** Đáp án đúng (chữ cái: A / B / C / D / ...)
- **Số lượng choices:** Không cố định — có thể 3, 4 hoặc nhiều hơn
- **Số lượng test:** ~2000 câu (private test)
- **Môi trường:** CPU only
- **Ngôn ngữ dữ liệu:** Tiếng Việt

### Cấu trúc input mẫu

```json
{
  "qid": "test_0001",
  "question": "Đoạn thông tin:\n[1] Tiêu đề: ...\nNội dung: ...\n\nCâu hỏi: ...",
  "choices": [
    "Lựa chọn A",
    "Lựa chọn B",
    "Lựa chọn C"
  ]
}
```

> ⚠️ **Lưu ý quan trọng:** Passage được nhúng trực tiếp trong trường `question`, không tách riêng. Pipeline cần tự phát hiện và tách passage vs câu hỏi thực sự.

---

## 🗂️ Các Domain

| Domain | Đặc điểm nhận dạng |
|---|---|
| `rag` | Có đoạn văn/passage đi kèm trong question |
| `math` | Toán học, tính toán, phương trình, số liệu |
| `multi_domain` | Tổng hợp nhiều lĩnh vực, cần tư duy phản biện |
| `should_correct` | Yêu cầu kiểm tra tính đúng/sai của phát biểu |
| `ignore_answer` | Không có đáp án nào đúng (câu bẫy) |

---

## 🏗️ Kiến trúc Pipeline Tổng Thể

Qwen/Qwen3.5-4B

```
Input JSON: {qid, question, choices}
            ↓
┌─────────────────────────────────────┐
│         PRE-PROCESSING              │
│  - Tách passage vs câu hỏi thực     │
│  - Map choices → chữ cái động       │
│    (A, B, C, D, E... tuỳ số lượng)  │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│         AGENT ROUTER                │  ~100ms
│  Qwen3.5-4B (no-think mode)         │
│  → {domain, confidence}             │
│  Nếu confidence < 0.6 → multi_domain│
└──────────────┬──────────────────────┘
               ↓
    ┌──────────┴──────────┐
    │   Domain Branches   │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  RAG                │  BM25 + Qwen3.5-4B   ~350ms
    ├─────────────────────┤
    │  MATH               │  CoT + Qwen3.5-4B    ~800ms
    ├─────────────────────┤
    │  MULTI_DOMAIN       │  Decompose + Qwen     ~600ms
    ├─────────────────────┤
    │  SHOULD_CORRECT     │  CoT + JSON output    ~400ms
    ├─────────────────────┤
    │  IGNORE_ANSWER      │  Sanity check         ~250ms
    └──────────┬──────────┘
               ↓
┌─────────────────────────────────────┐
│         POST-PROCESSING             │
│  - Parse JSON output                │
│  - Map chữ cái → index choices      │
│  - Fallback nếu parse lỗi           │
└──────────────┬──────────────────────┘
               ↓
Output: {"qid": "test_0001", "answer": "A"}
```

---

## 🔧 Chi tiết từng bước

### Bước 0: Pre-processing (Tách passage & map choices)

Vì `choices` có thể có 3, 4 hoặc nhiều hơn options, cần map **động**:

```python
import re

def preprocess(item: dict) -> dict:
    question_raw = item["question"]
    choices = item["choices"]

    # Map choices động theo số lượng thực tế
    labels = [chr(ord('A') + i) for i in range(len(choices))]
    choice_map = {label: text for label, text in zip(labels, choices)}

    # Tách passage nếu có (dạng "Đoạn thông tin:" hoặc "[1] Tiêu đề:")
    passage = ""
    question_clean = question_raw
    
    passage_patterns = [
        r"(Đoạn thông tin:.*?)(?=Câu hỏi:)",
        r"(\[1\].*?)(?=Câu hỏi:)",
        r"(Nội dung:.*?)(?=Câu hỏi:)"
    ]
    for pattern in passage_patterns:
        match = re.search(pattern, question_raw, re.DOTALL)
        if match:
            passage = match.group(1).strip()
            question_clean = question_raw[match.end():].replace("Câu hỏi:", "").strip()
            break

    return {
        "qid": item["qid"],
        "passage": passage,
        "question": question_clean,
        "choices": choice_map,   # {"A": "...", "B": "...", "C": "..."}
        "num_choices": len(choices)
    }
```

---

### Bước 1: Agent Router

Dùng **Qwen3.5-4B ở no-think mode** để phân loại domain nhanh.

```
SYSTEM:
Bạn là bộ phân loại câu hỏi tiếng Việt. Chỉ trả về JSON, không giải thích.

Các domain:
- rag: câu hỏi dựa trên đoạn văn/passage được cung cấp
- math: toán học, tính toán, phương trình, xác suất, thống kê
- multi_domain: kết hợp nhiều lĩnh vực, cần tư duy tổng hợp
- should_correct: yêu cầu xác định phát biểu đúng hay sai
- ignore_answer: không có đáp án nào đúng trong các lựa chọn

USER:
Passage (nếu có): {passage[:200]}
Câu hỏi: {question}
Số lượng đáp án: {num_choices}

Trả lời JSON:
{"domain": "rag", "confidence": 0.95}
```

**Logic fallback:**
```python
def route(result: dict, passage: str) -> str:
    # Nếu có passage dài → ưu tiên RAG
    if passage and len(passage) > 100:
        if result["domain"] != "math":
            return "rag"
    # Confidence thấp → fallback multi_domain
    if result["confidence"] < 0.6:
        return "multi_domain"
    return result["domain"]
```

---

### Bước 2A: RAG — BM25 thay thế Embedding (CPU-friendly)

**Lý do bỏ BGE-m3 ở inference time:**
- BGE-m3 trên CPU: ~500ms-1s mỗi lần embed
- BM25 thuần Python: ~1-5ms, không cần GPU
- Với passage đi kèm sẵn, keyword overlap đã đủ tốt cho tiếng Việt

```python
from rank_bm25 import BM25Okapi
import re

def bm25_retrieve(passage: str, question: str, top_k: int = 5) -> str:
    # Tách câu
    sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)', passage)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if not sentences:
        return passage[:1000]  # fallback: lấy 1000 ký tự đầu
    
    # Tokenize đơn giản cho tiếng Việt (tách theo khoảng trắng)
    tokenized = [s.lower().split() for s in sentences]
    bm25 = BM25Okapi(tokenized)
    
    query_tokens = question.lower().split()
    scores = bm25.get_scores(query_tokens)
    
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    top_indices.sort()  # Giữ thứ tự gốc trong passage
    
    return " ".join([sentences[i] for i in top_indices])
```

**RAG Prompt:**
```
SYSTEM:
Bạn là trợ lý trả lời câu hỏi dựa trên tài liệu tiếng Việt.
Chỉ dựa vào thông tin được cung cấp. Trả lời JSON, không giải thích thêm.

USER:
### Thông tin liên quan:
{bm25_context}

### Câu hỏi:
{question}

### Các đáp án:
{choice_A}. {text_A}
{choice_B}. {text_B}
...

Đáp án đúng là chữ cái nào? 
{"answer": "A", "reason": "...ngắn gọn..."}
```

---

### Bước 2B: Math — CoT + Self-verify

```
SYSTEM:
Bạn là chuyên gia toán học. Giải từng bước, kiểm tra lại kết quả.
Trả về đúng format XML+JSON được yêu cầu.

USER:
Câu hỏi: {question}

Các đáp án:
A. {text_A}
B. {text_B}
...

<reasoning>
Phân tích bước 1: ...
Phân tích bước 2: ...
Kiểm tra lại: ...
Kết luận: đáp án là ...
</reasoning>

<answer>{"answer": "A", "reason": "..."}</answer>
```

**Lưu ý:** Dùng Qwen3.5 với `/think` tag nếu model hỗ trợ, hoặc Gemma-4 nếu benchmark cho thấy tốt hơn với bài toán số học.

---

### Bước 2C: Multi-domain — Decomposition Prompting

```
SYSTEM:
Bạn là trợ lý thông minh, giỏi phân tích câu hỏi tổng hợp tiếng Việt.

USER:
Câu hỏi: {question}

Các đáp án:
A. {text_A}
B. {text_B}
...

Hãy làm theo các bước:
1. Xác định câu hỏi đang hỏi về lĩnh vực/khía cạnh gì.
2. Phân tích từng đáp án dựa trên kiến thức liên quan.
3. Loại bỏ đáp án sai, chọn đáp án chính xác nhất.

{"answer": "A", "reason": "..."}
```

---

### Bước 2D: Should Correct — Kiểm tra tính đúng/sai

```
SYSTEM:
Bạn là chuyên gia kiểm tra tính chính xác của thông tin tiếng Việt.

USER:
Câu hỏi/Phát biểu cần kiểm tra: {question}

Các đáp án:
A. {text_A}
B. {text_B}
...

Suy nghĩ từng bước:
1. Phát biểu cần kiểm tra là gì?
2. Phát biểu này đúng hay sai? Tại sao?
3. Nếu sai, thông tin đúng là gì? Đáp án nào phản ánh điều này?

{"is_correct": false, "correction": "...", "answer": "B"}
```

---

### Bước 2E: Ignore Answer — Phát hiện câu bẫy

```
SYSTEM:
Bạn là trợ lý phân tích câu hỏi tiếng Việt.
Một số câu hỏi không có đáp án đúng — hãy cẩn thận kiểm tra từng lựa chọn.

USER:
Câu hỏi: {question}

Các đáp án:
A. {text_A}
B. {text_B}
...

Kiểm tra từng đáp án:
- Đáp án A: đúng hay sai? Tại sao?
- Đáp án B: đúng hay sai? Tại sao?
...

Nếu có đáp án đúng → chọn chữ cái đó.
Nếu không có đáp án nào đúng → tìm những đáp án "Tôi không thể cung cấp thông tin" hoặc tương tự".

{"answer": "A", "reason": "..."}
hoặc
{"answer": "NONE", "reason": "Không có đáp án nào chính xác vì..."}
```

---

### Bước 3: Post-processing & Fallback

```python
import re, json

def parse_answer(raw_output: str, num_choices: int) -> str:
    valid_labels = [chr(ord('A') + i) for i in range(num_choices)]
    
    # Thử parse JSON
    try:
        clean = re.sub(r"```json|```", "", raw_output).strip()
        # Tìm JSON cuối cùng trong output (tránh JSON trong reasoning)
        json_matches = re.findall(r'\{[^{}]*"answer"[^{}]*\}', clean)
        if json_matches:
            data = json.loads(json_matches[-1])
            answer = data.get("answer", "").strip().upper()
            if answer in valid_labels or answer == "NONE":
                return answer
    except Exception:
        pass

    # Fallback: tìm chữ cái đơn lẻ trong output
    for label in valid_labels:
        pattern = rf'\b{label}\b'
        if re.search(pattern, raw_output):
            return label

    # Fallback cuối: trả về đáp án đầu tiên
    return valid_labels[0]
```

---

## ⚡ Tối ưu Latency cho CPU + 2000 câu

### Ước tính thời gian

| Domain | Approach | Thời gian/câu (CPU) |
|---|---|---|
| Router | Qwen 4B no-think | ~100ms |
| RAG | BM25 + Qwen 4B | ~350ms |
| Math | Qwen 4B + CoT | ~800ms |
| Multi-domain | Qwen 4B | ~600ms |
| Should Correct | Qwen 4B CoT | ~400ms |
| Ignore Answer | Qwen 4B | ~250ms |

**Trung bình ~400ms/câu × 2000 câu = ~800 giây (~13 phút)** nếu tuần tự.

### Giải pháp: Multiprocessing

```python
from concurrent.futures import ThreadPoolExecutor
import os

def process_question(item: dict) -> dict:
    processed = preprocess(item)
    domain = route_question(processed)
    answer = run_domain_pipeline(domain, processed)
    return {"qid": item["qid"], "answer": answer}

# Dùng số core CPU thực tế
num_workers = min(os.cpu_count(), 8)

with ThreadPoolExecutor(max_workers=num_workers) as executor:
    results = list(executor.map(process_question, questions))
```

**Với 4 cores → ~3-4 phút | Với 8 cores → ~1.5-2 phút** ✅

### Các tối ưu khác

| Kỹ thuật | Mô tả | Tiết kiệm |
|---|---|---|
| `no-think` mode cho Router, RAG, Ignore | Chỉ dùng `/think` cho Math | ~30-40% |
| Giới hạn `max_tokens` output | Router: 50, RAG: 100, Math: 300 | ~20% |
| Bỏ BGE-m3/Qwen-Rerank ở inference | Thay bằng BM25 thuần | ~60% với RAG |
| Cascade Model | 4B trước, chỉ escalate khi confidence thấp | ~25% |
| Giới hạn BM25 context | Top-5 câu, max 600 ký tự | Giảm LLM input |

---

## 📊 So sánh Approach cũ vs mới

| Thành phần | Cũ | Mới | Lý do thay đổi |
|---|---|---|---|
| RAG retrieval | BGE-m3 embedding | BM25 thuần Python | CPU-friendly, 100x nhanh hơn |
| Chunking | Chunk overlap | Sentence split + BM25 | Không cần vector index |
| Rerank | Qwen-Rerank | Bỏ (BM25 score đủ tốt) | Quá chậm trên CPU |
| Thinking mode | Tất cả domain | Chỉ Math + Should Correct | Tiết kiệm latency |
| Choice mapping | Cố định A/B/C/D | Động theo num_choices | Xử lý 3, 4, 5+ options |
| Passage tách | Thủ công | Regex tự động từ `question` | Dữ liệu nhúng trong question |

---

## 🎯 Chiến lược Cascade Model (Sáng tạo)

Với mục tiêu cân bằng **Accuracy vs Latency**:

```
Câu hỏi
    ↓
Qwen3.5-4B (model nhỏ, nhanh)
    ↓
confidence > 0.85?
    ├── YES → Trả lời luôn ✓  (~300ms)
    └── NO  → Gemma-4 (model lớn hơn)  (~800ms)
                ↓
            Trả lời cuối cùng
```

**Lợi ích:** ~70-80% câu hỏi dễ được giải quyết bởi model nhỏ → tiết kiệm đáng kể tổng thời gian.

---

## 📁 Cấu trúc thư mục đề xuất

```
project/
├── pipeline.py          # Pipeline chính
├── router.py            # Agent router
├── data/                # data public/ private test
├── output/              # storage CSV format include: qid,answer 
├── domains/
│   ├── rag.py           # RAG domain handler
│   ├── math.py          # Math domain handler
│   ├── multi_domain.py  # Multi-domain handler
│   ├── should_correct.py
│   └── ignore_answer.py
├── utils/
│   ├── preprocess.py    # Tách passage, map choices
│   ├── bm25.py          # BM25 retrieval
│   └── postprocess.py   # Parse output, fallback
├── prompts/             # Prompt templates
└── run.py               # Entry-point: Đọc public_test.csv hoặc private_test.csv tại /data, ghi pred.csv vào /output, multiprocessing
```

---

*Pipeline được thiết kế tối ưu cho môi trường CPU, dữ liệu tiếng Việt, 2000 câu trắc nghiệm đa domain.*
