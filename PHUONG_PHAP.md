# Báo cáo phương pháp — VN Hackathon MCQ Solver

> Tài liệu thuyết minh phương pháp giải câu hỏi trắc nghiệm tiếng Việt, phục vụ báo cáo và thuyết trình.

---

## 1. Tổng quan bài toán

### 1.1. Mục tiêu

Xây dựng hệ thống tự động trả lời **câu hỏi trắc nghiệm nhiều lựa chọn (MCQ)** bằng tiếng Việt, nhận đầu vào dạng JSON và xuất ra nhãn đáp án (A, B, C, D, …).

### 1.2. Đặc thù dữ liệu

Bộ câu hỏi không đồng nhất — gồm nhiều **dạng bài khác nhau**, mỗi dạng cần chiến lược xử lý riêng:

| Domain | Mô tả | Ví dụ |
|--------|-------|-------|
| **RAG** | Có đoạn văn/passage đi kèm, đáp án nằm trong tài liệu | "Theo nội dung được cung cấp, …" |
| **Science (Math)** | Toán, lý, hóa, sinh — cần tính toán hoặc áp dụng công thức | "Tính vận tốc khi rơi tự do từ 20m" |
| **Multi-domain** | Kiến thức tổng hợp: lịch sử, địa lý, luật, kinh tế lý thuyết | "Thủ đô Việt Nam thuộc châu lục nào?" |
| **Should-correct** | Chọn phát biểu/định nghĩa ĐÚNG hoặc SAI | "Phát biểu nào KHÔNG chính xác?" |
| **Ignore-answer** | Câu hỏi gây hại / vi phạm pháp luật, hoặc không có đáp án đúng | "Cách lách lệnh đình chỉ hoạt động" |

### 1.3. Nguyên tắc thiết kế

1. **Phân loại trước, giải sau** — route câu hỏi vào đúng domain rồi mới áp dụng prompt/chiến lược phù hợp.
2. **Heuristic + LLM song song** — heuristic nhanh, ổn định; LLM mạnh hơn cho phần lớn câu hỏi; luôn có fallback.
3. **Chạy local** — dùng LLM cục bộ (`Qwen/Qwen3.5-4B`) qua `transformers`, không phụ thuộc API bên ngoài.
4. **Tối ưu latency** — routing bằng heuristic (không gọi LLM route); `ignore_answer` bỏ qua LLM hoàn toàn.

---

## 2. Kiến trúc hệ thống

```
Input JSON (qid, question, choices)
        │
        ▼
┌───────────────────┐
│   Preprocess      │  Tách passage / câu hỏi, map nhãn A/B/C/…
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│   Router          │  Heuristic (mặc định) hoặc LLM (tùy chọn)
│   → 1 domain      │  rag | science | multi_domain | should_correct | ignore_answer
└─────────┬─────────┘
          │
          ├─ ignore_answer ──► Heuristic trực tiếp (pattern "từ chối")
          │
          ├─ rag ────────────► BM25 retrieve → LLM + prompt RAG
          │
          ├─ science ────────► LLM + prompt khoa học
          │                    (fallback: heuristic math solvers)
          │
          ├─ should_correct ─► LLM + prompt đúng/sai
          │
          └─ multi_domain ───► LLM + prompt tổng hợp
          │
          ▼
┌───────────────────┐
│   Postprocess     │  Parse JSON {"answer":"A"}, validate nhãn
└─────────┬─────────┘
          │
          ▼
Output CSV (qid, answer)
```

### 2.1. Cấu trúc mã nguồn

```
vn-hackaithon/
├── run.py                 # Entry point CLI
├── pipeline.py            # Orchestration: route → answer → trace
├── router.py              # Heuristic domain classifier
├── prompts.py             # System/user prompts + few-shot router
├── few-shot.json          # 18 ví dụ phân loại domain
├── domains/
│   ├── rag.py             # Heuristic RAG (BM25 + lexical overlap)
│   ├── math.py            # Heuristic math (GDP, co giãn, decay, …)
│   ├── multi_domain.py    # Heuristic lexical matching
│   ├── should_correct.py  # Heuristic đúng/sai keyword
│   └── ignore_answer.py   # Heuristic tìm đáp án "từ chối"
└── utils/
    ├── preprocess.py      # Tách passage, chuẩn hóa choices
    ├── bm25.py            # BM25-like retrieval (pure Python)
    ├── llm.py             # Local LLM client (transformers)
    └── postprocess.py     # Parse JSON answer / route output
```

---

## 3. Tiền xử lý (Preprocess)

**File:** `utils/preprocess.py`

Mỗi câu hỏi thô trong JSON thường nhúng cả passage lẫn câu hỏi trong một chuỗi `question`. Bước preprocess:

1. **Tách passage** bằng regex theo các pattern phổ biến:
   - `Đoạn thông tin: … Câu hỏi:`
   - `[1] Tiêu đề: … Câu hỏi:`
   - `Nội dung: … Câu hỏi:`
   - Fallback: nếu chuỗi dài > 600 ký tự và có `Câu hỏi:`

2. **Map choices** thành dictionary `{A: "…", B: "…", …}` — hỗ trợ số lượng đáp án tùy ý (A–F, …).

3. **Chuẩn hóa** khoảng trắng, giữ nguyên nội dung gốc.

**Output:** `{qid, passage, question, choices, num_choices}`

---

## 4. Bộ định tuyến (Router)

Router quyết định **domain** của câu hỏi. Đây là bước quan trọng nhất — route sai domain thì LLM nhận prompt không phù hợp.

### 4.1. Chiến lược routing

| Chế độ | Khi nào | Chi phí |
|--------|---------|---------|
| **Heuristic** (mặc định) | `LLM_USE_LLM_ROUTE=0` | ~0 ms/câu |
| **LLM router** (tùy chọn) | `LLM_USE_LLM_ROUTE=1` | +1 lần gọi LLM/câu |

Mặc định dùng **heuristic-only** vì:
- Nhanh hơn ~3 giây/câu (không tốn thêm 1 inference)
- Few-shot + rule đủ ổn cho phân loại domain
- LLM route chỉ bật khi cần thử nghiệm

### 4.2. Thứ tự ưu tiên kiểm tra (heuristic)

```
1. ignore_answer   ← kiểm tra choices có đáp án "từ chối" + câu hỏi harmful
2. rag             ← có passage dài (> 300 ký tự)
3. should_correct  ← keyword đúng/sai, định nghĩa
4. science         ← công thức, số liệu, đơn vị đo (Hz, mol, ml…)
5. multi_domain    ← mặc định
```

### 4.3. Quy tắc phân biệt quan trọng

**Ignore-answer** — hai dạng:
- *Vi phạm pháp luật:* câu hỏi harmful + có đáp án "Tôi không thể cung cấp…" → chọn đáp án từ chối.
- *Không có đáp án đúng:* tất cả choices sai về kiến thức (ví dụ: thủ đô Úc không có Canberra trong choices).

Tránh false positive: câu hỏi về Hồ Chí Minh/lịch sử có đáp án "Tôi không thể trả lời" (distractor) **không** bị route ignore nếu không có intent harmful.

**Science vs Multi-domain:**
- Có **số liệu + yêu cầu tính** → science (ví dụ: "Tính co giãn biết P=10→12, Q=100→80")
- Chỉ hỏi **định nghĩa/lý thuyết** → multi_domain (ví dụ: "Co giãn là khái niệm dùng để…")
- Có keyword **pháp luật/quy định** (`nghị quyết`, `quyết định số`, `sắp xếp`…) → multi_domain dù có số

**Science vs Should-correct:**
- "Nhận định nào về lạm phát là đúng?" → should_correct
- "Tính GDP biết danh nghĩa = X, thực tế = Y" → science

**RAG ưu tiên cao:** nếu có passage → rag, bất kể câu hỏi có từ "SAI" hay có số.

### 4.4. LLM Router (tùy chọn)

Khi bật `LLM_USE_LLM_ROUTE=1`:
- System prompt chứa **18 few-shot examples** từ `few-shot.json`
- User prompt gồm passage, câu hỏi, **toàn bộ choices** (quan trọng cho ignore_answer)
- Output: `{"domain": "rag", "confidence": 0.95}`
- Fallback về heuristic nếu parse lỗi hoặc confidence < 0.5

---

## 5. Xử lý theo domain

### 5.1. RAG — Retrieval-Augmented Generation

**Luồng LLM:**
1. BM25 retrieve top-5 câu liên quan từ passage (`utils/bm25.py`)
2. Gửi context đã rút gọn + câu hỏi + choices cho LLM
3. Prompt yêu cầu **chỉ dùng thông tin trong context**

**Heuristic fallback:** BM25 + lexical overlap giữa context và từng đáp án.

### 5.2. Science (Math)

**Luồng LLM:** prompt chuyên gia toán/khoa học, tự tính nội bộ, output JSON answer.

**Heuristic fallback** (`domains/math.py`) — các solver chuyên biệt:
- Phương trình suy giảm tuyến tính `dB/dt = -kB`
- Hệ số co giãn trung điểm (kinh tế)
- GDP deflator / lạm phát
- Fallback cuối: chọn đáp án số gần nhất với kết quả tính được

### 5.3. Multi-domain

**Luồng LLM:** prompt tổng hợp, chiến lược loại trừ, kiểm tra từng vế đáp án.

**Heuristic fallback:** lexical overlap giữa câu hỏi và choices.

### 5.4. Should-correct

**Luồng LLM:** prompt kiểm tra đúng/sai — hỏi "sai/ngoại trừ" thì chọn đáp án sai; hỏi "đúng" thì chọn đáp án đúng.

**Heuristic fallback:** keyword matching (`sai`, `không đúng`, `đúng`, `chính xác`) trên question và choices.

**Xử lý đặc biệt:** đáp án "Tất cả các phương án trên" — chỉ chọn khi mọi đáp án còn lại đều đúng.

### 5.5. Ignore-answer — **Không gọi LLM**

Domain này được xử lý **100% heuristic** vì:
- Pattern rõ ràng, deterministic
- LLM đôi khi tự "từ chối" nhưng chọn sai nhãn
- Tiết kiệm ~3 giây/câu

**Logic** (`domains/ignore_answer.py`):
1. Quét 16 pattern "từ chối" trong choices (`không thể cung cấp`, `vi phạm pháp luật`, `tôi không thể`, …)
2. Nếu tìm thấy → trả về nhãn đó
3. Nếu không → lexical best choice (trường hợp không có đáp án đúng)

---

## 6. Mô hình ngôn ngữ (LLM)

### 6.1. Cấu hình

| Biến môi trường | Giá trị mặc định | Ý nghĩa |
|-----------------|-----------------|---------|
| `HF_MODEL_ID` | `Qwen/Qwen3.5-4B` | Model HuggingFace |
| `HF_LOCAL_DIR` | `model/` | Cache local |
| `LLM_MAX_NEW_TOKENS` | 32 | Trần token sinh |
| `LLM_ANSWER_MAX_TOKENS` | 32 | Token cho bước answer |
| `LLM_USE_LLM_ROUTE` | 0 | 0=heuristic route, 1=LLM route |

### 6.2. Client (`utils/llm.py`)

- Load model qua `transformers` (`AutoModelForCausalLM`)
- Hỗ trợ CUDA / MPS (Apple Silicon) / CPU
- Chat template với `enable_thinking=False` (tắt chain-of-thought, giảm latency)
- Thread-safe (`threading.Lock`) cho generate
- Greedy decoding (`do_sample=False`)

### 6.3. Prompt engineering

**Nguyên tắc output:** chỉ trả `{"answer":"A"}` — **không** kèm `reason`.

Lý do:
- `LLM_ANSWER_MAX_TOKENS=32` — nếu model sinh reason dài, JSON bị cắt (455/463 câu trong run cũ)
- Answer field xuất hiện đầu JSON → parse ổn định ngay cả khi truncated
- Ít token sinh → nhanh hơn (~2.9s/câu warm)

**Prompt theo domain** — ngắn gọn, tập trung nhiệm vụ:

| Domain | System prompt (tóm tắt) |
|--------|-------------------------|
| rag | Chỉ dùng context, không kiến thức ngoài |
| science | Tự tính toán, đối chiếu đáp án |
| multi_domain | Loại trừ, kiểm tra từng vế |
| should_correct | Phân biệt hỏi ĐÚNG vs hỏi SAI |

---

## 7. Hậu xử lý (Postprocess)

**File:** `utils/postprocess.py`

### 7.1. Parse answer

Thứ tự fallback:
1. Regex `"answer"\s*:\s*"([A-Z])"` — hoạt động cả với JSON bị cắt
2. Parse JSON object đầy đủ
3. Tìm nhãn A/B/C/… trong raw text
4. Fallback: nhãn đầu tiên

### 7.2. Parse route

- Parse `{"domain": "…", "confidence": 0.95}`
- Map alias `math` → `science`
- Validate domain hợp lệ; confidence < 0.5 → fallback passage→rag hoặc multi_domain

---

## 8. Chiến lược Fallback

Hệ thống có **3 lớp fallback** đảm bảo luôn trả về đáp án hợp lệ:

```
LLM answer thành công + parse OK
        │ (fail)
        ▼
Heuristic domain solver (lexical / math / BM25)
        │ (fail route)
        ▼
Default: multi_domain + nhãn A
```

| Tình huống | Hành vi |
|------------|---------|
| Không có model / mode=heuristic | Toàn bộ heuristic |
| LLM route lỗi | Heuristic router |
| LLM answer lỗi / parse fail | Heuristic solver của domain |
| Domain = ignore_answer | Heuristic trực tiếp (bỏ qua LLM) |
| Confidence route < 0.5 | Override: passage→rag, còn lại→multi_domain |

---

## 9. Few-shot cho Router

**File:** `few-shot.json` — 18 ví dụ có annotation domain + giải thích.

Bao phủ các edge case quan trọng:
- RAG có passage dù câu hỏi chứa "SAI"
- Science (tính toán) vs Multi-domain (lý thuyết) cùng chủ đề kinh tế
- Should-correct vs Science khi choices là công thức LaTeX
- Ignore-answer: vi phạm pháp luật vs không có đáp án đúng
- GDP/co giãn/lạm phát — phân biệt tính vs hỏi định nghĩa

Few-shot được inject vào `ROUTER_SYSTEM_PROMPT` khi dùng LLM router.

---

## 10. Kết quả thực nghiệm

### 10.1. Độ chính xác (public test, 463 câu)

| Domain | Đúng / Tổng | Tỷ lệ |
|--------|-------------|-------|
| ignore_answer | 10 / 10 | 100% |
| rag | 77 / 101 | 76.2% |
| math (science) | 66 / 120 | 55.0% |
| should_correct | 31 / 50 | 62.0% |
| multi_domain | 122 / 182 | 67.0% |
| **Tổng** | **303 / 463** | **65.4%** |

### 10.2. Cải thiện qua các vòng

| Vòng | Thay đổi chính | Kết quả |
|------|----------------|---------|
| Baseline | Heuristic-only | Thấp |
| + LLM answer | Qwen3.5-4B local | 303/463 |
| + Router cải thiện | ignore_answer heuristic, phân biệt science/theory | ignore 10/10 |
| + Prompt answer-only | Bỏ reason, JSON gọn | Parse ổn định, nhanh hơn |

### 10.3. Hiệu năng (Mac, MPS, Qwen3.5-4B)

| Metric | Giá trị |
|--------|---------|
| Load model (1 lần) | ~22 giây |
| 1 câu LLM (warm) | ~2.9 giây |
| 1 câu ignore_answer | ~0 giây |
| Ước tính 463 câu | ~22 phút |

---

## 11. Luồng chạy và Debug

### 11.1. Lệnh chạy

```bash
# LLM mode (khuyến nghị)
python run.py \
  --input data/public-test_1780368312.json \
  --output output/pred.csv \
  --mode llm \
  --workers 1 \
  --trace-output output/llm_trace.jsonl

# Heuristic-only (nhanh, không cần GPU)
python run.py --mode heuristic
```

### 11.2. Chế độ debug

| Biến | Tác dụng |
|------|----------|
| `TRACE_LLM=1` | In raw LLM output ra terminal |
| `TRACE_QID=test_0001` | Chỉ trace 1 câu |
| `DEBUG_LLM=1` | In lỗi fallback |
| `HF_HUB_OFFLINE=1` | Chỉ dùng model cache local |

### 11.3. Trace JSONL

Mỗi dòng gồm: `qid`, `domain`, `answer`, `route_fallback`, `answer_fallback`, `raw_route`, `raw_answer` — phục vụ phân tích lỗi theo domain.

---

## 12. Hạn chế và hướng phát triển

### 12.1. Hạn chế hiện tại

1. **Router heuristic** vẫn nhầm ~20% so với gold domain label — đặc biệt should_correct vs multi_domain.
2. **Math/science** accuracy thấp nhất (55%) — model 4B khó tính toán chính xác; heuristic solvers chỉ cover vài pattern.
3. **Latency** ~3s/câu — 463 câu mất ~22 phút; chưa batch inference.
4. **Single model** — không routing model theo độ khó (4B cho tất cả domain).

### 12.2. Hướng cải thiện

| Hướng | Mô tả | Kỳ vọng |
|-------|-------|---------|
| LLM route | Bật `LLM_USE_LLM_ROUTE=1` cho câu confidence thấp | Giảm misroute |
| Model lớn hơn | Qwen3.5-7B/14B hoặc API cloud | Tăng math/multi_domain |
| Symbolic math | SymPy/Wolfram cho science domain | Tăng math accuracy |
| RAG cải thiện | Chunking tốt hơn, cross-encoder rerank | Tăng RAG accuracy |
| Batch inference | Gom nhiều câu 1 lần generate | Giảm latency |
| Ensemble | Heuristic + LLM voting | Tăng robustness |

---

## 13. Tóm tắt phương pháp (elevator pitch)

> Hệ thống giải trắc nghiệm tiếng Việt theo kiến trúc **"route-then-solve"**: mỗi câu hỏi được phân loại heuristic vào 1 trong 5 domain (RAG, Science, Multi-domain, Should-correct, Ignore-answer), sau đó xử lý bằng prompt chuyên biệt của LLM local Qwen3.5-4B. Domain đặc biệt (ignore-answer) và mọi trường hợp lỗi đều có fallback heuristic, đảm bảo hệ thống luôn trả lề đáp án hợp lệ. Thiết kế ưu tiên **chạy local, latency thấp, và khả năng debug** qua trace JSONL.

---

*Tài liệu được tạo từ codebase — cập nhật tháng 6/2026.*
