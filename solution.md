# Solution Overview

## 1. Mô Tả Bài Toán

Bài toán yêu cầu xây dựng hệ thống trả lời câu hỏi trắc nghiệm tiếng Việt. Mỗi mẫu đầu vào gồm mã câu hỏi `qid`, nội dung câu hỏi, các lựa chọn đáp án và trong một số trường hợp có thêm đoạn ngữ cảnh dài đi kèm. Đầu ra cần là một file `pred.csv` gồm hai cột `qid,answer`, trong đó `answer` là nhãn đáp án được chọn như `A/B/C/D`.

Thách thức chính của bài toán là dữ liệu không thuộc một dạng cố định. Câu hỏi có thể là câu đọc hiểu cần bám sát context, câu tính toán khoa học, câu kiểm tra phát biểu đúng/sai, câu kiến thức tổng hợp nhiều lĩnh vực hoặc câu có thông tin không nên trả lời trực tiếp. Vì vậy, nếu dùng một prompt hoặc một chiến lược duy nhất cho toàn bộ dữ liệu thì mô hình dễ bị nhầm domain, đọc sai yêu cầu phủ định, hoặc trả lời không ổn định.

Do đó, hướng tiếp cận của nhóm là xây dựng một pipeline dạng agent: trước tiên phân tích câu hỏi để xác định domain, sau đó chọn chiến lược giải phù hợp cho từng domain. Pipeline luôn có cơ chế fallback heuristic để đảm bảo hệ thống vẫn trả về đáp án hợp lệ ngay cả khi LLM sinh lỗi, parse lỗi hoặc thiếu thông tin.

## 2. Phương Pháp Đề Xuất

### 2.1. Tổng Quan Pipeline

Pipeline tổng quát gồm các bước:

1. **Input Processing**: đọc câu hỏi, chuẩn hóa text, tách passage/context nếu câu hỏi có đoạn thông tin dài.
2. **Domain Router**: agent phân loại câu hỏi vào một trong các domain chính.
3. **Domain-Specific Solver**: mỗi domain dùng một chiến lược giải khác nhau, gồm RAG, solver tính toán, CoT, PoT, verifier hoặc heuristic.
4. **Answer Parsing**: chuẩn hóa output của mô hình về đúng một nhãn đáp án.
5. **Fallback Layer**: nếu LLM lỗi hoặc không parse được, hệ thống chuyển sang solver heuristic để vẫn tạo output hợp lệ.

Các domain chính:

| Domain | Loại câu hỏi | Chiến lược chính |
| --- | --- | --- |
| `RAG` | Câu hỏi có passage/context dài | Truy xuất evidence bằng BM25 + LLM bám context |
| `Science` | Toán, vật lý, hóa học, kinh tế định lượng, logic tính toán | Specialized solver + Program-of-Thought/Python + LLM |
| `Should Correct` | Câu hỏi chọn phát biểu đúng/sai, không đúng, ngoại trừ | Nhận diện polarity + CoT + verifier |
| `Multi-domain` | Câu hỏi kiến thức tổng hợp nhiều lĩnh vực | LLM tổng hợp + domain hints + CoT có điều kiện |
| `Ignore Answer` | Câu hỏi chứa yêu cầu không nên trả lời hoặc lựa chọn từ chối | Heuristic chọn đáp án từ chối phù hợp |

### 2.2. Domain Router Agent

Router là tầng quyết định câu hỏi nên được xử lý theo chiến lược nào. Thay vì đưa mọi câu vào cùng một prompt, hệ thống dùng router để giảm độ nhiễu và chọn đúng công cụ giải.

Router dựa trên các tín hiệu:

- Câu hỏi có passage dài, marker như “Đoạn thông tin”, “Theo thông tin”, hoặc context lớn -> route sang `RAG`.
- Câu hỏi có số liệu, đơn vị đo, công thức, biểu thức toán học, từ khóa như “tính”, “bao nhiêu”, “xác định” -> route sang `Science`.
- Câu hỏi có các cụm “phát biểu nào đúng/sai”, “không đúng”, “ngoại trừ”, “nhận định nào” -> route sang `Should Correct`.
- Câu hỏi lý thuyết, chính trị, pháp luật, xã hội, công nghệ, kiến thức chung không thuộc nhóm trên -> route sang `Multi-domain`.
- Câu hỏi có dấu hiệu yêu cầu hành vi không phù hợp và trong lựa chọn có phương án từ chối -> route sang `Ignore Answer`.

Router có thể dùng heuristic nhanh hoặc LLM-route tùy cấu hình. Trong cấu hình chạy chính, nhóm ưu tiên heuristic routing để tăng tốc độ, giảm chi phí suy luận và ổn định hơn trên local model.

### 2.3. Xử Lý Domain RAG

Domain `RAG` dành cho các câu hỏi cần dựa trên đoạn context dài. Mục tiêu là tránh để mô hình trả lời bằng kiến thức ngoài, đồng thời giúp mô hình tập trung vào phần evidence liên quan.

Chiến lược xử lý:

- Nếu passage ngắn, đưa toàn bộ context vào prompt.
- Nếu passage dài, dùng BM25 để truy xuất các đoạn liên quan nhất.
- Query truy xuất được mở rộng bằng cả câu hỏi và nội dung các lựa chọn để tăng khả năng tìm đúng evidence.
- LLM được yêu cầu phân tích từng lựa chọn dựa trên context, sau đó chọn đáp án cuối cùng.
- Sau khi có đáp án, verifier kiểm tra lại xem lựa chọn có thật sự được context hỗ trợ hay không.

Cách này giúp giảm lỗi thường gặp trong câu đọc hiểu: chọn đáp án theo kiến thức nền, bỏ sót phủ định, hoặc nhầm giữa các lựa chọn chỉ khác nhau một chi tiết nhỏ.

### 2.4. Xử Lý Domain Science

Domain `Science` bao gồm các câu hỏi toán học, vật lý, hóa học, kinh tế định lượng, xác suất, ma trận hoặc các bài có công thức/số liệu.

Chiến lược xử lý gồm nhiều tầng:

- **Specialized Solvers**: trước tiên kiểm tra các dạng bài quen thuộc bằng rule/công thức cứng, ví dụ co giãn cầu, GDP deflator, lãi suất, điện trở song song, ma trận, phản ứng trung hòa, Cournot, xác suất/tổ hợp.
- **Program-of-Thought**: với câu hỏi định lượng phức tạp, LLM sinh một đoạn Python ngắn để tính toán hoặc kiểm tra từng lựa chọn. Code được chạy trong sandbox giới hạn module và timeout.
- **LLM Reasoning**: nếu không khớp solver hoặc PoT không đưa ra đáp án hợp lệ, câu hỏi được đưa vào prompt chuyên biệt cho science.
- **Fallback Heuristic**: khi các bước trên thất bại, hệ thống chọn đáp án dựa trên matching lexical/numeric gần nhất.

Cách tiếp cận này giúp các câu tính toán bớt phụ thuộc vào khả năng tính nhẩm của LLM. Với những bài có số liệu rõ ràng, solver hoặc PoT thường cho kết quả ổn định hơn so với hỏi trực tiếp mô hình.

### 2.5. Xử Lý Domain Should Correct

Domain `Should Correct` tập trung vào các câu hỏi yêu cầu chọn phát biểu đúng, sai, không đúng, ngoại trừ hoặc nhận định phù hợp.

Vấn đề chính của domain này là mô hình dễ đọc nhầm hướng câu hỏi. Ví dụ “phát biểu nào không đúng” khác hoàn toàn với “phát biểu nào đúng”. Vì vậy pipeline xử lý theo hướng:

- Xác định polarity của câu hỏi: đang hỏi đáp án đúng, sai, không chính xác hay ngoại trừ.
- Dùng CoT ngắn để so sánh từng lựa chọn nếu câu hỏi phức tạp.
- Dùng verifier để kiểm tra lại đáp án ban đầu, đặc biệt với câu có phủ định hoặc nhiều lựa chọn gần giống nhau.
- Nếu LLM không chắc chắn, fallback về heuristic theo polarity và lexical matching.

Nhờ đó, hệ thống giảm lỗi chọn ngược đáp án trong các câu có từ khóa phủ định như “không”, “sai”, “ngoại trừ”.

### 2.6. Xử Lý Domain Multi-domain

Domain `Multi-domain` là nhóm câu hỏi kiến thức tổng hợp, không có passage dài và không phải bài tính toán rõ ràng. Nhóm này có thể bao gồm pháp luật, chính trị, kinh tế, công nghệ, sinh học, địa lý hoặc kiến thức xã hội.

Chiến lược xử lý:

- Dùng prompt tổng hợp để yêu cầu mô hình chọn một đáp án duy nhất.
- Bổ sung domain hints cho một số nhóm câu hỏi dễ nhầm, ví dụ pháp luật, hành chính công, môi trường, đất đai, tư tưởng chính trị.
- Với câu hỏi có nhiều lựa chọn hoặc wording phức tạp, bật CoT có điều kiện để mô hình phân tích ngắn từng lựa chọn trước khi kết luận.
- Với một số pattern đặc biệt, dùng specialized heuristic để early-exit trước khi gọi LLM.

Mục tiêu của domain này là giữ tính linh hoạt cho những câu không thuộc nhóm chuyên biệt, nhưng vẫn bổ sung gợi ý miền để giảm hallucination và giảm nhầm lẫn giữa các khái niệm gần nhau.

### 2.7. Xử Lý Domain Ignore Answer

Domain `Ignore Answer` dành cho các câu hỏi có dấu hiệu không nên trả lời trực tiếp hoặc có lựa chọn kiểu từ chối, ví dụ “tôi không thể cung cấp”, “không hỗ trợ”, “vi phạm pháp luật”.

Chiến lược xử lý:

- Router phát hiện các pattern về hành vi không phù hợp hoặc yêu cầu có tính harmful.
- Kiểm tra trong các lựa chọn có phương án từ chối phù hợp hay không.
- Nếu có, hệ thống chọn trực tiếp bằng heuristic, không gọi LLM.

Cách này giúp pipeline ổn định hơn với các câu bẫy alignment và tránh để LLM tự sinh lý do ngoài format đáp án.

### 2.8. Verifier Và Fallback

Sau khi solver hoặc LLM đưa ra đáp án, pipeline có thêm hai lớp bảo vệ:

- **Answer Parser**: trích xuất đáp án về đúng format một chữ cái, kể cả khi LLM trả lời dạng JSON, câu văn tiếng Việt hoặc có phần suy luận.
- **Verifier**: kiểm tra lại đáp án với các domain dễ sai như RAG, Should Correct và một số câu Multi-domain.

Nếu LLM lỗi, timeout, sinh sai format hoặc không parse được, hệ thống luôn fallback về heuristic solver. Điều này đảm bảo output cuối cùng luôn có đủ `qid,answer`, phù hợp yêu cầu submission.

## 3. Điểm Mạnh Của Phương Pháp

- **Chia nhỏ bài toán theo domain**: mỗi loại câu hỏi được xử lý bằng chiến lược phù hợp thay vì dùng một prompt chung.
- **Kết hợp LLM và heuristic**: tận dụng khả năng ngôn ngữ của LLM nhưng vẫn có rule-based solver cho các dạng bài dễ chuẩn hóa.
- **Tối ưu cho local model**: router heuristic và token budget nhỏ giúp pipeline chạy được với GGUF local.
- **Ổn định đầu ra**: có parser, verifier và fallback để luôn tạo file `pred.csv` đúng format.
- **Phù hợp yêu cầu Docker/BTC**: container tự đọc `/data/public_test.csv` hoặc `/data/private_test.csv`, ghi `/output/pred.csv`.

