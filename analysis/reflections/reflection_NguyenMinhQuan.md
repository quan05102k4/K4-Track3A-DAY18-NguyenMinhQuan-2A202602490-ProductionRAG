# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyễn Minh Quân  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Ngưỡng 0.85 trong `config.py` cắt câu theo độ tương đồng cosine của `all-MiniLM-L6-v2`. Cần chú ý: chỉ cắt câu bằng regex `[.!?]` và `\n\n`, nên bảng markdown không có dấu câu bị gom thành một đoạn dài. |
| Hierarchical (parent-child) chunking | M1 | `chunk_hierarchical()`, `_pack()` | Parent 2048 ký tự, child 256 ký tự, child giữ `parent_id`. Phần khó nhất là `_pack()`: khi một đoạn không có dấu câu vượt 256 ký tự, hàm cắt theo ký tự và làm cụt từ. Lỗi này xuất hiện trong câu hỏi về mua sắm (xem failure analysis, #2). |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | Tách theo heading `#`, `##`, `###` và lưu tên mục vào `metadata["section"]`. Phù hợp với tài liệu chính sách vì mỗi mục là một quy định độc lập. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()`, `BM25Search`, `DenseSearch` | RRF dùng thứ hạng nên không cần chuẩn hóa điểm BM25 và cosine. Lỗi `_` trong tách từ tiếng Việt (`nghỉ_phép`) được xử lý bằng `replace("_", " ")`. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Cross-encoder đưa đúng đoạn có đáp án lên đầu trong các test. Độ trễ trên CPU là khoảng 210 ms cho 3 tài liệu và 820 ms cho 20 tài liệu, vượt mục tiêu 150 ms, nên cần GPU hoặc giảm số ứng viên. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Faithfulness giảm từ 0.80 xuống 0.68 khi chuyển sang Production, Context Precision giữ ở 0.925 bằng baseline, còn Context Recall giảm từ 0.90 xuống 0.79. Điều này cho thấy Production chưa cải thiện so với baseline: câu trả lời ít bám tài liệu hơn do lỗi xung đột phiên bản và lỗi cắt chunk. |
| Contextual embeddings | M5 | `_enrich_single_call()`, `contextual_prepend()` | Một lệnh gọi LLM cho mỗi chunk sinh ra tóm tắt, câu hỏi giả định, câu bối cảnh và metadata. Chi phí là 125 lệnh gọi cho 125 chunk. Hàm có fallback không cần LLM để pipeline không dừng khi thiếu API. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải (exact error message):**
  - `Exception raised in Job[53]: BadRequestError(Error code: 400 - {'error': {'message': "'n' : number must be at most 1"`. Nguyên nhân: RAGAS `answer_relevancy` mặc định yêu cầu `n=3` completions, còn Groq chỉ nhận `n=1`. Cách xử lý: đặt `answer_relevancy.strictness = 1` và sau đó quay lại OpenAI.
  - `RateLimitError 429 ... quota ... limit: 20` (Gemini gói miễn phí). Nguyên nhân: gói miễn phí chỉ có 20 request mỗi ngày cho mỗi project, nên toàn bộ làm giàu 125 chunk và RAGAS không chạy được. Cách xử lý: chuyển sang Groq, sau đó sang OpenAI.
  - `rate_limit_exceeded ... output tokens per minute (OTPM): Limit 1000` (Groq). Nguyên nhân: mỗi lệnh gọi đặt `max_tokens=2048` nên bị tính trước quá nhiều token. Cách xử lý: giảm `max_tokens` xuống 600 và thử lại khi gặp 429.
  - `TimeoutError` và 67 job RAGAS lỗi 429 trong lần chạy Production đầu tiên, làm điểm Production thành 0.0. Cách xử lý: giảm số luồng RAGAS xuống 2, sau đó về 8 khi dùng OpenAI.
- **Nguyên nhân gốc rễ & Cách debug:**
  - Lỗi quota làm tôi mất khá nhiều thời gian vì mỗi lần đổi key lại gặp một giới hạn khác. Tôi đã thử từng key riêng lẻ bằng một lệnh gọi nhỏ để tách lỗi key khỏi lỗi quota.
  - Bài học quan trọng nhất: hàm `evaluate_ragas()` bắt mọi exception và trả về 0, nên một lần chạy lỗi vẫn in ra bảng điểm trông bình thường. Tôi chỉ phát hiện điều này khi đọc `reports/ragas_report.json` và thấy toàn bộ giá trị là 0.0. Từ đó, tôi kiểm tra số lượng `Exception raised in Job` trong log trước khi tin vào điểm số.
  - Lỗi cắt từ trong M1 được phát hiện khi phân tích failure: câu trả lời "Tổn" cho thấy ngữ cảnh đã bị cắt, và tôi phải lấy chunk thực tế để xác nhận.
- **Kiến thức còn thiếu & Cách khắc phục:**
  - Cách RAGAS dùng LLM làm judge và embedding để tính `answer_relevancy`, và tại sao nó nhạy với `n` và quota.
  - Hiểu rõ giới hạn của các nhà cung cấp API miễn phí (RPM, TPM, RPD, OTPM) để thiết kế retry và giới hạn tốc độ. Tôi sẽ đọc lại tài liệu rate limit của nhà cung cấp trước khi chạy batch lớn.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

Dựa trên những kỹ thuật đã học và thực hành, lập kế hoạch cụ thể áp dụng vào project của bạn:

### Project: Trợ lý tra cứu chính sách nội bộ

#### 1. Hiện trạng
- **Pipeline hiện tại:** Naive RAG, cắt theo đoạn văn, chỉ dùng dense search, không có reranker và không có đánh giá tự động.
- **Vấn đề / Bottlenecks đang gặp:** Câu trả lời sai khi tài liệu có nhiều phiên bản (cũ và mới). Các bảng phê duyệt bị cắt làm mất thông tin. Chưa có số liệu để so sánh các thay đổi.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Structure-aware theo heading, cộng với child chunk không cắt giữa từ và giữ nguyên dòng bảng. Lý do: tài liệu chính sách có cấu trúc rõ ràng và mỗi mục là một quy định độc lập.
2. **Search retrieval:** Hybrid BM25 + Dense + RRF. Lý do: BM25 bắt đúng số hiệu và số ngày, Dense bắt được cách diễn đạt khác nhau.
3. **Reranking:** Cross-encoder `bge-reranker-v2-m3` trên top-20 để lấy top-3. Với độ trễ cao trên CPU, sẽ dùng GPU hoặc giảm số ứng viên.
4. **Evaluation:** RAGAS 4 metrics trên bộ 20 câu hỏi, cộng thêm kiểm tra số job lỗi trước khi tin điểm số. Mỗi lần thay đổi phải chạy lại baseline để so sánh.
5. **Enrichment:** Metadata `version`/`status` và câu bối cảnh (contextual prepend), để LLM biết đoạn nào đã bị thay thế.

#### 3. Timeline triển khai
- **Tuần 1:** Sửa chunking để không cắt giữa bảng và từ, thêm metadata phiên bản, thêm test cho các trường hợp này.
- **Tuần 2:** Thêm quy tắc chọn phiên bản hiện hành trong prompt, tăng top-k cho câu hỏi nhiều bước, chạy lại đánh giá và so sánh với baseline.
