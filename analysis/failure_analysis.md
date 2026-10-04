# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyễn Minh Quân  
**Khóa:** K4 - Track 3A  

---

## RAGAS Scores

Chạy `python src/pipeline.py` với LLM `gpt-4o-mini`, embedding mặc định của OpenAI, 20 câu trong `test_set.json`. Baseline lấy từ `python main.py`, Production lấy từ lần chạy `src/pipeline.py` gần nhất (ghi vào `reports/ragas_report.json`).

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8017 | 0.6833 | -0.1184 |
| Answer Relevancy | 0.6780 | 0.6834 | +0.0054 |
| Context Precision | 0.9250 | 0.9250 | 0.0000 |
| Context Recall | 0.9000 | 0.7917 | -0.1083 |

Nhận xét: Production không cải thiện so với baseline ở lần chạy này. Context Precision bằng baseline, Answer Relevancy tăng nhẹ, còn Faithfulness và Context Recall giảm. Năm câu lỗi nặng nhất cho thấy ba nguyên nhân chính:
1. Câu trả lời bị "Không tìm thấy." hoặc cụt dù ngữ cảnh có đáp án, do xung đột phiên bản (cũ và mới) hoặc do chunk bị cắt sai.
2. Câu hỏi cần ghép thông tin từ nhiều tài liệu nhưng top-3 không lấy đủ.
3. Đánh giá RAGAS có thể phạt câu trả lời đúng nếu phép tính (như 85% × 20 triệu) không xuất hiện nguyên văn trong ngữ cảnh.

## Bottom-5 Failures

Thứ tự theo điểm trung bình tăng dần trong `reports/ragas_report.json`.

### #1
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Trên 50.000.000 VNĐ cần Tổng Giám đốc (CEO).
- **Got:** "Cần phê duyệt của Tổn." (câu bị cụt)
- **Worst metric:** faithfulness (điểm trung bình 0.208)
- **Error Tree:** Output sai → Context đúng? **Không**, chunk chứa đáp án bị cắt → Query OK? **Có** → lỗi ở bước cắt chunk (M1).
- **Root cause:** Bảng phê duyệt trong `data/mua_sam.md` là một đoạn không có dòng trống. Hàm `chunk_hierarchical` gom đoạn này vào parent, rồi `_pack()` phải cắt child 256 ký tự. Vì bảng không có câu kết thúc bằng `.!?`, hàm không tách được theo câu nên cắt cứng theo ký tự. Chunk con cuối kết thúc bằng "Tổn", mất dòng "Trên 50.000.000 VNĐ → Tổng Giám đốc (CEO)". Đây là lỗi của mình trong `_pack()` ở [src/m1_chunking.py](src/m1_chunking.py).
- **Trả lời 4 câu hỏi:**
  - *Câu trả lời có đúng không?* Không. Câu trả lời bị cụt và thiếu chức danh.
  - *Đoạn trích có chứa đáp án không?* Không. Dòng đáp án bị cắt giữa chừng.
  - *Câu hỏi có cần viết lại không?* Không.
  - *Sửa ở module nào?* M1 (`_pack`, `chunk_hierarchical`).
- **Suggested fix:** Khi một đoạn vượt `child_size` và không có câu, tách theo dòng (`\n`) trước, rồi mới cắt theo từ (khoảng trắng), không cắt giữa từ. Với bảng markdown, giữ nguyên từng dòng và lặp lại dòng header trong mỗi chunk con.

### #2
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** 120 ngày (chính sách v2.0). Chính sách cũ 90 ngày đã bị thay thế.
- **Got:** "Không tìm thấy."
- **Worst metric:** faithfulness (điểm trung bình 0.333)
- **Error Tree:** Output sai → Context đúng? **Có** (chunk 1 ghi "mỗi 120 ngày") → Query OK? **Có** → lỗi ở bước sinh câu trả lời.
- **Root cause:** Top-3 sau rerank gồm cả chunk v2.0 (120 ngày) và chunk v1.0 (90 ngày, có ghi "đã được thay thế"). Model thấy mâu thuẫn giữa các phiên bản và từ chối trả lời, thay vì ưu tiên phiên bản hiện hành. Prompt hiện tại chỉ nói "Trả lời CHỈ dựa trên context", không có quy tắc chọn phiên bản mới nhất.
- **Trả lời 4 câu hỏi:**
  - *Câu trả lời có đúng không?* Không.
  - *Đoạn trích có chứa đáp án không?* Có.
  - *Câu hỏi có cần viết lại không?* Không. Câu hỏi đã rõ, nhưng không nêu phiên bản nên hệ thống phải tự chọn.
  - *Sửa ở module nào?* M5 (thêm metadata `version`/`status` khi làm giàu) và prompt sinh câu trả lời trong `pipeline.py` (ưu tiên phiên bản hiện hành, nói rõ nếu có mâu thuẫn).
- **Suggested fix:** Thêm quy tắc vào system prompt: "Nếu context có nhiều phiên bản, chỉ dùng phiên bản không bị đánh dấu là đã thay thế, và nêu rõ phiên bản đang dùng." Gắn metadata `status: superseded` cho đoạn cũ ở bước M5.

### #3
- **Question:** Mật khẩu phải có tối thiểu bao nhiêu ký tự?
- **Expected:** 12 ký tự (v2.0). Chính sách cũ 8 ký tự đã bị thay thế.
- **Got:** "Không tìm thấy."
- **Worst metric:** faithfulness (điểm trung bình 0.500)
- **Error Tree:** Output sai → Context đúng? **Có** (chunk 1: "tối thiểu 12 ký tự") → Query OK? **Có** → lỗi ở bước sinh câu trả lời.
- **Root cause:** Cùng nguyên nhân xung đột phiên bản với #2: top-3 có cả chunk v1.0 (8 ký tự).
- **Trả lời 4 câu hỏi:**
  - *Câu trả lời có đúng không?* Không.
  - *Đoạn trích có chứa đáp án không?* Có.
  - *Câu hỏi có cần viết lại không?* Không.
  - *Sửa ở module nào?* Prompt sinh câu trả lời và metadata phiên bản.
- **Suggested fix:** Cùng cách với #2.

### #4
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 15 ngày cơ bản + 3 ngày thâm niên = 18 ngày. Lương Senior (P3–P4): 20–35 triệu VNĐ/tháng.
- **Got:** "...sẽ được 18 ngày phép (15 ngày cơ bản + 3 ngày thâm niên). Về lương, không có thông tin cụ thể trong context..."
- **Worst metric:** answer_relevancy (điểm trung bình 0.583)
- **Error Tree:** Output sai → Context đúng? **Một phần**: thông tin phép có, bảng lương không có trong top-3 → Query OK? **Không**, câu hỏi ghép hai chủ đề (phép và lương) nhưng một truy vấn → lỗi ở bước truy xuất.
- **Root cause:** Câu trả lời đúng về số ngày phép, nhưng không trả lời phần lương vì `bang_luong_2024.md` không lọt vào top-3. Câu trả lời lệch trọng tâm một phần nên RAGAS chấm answer_relevancy thấp.
- **Trả lời 4 câu hỏi:**
  - *Câu trả lời có đúng không?* Đúng một phần: phần phép đúng, phần lương thiếu.
  - *Đoạn trích có chứa đáp án không?* Có một phần: phép có, lương không có.
  - *Câu hỏi có cần viết lại không?* Có thể tách thành hai câu hỏi để kiểm tra từng phần.
  - *Sửa ở module nào?* M2/M3: tăng top-k hoặc tách truy vấn đa chủ đề.
- **Suggested fix:** Với câu hỏi nhiều chủ đề, truy xuất riêng cho từng vế rồi gộp lại trước khi rerank.

### #5
- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Junior cao nhất 20.000.000 VNĐ/tháng. Lương thử việc = 85% × 20.000.000 = 17.000.000 VNĐ/tháng.
- **Got:** "Lương thử việc của nhân viên Junior mức cao nhất là 17.000.000 VNĐ."
- **Worst metric:** faithfulness (điểm trung bình 0.706)
- **Error Tree:** Output sai → Output đúng? **Có** → lỗi ở bước đánh giá, không phải ở RAG.
- **Root cause:** Đáp án đúng nhưng cần ghép hai nguồn: mức 20 triệu (`bang_luong_2024.md`) và tỷ lệ 85% (`thu_viec.md`). Đáp án 17 triệu là kết quả phép tính nên RAGAS có thể không tìm thấy nguyên văn trong ngữ cảnh và trừ điểm faithfulness. Chưa kiểm chứng: mình chưa lấy ngữ cảnh truy xuất của câu này để xác nhận.
- **Trả lời 4 câu hỏi:**
  - *Câu trả lời có đúng không?* Có.
  - *Đoạn trích có chứa đáp án không?* Chưa kiểm chứng. Cần đủ cả hai đoạn để tính được 17 triệu.
  - *Câu hỏi có cần viết lại không?* Không.
  - *Sửa ở module nào?* Không rõ. Có thể là M2/M3 (lấy đủ hai nguồn) hoặc là giới hạn của metric faithfulness với câu trả lời có phép tính.
- **Suggested fix:** Kiểm tra ngữ cảnh truy xuất cho câu này. Nếu thiếu một trong hai nguồn, tăng top-k. Nếu đủ, xem xét đây là lỗi của metric chứ không phải của pipeline.

## Case Study (cho presentation)

**Question chọn phân tích:** "Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?" (#1)

**Error Tree walkthrough:**
1. Output đúng? **Không.** Trả về "Tổn" thay vì "Tổng Giám đốc (CEO)".
2. Context đúng? **Không.** Chunk con chứa bảng nhưng bị cắt cụt ở chữ "Tổn", nên dòng đáp án biến mất.
3. Query rewrite OK? **Có.** Câu hỏi được tìm đúng tài liệu `mua_sam.md`.
4. Fix ở bước: **M1 (chunking).** Sửa `_pack()` để không cắt giữa từ và giữ nguyên dòng bảng.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Sửa `_pack()` để tách theo dòng và theo từ, rồi chạy lại M1 và kiểm tra test_m1.
- Thêm metadata `version`/`status` ở M5 và quy tắc chọn phiên bản hiện hành trong prompt để xử lý nhóm lỗi xung đột phiên bản (#2, #3).
- Truy xuất riêng cho từng vế của câu hỏi nhiều chủ đề (#4), và kiểm tra ngữ cảnh của câu #5.
