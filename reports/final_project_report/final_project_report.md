# Báo cáo tổng hợp cuối hệ thống Chatbot Lịch sử Việt Nam

## 1. Tóm tắt kết luận

Báo cáo này tổng hợp kết quả cuối để đóng gói project chatbot. Phiên bản hệ thống được chốt là **Cloud Primary**: truy xuất lai cục bộ, dựng ngữ cảnh từ nguồn nội bộ, gọi LLM cloud qua 9Router `chatx`, hiển thị citation và có guard/OOS.

Kết quả chính được lấy từ hai nhóm đánh giá:

- **Answer quality 500 câu**: chấm độc lập bằng rubric 10 điểm trên câu trả lời thật, nguồn truy xuất thật và citation hiển thị.
- **Retrieval benchmark 300 query**: đo chất lượng truy xuất bằng Recall@K, Precision@5, MRR@5, NDCG@5, evidence coverage và citation alignment.

Các số liệu trong báo cáo được trình bày theo runtime chốt cuối của project. Đây là benchmark nội bộ phục vụ đánh giá đồ án, không phải đánh giá độc lập bởi bên thứ ba.

## 2. Phạm vi và cách hiểu kết quả

| Thành phần | Cách sử dụng trong báo cáo |
|---|---|
| Bộ 500 câu answer quality | Là nền chính để đánh giá chất lượng câu trả lời cuối. |
| Guard/OOS/invalid | Trình bày theo kết quả runtime chốt cuối. |
| Retrieval 300 query | Sử dụng kết quả benchmark truy xuất 300 query. |
| Auto score cũ | Không dùng làm kết luận chính. |

Câu claim nên dùng khi nộp báo cáo: **Hệ thống hỗ trợ tốt các chủ đề trọng tâm đã có nguồn kiểm chứng trong tư liệu nội bộ về lịch sử Việt Nam giai đoạn 1930-1975.** Không nên tuyên bố hệ thống bao phủ toàn bộ mọi nội dung lịch sử Việt Nam 1930-1975.

## 3. Kết quả answer quality 500 câu

| Chỉ số | Kết quả | Ghi chú |
|---|---:|---|
| Số câu đánh giá | 500 | 500 câu tự nhiên/đa dạng |
| Điểm trung bình | 8.766/10 | Rubric độc lập 10 điểm |
| Median | 10.00/10 | Trung vị rất cao do nhiều câu trả lời đạt chuẩn |
| P25 / P75 | 8.82 / 10.00 | Độ phân bố điểm |
| Tỷ lệ >= 9 điểm | 73.80% | Câu trả lời rất tốt |
| Tỷ lệ >= 8 điểm | 79.20% | Mốc đạt tốt |
| Tỷ lệ >= 7 điểm | 82.80% | Mốc dùng được trở lên |
| Tỷ lệ < 5 điểm | 13.00% | Còn lỗi rõ cần ghi nhận |
| Critical fail | 65/500 = 13.00% | Lỗi nặng theo rubric |
| HTTP error | 11/500 = 2.20% | Lỗi vận hành còn lại |

Kết quả cho thấy hệ thống đạt mức khá tốt trên bộ câu hỏi tự nhiên: điểm trung bình **8.766/10**, median **10/10**, tỷ lệ câu từ 8 điểm trở lên **79.20%**.

### 3.1. Bảng theo nhóm câu hỏi

| Nhóm câu hỏi | Số câu | Điểm TB | Median | >=8 điểm | <5 điểm | Critical fail | HTTP error |
|---|---:|---:|---:|---:|---:|---:|---:|
| fact_date | 60 | 8.277 | 10.00 | 71.67% | 23.33% | 14 | 2 |
| explanation | 60 | 8.707 | 10.00 | 80.00% | 18.33% | 11 | 1 |
| timeline | 65 | 9.948 | 10.00 | 100.00% | 0.00% | 0 | 0 |
| comparison | 65 | 9.488 | 10.00 | 90.77% | 3.08% | 2 | 1 |
| misconception_correction | 55 | 7.455 | 8.35 | 58.18% | 20.00% | 11 | 1 |
| natural_paraphrase | 75 | 8.867 | 10.00 | 82.67% | 13.33% | 10 | 4 |
| memory_multiturn | 80 | 7.848 | 9.20 | 58.75% | 21.25% | 17 | 2 |
| guard_oos_invalid | 40 | 9.938 | 9.90 | 100.00% | 0.00% | 0 | 0 |

### 3.2. Thành phần điểm trung bình

| Thành phần rubric | Điểm TB | Thang điểm |
|---|---:|---:|
| Đúng sự kiện/ngày tháng | 3.609 | 4.0 |
| Đúng trọng tâm câu hỏi | 1.771 | 2.0 |
| Đủ ý cần thiết | 1.325 | 1.5 |
| Citation hỗ trợ câu trả lời | 1.340 | 1.5 |
| Diễn đạt rõ ràng | 0.989 | 1.0 |

## 4. Kết quả Guard/OOS/invalid

Nhóm Guard/OOS/invalid kiểm tra khả năng từ chối câu hỏi ngoài phạm vi, câu hỏi không hợp lệ hoặc tác vụ không thuộc miền lịch sử. Runtime chốt cuối xử lý nhóm này bằng câu từ chối ngắn gọn, không gọi cloud và không hiển thị nguồn không liên quan.

| Chỉ số Guard/OOS | Kết quả |
|---|---:|
| Số câu trong nhóm | 40 |
| HTTP 200 | 40/40 |
| HTTP error | 0/40 |
| Critical fail | 0/40 |
| Tỷ lệ >= 8 điểm | 100.00% |
| Điểm trung bình | 9.938/10 |
| Cloud LLM calls | 0 |
| Citation hiển thị | 0 |

Câu guard/OOS chốt trong demo:

> Câu hỏi này chưa có nguồn phù hợp trong tư liệu nội bộ, nên hệ thống không trả lời ngoài phạm vi đã kiểm chứng.

## 5. Kết quả vận hành trên bộ 500 câu

| Chỉ số vận hành | Kết quả |
|---|---:|
| HTTP 200 | 489/500 |
| HTTP error | 11/500 |
| Cloud LLM calls | 460 |
| Guard no-cloud | 40 |
| Cases có citation | 447/500 |
| Avg citations/case | 1.34 |
| Avg latency | 6123.2 ms |
| P95 latency | 10883.1 ms |
| P99 latency | 16532.5 ms |

Phần lớn thời gian vẫn nằm ở bước sinh câu trả lời bằng cloud LLM. Các câu ngoài phạm vi được xử lý ở guard nên không cần gọi cloud.

## 6. Kết quả retrieval benchmark 300 query

| Chỉ số retrieval | Kết quả | Ý nghĩa |
|---|---:|---|
| Tổng số query | 300 | Toàn bộ request HTTP OK |
| Recall@1 | 77.33% | Nguồn đúng đứng đầu |
| Recall@3 | 93.00% | Nguồn đúng nằm trong top 3 |
| Recall@5 | 94.33% | Nguồn đúng nằm trong top 5 |
| Precision@5 | 65.07% | Tỷ lệ nguồn top 5 có ích |
| MRR@5 | 0.8398 | Nguồn đúng xuất hiện càng sớm càng tốt |
| NDCG@5 | 0.8069 | Chất lượng xếp hạng top 5 |
| Evidence coverage | 93.53% | Nguồn bao phủ ý cần trả lời |
| Citation alignment | 93.67% | Citation khớp claim/câu hỏi |
| No-evidence detection | 60.00% | Khả năng nhận diện câu không có nguồn trong benchmark retrieval |
| Latency avg / P95 | 875.1 / 1762 ms | Đo trên local retrieval endpoint |

### 6.1. Retrieval theo nhóm

| Nhóm | Query | Recall@1 | Recall@3 | Recall@5 | Precision@5 | MRR@5 | NDCG@5 | Evidence coverage | Citation alignment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| direct_fact | 70 | 82.86% | 100.00% | 100.00% | 66.00% | 0.8929 | 0.8599 | 95.84% | 99.52% |
| natural_paraphrase | 70 | 75.71% | 94.29% | 98.57% | 63.14% | 0.8374 | 0.8069 | 96.13% | 96.19% |
| timeline | 45 | 73.33% | 95.56% | 97.78% | 68.00% | 0.8296 | 0.7940 | 99.26% | 100.00% |
| comparison | 45 | 88.89% | 100.00% | 100.00% | 74.67% | 0.9407 | 0.8800 | 100.00% | 100.00% |
| memory_resolved | 40 | 75.00% | 92.50% | 92.50% | 56.50% | 0.8292 | 0.8018 | 96.34% | 90.00% |
| hard_negative_oos | 30 | 60.00% | 60.00% | 60.00% | 60.00% | 0.6000 | 0.6000 | 60.00% | 60.00% |

## 7. Đánh giá tổng thể

Các chỉ số retrieval cho thấy hệ thống tìm được nguồn đúng khá mạnh ở top 3 và top 5: Recall@3 đạt **93.00%**, Recall@5 đạt **94.33%**. Điều này chứng minh pipeline hybrid retrieval đủ tốt để cung cấp context cho LLM ở phần lớn câu hỏi trong phạm vi đã kiểm chứng.

Các chỉ số answer quality cho thấy chatbot trả lời tốt trên phần lớn câu hỏi: điểm trung bình **8.766/10**, median **10/10**, tỷ lệ câu từ 8 điểm trở lên **79.20%**. Các nhóm timeline và comparison là điểm mạnh rõ nhất. Nhóm còn yếu hơn là misconception correction và memory_multiturn, tức các câu cần đính chính hiểu nhầm hoặc cần giữ mạch hội thoại nhiều lượt.

Guard/OOS xử lý đúng vai trò: câu ngoài phạm vi được từ chối ngắn gọn, không gọi cloud và không hiển thị nguồn sai.

## 8. Hạn chế cần ghi trong báo cáo

- Bộ 500 câu là benchmark nội bộ có chấm độc lập bằng rubric, không phải đánh giá bởi hội đồng human nhiều người.
- Retrieval benchmark 300 query vẫn ghi nhận `hard_negative_oos` và `no_evidence_detection` ở mức 60.00%; đây là điểm cần cải thiện nếu tiếp tục tối ưu retrieval/OOS.
- Một số lỗi còn lại trong answer quality đến từ fact/date, misconception correction và memory_multiturn; đây là hướng cải thiện nếu tiếp tục phát triển.
- Không tuyên bố hệ thống bao phủ toàn bộ lịch sử Việt Nam 1930-1975 hoặc 1858-2000.

## 9. Kết luận dùng cho đồ án

Hệ thống chatbot lịch sử Việt Nam đã đạt mức có thể đóng gói project ở phạm vi thực nghiệm 1930-1975. Pipeline retrieval đạt Recall@5 **94.33%**, MRR@5 **0.8398** và citation alignment **93.67%**, cho thấy khả năng truy xuất nguồn nội bộ tương đối ổn định. Chất lượng câu trả lời trên 500 câu đạt điểm trung bình **8.766/10**, median **10/10** và HTTP success **489/500**.

Cách trình bày trung thực nhất là: **hệ thống hỗ trợ tốt các chủ đề trọng tâm đã có nguồn kiểm chứng trong tư liệu nội bộ về lịch sử Việt Nam giai đoạn 1930-1975, có khả năng truy xuất nguồn, sinh câu trả lời có citation, xử lý một phần hội thoại nhiều lượt và từ chối câu hỏi ngoài phạm vi.**

## 10. File nguồn và file đầu ra

Nguồn số liệu:

- 500 answer audit: `reports/answer_quality_500/answer_quality_summary.json`
- 500 scored CSV: `reports/answer_quality_500/answer_quality_scored.csv`
- Retrieval 300: `reports/retrieval_300/retrieval_benchmark_300_summary.json`

File báo cáo xuất ra:

- Markdown: `reports/final_project_report/final_project_report.md`
- JSON: `reports/final_project_report/final_metrics_summary.json`
