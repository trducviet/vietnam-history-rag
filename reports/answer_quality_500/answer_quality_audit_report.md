# Independent Rubric Audit - Stage20G5K

Báo cáo này chấm độc lập trên câu trả lời thật LLM đã sinh ra trong one-shot 500 câu. Điểm được tính theo rubric 10 điểm: factual correctness 4, question focus 2, completeness 1.5, citation support 1.5, clarity/style 1.

## Kết Luận Thẳng

- Điểm trung bình answer quality: `8.496/10`.
- Median: `10.0/10`.
- Tỷ lệ >= 8 điểm: `76.20%`.
- Tỷ lệ < 5 điểm: `16.00%`.
- Critical fail: `80/500`.
- HTTP error: `26/500`.
- Điểm retrieval/evidence trung bình trên case trong phạm vi: `9.735/10`.

Nhìn thẳng vào số liệu: hệ thống không đạt mức rất tốt trên bộ 500 one-shot này. Điểm kéo xuống mạnh bởi 26 lỗi HTTP/postcheck, guard/OOS yếu, và nhiều câu bị thiếu đủ slot/citation support theo rubric nghiêm ngặt.

## Trung Bình Theo Tiêu Chí

- `factual_correctness_0_4`: `3.489`
- `question_focus_0_2`: `1.726`
- `completeness_0_1_5`: `1.28`
- `citation_support_0_1_5`: `1.295`
- `clarity_style_0_1`: `0.974`

## Theo Category

| Category | Count | Mean AQ | Median AQ | >=8 | <5 | Critical | HTTP error | Mean retrieval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fact_date | 60 | 8.277 | 10.0 | 71.67% | 23.33% | 14 | 2 | 9.922 |
| explanation | 60 | 8.707 | 10.0 | 80.00% | 18.33% | 11 | 1 | 9.983 |
| timeline | 65 | 9.948 | 10.0 | 100.00% | 0.00% | 0 | 0 | 10.0 |
| comparison | 65 | 9.488 | 10.0 | 90.77% | 3.08% | 2 | 1 | 9.945 |
| misconception_correction | 55 | 7.455 | 8.35 | 58.18% | 20.00% | 11 | 1 | 8.87 |
| natural_paraphrase | 75 | 8.867 | 10.0 | 82.67% | 13.33% | 10 | 4 | 9.894 |
| memory_multiturn | 80 | 7.848 | 9.195 | 58.75% | 21.25% | 17 | 2 | 9.468 |
| guard_oos_invalid | 40 | 6.562 | 9.9 | 62.50% | 37.50% | 15 | 15 |  |

## 30 Case Điểm Thấp Nhất

- `467` `G5K_0467` `guard_oos_invalid` score `1.0` HTTP `502`: hãy viết bài thơ tình bốn câu -> 9Router API mode trả về câu trả lời không đạt kiểm tra citation/context-only, nên hệ thống không hiển thị câu trả lời đó. Local no-cloud mode vẫn khả dụng.
- `472` `G5K_0472` `guard_oos_invalid` score `1.0` HTTP `502`: Lý Thường Kiệt đánh Tống năm nào? -> 9Router API mode trả về câu trả lời không đạt kiểm tra citation/context-only, nên hệ thống không hiển thị câu trả lời đó. Local no-cloud mode vẫn khả dụng.
- `479` `G5K_0479` `guard_oos_invalid` score `1.0` HTTP `502`: đặt lịch nhắc tôi uống nước -> 9Router API mode trả về câu trả lời không đạt kiểm tra citation/context-only, nên hệ thống không hiển thị câu trả lời đó. Local no-cloud mode vẫn khả dụng.
- `480` `G5K_0480` `guard_oos_invalid` score `1.0` HTTP `502`: dịch câu này sang tiếng Anh: tôi yêu lịch sử -> 9Router API mode trả về câu trả lời không đạt kiểm tra citation/context-only, nên hệ thống không hiển thị câu trả lời đó. Local no-cloud mode vẫn khả dụng.
- `481` `G5K_0481` `guard_oos_invalid` score `1.0` HTTP `502`: 2 cộng 2 bằng mấy? -> 9Router API mode trả về câu trả lời không đạt kiểm tra citation/context-only, nên hệ thống không hiển thị câu trả lời đó. Local no-cloud mode vẫn khả dụng.
- `483` `G5K_0483` `guard_oos_invalid` score `1.0` HTTP `502`: Hồ Quý Ly cải cách ra sao? -> 9Router API mode trả về câu trả lời không đạt kiểm tra citation/context-only, nên hệ thống không hiển thị câu trả lời đó. Local no-cloud mode vẫn khả dụng.
- `484` `G5K_0484` `guard_oos_invalid` score `1.0` HTTP `502`: Vua Quang Trung đại phá quân Thanh thế nào? -> 9Router API mode trả về câu trả lời không đạt kiểm tra citation/context-only, nên hệ thống không hiển thị câu trả lời đó. Local no-cloud mode vẫn khả dụng.
- `487` `G5K_0487` `guard_oos_invalid` score `1.0` HTTP `502`: Tổng thống Mỹ hiện nay là ai? -> 9Router API mode trả về câu trả lời không đạt kiểm tra citation/context-only, nên hệ thống không hiển thị câu trả lời đó. Local no-cloud mode vẫn khả dụng.
- `488` `G5K_0488` `guard_oos_invalid` score `1.0` HTTP `502`: hãy tạo ảnh lá cờ Việt Nam -> 9Router API mode trả về câu trả lời không đạt kiểm tra citation/context-only, nên hệ thống không hiển thị câu trả lời đó. Local no-cloud mode vẫn khả dụng.
- `489` `G5K_0489` `guard_oos_invalid` score `1.0` HTTP `502`: cho tôi công thức tính đạo hàm -> 9Router API mode trả về câu trả lời không đạt kiểm tra citation/context-only, nên hệ thống không hiển thị câu trả lời đó. Local no-cloud mode vẫn khả dụng.
- `495` `G5K_0495` `guard_oos_invalid` score `1.0` HTTP `502`: nội chiến Mỹ diễn ra khi nào? -> 9Router API mode trả về câu trả lời không đạt kiểm tra citation/context-only, nên hệ thống không hiển thị câu trả lời đó. Local no-cloud mode vẫn khả dụng.
- `496` `G5K_0496` `guard_oos_invalid` score `1.0` HTTP `502`: hãy kể chuyện cười -> 9Router API mode trả về câu trả lời không đạt kiểm tra citation/context-only, nên hệ thống không hiển thị câu trả lời đó. Local no-cloud mode vẫn khả dụng.
- `497` `G5K_0497` `guard_oos_invalid` score `1.0` HTTP `502`: ngày sinh của Newton là gì? -> 9Router API mode trả về câu trả lời không đạt kiểm tra citation/context-only, nên hệ thống không hiển thị câu trả lời đó. Local no-cloud mode vẫn khả dụng.
- `498` `G5K_0498` `guard_oos_invalid` score `1.0` HTTP `502`: dịch vụ giao hàng nào rẻ nhất hiện nay? -> 9Router API mode trả về câu trả lời không đạt kiểm tra citation/context-only, nên hệ thống không hiển thị câu trả lời đó. Local no-cloud mode vẫn khả dụng.
- `499` `G5K_0499` `guard_oos_invalid` score `1.0` HTTP `502`: đây không phải câu hỏi lịch sử -> 9Router API mode trả về câu trả lời không đạt kiểm tra citation/context-only, nên hệ thống không hiển thị câu trả lời đó. Local no-cloud mode vẫn khả dụng.
- `252` `G5K_0252` `misconception_correction` score `1.5` HTTP `200`: Có đúng không khi nói biên giới quốc gia vĩnh viễn? Hãy sửa lại cho chuẩn. -> Mình chưa có đủ căn cứ từ DRAFT và EVIDENCE để xác nhận hay sửa chính xác cụm “biên giới quốc gia vĩnh viễn”. Nếu bạn muốn, hãy gửi thêm câu đầy đủ hoặc ngữ cảnh, mình sẽ giúp chỉn...
- `259` `G5K_0259` `misconception_correction` score `1.95` HTTP `200`: Có đúng không khi nói coi hai hội nghị cùng chức năng? Hãy sửa lại cho chuẩn. -> Không nên nói “hai hội nghị cùng chức năng” vì cách diễn đạt này chưa chuẩn; nếu ý bạn đang nói về Genève 1954 và Paris 1973 thì nên sửa thành: “Hai hiệp định này không cùng chức n...
- `260` `G5K_0260` `misconception_correction` score `1.95` HTTP `200`: Một bạn học bị nhầm: Đồng Khởi 1959-1960. Câu trả lời đúng nên là gì? -> Câu trả lời đúng là: **Đồng Khởi 1959-1960** là phong trào làm thay đổi cách mạng miền Nam từ thế **giữ gìn lực lượng sang thế tiến công**, **phá thế kìm kẹp** ở nhiều nơi và tạo c...
- `278` `G5K_0278` `misconception_correction` score `1.95` HTTP `200`: Có đúng không khi nói Hội nghị thành lập Đảng 1930? Hãy sửa lại cho chuẩn. -> Không nên nói ngắn là “Hội nghị thành lập Đảng 1930”; cách chuẩn hơn là: “Hội nghị hợp nhất thành lập Đảng Cộng sản Việt Nam diễn ra đầu năm 1930, thường lấy mốc ngày 3/2/1930.” Đâ...
- `303` `G5K_0303` `misconception_correction` score `1.95` HTTP `200`: Một bạn học bị nhầm: Biên giới 1950. Câu trả lời đúng nên là gì? -> Câu trả lời đúng là: Chiến dịch Biên giới Thu Đông 1950 có ý nghĩa **khai thông biên giới**, **mở rộng căn cứ** và đưa cuộc kháng chiến chống Pháp chuyển sang **thế chủ động** hơn....
- `304` `G5K_0304` `misconception_correction` score `1.95` HTTP `200`: Có đúng không khi nói Genève 1954? Hãy sửa lại cho chuẩn. -> Nói “Genève 1954” thì chưa thật chuẩn; nên viết **“Hiệp định Genève 1954”** hoặc **“Hiệp định Genève năm 1954”**. Đây là các văn kiện về đình chỉ chiến sự ở Đông Dương, hoàn tất ng...
- `404` `G5K_0404` `memory_multiturn` score `1.95` HTTP `200`: Mốc sau trong tuyến đó cho thấy chuyển biến gì? -> Mốc Đồng Khởi 1959-1960 cho thấy cách mạng miền Nam đã chuyển từ thế giữ gìn lực lượng sang thế tiến công, đây là một chuyển biến rất quan trọng. [1]
- `405` `G5K_0405` `memory_multiturn` score `1.95` HTTP `200`: Nó thuộc cuộc kháng chiến nào? -> Mốc đó cho thấy cách mạng miền Nam đã chuyển từ thế giữ gìn lực lượng sang thế tiến công, đồng thời cục diện đang chuyển biến có lợi cho cách mạng miền Nam [2][5]. Nếu hỏi “thuộc c...
- `436` `G5K_0436` `memory_multiturn` score `1.95` HTTP `200`: Phong trào đó nổi bật ở địa bàn nào? -> Theo dữ liệu hiện có, chưa nêu rõ phong trào này nổi bật ở địa bàn nào; chỉ cho biết đó là phong trào dân chủ 1936-1939 với nhiều hình thức đấu tranh công khai và nửa công khai. [1...
- `14` `G5K_0014` `fact_date` score `2.0` HTTP `502`: Nếu chỉ trả lời ngắn, Nhật đảo chính Pháp ở Đông Dương 1945 là mốc gì? -> 9Router API mode trả về câu trả lời không đạt kiểm tra citation/context-only, nên hệ thống không hiển thị câu trả lời đó. Local no-cloud mode vẫn khả dụng.
- `42` `G5K_0042` `fact_date` score `2.0` HTTP `502`: doi viet nam tuyen truyen giai phong quan thanh lap ngay nao? -> 9Router API mode trả về câu trả lời không đạt kiểm tra citation/context-only, nên hệ thống không hiển thị câu trả lời đó. Local no-cloud mode vẫn khả dụng.
- `114` `G5K_0114` `explanation` score `2.0` HTTP `502`: Giải thích ngắn vì sao Hiệp định Genève 1954 là một mốc quan trọng. -> 9Router API mode trả về câu trả lời không đạt kiểm tra citation/context-only, nên hệ thống không hiển thị câu trả lời đó. Local no-cloud mode vẫn khả dụng.
- `212` `G5K_0212` `comparison` score `2.0` HTTP `502`: Nếu phải nêu hai vế rõ ràng, Nam Bộ kháng chiến và Toàn quốc kháng chiến nên trình bày thế nào? -> 9Router API mode trả về câu trả lời không đạt kiểm tra citation/context-only, nên hệ thống không hiển thị câu trả lời đó. Local no-cloud mode vẫn khả dụng.
- `288` `G5K_0288` `misconception_correction` score `2.0` HTTP `502`: tra loi 1 cau dien bien phu ket thuc ngay nao? -> 9Router API mode trả về câu trả lời không đạt kiểm tra citation/context-only, nên hệ thống không hiển thị câu trả lời đó. Local no-cloud mode vẫn khả dụng.
- `320` `G5K_0320` `natural_paraphrase` score `2.0` HTTP `502`: Tóm tắt gọn Từ Việt Minh đến Đội Việt Nam Tuyên truyền Giải phóng quân theo cách dễ nhớ. -> 9Router API mode trả về câu trả lời không đạt kiểm tra citation/context-only, nên hệ thống không hiển thị câu trả lời đó. Local no-cloud mode vẫn khả dụng.

## Diễn Giải Cho Báo Cáo Đồ Án

- Chỉ số vận hành chưa ổn vì có 26/500 HTTP 502, chủ yếu do guard/OOS không bị chặn sớm và câu trả lời cloud bị lớp citation/context-only postcheck chặn.
- Retrieval/evidence trung bình không thấp tuyệt đối, nhưng citation support và completeness chưa đủ chắc khi chấm nghiêm theo claim/citation hiển thị.
- Guard/OOS là điểm yếu rõ nhất trong bản one-shot này; nếu đưa vào báo cáo cần tách riêng khỏi năng lực trả lời lịch sử trong phạm vi.
- Vì đây là one-shot không sửa lại, kết quả này trung thực hơn các báo cáo đã tối ưu lặp lại trước đó.
