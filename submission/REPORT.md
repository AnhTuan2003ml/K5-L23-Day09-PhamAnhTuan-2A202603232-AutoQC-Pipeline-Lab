# Báo cáo Day 09 — Phạm Anh Tuấn, 2A202603232

## 1. Kết luận trước
Sau khi chạy pipeline trên 224 ảnh đầu vào (`data/raw`), hệ thống loại 22 ảnh không đạt yêu cầu (2 ảnh hỏng, 3 ảnh quá nhỏ, 8 ảnh trùng byte, 4 ảnh gần trùng, 5 ảnh rò rỉ từ tập test/val) và giữ lại 202 ảnh. Trong quá trình xử lý, hệ thống sử dụng ngân sách duyệt 80 ảnh để giải quyết các trường hợp ưu tiên trong hàng đợi. Tuy nhiên, sau khi hoàn thành vẫn còn 38/202 ảnh mang cảnh báo chưa được duyệt, tương đương 18.81%. Vì tỷ lệ này vượt ngưỡng 10.00% (`gate.warn_ratio_hold`), pipeline đưa ra kết luận **HOLD** (mã thoát `10`) và chỉ xuất dataset bản nháp tại `submission/export/clean_v1`, chưa đủ điều kiện phát hành chính thức.

## 2. Lỗi và phân việc
Tổng hợp từ `submission/reports/02_error_list_v1.csv` (vòng 1 trên 202 ảnh đã nhận + 22 ảnh bị loại ở bước 1):

| Mã lỗi | Số ảnh/dòng | Ai sửa (máy/người) | Vì sao |
|---|---:|---|---|
| `CORRUPT_IMAGE`, `BAD_SIZE` | 2 + 3 ảnh | Máy (`remove_image`) | Ảnh hỏng/quá nhỏ (< 250px), loại ngay ở cửa vào |
| `DUP_IMAGE_EXACT`, `DUP_IMAGE_NEAR`, `LEAK_TEST` | 8 + 4 + 5 ảnh | Máy (`remove_image`) | Trùng lặp hoặc rò rỉ sang `test`/`gold_val`, loại để tránh sai lệch đánh giá |
| `BAD_CLASS_NAME` | 2 mục (`data.yaml`) | Máy (`rename_classes`) | Chuẩn hoá tên lớp theo `cfg["classes"]` |
| `BAD_FORMAT`, `BAD_CLASS_ID`, `BOX_ZERO`, `DUP_BOX` | 3 + 6 + 4 + 7 dòng | Máy (`drop_line`) | Lỗi cú pháp/ID sai/box rỗng/box lặp, xoá dòng an toàn |
| `BOX_OUT_OF_IMAGE` | 8 dòng | Máy (`clip_box`) | Cắt toạ độ tràn biên về `[0, 1]` |
| `MISSING_LABEL_FILE` | 6 ảnh | Người (`needs_label`) | Máy không tự bịa nhãn, chuyển người gán |
| `EMPTY_LABEL`, `BOX_TINY`, `BOX_HUGE`, `CLASS_RARE` | 12 + 6 + 3 + 4 | Người | Cảnh báo chất lượng/phân bố cần mắt người kiểm tra |
| `POSSIBLE_MISSING`, `POSSIBLE_LOOSE_BOX`, `POSSIBLE_WRONG_CLASS`, `POSSIBLE_FALSE_POS` | 53 + 28 + 18 + 29 | Người | Nghi ngờ ngữ nghĩa từ `second_opinion`, model có thể sai nên không tự sửa |

`05_review_queue.csv` sắp xếp 125 ảnh theo điểm ưu tiên giảm dần để tận dụng hiệu quả ngân sách 80 ảnh duyệt tay. Các ảnh vừa bị xoá dòng lỗi kèm cảnh báo ngữ nghĩa (`img_0020`, `priority=11.524`) và 6 ảnh thiếu file nhãn (`priority=10.0`) được đẩy lên đầu vì đây là những lỗi nghiêm trọng nhất cần người xử lý dứt điểm. Tiếp theo là những ảnh có nhiều cảnh báo ngữ nghĩa hoặc độ tin cậy lệch cao; nhờ vậy 80 lượt duyệt đã giải quyết trọn vẹn 6 ảnh thiếu nhãn và 74 ảnh nghi ngờ nặng nhất, để lại 45 ảnh ít nghiêm trọng hơn (trong đó 38 ảnh còn cảnh báo sau vòng 3).

## 3. Calibration (2 ngưỡng)
| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | `10` | `DUP_IMAGE_NEAR`: `1.00 / 0.50` → `1.00 / 0.67` (`DUP_IMAGE_EXACT`: `1.00 / 1.00`) | Tăng recall ảnh gần trùng từ 0.50 lên 0.67 mà không phát sinh nhiều cảnh báo sai như mức `16` (18 cặp trên `raw`) hay làm hỏng `EXACT` như mức `20` |
| `second_opinion.conf` | 0.25 | `0.60` | `WRONG_CLASS`: `0.56 / 0.83` → `0.62 / 0.83`; `MISSING`: `0.50 / 1.00` → `0.57 / 1.00`; `LOOSE_BOX`: `0.29 / 0.40` → `0.20 / 0.20` | Giảm cảnh báo sai cho `WRONG_CLASS` và `MISSING` trong khi giữ nguyên recall; đánh đổi là giảm khả năng bắt `LOOSE_BOX` (vốn đã yếu ở F1 = 0.20) |

**Hạn chế khi lựa chọn ngưỡng:** Tập `data/calib/` chỉ có 39 ảnh với số mẫu mỗi loại lỗi khá ít (2–8 mẫu), nên tại `phash_dist = 16` chỉ số calib đạt F1 cao nhất (`0.91`), nhưng khi đối chiếu trực tiếp trên `data/raw` lại phát sinh nhiều cảnh báo sai (18 cặp bị cờ trùng/rò rỉ ở `dist=16`). Tương tự, tăng `second_opinion.conf` lên `0.60` cải thiện precision cho `MISSING` và `WRONG_CLASS` trên calib nhưng làm `LOOSE_BOX` giảm recall từ `0.40` xuống `0.20`.

## 4. Cửa chặn và điều pipeline KHÔNG bắt được
- **(a) Chứng minh gate chặn đúng:** Chạy `python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb` (không có người duyệt nên 6 ảnh `MISSING_LABEL_FILE` vẫn còn nguyên): pipeline kết luận **REWORK** (`Còn 6 lỗi bắt buộc (ERROR)`), trả về mã thoát **20** và **không xuất** thư mục `work/exp_nb`.
- **(b) Điểm mù đã biết:**
  1. **Box lỏng nhẹ (`0.5 <= IoU < 0.7`) hoặc box lệch ở vật nhỏ (`area < 600 px²`)**: `second_opinion` bỏ qua dự đoán `< min_area_px` và chỉ cờ khi `IoU < 0.5`, đồng thời tại `conf=0.60` F1 của `POSSIBLE_LOOSE_BOX` trên calib chỉ đạt `0.20`.
  2. **Ảnh trùng biến đổi mạnh hoặc cắt xén lớn**: ở `phash_dist = 10`, vẫn sót `2/6` ca `DUP_IMAGE_NEAR` trên calib (recall `0.67`) vì biến đổi mạnh làm khoảng cách Hamming vượt ngưỡng.

## 5. Train: raw vs clean (tùy chọn, chấm riêng)
Chưa chạy (chưa thực hiện bước train trên Google Colab).

## 6. Quyết định và ghi chú AI
- **(a) Quyết định cuối:** **HOLD** (`18.81%` ảnh còn cảnh báo chưa duyệt `>= 10.00%`). Để chuyển thành **Release**, cần tăng ngân sách duyệt tay thêm ít nhất ~18–20 ảnh cảnh báo ưu tiên cao (đưa `n_warn_images` xuống `< 20/202`, tức `< 10%`) rồi chạy lại pipeline.
- **(b) Ghi chú AI:**
  - Prompt chính: triển khai 4 module `label_format.py`, `gate.py`, `autofix.py`, `second_opinion.py` theo docstring đặc tả, chạy calibration và hoàn thiện báo cáo từ số liệu thực tế.
  - Cách kiểm: chạy toàn bộ `pytest -q` (79/79 passed), chạy `pipeline.calibrate`, `pipeline.run` (cả `--budget 0` và `--budget 80`) và `scripts/check_submission.py --no-train`.
  - 1 lần AI sai: khi dùng công cụ AI đánh giá repo trên GitHub, AI đọc nhầm bản cache cũ và báo sai rằng 4 module vẫn còn `NotImplementedError` ở các dòng 245–315 và `configs/pipeline.yaml` vẫn giữ ngưỡng cũ; tôi đã kiểm tra trực tiếp bằng `git show origin/main:...` và `Select-String` để xác nhận mã nguồn thực tế trên `main` đã cập nhật đúng.
- **(c) CVAT (thưởng +2):**
  - Tôi đã upload 222 ảnh hợp lệ của tập `raw` lên CVAT local, trực tiếp rà soát các ca thiếu box hoặc gán nhầm trên từng ảnh và export lại thành `data.zip` (YOLO 1.1: **750** box trên 222 ảnh so với **732** box ở `raw` và **748** box ở `clean_v1`). Trong quá trình kiểm tra, tôi đã gán bổ sung nhãn cho ảnh bị bỏ sót như `img_0031` (từ 0 lên 6 box), vẽ thêm đối tượng bị thiếu ở `img_0001` (`3 → 4`), `img_0007` (`6 → 8`) và xoá box nhiễu ở `img_0050` (`1 → 0`), tổng cộng có **61** ảnh thay đổi số lượng box.
  - Khi dùng bộ lọc `width < 20` hoặc `height < 20` trên tập CVAT, có **64** box trên **32** ảnh khớp điều kiện (ví dụ `img_0005`, `img_0007`, `img_0020`, `img_0026`, `img_0051`), trong khi pipeline dùng ngưỡng `label_format.min_box_px = 12` chỉ gắn cờ `BOX_TINY` **6** dòng (`img_0026` dòng 6, `img_0051` dòng 5, `img_0069` dòng 2, `img_0161` dòng 2–4) do ngưỡng 20 px chặt hơn ngưỡng 12 px của pipeline.
  - Về thuộc tính `can_review`: người gán đánh dấu `can_review = true` khi gặp ca mơ hồ (nhầm lớp `car`/`truck`/`bus`, bị che khuất, ranh giới box không chắc chắn) để đưa vào diện kiểm duyệt — tương ứng với việc pipeline dùng các cảnh báo `second_opinion` và điểm ưu tiên `05_review_queue.csv` để đẩy ảnh khó vào hàng đợi người duyệt.
