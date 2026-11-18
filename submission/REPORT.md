# Báo cáo Day 09 — Phạm Anh Tuấn, 2A202603232

## 1. Kết luận trước
Pipeline kết luận **HOLD** (mã thoát `10`). Trong **224** ảnh đầu vào (`data/raw`), cửa nhận loại **22** ảnh (2 hỏng, 3 kích thước nhỏ, 8 trùng byte, 4 gần trùng, 5 rò rỉ test/val), giữ **202** ảnh và xuất bản nháp `submission/export/clean_v1` (chưa phát hành chính thức). Lý do: sau sửa tự động và duyệt mô phỏng **80/125** ảnh trong hàng đợi, vẫn còn **38/202 (18.81%)** ảnh mang cảnh báo chưa duyệt (`>= 10.00%` ngưỡng `gate.warn_ratio_hold`).

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

`05_review_queue.csv` xếp **125** ảnh theo `priority` giảm dần: ảnh vừa bị `FIXED_DROP_LINE` kèm cảnh báo ngữ nghĩa (`img_0020`, `priority=11.524`) và **6** ảnh `NEEDS_LABEL` (`priority=10.0`) đứng đầu để người duyệt xử lý dứt điểm lỗi thiếu nhãn trước, tiếp theo là các ảnh có tổng trọng số cảnh báo ngữ nghĩa cao (`POSSIBLE_MISSING`, `POSSIBLE_WRONG_CLASS`, `POSSIBLE_FALSE_POS`, `POSSIBLE_LOOSE_BOX`). Với ngân sách **80** ảnh, người duyệt xử lý được 6 ảnh thiếu nhãn và 74 ảnh cảnh báo ngữ nghĩa ưu tiên cao nhất, còn 45 ảnh trong hàng đợi (38 ảnh còn cảnh báo sau vòng 3).

## 3. Calibration (2 ngưỡng)
| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | `10` | `DUP_IMAGE_NEAR`: `1.00 / 0.50` → `1.00 / 0.67` (`DUP_IMAGE_EXACT`: `1.00 / 1.00`) | Tăng recall ảnh gần trùng từ 0.50 lên 0.67 mà không gây nổ báo động giả như mức `16` (18 cặp trên `raw`) hay làm hỏng `EXACT` như mức `20` |
| `second_opinion.conf` | 0.25 | `0.60` | `WRONG_CLASS`: `0.56 / 0.83` → `0.62 / 0.83`; `MISSING`: `0.50 / 1.00` → `0.57 / 1.00`; `LOOSE_BOX`: `0.29 / 0.40` → `0.20 / 0.20` | Giảm báo động giả cho `WRONG_CLASS` và `MISSING` trong khi giữ nguyên recall; đánh đổi là giảm khả năng bắt `LOOSE_BOX` (vốn đã yếu ở F1 = 0.20) |

**Chỗ calibration có thể đánh lừa:** Tập `data/calib/` chỉ có 39 ảnh với số mẫu mỗi loại lỗi nhỏ (2–8 mẫu), nên tại `phash_dist = 16` chỉ số calib đạt F1 cao nhất (`0.91`), nhưng khi kiểm tra thực tế trên `data/raw` thì bị báo động giả hàng loạt (18 cặp `LEAK_TEST`/`DUP_IMAGE_NEAR` ở `dist=16`). Tương tự, tăng `second_opinion.conf` lên `0.60` cải thiện F1 cho `MISSING`/`WRONG_CLASS` trên calib nhưng làm `LOOSE_BOX` giảm recall từ `0.40` xuống `0.20`.

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
  - 1 lần AI sai: ở lần kiểm tra `phash_dist`, nếu chỉ nhìn F1 trên `calib` dễ chọn nhầm `16`, cần đối chiếu thêm danh sách cặp trùng thực tế trên `data/raw` để phát hiện báo động giả.
