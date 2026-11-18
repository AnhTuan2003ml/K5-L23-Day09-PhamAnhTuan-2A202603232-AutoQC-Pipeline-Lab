# Annotation & AutoQC Guideline

## 1. Danh sách nhãn (Classes) & Thuộc tính

Thứ tự ID cố định (YOLO):

| ID | Tên nhãn | Màu CVAT | Thuộc tính | Mô tả |
|---:|---|---|---|---|
| 0 | `car` | `#ff8a3d` | `can_review` (checkbox, mặc định `false`) | Xe ô tô con, SUV, bán tải nhỏ, van chở khách |
| 1 | `person` | `#2fbf71` | `can_review` (checkbox, mặc định `false`) | Người đi bộ, người điều khiển phương tiện (tách riêng với xe) |
| 2 | `truck` | `#7b61ff` | `can_review` (checkbox, mặc định `false`) | Xe tải, xe đầu kéo, xe chở hàng |
| 3 | `bus` | `#e5484d` | `can_review` (checkbox, mặc định `false`) | Xe buýt, xe khách cỡ lớn |

- **Khi nào bật `can_review = true`**: Đánh dấu khi gặp ca khó/mơ hồ (nhầm giữa `car`–`truck`–`bus`, bị che khuất nặng, ranh giới box không chắc chắn) để đưa vào hàng đợi kiểm duyệt.

## 2. Quy chuẩn Bounding Box

- **Ôm sát đối tượng**: Box phải bao khít phần nhìn thấy của đối tượng, không phình rộng quá mức (`IoU >= 0.5` so với đối tượng thật) và không cắt xén thân đối tượng.
- **Giới hạn toạ độ**: Box nằm gọn trong khung ảnh (`0 <= x, y, w, h <= 1`), độ lệch tràn biên tối đa cho phép `overflow_tol = 0.002`.
- **Kích thước tối thiểu**:
  - **Upload lên CVAT**: Lọc bỏ các nhãn có `width < 20 px` hoặc `height < 20 px` (`--min-box-px 20`).
  - **Trong pipeline AutoQC**:
    - Cạnh ngắn `< 12 px` (`label_format.min_box_px: 12`) → cảnh báo `BOX_TINY`.
    - Diện tích `< 600 px²` (`second_opinion.min_area_px: 600`) → bỏ qua dự đoán của model kiểm tra (vật quá nhỏ không gán).
- **Kích thước tối đa**: Box chiếm `> 95%` khung ảnh (`max_box_frac: 0.95`) → cảnh báo `BOX_HUGE`.
- **Trùng lặp box**: Hai box cùng lớp chồng lấn `IoU >= 0.9` (`dup_box_iou: 0.9`) → cảnh báo/loại bỏ box lặp.

## 3. Quy chuẩn Ảnh đầu vào (Intake)

- Ảnh phải đọc được nguyên vẹn (không hỏng file, không cắt cụt).
- Cạnh ngắn tối thiểu `250 px` (`intake.min_side_px: 250`).
- Không trùng lặp byte hoặc trùng gần đúng (pHash) với ảnh trong `data/test/images` hoặc `data/gold_val/images` để tránh rò rỉ dữ liệu.

## 4. Quy trình Kiểm duyệt & Cổng phát hành (Gate)

- **Ngân sách duyệt tay**: Tối đa `80` ảnh/lô (`review.budget_images: 80`), ưu tiên ảnh có tổng trọng số lỗi cao (`NEEDS_LABEL: 10`, `POSSIBLE_WRONG_CLASS: 3`, `POSSIBLE_MISSING: 2`, `POSSIBLE_FALSE_POS: 2`, `POSSIBLE_LOOSE_BOX: 1.5`, ...).
- **Cổng Hold**: Nếu tỉ lệ ảnh còn cảnh báo chưa được người duyệt `>= 10%` (`gate.warn_ratio_hold: 0.10`) → **Hold** (chưa phát hành).
- **Cổng Rework**: Nếu còn ảnh thiếu file nhãn (`MISSING_LABEL_FILE`) chưa được gán → **Rework**.

## 5. Quy tắc lọc nhanh trên CVAT

Tham khảo chi tiết tại [CVAT_FILTER.md](CVAT_FILTER.md):
- Box nhỏ: `width < 20` (hoặc `height < 20`)
- Theo lớp: `label=="truck"`
- Ca cần duyệt lại: `attr["can_review"]==true`
