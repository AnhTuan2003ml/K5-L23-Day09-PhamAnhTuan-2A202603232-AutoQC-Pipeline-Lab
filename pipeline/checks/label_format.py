"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |

Gợi ý
- Mép trái = x - w/2, mép phải = x + w/2 (xem câu hỏi "dòng YOLO nào lỗi?" trong slide).
- Một dòng hỏng (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO) thì `continue`, đừng kiểm tiếp hình học của dòng đó.
- Dòng thiếu cột làm script của slide dừng giữa chừng: đây là lý do phải `try/except`.
- `iou_xywh(a, b)` có sẵn trong common.py.
- Ngưỡng nằm trong `cfg["label_format"]` — đừng viết cứng số vào code.
"""
from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401


def run(ds, cfg):
    c = cfg["label_format"]
    tol = c["overflow_tol"]
    min_box_px = c["min_box_px"]
    max_box_frac = c["max_box_frac"]
    dup_box_iou = c["dup_box_iou"]
    issues = []
    stems = sorted({p.stem for p in ds.image_files()} | {p.stem for p in ds.label_files()})
    for stem in stems:
        lines = ds.read_label_lines(stem)
        if lines is None:
            continue
        size = ds.image_size(stem)
        seen_boxes = []
        for line_no, text in lines:
            try:
                cid, cx, cy, w, h = parse_yolo_line(text)
            except ValueError as exc:
                issues.append(Issue("label_format", ERROR, "BAD_FORMAT", stem, str(exc), line=line_no))
                continue
            if not (0 <= cid < len(ds.classes)):
                issues.append(Issue("label_format", ERROR, "BAD_CLASS_ID", stem, f"id lớp không hợp lệ: {cid}", line=line_no))
                continue
            if w <= 0 or h <= 0:
                issues.append(Issue("label_format", ERROR, "BOX_ZERO", stem, f"kích thước box <= 0: ({w}, {h})", line=line_no))
                continue
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            overflow = max(0.0, -x1, -y1, x2 - 1.0, y2 - 1.0)
            if overflow > tol:
                issues.append(Issue("label_format", ERROR, "BOX_OUT_OF_IMAGE", stem, f"box tràn biên {overflow:.4f}", line=line_no, extra={"overflow": round(overflow, 6)}))
            box = (cx, cy, w, h)
            if any(prev_cid == cid and iou_xywh(box, prev_box) > dup_box_iou for prev_cid, prev_box in seen_boxes):
                issues.append(Issue("label_format", ERROR, "DUP_BOX", stem, "box trùng lặp", line=line_no))
            seen_boxes.append((cid, box))
            if size is not None:
                im_w, im_h = size
                if min(w * im_w, h * im_h) < min_box_px:
                    issues.append(Issue("label_format", WARN, "BOX_TINY", stem, "box quá nhỏ", line=line_no))
            if w * h > max_box_frac:
                issues.append(Issue("label_format", WARN, "BOX_HUGE", stem, "box quá lớn", line=line_no))
    return issues
