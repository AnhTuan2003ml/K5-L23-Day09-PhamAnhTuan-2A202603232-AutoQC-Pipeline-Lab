"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q

Máy kiểm quy tắc, người kiểm ý nghĩa (slide). Các lỗi như thiếu nhãn / gán nhầm lớp hợp lệ / box lỏng
KHÔNG bắt được bằng quy tắc cú pháp. Ta nhờ một model mạnh hơn "nhìn lại" và CHỈ CẢNH BÁO (model cũng có thể sai):
những ảnh bị cờ sẽ vào hàng đợi cho NGƯỜI duyệt.

`preds[stem]` = list `[id_lớp, conf, cx, cy, w, h]` của model (lấy qua `get_preds(ds, cfg, "checker")` nếu preds=None).
`L = ds.valid_boxes(stem)` = list `(id_lớp, cx, cy, w, h)` của nhãn (đã bỏ dòng hỏng).
Ngưỡng ở `cfg["second_opinion"]`: conf, min_area_px, fp_conf, iou_missing, iou_match, iou_loose.

Thuật toán gợi ý, với từng ảnh (bỏ qua ảnh không có trong preds hoặc không có file nhãn):
  P = dự đoán có conf >= conf  VÀ diện tích px (w*W * h*H) >= min_area_px
  Với mỗi dự đoán r trong P: lấy các nhãn có IoU(r, nhãn) >= iou_missing  (cands)
   - không có cands                            -> POSSIBLE_MISSING     (model thấy vật mà nhãn không có box)
   - có cands cùng lớp, tốt nhất IoU < iou_loose -> POSSIBLE_LOOSE_BOX  (đúng vật, box lỏng/lệch)
   - không có cands cùng lớp, nhưng cands tốt nhất IoU >= iou_match (khác lớp) -> POSSIBLE_WRONG_CLASS
  Với mỗi dòng nhãn hợp lệ: nếu IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) đều < iou_missing
                                             -> POSSIBLE_FALSE_POS  (box nhãn không khớp vật nào model thấy; Issue.line = số dòng)

Chi tiết các ca biên (test kiểm đúng các ca này)
- `cands` gồm nhãn MỌI lớp có IoU >= iou_missing; "tốt nhất cùng lớp" = cands cùng lớp có IoU lớn nhất; "tốt nhất khác lớp" tương tự với cands khác lớp.
- Thứ tự quyết định cho mỗi dự đoán: (1) cands rỗng -> MISSING. (2) có cands cùng lớp: IoU tốt nhất < iou_loose -> LOOSE_BOX, ngược lại KHÔNG cờ
  (nhãn đúng lớp đã phủ vật này, kể cả khi còn nhãn khác lớp chồng lên). (3) không có cands cùng lớp: nếu IoU tốt nhất khác lớp >= iou_match -> WRONG_CLASS;
  nếu chỉ có khác lớp với IoU trong [iou_missing, iou_match) -> không cờ gì (mơ hồ: để cho FALSE_POS / lần duyệt khác).
- Một dự đoán sinh tối đa MỘT issue (MISSING / LOOSE_BOX / WRONG_CLASS); một ảnh có thể có nhiều issue.
- Thứ tự issue: theo `ds.image_files()` rồi theo thứ tự dự đoán trong `preds[stem]`, cuối cùng là các FALSE_POS theo số dòng; `Issue.image` = stem, `Issue.line` chỉ đặt cho FALSE_POS.

Lưu ý số dòng: dùng `ds.read_label_lines(stem)` (giữ số dòng gốc, kể cả dòng hỏng/trống) để lấy `Issue.line`; `valid_boxes` KHÔNG giữ số dòng.

Tất cả severity WARN; `Issue.extra` nên có {"conf": ...} hoặc {"iou": ...} (hàng đợi duyệt dùng conf để xếp ưu tiên).

Gợi ý
- `iou_xywh` có sẵn. Dòng nhãn hỏng không được làm script chết (dùng `ds.valid_boxes` và `parse_yolo_line` trong try).
- Kết quả phụ thuộc mạnh vào ngưỡng: calibrate chúng ở Bước 3 thay vì đoán.
"""
from ..common import WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401
from ..model_cache import get_preds  # noqa: F401


def run(ds, cfg, preds=None):
    preds = get_preds(ds, cfg, "checker") if preds is None else preds
    c = cfg["second_opinion"]
    conf_thr = c["conf"]
    min_area_px = c["min_area_px"]
    fp_conf = c["fp_conf"]
    iou_missing = c["iou_missing"]
    iou_match = c["iou_match"]
    iou_loose = c["iou_loose"]
    issues = []

    for img_path in ds.image_files():
        stem = img_path.stem
        if stem not in preds:
            continue
        raw_lines = ds.read_label_lines(stem)
        if raw_lines is None:
            continue
        size = ds.image_size(stem)
        if size is None:
            continue
        im_w, im_h = size

        label_boxes = []
        for line_no, text in raw_lines:
            try:
                cid, cx, cy, w, h = parse_yolo_line(text)
            except ValueError:
                continue
            if 0 <= cid < len(ds.classes):
                label_boxes.append((line_no, cid, cx, cy, w, h))

        img_preds = preds[stem]
        for pr in img_preds:
            pcid, pconf, pcx, pcy, pw, ph = pr
            if pconf < conf_thr or (pw * im_w) * (ph * im_h) < min_area_px:
                continue
            pbox = (pcx, pcy, pw, ph)
            cands = []
            for lno, lcid, lcx, lcy, lw, lh in label_boxes:
                iou = iou_xywh(pbox, (lcx, lcy, lw, lh))
                if iou >= iou_missing:
                    cands.append((lno, lcid, iou))
            if not cands:
                issues.append(Issue(
                    "second_opinion",
                    WARN,
                    "POSSIBLE_MISSING",
                    stem,
                    f"model thấy {ds.classes[pcid]} ({pconf:.2f})",
                    extra={"conf": round(float(pconf), 4)},
                ))
                continue
            same = [x for x in cands if x[1] == pcid]
            if same:
                best_same_iou = max(x[2] for x in same)
                if best_same_iou < iou_loose:
                    issues.append(Issue(
                        "second_opinion",
                        WARN,
                        "POSSIBLE_LOOSE_BOX",
                        stem,
                        f"box {ds.classes[pcid]} có IoU={best_same_iou:.2f}",
                        extra={"conf": round(float(pconf), 4), "iou": round(float(best_same_iou), 4)},
                    ))
            else:
                best_other = max(cands, key=lambda x: x[2])
                if best_other[2] >= iou_match:
                    issues.append(Issue(
                        "second_opinion",
                        WARN,
                        "POSSIBLE_WRONG_CLASS",
                        stem,
                        f"nhãn {ds.classes[best_other[1]]} vs model {ds.classes[pcid]}",
                        extra={"conf": round(float(pconf), 4), "iou": round(float(best_other[2]), 4)},
                    ))

        fp_preds = [pr for pr in img_preds if pr[1] >= fp_conf]
        for lno, lcid, lcx, lcy, lw, lh in label_boxes:
            lbox = (lcx, lcy, lw, lh)
            max_iou = max((iou_xywh(lbox, (pr[2], pr[3], pr[4], pr[5])) for pr in fp_preds), default=0.0)
            if max_iou < iou_missing:
                issues.append(Issue(
                    "second_opinion",
                    WARN,
                    "POSSIBLE_FALSE_POS",
                    stem,
                    f"box {ds.classes[lcid]} không khớp dự đoán nào",
                    line=lno,
                    extra={"iou": round(float(max_iou), 4)},
                ))

    return issues
