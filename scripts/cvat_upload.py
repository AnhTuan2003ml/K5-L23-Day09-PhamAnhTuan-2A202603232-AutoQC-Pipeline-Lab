import argparse
import io
import json
import mimetypes
import os
import time
import zipfile
from pathlib import Path

import requests
from PIL import Image


def get_login_session(cvat_url, username, password):
    base = cvat_url.rstrip("/")
    session = requests.Session()
    resp = session.post(
        f"{base}/api/auth/login",
        json={"username": username, "password": password},
        timeout=15,
    )
    resp.raise_for_status()
    token = resp.json().get("key")
    if token:
        session.headers.update({"Authorization": f"Token {token}"})
    csrf = session.cookies.get("csrftoken", "")
    if csrf:
        session.headers.update({"X-CSRFToken": csrf})
    session.headers.update({"Referer": f"{base}/"})
    return session


def wait_request(session, cvat_url, rq_id, label="request", timeout=300):
    base = cvat_url.rstrip("/")
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = session.get(f"{base}/api/requests/{rq_id}", timeout=15)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        if status == "finished":
            return data
        if status == "failed":
            raise RuntimeError(f"{label} failed: {data.get('message') or data}")
        time.sleep(1.5)
    raise TimeoutError(f"{label} timed out after {timeout}s")


def get_or_create_project(session, cvat_url, name, labels):
    base = cvat_url.rstrip("/")
    resp = session.get(f"{base}/api/projects", params={"search": name, "page_size": 20}, timeout=15)
    resp.raise_for_status()
    for project in resp.json().get("results", []):
        if project["name"] == name:
            return project["id"]
    resp = session.post(f"{base}/api/projects", json={"name": name, "labels": labels}, timeout=30)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Failed to create project: {resp.status_code} {resp.text}")
    return resp.json()["id"]


def collect_readable_images(img_dir):
    good = []
    skipped = []
    for path in sorted(img_dir.iterdir()):
        if not path.is_file():
            continue
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                im.load()
                width, height = im.size
            good.append((path, width, height))
        except Exception as exc:
            skipped.append((path.name, str(exc)))
    return good, skipped


def build_yolo11_zip(images, lbl_dir, class_names, min_box_px=0.0):
    buf = io.BytesIO()
    kept_boxes = 0
    skipped_lines = 0
    filtered_small = 0
    valid_ids = set(range(len(class_names)))
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("obj.names", "\n".join(class_names) + "\n")
        zf.writestr(
            "obj.data",
            f"classes = {len(class_names)}\ntrain = train.txt\nnames = obj.names\nbackup = backup/\n",
        )
        train_lines = []
        for img_path, img_w, img_h in images:
            train_lines.append(f"obj_train_data/{img_path.name}")
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            valid_lines = []
            if lbl_path.exists():
                for raw_ln in lbl_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    ln = raw_ln.strip()
                    if not ln:
                        continue
                    parts = ln.split()
                    if len(parts) != 5:
                        skipped_lines += 1
                        continue
                    try:
                        cid = int(parts[0])
                        cx, cy, w, h = (float(x) for x in parts[1:])
                        if cid not in valid_ids or any(v != v for v in (cx, cy, w, h)) or w <= 0 or h <= 0:
                            skipped_lines += 1
                            continue
                        if min_box_px > 0 and (w * img_w < min_box_px or h * img_h < min_box_px):
                            filtered_small += 1
                            continue
                        valid_lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                        kept_boxes += 1
                    except Exception:
                        skipped_lines += 1
            zf.writestr(
                f"obj_train_data/{img_path.stem}.txt",
                ("\n".join(valid_lines) + "\n") if valid_lines else "",
            )
        zf.writestr("train.txt", "\n".join(train_lines) + "\n")
    return buf.getvalue(), kept_boxes, skipped_lines, filtered_small


def delete_tasks_by_name(session, cvat_url, project_id, task_name):
    base = cvat_url.rstrip("/")
    resp = session.get(
        f"{base}/api/tasks",
        params={"project_id": project_id, "search": task_name, "page_size": 50},
        timeout=15,
    )
    resp.raise_for_status()
    for task in resp.json().get("results", []):
        if task.get("name") == task_name:
            del_resp = session.delete(f"{base}/api/tasks/{task['id']}", timeout=30)
            if del_resp.status_code not in (200, 204):
                raise RuntimeError(f"Delete task {task['id']} failed: {del_resp.status_code} {del_resp.text}")


def upload_split(session, cvat_url, project_id, split_dir, task_name, class_names, min_box_px=0.0, replace=False):
    base = cvat_url.rstrip("/")
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"

    good_imgs, skipped_imgs = collect_readable_images(img_dir)
    if not good_imgs:
        raise RuntimeError(f"No readable images found in {img_dir}")

    if replace:
        delete_tasks_by_name(session, base, project_id, task_name)

    resp = session.post(f"{base}/api/tasks", json={"name": task_name, "project_id": project_id}, timeout=30)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Create task failed: {resp.status_code} {resp.text}")
    task_id = resp.json()["id"]

    files = []
    open_handles = []
    try:
        for idx, (path, _, _) in enumerate(good_imgs):
            mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            fh = open(path, "rb")
            open_handles.append(fh)
            files.append((f"client_files[{idx}]", (path.name, fh, mime)))

        resp = session.post(
            f"{base}/api/tasks/{task_id}/data",
            data={"image_quality": "70", "sorting_method": "lexicographical", "use_cache": "true"},
            files=files,
            timeout=180,
        )
        if resp.status_code not in (200, 201, 202):
            raise RuntimeError(f"Upload data failed: {resp.status_code} {resp.text}")
        rq_id = resp.json().get("rq_id")
        if rq_id:
            wait_request(session, base, rq_id, label=f"task {task_id} data")
    finally:
        for fh in open_handles:
            fh.close()

    kept_boxes = 0
    skipped_lines = 0
    filtered_small = 0
    if lbl_dir.exists():
        zip_bytes, kept_boxes, skipped_lines, filtered_small = build_yolo11_zip(
            good_imgs, lbl_dir, class_names, min_box_px=min_box_px
        )
        resp = session.post(
            f"{base}/api/tasks/{task_id}/annotations",
            params={"format": "YOLO 1.1", "conv_mask_to_poly": "false"},
            files={"annotation_file": ("annotations.zip", zip_bytes, "application/zip")},
            timeout=120,
        )
        if resp.status_code not in (200, 201, 202):
            raise RuntimeError(f"Upload annotations failed: {resp.status_code} {resp.text}")
        rq_id = resp.json().get("rq_id")
        if rq_id:
            wait_request(session, base, rq_id, label=f"task {task_id} annotations")

    return {
        "task_id": task_id,
        "task_name": task_name,
        "url": f"{base}/tasks/{task_id}",
        "images_uploaded": len(good_imgs),
        "images_skipped": [name for name, _ in skipped_imgs],
        "boxes_imported": kept_boxes,
        "boxes_filtered_small": filtered_small,
        "label_lines_skipped": skipped_lines,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cvat-url", default="http://100.104.206.55:8080")
    parser.add_argument("--username", default=os.environ.get("CVAT_USERNAME", ""))
    parser.add_argument("--password", default=os.environ.get("CVAT_PASSWORD", ""))
    parser.add_argument("--project", default="AutoQC-Pipeline-Lab")
    parser.add_argument("--labels", default="cvat/labels.json")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--splits", nargs="+", default=["raw", "calib", "gold_val"])
    parser.add_argument("--min-box-px", type=float, default=0.0)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.username or not args.password:
        raise ValueError("Missing credentials: pass --username and --password (or set CVAT_USERNAME/CVAT_PASSWORD)")
    session = get_login_session(args.cvat_url, args.username, args.password)

    me = session.get(f"{args.cvat_url.rstrip('/')}/api/users/self", timeout=15)
    me.raise_for_status()
    print(f"Logged in as: {me.json()['username']}")

    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    class_names = [item["name"] for item in labels]
    project_id = get_or_create_project(session, args.cvat_url, args.project, labels)
    print(f"Project '{args.project}' id={project_id}")

    for split in args.splits:
        result = upload_split(
            session=session,
            cvat_url=args.cvat_url,
            project_id=project_id,
            split_dir=Path(args.data_root) / split,
            task_name=f"data-{split}",
            class_names=class_names,
            min_box_px=args.min_box_px,
            replace=args.replace,
        )
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
