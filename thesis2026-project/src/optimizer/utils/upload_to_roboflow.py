import os, json, argparse, logging, time
from pathlib import Path
from roboflow import Roboflow

"""
Upload pictures to Roboflow.

Usage (examples):
  poetry run python upload_to_roboflow.py --api_key "Roboflow API key" --workspace my-workspace --project my-project --run_dir "Path to run folder (…/YYYY-MM-DD/<tag>/runXX)"
"""

def infer_run_parts(run_dir: Path):
    # Expect: <out_root>/<YYYY-MM-DD>/<tag>/runXX/
    date = run_dir.parents[1].name
    tag  = run_dir.parent.name
    run  = run_dir.name
    return date, tag, run

def is_valid_yolo_txt(p: Path) -> bool:
    """Return True iff file exists, non-empty, and every non-empty line is 'cls x y w h' with coords in [0,1]."""
    try:
        if not p.exists() or p.stat().st_size == 0:
            return False
        ok_line = False
        with p.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    return False
                # class id can be int; coords must be floats in [0,1]
                _cls = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:])
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                    return False
                ok_line = True
        return ok_line
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser(description="Upload YOLO frames+labels from a run folder to Roboflow")
    ap.add_argument("--api_key", required=True, help="Roboflow API key")
    ap.add_argument("--workspace", required=True, help="Workspace ID (URL slug)")
    ap.add_argument("--project", required=True, help="Project ID (URL slug)")
    ap.add_argument("--run_dir", required=True, help="Path to run folder (…/YYYY-MM-DD/<tag>/runXX)")
    ap.add_argument("--split", default="train", choices=["train","valid","test"], help="Dataset split")
    ap.add_argument("--batch_name", default="", help="Optional Roboflow batch name")
    ap.add_argument("--limit", type=int, default=0, help="Upload only first N images (0 = all)")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between uploads (avoid API bursts)")
    ap.add_argument("--dry_run", action="store_true", help="List what would be uploaded, do not send")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    run_dir = Path(args.run_dir).resolve()
    frames_dir = run_dir / "frames"
    labels_dir = run_dir / "labels"
    meta_path  = run_dir / "meta.json"

    if not frames_dir.is_dir():
        raise SystemExit(f"Frames folder not found: {frames_dir}")
    if not labels_dir.is_dir():
        logging.warning("Labels folder not found; proceeding with images only (no annotations).")

    # derive tags from folder structure + meta.json
    date, tag, run = infer_run_parts(run_dir)
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception as e:
            logging.warning(f"Could not parse meta.json: {e}")

    # set batch name default if not provided
    batch_name = args.batch_name or f"{date}_{tag}_{run}"

    # Connect to Roboflow
    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace(args.workspace).project(args.project)
    logging.info(f"Uploading to workspace='{args.workspace}', project='{args.project}', split='{args.split}', batch='{batch_name}'")

    # Collect images (.jpg only; extend here if you also save .png)
    images = sorted(frames_dir.glob("*.jpg"))
    if args.limit > 0:
        images = images[:args.limit]

    # Upload loop
    uploaded = 0
    for i, img_path in enumerate(images, 1):
        label_path = (labels_dir / (img_path.stem + ".txt"))
        ann_ok = is_valid_yolo_txt(label_path)
        annotation_path = str(label_path) if ann_ok else None

        # Roboflow supports sending paired annotations; if None, uploads image only.
        kwargs = {
            "image_path": str(img_path),
            "annotation_path": annotation_path,  # None -> image-only upload
            "split": args.split,
            "batch_name": batch_name,
            "tag_names": [f"date:{date}", f"tag:{tag}", f"run:{run}"]
        }

        if args.dry_run:
            logging.info(f"[DRY] {img_path.name}  ann={'valid' if ann_ok else 'none'}  -> split={args.split} batch={batch_name}")
        else:
            try:
                project.upload(**kwargs)  # image + (optional) YOLO label
                uploaded += 1
                if not ann_ok and label_path.exists() and label_path.stat().st_size == 0:
                    logging.debug(f"Skipped empty label for {img_path.name}")
                if args.sleep > 0:
                    time.sleep(args.sleep)
            except Exception as e:
                logging.error(f"Upload failed for {img_path.name}: {e}")

        if i % 100 == 0:
            logging.info(f"Progress: {i}/{len(images)}")

    logging.info(f"Done. Uploaded {uploaded}/{len(images)} images to {args.split} (batch: {batch_name}).")

if __name__ == "__main__":
    main()
