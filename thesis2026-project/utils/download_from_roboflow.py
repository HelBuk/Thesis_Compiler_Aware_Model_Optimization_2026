#!/usr/bin/env python3
"""
Download a Roboflow dataset version in YOLOv8 format.

Usage (examples):
  --export ROBOFLOW_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
  poetry run python download_from_roboflow.py --api_key "Roboflow API key" --workspace my-workspace --project my-project --version 7

Options:
  --api_key can be omitted if ROBOFLOW_API_KEY is set.
  --dest chooses the parent folder (default: ./datasets)
  --format can be yolov8, coco, etc. (default: yolov8)
"""

import os
import argparse
import logging
import shutil
from pathlib import Path
from datetime import datetime

def main():
    ap = argparse.ArgumentParser(description="Download Roboflow dataset version (YOLOv8) into ./datasets/")
    ap.add_argument("--api_key", required=True, help="Roboflow API key")
    ap.add_argument("--workspace", required=True, help="Roboflow workspace slug")
    ap.add_argument("--project", required=True, help="Roboflow project slug")
    ap.add_argument("--version", type=int, required=True, help="Dataset version number (integer)")
    ap.add_argument("--format", default="yolov8", help="Export format (default: yolov8)")
    ap.add_argument("--dest", default="datasets", help="Destination root folder (default: ./datasets)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing destination if it exists")
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("ERROR: Provide --api_key or set ROBOFLOW_API_KEY env var.")

    # Lazy import so we only require roboflow when used
    try:
        from roboflow import Roboflow
    except Exception:
        raise SystemExit("Missing dependency: pip install roboflow")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    rf = Roboflow(api_key=args.api_key)
    logging.info(f"Connecting to workspace='{args.workspace}', project='{args.project}', version={args.version}...")
    project = rf.workspace(args.workspace).project(args.project)

    # Download to a temp location chosen by the SDK (usually ./<project>-<version>/)
    logging.info(f"Requesting download in format '{args.format}' from Roboflow...")
    ds = project.version(args.version).download(args.format)

    src_path = Path(ds.location).resolve()          # e.g., ./my-project-3
    if not src_path.exists():
        raise SystemExit(f"ERROR: Roboflow returned location that doesn't exist: {src_path}")

    # Final destination: datasets/<project>-v<version>-<format>/
    dest_root = Path(args.dest).resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    dest_name = f"{args.project}-v{args.version}-{args.format}"
    dest_path = dest_root / dest_name

    if dest_path.exists():
        if args.overwrite:
            logging.info(f"Destination exists, overwriting: {dest_path}")
            shutil.rmtree(dest_path)
        else:
            logging.info(f"Destination exists, leaving as-is: {dest_path}")
            print_next_steps(dest_path)
            return

    # Move the downloaded folder into datasets/
    logging.info(f"Moving '{src_path}' -> '{dest_path}'")
    shutil.move(str(src_path), str(dest_path))

    # Nice extras: record a small receipt
    receipt = dest_path / "_download_receipt.txt"
    receipt.write_text(
        f"workspace: {args.workspace}\n"
        f"project:   {args.project}\n"
        f"version:   {args.version}\n"
        f"format:    {args.format}\n"
        f"when:      {datetime.now().isoformat(timespec='seconds')}\n"
    )

    logging.info("Download complete.")
    print_next_steps(dest_path)

def print_next_steps(dest_path: Path):
    data_yaml = dest_path / "data.yaml"
    print("\nDataset ready\n")
    print(f"Location: {dest_path}")
    if data_yaml.exists():
        print(f"data.yaml: {data_yaml}\n")
        print("Train with YOLOv8 (example):")
        print(f" yolo detect train data='{data_yaml}' model=yolov8n.pt imgsz=1280 epochs=50 device=0\n")
    else:
        print("Note: data.yaml not found (format might not be 'yolov8').")

if __name__ == "__main__":
    main()
