#!/usr/bin/env python3
"""
Convert Exceptional Trajectories (ET) dataset into universal evaluator format.

Input (ET):
  - traj/<scene>.txt       (KITTI-like 12 floats per frame, w2c)
  - caption/<scene>.txt    (plain caption text)
  - optional split files:
      clatr_train_split.txt / clatr_val_split.txt / clatr_test_split.txt

Output (universal root):
  - new_joint_vecs/et_<scene>.npy   (T, 9): [rot6d(6), tx, ty, tz]
  - texts/et_<scene>.txt
  - untagged_text/et_<scene>.txt
  - metadata/et_<scene>.json
  - source_splits/et_{train,val,test}.txt
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from utils.camera_geometry import matrix_to_sixd, validate_rotation


def read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    with open(path, "r") as f:
        return [ln.strip() for ln in f.readlines() if ln.strip()]


def parse_et_pose_file(traj_path: Path) -> np.ndarray:
    """
    Parse ET trajectory text file and return 9D [rot6d, position] in T_wc frame.

    ET line format (w2c):
      R11 R12 R13 tx R21 R22 R23 ty R31 R32 R33 tz
    """
    rows = []
    with open(traj_path, "r") as f:
        for ln in f:
            vals = ln.strip().split()
            if len(vals) != 12:
                continue
            row = np.array([float(v) for v in vals], dtype=np.float64)
            block = row.reshape(3, 4)  # w2c

            r_w2c = block[:, :3]
            t_w2c = block[:, 3]

            # Convert w2c -> c2w (T_wc), consistent with OpenGL/T_wc pipeline.
            r_wc = r_w2c.T
            t_wc = -r_w2c.T @ t_w2c
            r_wc = validate_rotation(r_wc)

            rot6d = matrix_to_sixd(r_wc[None, ...]).reshape(-1)  # (6,)
            rows.append(np.concatenate([rot6d, t_wc], axis=0))

    if not rows:
        return np.empty((0, 9), dtype=np.float32)
    return np.stack(rows, axis=0).astype(np.float32)


def load_et_caption(caption_path: Path) -> str:
    lines = read_lines(caption_path)
    if not lines:
        return ""
    # Keep first non-empty line to match single-caption training usage.
    return lines[0]


def load_et_split_stems(et_root: Path) -> Dict[str, List[str]]:
    return {
        "train": read_lines(et_root / "clatr_train_split.txt"),
        "val": read_lines(et_root / "clatr_val_split.txt"),
        "test": read_lines(et_root / "clatr_test_split.txt"),
    }


def ensure_output_dirs(out_root: Path):
    for sub in ("new_joint_vecs", "texts", "untagged_text", "metadata", "source_splits"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)


def write_caption_files(out_root: Path, sid: str, caption: str):
    for sub in ("texts", "untagged_text"):
        with open(out_root / sub / f"{sid}.txt", "w") as f:
            f.write(caption + "\n")


def upsert_lines(path: Path, new_items: List[str]):
    existing = read_lines(path)
    seen = set(existing)
    merged = existing[:]
    for item in new_items:
        if item not in seen:
            merged.append(item)
            seen.add(item)
    with open(path, "w") as f:
        for item in merged:
            f.write(item + "\n")


def main():
    parser = argparse.ArgumentParser(description="Convert ET data to universal 9D+text format")
    parser.add_argument("--et_root", type=str, default="/data4/haozhe/CamTraj/data/ET/et-data")
    parser.add_argument("--out_root", type=str, default="./dataset/RE10K_ET_GenDoP")
    parser.add_argument("--caption_subdir", type=str, default="caption",
                        help="ET caption folder name under et_root")
    parser.add_argument("--min_frames", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--prefix", type=str, default="et")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--append_global_splits", action="store_true",
                        help="Append converted IDs into out_root/{train,val,test}.txt")
    args = parser.parse_args()

    et_root = Path(args.et_root)
    out_root = Path(args.out_root)
    traj_dir = et_root / "traj"
    caption_dir = et_root / args.caption_subdir

    if not traj_dir.exists():
        raise FileNotFoundError(f"ET traj dir not found: {traj_dir}")
    if not caption_dir.exists():
        raise FileNotFoundError(f"ET caption dir not found: {caption_dir}")

    ensure_output_dirs(out_root)

    split_stems = load_et_split_stems(et_root)
    ordered_stems = split_stems["train"] + split_stems["val"] + split_stems["test"]

    if not ordered_stems:
        ordered_stems = sorted([p.stem for p in traj_dir.glob("*.txt")])

    converted_ids: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    processed = 0
    skipped = 0

    train_set = set(split_stems["train"])
    val_set = set(split_stems["val"])
    test_set = set(split_stems["test"])

    for stem in tqdm(ordered_stems, desc="Converting ET"):
        if processed >= args.max_samples:
            break

        traj_path = traj_dir / f"{stem}.txt"
        cap_path = caption_dir / f"{stem}.txt"
        if not traj_path.exists() or not cap_path.exists():
            skipped += 1
            continue

        sid = f"{args.prefix}_{stem}"
        out_npy = out_root / "new_joint_vecs" / f"{sid}.npy"
        if out_npy.exists() and not args.overwrite:
            # Still register split for bookkeeping.
            if stem in train_set:
                converted_ids["train"].append(sid)
            elif stem in val_set:
                converted_ids["val"].append(sid)
            elif stem in test_set:
                converted_ids["test"].append(sid)
            skipped += 1
            continue

        traj9 = parse_et_pose_file(traj_path)
        if len(traj9) < args.min_frames:
            skipped += 1
            continue

        caption = load_et_caption(cap_path)
        if not caption:
            skipped += 1
            continue

        np.save(out_npy, traj9)
        write_caption_files(out_root, sid, caption)

        meta = {
            "id": sid,
            "source": "ET",
            "original_id": stem,
            "traj_file": str(traj_path),
            "caption_file": str(cap_path),
            "num_frames": int(traj9.shape[0]),
            "feature_dim": 9,
            "pose_semantics": "T_wc (converted from ET w2c)",
        }
        with open(out_root / "metadata" / f"{sid}.json", "w") as f:
            json.dump(meta, f, indent=2)

        if stem in train_set:
            converted_ids["train"].append(sid)
        elif stem in val_set:
            converted_ids["val"].append(sid)
        elif stem in test_set:
            converted_ids["test"].append(sid)
        else:
            # If split files are unavailable/incomplete, place in train by default.
            converted_ids["train"].append(sid)

        processed += 1

    # Write ET source split files
    for split_name in ("train", "val", "test"):
        upsert_lines(out_root / "source_splits" / f"et_{split_name}.txt", converted_ids[split_name])

    # Optionally merge into global train/val/test files.
    if args.append_global_splits:
        for split_name in ("train", "val", "test"):
            upsert_lines(out_root / f"{split_name}.txt", converted_ids[split_name])

    print("\nET conversion complete")
    print(f"  processed: {processed}")
    print(f"  skipped:   {skipped}")
    print(f"  out_root:  {out_root}")


if __name__ == "__main__":
    main()

