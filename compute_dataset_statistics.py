#!/usr/bin/env python3
"""
Dataset Statistics Calculator for RealEstate10K Camera Trajectory Dataset

Computes statistics aligned with the actual processing pipeline:
  - Scene/frame counts, duration distributions
  - Motion type distribution (using the same forward-vector analysis
    as realestate10k_processor._generate_guidance())
  - Vocabulary richness metrics
  - Caption quality metrics

Supports both 12D rotmat and 10D quaternion formats.

Usage:
    python compute_dataset_statistics.py ./dataset/RealEstate10K_rotmat
    python compute_dataset_statistics.py ./dataset/RealEstate10K_rotmat --format rotmat --output stats.json
    python compute_dataset_statistics.py ./dataset/RealEstate10K_rotmat --text-dir untagged_text
"""

import sys
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from utils.unified_data_format import CameraDataFormat, detect_format_from_dataset_name
from utils.camera_geometry import forward_from_sixd


# ============================================================================
# Motion Classification (mirrors _generate_guidance in realestate10k_processor)
# ============================================================================

# Thresholds — same as realestate10k_processor._generate_guidance
THRESHOLD_TRANSLATION = 0.10      # minimum total displacement to count
THRESHOLD_DOMINANCE = 0.6         # axis must be >= 60% of max displacement
THRESHOLD_YAW = 0.12              # radians
THRESHOLD_PITCH = 0.10            # radians


def _quat_to_forward(quat: np.ndarray) -> np.ndarray:
    """Compute camera forward vector from quaternion [qw, qx, qy, qz].

    Converts to rotation matrix, then uses -col2 (OpenGL forward).
    """
    from scipy.spatial.transform import Rotation as R
    # scipy expects [qx, qy, qz, qw]
    quat_scipy = np.concatenate([quat[1:4], quat[0:1]])
    rot_mat = R.from_quat(quat_scipy).as_matrix()  # (3, 3)
    return -rot_mat[:, 2]  # forward = -col2


def classify_motion(trajectory: np.ndarray,
                    fmt: CameraDataFormat) -> Dict[str, bool]:
    """Classify camera motion using the same logic as the data processor.

    Translation: OpenGL convention (+X right, +Y up, -Z forward).
    Rotation: forward-vector yaw/pitch analysis (no Euler conversion).

    Returns dict of motion flags aligned with guidance labels:
        tracks_left, tracks_right, moves_up, moves_down,
        dollies_forward, dollies_backward,
        pans_left, pans_right, tilts_up, tilts_down, static
    """
    positions = trajectory[:, :3]

    # Extract forward vector for first and last frame
    if fmt == CameraDataFormat.FULL_12_ROTMAT:
        fwd0 = forward_from_sixd(trajectory[0, 6:12])
        fwd1 = forward_from_sixd(trajectory[-1, 6:12])
    elif fmt == CameraDataFormat.QUATERNION_10:
        fwd0 = _quat_to_forward(trajectory[0, 6:10])
        fwd1 = _quat_to_forward(trajectory[-1, 6:10])
    else:
        raise ValueError(f"Unsupported format for motion classification: {fmt.name}")

    flags = {
        "static": False,
        "dollies_forward": False,
        "dollies_backward": False,
        "tracks_left": False,
        "tracks_right": False,
        "moves_up": False,
        "moves_down": False,
        "pans_left": False,
        "pans_right": False,
        "tilts_up": False,
        "tilts_down": False,
    }

    # --- Translation (same as _generate_guidance) ---
    total_t = positions[-1] - positions[0]
    abs_t = np.abs(total_t)
    max_t = float(np.max(abs_t))

    has_translation = False
    if max_t > THRESHOLD_TRANSLATION:
        # X: right / left
        if abs_t[0] > THRESHOLD_TRANSLATION and abs_t[0] >= THRESHOLD_DOMINANCE * max_t:
            has_translation = True
            if total_t[0] > 0:
                flags["tracks_right"] = True
            else:
                flags["tracks_left"] = True
        # Y: up / down
        if abs_t[1] > THRESHOLD_TRANSLATION and abs_t[1] >= THRESHOLD_DOMINANCE * max_t:
            has_translation = True
            if total_t[1] > 0:
                flags["moves_up"] = True
            else:
                flags["moves_down"] = True
        # Z: forward / backward (forward = -Z in OpenGL)
        if abs_t[2] > THRESHOLD_TRANSLATION and abs_t[2] >= THRESHOLD_DOMINANCE * max_t:
            has_translation = True
            if total_t[2] < 0:
                flags["dollies_forward"] = True
            else:
                flags["dollies_backward"] = True

    # --- Rotation (forward-vector method, same as _generate_guidance) ---
    yaw0 = np.arctan2(fwd0[0], -fwd0[2])
    yaw1 = np.arctan2(fwd1[0], -fwd1[2])
    dyaw = np.arctan2(np.sin(yaw1 - yaw0), np.cos(yaw1 - yaw0))  # wrap to [-π, π]

    pitch0 = np.arcsin(np.clip(fwd0[1], -1, 1))
    pitch1 = np.arcsin(np.clip(fwd1[1], -1, 1))
    dpitch = pitch1 - pitch0

    has_rotation = False
    if abs(dyaw) > THRESHOLD_YAW:
        has_rotation = True
        if dyaw > 0:
            flags["pans_right"] = True
        else:
            flags["pans_left"] = True
    if abs(dpitch) > THRESHOLD_PITCH:
        has_rotation = True
        if dpitch > 0:
            flags["tilts_up"] = True
        else:
            flags["tilts_down"] = True

    # Static only if no translation AND no rotation detected
    if not has_translation and not has_rotation:
        flags["static"] = True

    return flags


# ============================================================================
# Vocabulary Analysis
# ============================================================================

def analyze_vocabulary(captions: List[str]) -> Dict:
    """Analyze vocabulary richness and diversity."""
    vocabulary = Counter()
    caption_lengths = []

    for caption in captions:
        words = re.findall(r'\b[a-z]+\b', caption.lower())
        vocabulary.update(words)
        caption_lengths.append(len(words))

    total_tokens = sum(vocabulary.values())
    unique_tokens = len(vocabulary)

    # Type-Token Ratio
    ttr = unique_tokens / total_tokens if total_tokens > 0 else 0

    # Root TTR (more stable for large corpora)
    rttr = unique_tokens / np.sqrt(total_tokens) if total_tokens > 0 else 0

    # Moving-Average Type-Token Ratio (100-word windows)
    all_words = []
    for caption in captions:
        all_words.extend(re.findall(r'\b[a-z]+\b', caption.lower()))

    window_size = 100
    if len(all_words) >= window_size:
        mattr_scores = [
            len(set(all_words[i:i + window_size])) / window_size
            for i in range(len(all_words) - window_size + 1)
        ]
        mattr = float(np.mean(mattr_scores))
    else:
        mattr = ttr

    return {
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "type_token_ratio": round(ttr, 4),
        "root_ttr": round(rttr, 4),
        "mattr": round(mattr, 4),
        "avg_caption_length": round(float(np.mean(caption_lengths)), 1) if caption_lengths else 0,
        "median_caption_length": float(np.median(caption_lengths)) if caption_lengths else 0,
        "most_common_words": vocabulary.most_common(20),
    }


# ============================================================================
# Main Statistics Calculator
# ============================================================================

class DatasetStatistics:
    """Compute comprehensive statistics for a processed camera trajectory dataset."""

    def __init__(self, dataset_root: Path, fmt: CameraDataFormat,
                 text_subdir: str = "texts", fps: float = 24.0):
        self.dataset_root = dataset_root
        self.fmt = fmt
        self.fps = fps

        self.motion_dir = dataset_root / "new_joint_vecs"
        self.text_dir = dataset_root / text_subdir

        self.stats: Dict = {}

    def compute(self) -> Dict:
        """Compute all statistics. Returns dict suitable for JSON serialization."""
        print(f"Dataset:  {self.dataset_root}")
        print(f"Format:   {self.fmt.name}")
        print(f"Text dir: {self.text_dir.name}")
        print(f"FPS:      {self.fps}")

        npy_files = sorted(self.motion_dir.glob("*.npy"))
        num_scenes = len(npy_files)
        print(f"\nFound {num_scenes} scenes")
        if num_scenes == 0:
            print("No scenes found, nothing to compute.")
            return {}

        # Accumulators
        seq_lengths: List[int] = []
        captions: List[str] = []
        motion_counts: Dict[str, int] = defaultdict(int)
        combo_counts: Dict[str, int] = defaultdict(int)

        print("Processing scenes...")
        for i, npy_file in enumerate(npy_files):
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{num_scenes}...")

            scene_id = npy_file.stem
            trajectory = np.load(npy_file)
            seq_lengths.append(len(trajectory))

            # Motion classification
            try:
                flags = classify_motion(trajectory, self.fmt)
            except Exception as e:
                print(f"  Warning: motion classification failed for {scene_id}: {e}")
                continue

            active = [k for k, v in flags.items() if v]
            for m in active:
                motion_counts[m] += 1
            if len(active) > 1:
                combo_key = "+".join(sorted(active))
                combo_counts[combo_key] += 1

            # Caption
            txt_file = self.text_dir / f"{scene_id}.txt"
            if txt_file.exists():
                caption = txt_file.read_text().strip().split("#")[0].strip()
                if caption:
                    captions.append(caption)

        print(f"  Completed {num_scenes} scenes")

        seq_arr = np.array(seq_lengths)
        dur_arr = seq_arr / self.fps

        # --- Aggregate stats ---
        self.stats = {
            "dataset_root": str(self.dataset_root),
            "format": self.fmt.name,
            "fps": self.fps,
            "num_scenes": num_scenes,
            "num_captions": len(captions),
            "total_frames": int(seq_arr.sum()),
            "sequence_length": {
                "mean": round(float(seq_arr.mean()), 1),
                "median": float(np.median(seq_arr)),
                "min": int(seq_arr.min()),
                "max": int(seq_arr.max()),
                "std": round(float(seq_arr.std()), 1),
            },
            "duration_seconds": {
                "mean": round(float(dur_arr.mean()), 2),
                "median": round(float(np.median(dur_arr)), 2),
                "min": round(float(dur_arr.min()), 2),
                "max": round(float(dur_arr.max()), 2),
            },
        }

        # --- Motion distribution (percentages) ---
        print("\nComputing motion distribution...")
        basic = {
            k: round(v / num_scenes * 100, 2)
            for k, v in sorted(motion_counts.items(), key=lambda x: -x[1])
        }
        combos = {
            k: round(v / num_scenes * 100, 2)
            for k, v in sorted(combo_counts.items(), key=lambda x: -x[1])[:20]
        }
        self.stats["motion_distribution"] = {
            "counts": dict(motion_counts),
            "percentages": basic,
            "top_combinations": combos,
        }

        # --- Vocabulary ---
        print("Analyzing vocabulary...")
        self.stats["vocabulary"] = analyze_vocabulary(captions)

        return self.stats

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self):
        s = self.stats
        if not s:
            print("No statistics computed yet.")
            return

        print("\n" + "=" * 80)
        print("DATASET STATISTICS REPORT")
        print("=" * 80)

        print(f"\n{'BASIC STATISTICS':^80}")
        print("-" * 80)
        print(f"  Dataset:              {s['dataset_root']}")
        print(f"  Format:               {s['format']}")
        print(f"  Total scenes:         {s['num_scenes']:,}")
        print(f"  Total captions:       {s['num_captions']:,}")
        print(f"  Total frames:         {s['total_frames']:,}")

        sl = s["sequence_length"]
        dur = s["duration_seconds"]
        print(f"\n{'SEQUENCE LENGTH':^80}")
        print("-" * 80)
        print(f"  Mean:   {sl['mean']:.1f} frames  ({dur['mean']:.2f}s)")
        print(f"  Median: {sl['median']:.0f} frames  ({dur['median']:.2f}s)")
        print(f"  Min:    {sl['min']} frames  ({dur['min']:.2f}s)")
        print(f"  Max:    {sl['max']} frames  ({dur['max']:.2f}s)")
        print(f"  Std:    {sl['std']:.1f} frames")
        print(f"  FPS:    {s['fps']}")

        md = s["motion_distribution"]
        pcts = md["percentages"]

        # Separate translation, rotation, static
        trans_keys = ["dollies_forward", "dollies_backward", "tracks_left",
                      "tracks_right", "moves_up", "moves_down"]
        rot_keys = ["pans_left", "pans_right", "tilts_up", "tilts_down"]

        print(f"\n{'MOTION DISTRIBUTION':^80}")
        print("-" * 80)

        trans = {k: pcts[k] for k in trans_keys if k in pcts}
        rot = {k: pcts[k] for k in rot_keys if k in pcts}
        other = {k: pcts[k] for k in pcts if k not in trans_keys and k not in rot_keys}

        if trans:
            print("\n  Translation:")
            for k, v in sorted(trans.items(), key=lambda x: -x[1]):
                cnt = md["counts"].get(k, 0)
                print(f"    {k:25s} {v:6.2f}%  ({cnt:,})")
        if rot:
            print("\n  Rotation:")
            for k, v in sorted(rot.items(), key=lambda x: -x[1]):
                cnt = md["counts"].get(k, 0)
                print(f"    {k:25s} {v:6.2f}%  ({cnt:,})")
        if other:
            print("\n  Other:")
            for k, v in sorted(other.items(), key=lambda x: -x[1]):
                cnt = md["counts"].get(k, 0)
                print(f"    {k:25s} {v:6.2f}%  ({cnt:,})")

        combos = md["top_combinations"]
        if combos:
            print(f"\n{'TOP MOTION COMBINATIONS':^80}")
            print("-" * 80)
            for i, (combo, pct) in enumerate(combos.items(), 1):
                print(f"  {i:2}. {combo:55s} {pct:6.2f}%")

        vocab = s["vocabulary"]
        print(f"\n{'VOCABULARY':^80}")
        print("-" * 80)
        print(f"  Total tokens:   {vocab['total_tokens']:,}")
        print(f"  Unique tokens:  {vocab['unique_tokens']:,}")
        print(f"  TTR:            {vocab['type_token_ratio']:.4f}")
        print(f"  Root TTR:       {vocab['root_ttr']:.4f}")
        print(f"  MATTR:          {vocab['mattr']:.4f}")
        print(f"  Avg caption:    {vocab['avg_caption_length']:.1f} words")
        print(f"  Median caption: {vocab['median_caption_length']:.1f} words")
        print(f"\n  Most common words:")
        for word, count in vocab["most_common_words"][:15]:
            pct = count / vocab["total_tokens"] * 100 if vocab["total_tokens"] > 0 else 0
            print(f"    {word:20s} {count:6,}  ({pct:.2f}%)")

        print("\n" + "=" * 80)

    def save_json(self, output_path: Path):
        """Save statistics to JSON (convert Counter tuples for serialization)."""
        serializable = json.loads(json.dumps(self.stats, default=str))
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nSaved: {output_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute statistics for a processed RealEstate10K dataset",
    )
    parser.add_argument(
        "dataset_root",
        help="Root directory (contains new_joint_vecs/, texts/, metadata/)",
    )
    parser.add_argument(
        "--format",
        choices=["rotmat", "quat", "auto"],
        default="auto",
        help="Data format, or 'auto' to detect from directory name (default: auto)",
    )
    parser.add_argument(
        "--text-dir",
        default="texts",
        help="Subdirectory for captions (default: texts)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Frame rate for duration calculations (default: 24.0)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file (optional)",
    )

    args = parser.parse_args()
    dataset_root = Path(args.dataset_root)

    if not dataset_root.exists():
        print(f"Error: {dataset_root} does not exist")
        return 1
    if not (dataset_root / "new_joint_vecs").is_dir():
        print(f"Error: {dataset_root / 'new_joint_vecs'} not found")
        return 1

    # Resolve format
    if args.format == "auto":
        fmt = detect_format_from_dataset_name(dataset_root.name)
        if fmt is None:
            print("Cannot auto-detect format, defaulting to rotmat")
            fmt = CameraDataFormat.FULL_12_ROTMAT
    else:
        fmt = (CameraDataFormat.FULL_12_ROTMAT if args.format == "rotmat"
               else CameraDataFormat.QUATERNION_10)

    analyzer = DatasetStatistics(dataset_root, fmt, args.text_dir, args.fps)
    analyzer.compute()
    analyzer.print_report()

    if args.output:
        analyzer.save_json(Path(args.output))

    return 0


if __name__ == "__main__":
    sys.exit(main())