#!/usr/bin/env python3
"""
Convert 6D camera trajectories to 12D by adding velocity information

Reads existing 6D processed data and adds derivatives (dx, dy, dz, dpitch, dyaw, droll)
using timestamps from the original RealEstate10K data.

6D format: [x, y, z, pitch, yaw, roll]
12D format: [x, y, z, dx, dy, dz, pitch, yaw, roll, dpitch, dyaw, droll]
"""

import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil


def parse_camera_file_timestamps(file_path: str):
    """
    Parse timestamps from RealEstate10K camera file
    Returns list of timestamps in seconds
    """
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    timestamps = []
    for line in lines[1:]:  # Skip URL line
        values = line.split()
        if len(values) >= 19:
            timestamp_us = int(float(values[0]))
            timestamp_s = timestamp_us / 1_000_000.0
            timestamps.append(timestamp_s)
    
    return np.array(timestamps)


def compute_velocities(trajectory_6d: np.ndarray, timestamps: np.ndarray):
    """
    Compute velocities from 6D trajectory and timestamps
    
    Args:
        trajectory_6d: (seq_len, 6) array [x, y, z, pitch, yaw, roll]
        timestamps: (seq_len,) array of timestamps in seconds
            Note: If trajectory is in relative coordinates (starts with identity frame),
            timestamps should already include the prepended first timestamp
    
    Returns:
        trajectory_12d: (seq_len, 12) array [x, y, z, dx, dy, dz, pitch, yaw, roll, dpitch, dyaw, droll]
    """
    seq_len = trajectory_6d.shape[0]
    
    if seq_len != len(timestamps):
        raise ValueError(f"Trajectory length {seq_len} doesn't match timestamps length {len(timestamps)}")
    
    # Compute time differences
    dt = np.diff(timestamps)
    
    # For the first frame, use the same dt as between frames 1-2
    # (This avoids dt=0 when first timestamp is duplicated for identity frame)
    if dt[0] == 0 and len(dt) > 1:
        dt[0] = dt[1]
    
    # Prepend first dt for velocity computation
    dt = np.concatenate([[dt[0] if len(dt) > 0 else 1.0/30], dt])
    dt[dt == 0] = 1e-6  # Avoid any remaining division by zero
    
    # Split position and orientation
    positions = trajectory_6d[:, :3]  # [x, y, z]
    orientations = trajectory_6d[:, 3:]  # [pitch, yaw, roll]
    
    # Compute velocities (forward differences)
    position_diffs = np.diff(positions, axis=0)
    position_velocities = position_diffs / dt[1:, np.newaxis]
    # Use first velocity for first frame
    position_velocities = np.vstack([position_velocities[0:1], position_velocities])
    
    orientation_diffs = np.diff(orientations, axis=0)
    orientation_velocities = orientation_diffs / dt[1:, np.newaxis]
    # Use first velocity for first frame
    orientation_velocities = np.vstack([orientation_velocities[0:1], orientation_velocities])
    
    # Construct 12D trajectory
    trajectory_12d = np.concatenate([
        positions,              # x, y, z
        position_velocities,    # dx, dy, dz
        orientations,           # pitch, yaw, roll
        orientation_velocities  # dpitch, dyaw, droll
    ], axis=1)
    
    return trajectory_12d


def convert_dataset(
    input_dir: str,
    output_dir: str,
    original_data_dir: str,
    scene_mapping_file: str
):
    """
    Convert 6D dataset to 12D dataset
    
    Args:
        input_dir: Directory containing 6D data
        output_dir: Directory to write 12D data
        original_data_dir: Directory containing original RealEstate10K data (test/train folders)
        scene_mapping_file: JSON file mapping numeric IDs to original hex IDs
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Load scene ID mapping
    print(f"Loading scene ID mapping from {scene_mapping_file}...")
    with open(scene_mapping_file, 'r') as f:
        scene_mapping = json.load(f)
    
    print(f"Found {len(scene_mapping)} scenes in mapping")
    
    # Create output directories
    motion_output_dir = output_path / "new_joint_vecs"
    motion_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy text captions and other files
    print("Copying text captions and metadata...")
    for dir_name in ['texts', 'untagged_text', 'metadata']:
        src_dir = input_path / dir_name
        dst_dir = output_path / dir_name
        if src_dir.exists():
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)
            print(f"  Copied {dir_name}/")
    
    # Copy split files
    for split_file in ['train.txt', 'val.txt', 'test.txt']:
        src = input_path / split_file
        dst = output_path / split_file
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Copied {split_file}")
    
    # Copy scene mapping
    dst_mapping = output_path / "scene_id_mapping.json"
    shutil.copy2(scene_mapping_file, dst_mapping)
    print(f"  Copied scene_id_mapping.json")
    
    # Process motion files
    motion_input_dir = input_path / "new_joint_vecs"
    motion_files = sorted(motion_input_dir.glob("*.npy"))
    
    print(f"\nProcessing {len(motion_files)} motion files...")
    
    stats = {
        'processed': 0,
        'failed': 0,
        'not_found': 0
    }
    
    for motion_file in tqdm(motion_files, desc="Converting 6D → 12D"):
        scene_id = motion_file.stem  # e.g., "000000"
        
        try:
            # Get original scene ID
            if scene_id not in scene_mapping:
                print(f"  ⚠️  Scene {scene_id} not in mapping, skipping")
                stats['not_found'] += 1
                continue
            
            original_id = scene_mapping[scene_id]
            
            # Find original camera file (try test and train folders)
            original_file = None
            for split in ['test', 'train']:
                candidate = Path(original_data_dir) / split / f"{original_id}.txt"
                if candidate.exists():
                    original_file = candidate
                    break
            
            if original_file is None:
                print(f"  ⚠️  Original file not found for {scene_id} ({original_id}), skipping")
                stats['not_found'] += 1
                continue
            
            # Load 6D trajectory
            trajectory_6d = np.load(motion_file)
            
            # Parse timestamps from original file
            timestamps = parse_camera_file_timestamps(str(original_file))
            
            # Handle relative coordinate case: processor adds identity frame at start
            # So trajectory has N+1 frames but original file has N frames
            if len(trajectory_6d) == len(timestamps) + 1:
                # Prepend first timestamp for the identity frame
                timestamps = np.concatenate([[timestamps[0]], timestamps])
            
            # Check length consistency
            if len(trajectory_6d) != len(timestamps):
                print(f"  ⚠️  Length mismatch for {scene_id}: trajectory={len(trajectory_6d)}, timestamps={len(timestamps)}")
                stats['failed'] += 1
                continue
            
            # Compute 12D trajectory with velocities
            trajectory_12d = compute_velocities(trajectory_6d, timestamps)
            
            # Save 12D trajectory
            output_file = motion_output_dir / f"{scene_id}.npy"
            np.save(output_file, trajectory_12d)
            
            stats['processed'] += 1
            
        except Exception as e:
            print(f"  ✗ Error processing {scene_id}: {e}")
            stats['failed'] += 1
            import traceback
            traceback.print_exc()
    
    # Calculate and save dataset statistics for 12D data
    print("\nCalculating dataset statistics...")
    all_data = []
    for scene_id in tqdm(scene_mapping.keys(), desc="Loading trajectories"):
        motion_file = motion_output_dir / f"{scene_id}.npy"
        if motion_file.exists():
            data = np.load(motion_file)
            all_data.append(data)
    
    if all_data:
        all_data = np.concatenate(all_data, axis=0)
        mean = np.mean(all_data, axis=0)
        std = np.std(all_data, axis=0)
        
        np.save(output_path / "Mean.npy", mean)
        np.save(output_path / "Std.npy", std)
        
        print(f"\nDataset statistics (12D):")
        print(f"  Mean shape: {mean.shape}")
        print(f"  Std shape: {std.shape}")
        print(f"  Mean values: {mean}")
        print(f"  Std values: {std}")
    
    print(f"\n" + "="*60)
    print(f"Conversion completed!")
    print(f"  Processed: {stats['processed']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Not found: {stats['not_found']}")
    print(f"  Total: {len(motion_files)}")
    print(f"="*60)
    
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert 6D camera trajectories to 12D")
    parser.add_argument("--input-dir", default="/home/haozhe/CamTraj/momask-codes/dataset/RealEstate10K_6feat",
                       help="Input directory with 6D data")
    parser.add_argument("--output-dir", default="/home/haozhe/CamTraj/momask-codes/dataset/RealEstate10K_12feat",
                       help="Output directory for 12D data")
    parser.add_argument("--original-data", default="/data5/haozhe/CamTraj/data/real-state-10k/RealEstate10K",
                       help="Directory containing original RealEstate10K data (with test/ and train/ folders)")
    parser.add_argument("--scene-mapping", 
                       default="/home/haozhe/CamTraj/momask-codes/dataset/RealEstate10K_6feat/scene_id_mapping.json",
                       help="Scene ID mapping JSON file")
    
    args = parser.parse_args()
    
    print("="*60)
    print("6D → 12D Camera Trajectory Conversion")
    print("="*60)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Original data: {args.original_data}")
    print(f"Scene mapping: {args.scene_mapping}")
    print("="*60)
    print()
    
    stats = convert_dataset(
        args.input_dir,
        args.output_dir,
        args.original_data,
        args.scene_mapping
    )


if __name__ == "__main__":
    main()
