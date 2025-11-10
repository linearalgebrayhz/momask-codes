#!/usr/bin/env python3
"""Camera Motion Sanity Checker

Given a RealEstate10K camera parameter file, this script:
- Parses poses (same assumptions as RealEstate10KProcessor)
- Computes cumulative translation vector relative to first frame
- Computes per-frame incremental translations and their sign patterns
- Computes Euler angle deltas (pitch, yaw) and range
- Reports dominant translation axis, direction labels (OpenGL: +X right, +Y up, -Z forward)
- Flags possible left/right inversion or forward/backward confusion
- Prints a compact summary for debugging caption/visualization mismatches.

Usage:
    python utils/camera_sanity_check.py path/to/camera_file.txt --transform relative
"""
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

OPENGL_DOC = "+X right, +Y up, -Z forward (negative Z means forward motion)"

def parse_camera_file(file_path: Path, transform: str = "relative"):
    lines = file_path.read_text().strip().splitlines()
    if not lines:
        raise ValueError("Empty camera file")
    video_url = lines[0]
    first_pose = None
    data = []
    for idx, line in enumerate(lines[1:], start=1):
        parts = line.split()
        if len(parts) < 19:
            continue
        pose_vals = [float(v) for v in parts[7:]]
        pose_mat = np.array(pose_vals).reshape(3,4)
        if first_pose is None:
            first_pose = pose_mat.copy()
            if transform == "relative":
                rel_pose = np.concatenate([np.eye(3), np.zeros((3,1))], axis=1)
                data.append(rel_pose)
                continue  # skip using first frame absolute again
        if transform == "relative":
            abs_rot = pose_mat[:,:3]
            abs_trans = pose_mat[:,3]
            base_rot = first_pose[:,:3]
            base_trans = first_pose[:,3]
            rel_rot = base_rot.T @ abs_rot
            rel_trans = base_rot.T @ (abs_trans - base_trans)
            pose_mat = np.concatenate([rel_rot, rel_trans.reshape(3,1)], axis=1)
        data.append(pose_mat)
    poses = np.stack(data, axis=0)  # (N,3,4)
    return video_url, poses

def rotation_to_euler_xyz(rot_mats):
    eulers = []
    for m in rot_mats:
        r = R.from_matrix(m)
        # scipy xyz returns intrinsic rotations about x,y,z: treat as roll,pitch,yaw then reorder
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)
        eulers.append([pitch, yaw, roll])  # store [pitch,yaw,roll]
    return np.array(eulers)

def analyze(poses):
    translations = poses[:,:3,3]
    rotations = poses[:,:3,:3]
    eulers = rotation_to_euler_xyz(rotations)  # (N,3) pitch,yaw,roll

    total = translations[-1] - translations[0]
    deltas = np.diff(translations, axis=0)
    step_sign = np.sign(deltas)
    euler_total = eulers[-1] - eulers[0]

    # Dominant translation axis
    abs_total = np.abs(total)
    dominant_axis = np.argmax(abs_total)
    axis_names = ['X(lateral)','Y(vertical)','Z(depth)']
    axis_label = axis_names[dominant_axis]

    # Direction label (OpenGL semantics)
    if dominant_axis == 0:
        dir_label = 'right' if total[0] > 0 else 'left'
    elif dominant_axis == 1:
        dir_label = 'up' if total[1] > 0 else 'down'
    else:
        dir_label = 'forward' if total[2] < 0 else 'backward'

    # Heuristic: check inversion suspicion
    inversion_flags = []
    # If lateral movement caption says right but total[0]<0, etc. (placeholder; real caption cross-check happens externally)
    if abs_total[0] > 0.05 and abs_total[2] < 0.02:
        # Predominantly lateral; verify sign consistency by majority of per-step signs
        lateral_majority = np.mean(step_sign[:,0] > 0)  # fraction of positive steps
        if (total[0] > 0 and lateral_majority < 0.3) or (total[0] < 0 and lateral_majority > 0.7):
            inversion_flags.append('Lateral direction inconsistent between cumulative and step-wise increments')

    # Pitch/Yaw analysis
    pitch_range = np.ptp(eulers[:,0])
    yaw_range = np.ptp(eulers[:,1])

    # Build report
    report = {
        'num_frames': int(poses.shape[0]),
        'total_translation': total.tolist(),
        'dominant_axis': axis_label,
        'dominant_direction_label': dir_label,
        'translation_magnitude': float(np.linalg.norm(total)),
        'pitch_total_change': float(euler_total[0]),
        'yaw_total_change': float(euler_total[1]),
        'pitch_range': float(pitch_range),
        'yaw_range': float(yaw_range),
        'potential_inversions': inversion_flags,
        'opengl_convention': OPENGL_DOC
    }
    return report

def main():
    ap = argparse.ArgumentParser(description='Sanity check a RealEstate10K camera file')
    ap.add_argument('camera_file')
    ap.add_argument('--transform', default='relative', choices=['relative','absolute'], help='Interpretation mode matching dataset processing')
    args = ap.parse_args()
    camera_path = Path(args.camera_file)
    if not camera_path.exists():
        print(f"Error: file not found {camera_path}")
        return 1
    try:
        _, poses = parse_camera_file(camera_path, transform=args.transform)
        rep = analyze(poses)
        print("=== Camera Motion Sanity Report ===")
        for k,v in rep.items():
            print(f"{k}: {v}")
    except Exception as e:
        print(f"Failed to analyze {camera_path}: {e}")
        return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
