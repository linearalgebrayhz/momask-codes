"""
Camera process utilities.

Provides camera-specific metric computation used by camera_eval.py.
Format handling is delegated to unified_data_format.py.
"""

import numpy as np

from .unified_data_format import UnifiedCameraData


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def orientation_to_vector(orientations: np.ndarray) -> np.ndarray:
    """Convert pitch/yaw angles to unit direction vectors.

    Args:
        orientations: (..., 2+) array where dim-0 = pitch, dim-1 = yaw.

    Returns:
        Unit direction vectors (..., 3).
    """
    pitch = orientations[..., 0]
    yaw = orientations[..., 1]

    x = np.cos(pitch) * np.sin(yaw)
    y = -np.sin(pitch)
    z = np.cos(pitch) * np.cos(yaw)

    return np.stack([x, y, z], axis=-1)


def calculate_trajectory_smoothness(
    positions: np.ndarray, orientations: np.ndarray
) -> float:
    """Smoothness score based on acceleration magnitude (lower = smoother).

    Args:
        positions: (T, 3) camera positions.
        orientations: (T, 2+) camera orientations.

    Returns:
        Combined positional + orientational acceleration magnitude.
    """
    if positions.shape[0] < 3:
        return float("inf")

    pos_accel = np.diff(positions, n=2, axis=0)
    pos_smoothness = np.mean(np.linalg.norm(pos_accel, axis=-1))

    ori_accel = np.diff(orientations, n=2, axis=0)
    ori_smoothness = np.mean(np.linalg.norm(ori_accel, axis=-1))

    return pos_smoothness + ori_smoothness


def calculate_camera_metrics(
    pred_data: np.ndarray, gt_data: np.ndarray
) -> dict:
    """Calculate camera-specific evaluation metrics.

    Supports all formats handled by UnifiedCameraData (10D quat, 12D rotmat, etc.).

    Args:
        pred_data: (batch, seq_len, features) predicted camera data.
        gt_data:   (batch, seq_len, features) ground-truth camera data.

    Returns:
        Dictionary of metric values.
    """
    # Collapse batch dimension if present (UnifiedCameraData expects 2D)
    pred_seq = pred_data[0] if pred_data.ndim == 3 else pred_data
    gt_seq = gt_data[0] if gt_data.ndim == 3 else gt_data

    pred_unified = UnifiedCameraData(pred_seq)
    gt_unified = UnifiedCameraData(gt_seq)

    pred_pos = pred_unified.positions.numpy()
    gt_pos = gt_unified.positions.numpy()
    pred_ori = pred_unified.orientations.numpy()
    gt_ori = gt_unified.orientations.numpy()

    # Position error
    pos_error = np.linalg.norm(pred_pos - gt_pos, axis=-1)

    # Orientation error via direction-vector angular distance
    pred_vec = orientation_to_vector(pred_ori)
    gt_vec = orientation_to_vector(gt_ori)
    dot = np.clip(np.sum(pred_vec * gt_vec, axis=-1), -1.0, 1.0)
    angle_error = np.arccos(dot)

    # Smoothness
    pred_smoothness = calculate_trajectory_smoothness(pred_pos, pred_ori)
    gt_smoothness = calculate_trajectory_smoothness(gt_pos, gt_ori)

    # Velocity error (finite-difference)
    vel_error = np.mean(
        np.linalg.norm(np.diff(pred_pos, axis=0) - np.diff(gt_pos, axis=0), axis=-1)
    )

    metrics = {
        "mean_position_error": float(np.mean(pos_error)),
        "mean_orientation_error": float(np.mean(angle_error)),
        "pred_smoothness": pred_smoothness,
        "gt_smoothness": gt_smoothness,
        "velocity_error": vel_error,
        "format": gt_unified.num_features,
    }

    # Direct velocity metrics when available (10D/12D)
    pred_vel = pred_unified.velocities
    gt_vel = gt_unified.velocities
    if pred_vel is not None and gt_vel is not None:
        pv = pred_vel.numpy()
        gv = gt_vel.numpy()
        vel_err = np.linalg.norm(pv - gv, axis=-1)
        metrics["direct_velocity_error"] = float(np.mean(vel_err))
        metrics["direct_velocity_error_std"] = float(np.std(vel_err))

    return metrics
