import numpy as np
import torch
from os.path import join as pjoin
import math

def process_camera_file(positions, orientations):
    """
    Process camera trajectory data
    
    Args:
        positions: (seq_len, 3) - camera positions [x, y, z]
        orientations: (seq_len, 2) - camera orientations [pitch, yaw] in radians
    
    Returns:
        data: (seq_len, 5) - processed camera data [x, y, z, pitch, yaw]
    """
    seq_len = positions.shape[0]
    
    # Ensure orientations are in valid ranges
    orientations[:, 0] = np.clip(orientations[:, 0], -np.pi/2, np.pi/2)  # pitch: [-90, 90] degrees
    orientations[:, 1] = np.mod(orientations[:, 1], 2*np.pi)  # yaw: [0, 360] degrees
    
    # Calculate velocities
    if seq_len > 1:
        pos_velocity = positions[1:] - positions[:-1]  # (seq_len-1, 3)
        ori_velocity = orientations[1:] - orientations[:-1]  # (seq_len-1, 2)
        
        # Handle yaw wrapping for velocity calculation
        ori_velocity[:, 1] = np.mod(ori_velocity[:, 1] + np.pi, 2*np.pi) - np.pi
        
        # Pad to match original length
        pos_velocity = np.concatenate([np.zeros((1, 3)), pos_velocity], axis=0)
        ori_velocity = np.concatenate([np.zeros((1, 2)), ori_velocity], axis=0)
    else:
        pos_velocity = np.zeros((seq_len, 3))
        ori_velocity = np.zeros((seq_len, 2))
    
    # Combine all features
    data = np.concatenate([
        positions,  # (seq_len, 3) - x, y, z
        orientations,  # (seq_len, 2) - pitch, yaw
    ], axis=-1)
    
    return data

def recover_from_camera_data(data):
    """
    Recover camera positions and orientations from processed data
    
    Args:
        data: (seq_len, 5) - processed camera data [x, y, z, pitch, yaw]
    
    Returns:
        positions: (seq_len, 3) - camera positions
        orientations: (seq_len, 2) - camera orientations
    """
    positions = data[..., :3]  # x, y, z
    orientations = data[..., 3:]  # pitch, yaw
    
    return positions, orientations

def calculate_camera_metrics(pred_data, gt_data):
    """
    Calculate camera-specific evaluation metrics
    
    Args:
        pred_data: (batch, seq_len, 5) - predicted camera data
        gt_data: (batch, seq_len, 5) - ground truth camera data
    
    Returns:
        metrics: dict with various camera metrics
    """
    pred_pos, pred_ori = recover_from_camera_data(pred_data)
    gt_pos, gt_ori = recover_from_camera_data(gt_data)
    
    # Handle batch dimension - take the first (and should be only) batch
    if pred_pos.ndim == 3:  # (batch, seq_len, 3)
        pred_pos = pred_pos[0]  # (seq_len, 3)
        pred_ori = pred_ori[0]  # (seq_len, 2)
        gt_pos = gt_pos[0]      # (seq_len, 3)
        gt_ori = gt_ori[0]      # (seq_len, 2)
    
    # Position error (Euclidean distance)
    pos_error = np.sqrt(np.sum((pred_pos - gt_pos) ** 2, axis=-1))
    mean_pos_error = np.mean(pos_error)
    
    # Orientation error (angular distance)
    # Convert to unit vectors and calculate angle
    pred_vectors = orientation_to_vector(pred_ori)
    gt_vectors = orientation_to_vector(gt_ori)
    
    # Calculate angle between vectors
    dot_product = np.sum(pred_vectors * gt_vectors, axis=-1)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle_error = np.arccos(dot_product)
    mean_angle_error = np.mean(angle_error)
    
    # Smoothness metrics
    pred_smoothness = calculate_trajectory_smoothness(pred_pos, pred_ori)
    gt_smoothness = calculate_trajectory_smoothness(gt_pos, gt_ori)
    
    # Velocity consistency
    pred_velocity = np.diff(pred_pos, axis=0)
    gt_velocity = np.diff(gt_pos, axis=0)
    velocity_error = np.mean(np.sqrt(np.sum((pred_velocity - gt_velocity) ** 2, axis=-1)))
    
    return {
        'mean_position_error': mean_pos_error,
        'mean_orientation_error': mean_angle_error,
        'pred_smoothness': pred_smoothness,
        'gt_smoothness': gt_smoothness,
        'velocity_error': velocity_error
    }

def orientation_to_vector(orientations):
    """
    Convert pitch, yaw to unit direction vector
    
    Args:
        orientations: (..., 2) - pitch, yaw in radians
    
    Returns:
        vectors: (..., 3) - unit direction vectors
    """
    pitch, yaw = orientations[..., 0], orientations[..., 1]
    
    # Convert to unit vector
    x = np.cos(pitch) * np.sin(yaw)
    y = -np.sin(pitch)
    z = np.cos(pitch) * np.cos(yaw)
    
    vectors = np.stack([x, y, z], axis=-1)
    return vectors

def calculate_trajectory_smoothness(positions, orientations):
    """
    Calculate trajectory smoothness based on acceleration
    
    Args:
        positions: (seq_len, 3) - camera positions
        orientations: (seq_len, 2) - camera orientations
    
    Returns:
        smoothness: float - smoothness score (lower is smoother)
    """
    seq_len = positions.shape[0]
    
    # Handle edge cases
    if seq_len < 3:
        # For sequences too short to calculate acceleration, return a high penalty
        # This prevents the metric from being misleadingly low (0.0)
        return float('inf')  # or a large penalty value like 999.0
    
    # Position acceleration
    pos_velocity = np.diff(positions, axis=0)
    pos_acceleration = np.diff(pos_velocity, axis=0)
    pos_smoothness = np.mean(np.sqrt(np.sum(pos_acceleration ** 2, axis=-1)))
    
    # Orientation acceleration
    ori_velocity = np.diff(orientations, axis=0)
    ori_acceleration = np.diff(ori_velocity, axis=0)
    ori_smoothness = np.mean(np.sqrt(np.sum(ori_acceleration ** 2, axis=-1)))
    
    return pos_smoothness + ori_smoothness

def normalize_camera_data(data, mean=None, std=None):
    """
    Normalize camera data using z-score normalization
    
    Args:
        data: (seq_len, 5) - camera data
        mean: (5,) - mean values for normalization
        std: (5,) - standard deviation values for normalization
    
    Returns:
        normalized_data: (seq_len, 5) - normalized data
        mean: (5,) - computed mean
        std: (5,) - computed standard deviation
    """
    if mean is None or std is None:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        # Avoid division by zero
        std = np.where(std == 0, 1.0, std)
    
    normalized_data = (data - mean) / std
    return normalized_data, mean, std

def denormalize_camera_data(normalized_data, mean, std):
    """
    Denormalize camera data
    
    Args:
        normalized_data: (seq_len, 5) - normalized data
        mean: (5,) - mean values
        std: (5,) - standard deviation values
    
    Returns:
        data: (seq_len, 5) - denormalized data
    """
    return normalized_data * std + mean 