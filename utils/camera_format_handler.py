#!/usr/bin/env python3
"""
Camera Format Handler for Multi-Feature Support

This module provides utilities to handle different camera trajectory formats:
- 5-feature: [x, y, z, pitch, yaw] (legacy cam dataset)
- 6-feature: [x, y, z, pitch, yaw, roll] (position + full orientation)  
- 12-feature: [x, y, z, dx, dy, dz, pitch, yaw, roll, dpitch, dyaw, droll] (full dynamics)

Provides format detection, feature extraction, and compatibility functions.
"""

import numpy as np
import torch
from typing import Tuple, Optional, Union
from enum import Enum

class CameraFormat(Enum):
    """Camera trajectory data formats"""
    LEGACY_5 = 5    # [x, y, z, pitch, yaw]
    STANDARD_6 = 6  # [x, y, z, pitch, yaw, roll]
    FULL_12 = 12    # [x, y, z, dx, dy, dz, pitch, yaw, roll, dpitch, dyaw, droll]

def detect_camera_format(data: Union[np.ndarray, torch.Tensor]) -> CameraFormat:
    """
    Detect camera data format based on feature dimensions
    
    Args:
        data: Camera trajectory data (..., features)
        
    Returns:
        CameraFormat enum indicating the detected format
    """
    if isinstance(data, torch.Tensor):
        feature_dim = data.shape[-1]
    else:
        feature_dim = data.shape[-1]
    
    if feature_dim == 5:
        return CameraFormat.LEGACY_5
    elif feature_dim == 6:
        return CameraFormat.STANDARD_6
    elif feature_dim == 12:
        return CameraFormat.FULL_12
    else:
        raise ValueError(f"Unsupported feature dimension: {feature_dim}. Expected 5, 6, or 12.")

def extract_position(data: Union[np.ndarray, torch.Tensor], format_type: Optional[CameraFormat] = None) -> Union[np.ndarray, torch.Tensor]:
    """
    Extract position [x, y, z] from camera data
    
    Args:
        data: Camera trajectory data (..., features)
        format_type: Optional format specification
        
    Returns:
        Position data (..., 3)
    """
    if format_type is None:
        format_type = detect_camera_format(data)
    
    # Position is always the first 3 features for all formats
    return data[..., :3]

def extract_orientation(data: Union[np.ndarray, torch.Tensor], format_type: Optional[CameraFormat] = None) -> Union[np.ndarray, torch.Tensor]:
    """
    Extract orientation from camera data
    
    Args:
        data: Camera trajectory data (..., features)
        format_type: Optional format specification
        
    Returns:
        Orientation data (..., 2 or 3) depending on format
        - 5-feature: [pitch, yaw] (..., 2)
        - 6-feature: [pitch, yaw, roll] (..., 3)
        - 12-feature: [pitch, yaw, roll] (..., 3)
    """
    if format_type is None:
        format_type = detect_camera_format(data)
    
    if format_type == CameraFormat.LEGACY_5:
        return data[..., 3:5]  # [pitch, yaw]
    elif format_type == CameraFormat.STANDARD_6:
        return data[..., 3:6]  # [pitch, yaw, roll]
    elif format_type == CameraFormat.FULL_12:
        return data[..., 6:9]  # [pitch, yaw, roll]
    else:
        raise ValueError(f"Unsupported format: {format_type}")

def extract_velocity(data: Union[np.ndarray, torch.Tensor], format_type: Optional[CameraFormat] = None) -> Optional[Union[np.ndarray, torch.Tensor]]:
    """
    Extract velocity from camera data (only available for 12-feature format)
    
    Args:
        data: Camera trajectory data (..., features)
        format_type: Optional format specification
        
    Returns:
        Velocity data (..., 3) or None if not available
        - 5-feature: None
        - 6-feature: None  
        - 12-feature: [dx, dy, dz] (..., 3)
    """
    if format_type is None:
        format_type = detect_camera_format(data)
    
    if format_type == CameraFormat.FULL_12:
        return data[..., 3:6]  # [dx, dy, dz]
    else:
        return None

def extract_angular_velocity(data: Union[np.ndarray, torch.Tensor], format_type: Optional[CameraFormat] = None) -> Optional[Union[np.ndarray, torch.Tensor]]:
    """
    Extract angular velocity from camera data (only available for 12-feature format)
    
    Args:
        data: Camera trajectory data (..., features)
        format_type: Optional format specification
        
    Returns:
        Angular velocity data (..., 3) or None if not available
        - 5-feature: None
        - 6-feature: None
        - 12-feature: [dpitch, dyaw, droll] (..., 3)
    """
    if format_type is None:
        format_type = detect_camera_format(data)
    
    if format_type == CameraFormat.FULL_12:
        return data[..., 9:12]  # [dpitch, dyaw, droll]
    else:
        return None

def convert_to_legacy_5(data: Union[np.ndarray, torch.Tensor], format_type: Optional[CameraFormat] = None) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert camera data to legacy 5-feature format [x, y, z, pitch, yaw]
    
    Args:
        data: Camera trajectory data (..., features)
        format_type: Optional format specification
        
    Returns:
        Legacy 5-feature data (..., 5)
    """
    if format_type is None:
        format_type = detect_camera_format(data)
    
    position = extract_position(data, format_type)  # (..., 3)
    orientation = extract_orientation(data, format_type)  # (..., 2 or 3)
    
    # Take only pitch and yaw for legacy format
    pitch_yaw = orientation[..., :2]  # (..., 2)
    
    if isinstance(data, torch.Tensor):
        return torch.cat([position, pitch_yaw], dim=-1)
    else:
        return np.concatenate([position, pitch_yaw], axis=-1)

def convert_to_standard_6(data: Union[np.ndarray, torch.Tensor], format_type: Optional[CameraFormat] = None) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert camera data to standard 6-feature format [x, y, z, pitch, yaw, roll]
    
    Args:
        data: Camera trajectory data (..., features)
        format_type: Optional format specification
        
    Returns:
        Standard 6-feature data (..., 6)
    """
    if format_type is None:
        format_type = detect_camera_format(data)
    
    if format_type == CameraFormat.STANDARD_6:
        return data  # Already in correct format
    
    position = extract_position(data, format_type)  # (..., 3)
    
    if format_type == CameraFormat.LEGACY_5:
        # Add zero roll to 5-feature data
        orientation = extract_orientation(data, format_type)  # (..., 2)
        if isinstance(data, torch.Tensor):
            zero_roll = torch.zeros_like(orientation[..., :1])
            full_orientation = torch.cat([orientation, zero_roll], dim=-1)
            return torch.cat([position, full_orientation], dim=-1)
        else:
            zero_roll = np.zeros_like(orientation[..., :1])
            full_orientation = np.concatenate([orientation, zero_roll], axis=-1)
            return np.concatenate([position, full_orientation], axis=-1)
    
    elif format_type == CameraFormat.FULL_12:
        # Extract position and orientation from 12-feature data
        orientation = extract_orientation(data, format_type)  # (..., 3)
        if isinstance(data, torch.Tensor):
            return torch.cat([position, orientation], dim=-1)
        else:
            return np.concatenate([position, orientation], axis=-1)
    
    else:
        raise ValueError(f"Unsupported format conversion: {format_type} -> STANDARD_6")

def get_feature_names(format_type: CameraFormat) -> list:
    """
    Get feature names for a given camera format
    
    Args:
        format_type: Camera format
        
    Returns:
        List of feature names
    """
    if format_type == CameraFormat.LEGACY_5:
        return ['x', 'y', 'z', 'pitch', 'yaw']
    elif format_type == CameraFormat.STANDARD_6:
        return ['x', 'y', 'z', 'pitch', 'yaw', 'roll']
    elif format_type == CameraFormat.FULL_12:
        return ['x', 'y', 'z', 'dx', 'dy', 'dz', 'pitch', 'yaw', 'roll', 'dpitch', 'dyaw', 'droll']
    else:
        raise ValueError(f"Unsupported format: {format_type}")

def recover_from_camera_data_universal(data: Union[np.ndarray, torch.Tensor], format_type: Optional[CameraFormat] = None) -> Tuple:
    """
    Universal function to recover camera components from any format
    
    Args:
        data: Camera trajectory data (..., features)
        format_type: Optional format specification
        
    Returns:
        Tuple containing (position, orientation, velocity, angular_velocity)
        - position: (..., 3) - always available
        - orientation: (..., 2 or 3) - always available  
        - velocity: (..., 3) or None - only for 12-feature
        - angular_velocity: (..., 3) or None - only for 12-feature
    """
    if format_type is None:
        format_type = detect_camera_format(data)
    
    position = extract_position(data, format_type)
    orientation = extract_orientation(data, format_type)
    velocity = extract_velocity(data, format_type)
    angular_velocity = extract_angular_velocity(data, format_type)
    
    return position, orientation, velocity, angular_velocity

def is_compatible_format(data: Union[np.ndarray, torch.Tensor], expected_dim: int) -> bool:
    """
    Check if camera data is compatible with expected dimension
    
    Args:
        data: Camera trajectory data
        expected_dim: Expected feature dimension
        
    Returns:
        True if compatible, False otherwise
    """
    actual_dim = data.shape[-1]
    return actual_dim == expected_dim

def get_format_info(format_type: CameraFormat) -> dict:
    """
    Get detailed information about a camera format
    
    Args:
        format_type: Camera format
        
    Returns:
        Dictionary with format information
    """
    info = {
        'dimension': format_type.value,
        'features': get_feature_names(format_type),
        'has_roll': format_type != CameraFormat.LEGACY_5,
        'has_velocity': format_type == CameraFormat.FULL_12,
        'has_angular_velocity': format_type == CameraFormat.FULL_12,
    }
    
    return info

# Backward compatibility aliases
def recover_from_camera_data(data: Union[np.ndarray, torch.Tensor]) -> Tuple:
    """
    Backward compatibility function - assumes legacy 5-feature format
    
    Args:
        data: Camera trajectory data (..., 5)
        
    Returns:
        Tuple of (positions, orientations) for legacy format
    """
    position, orientation, _, _ = recover_from_camera_data_universal(data, CameraFormat.LEGACY_5)
    return position, orientation
