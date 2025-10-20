"""
Unified Data Format Module for Camera Trajectory Processing

This module provides a unified interface for handling different camera trajectory data formats:
- 5-feature: [x, y, z, pitch, yaw] (legacy cam dataset)
- 6-feature: [x, y, z, pitch, yaw, roll] (position + full orientation)
- 12-feature: [x, y, z, dx, dy, dz, pitch, yaw, roll, dpitch, dyaw, droll] (position + velocity + orientation + angular velocity)

The module automatically detects format and provides conversion utilities.
"""

import numpy as np
import torch
from typing import Union, Tuple, Dict, Optional
from enum import Enum

class CameraDataFormat(Enum):
    """Enumeration of supported camera data formats"""
    LEGACY_5 = 5  # [x, y, z, pitch, yaw]
    POSITION_ORIENTATION_6 = 6  # [x, y, z, pitch, yaw, roll]
    FULL_12 = 12  # [x, y, z, dx, dy, dz, pitch, yaw, roll, dpitch, dyaw, droll]

class UnifiedCameraData:
    """
    Unified camera data representation that can handle multiple formats
    """
    
    def __init__(self, data: Union[np.ndarray, torch.Tensor], format_type: Optional[CameraDataFormat] = None):
        """
        Initialize unified camera data
        
        Args:
            data: Camera trajectory data of shape (seq_len, features)
            format_type: Explicit format type, if None will auto-detect
        """
        self.data = torch.tensor(data) if isinstance(data, np.ndarray) else data
        self.seq_len, self.num_features = self.data.shape
        
        # Auto-detect format if not provided
        if format_type is None:
            self.format_type = self._detect_format()
        else:
            self.format_type = format_type
            
        self._validate_format()
        
    def _detect_format(self) -> CameraDataFormat:
        """Auto-detect data format based on number of features"""
        if self.num_features == 5:
            return CameraDataFormat.LEGACY_5
        elif self.num_features == 6:
            return CameraDataFormat.POSITION_ORIENTATION_6
        elif self.num_features == 12:
            return CameraDataFormat.FULL_12
        else:
            raise ValueError(f"Unsupported data format with {self.num_features} features. "
                           f"Supported formats: 5, 6, or 12 features.")
    
    def _validate_format(self):
        """Validate that data matches declared format"""
        expected_features = self.format_type.value
        if self.num_features != expected_features:
            raise ValueError(f"Data has {self.num_features} features but format expects {expected_features}")
    
    @property
    def positions(self) -> torch.Tensor:
        """Extract position data [x, y, z]"""
        return self.data[:, :3]
    
    @property
    def orientations(self) -> torch.Tensor:
        """Extract orientation data based on format"""
        if self.format_type == CameraDataFormat.LEGACY_5:
            return self.data[:, 3:5]  # [pitch, yaw]
        elif self.format_type == CameraDataFormat.POSITION_ORIENTATION_6:
            return self.data[:, 3:6]  # [pitch, yaw, roll]
        else:  # FULL_12
            return self.data[:, 6:9]  # [pitch, yaw, roll]
    
    @property
    def velocities(self) -> Optional[torch.Tensor]:
        """Extract velocity data if available"""
        if self.format_type == CameraDataFormat.FULL_12:
            return self.data[:, 3:6]  # [dx, dy, dz]
        return None
    
    @property
    def angular_velocities(self) -> Optional[torch.Tensor]:
        """Extract angular velocity data if available"""
        if self.format_type == CameraDataFormat.FULL_12:
            return self.data[:, 9:12]  # [dpitch, dyaw, droll]
        return None
    
    def to_format(self, target_format: CameraDataFormat) -> 'UnifiedCameraData':
        """
        Convert to target format
        
        Args:
            target_format: Target data format
            
        Returns:
            New UnifiedCameraData in target format
        """
        if self.format_type == target_format:
            return UnifiedCameraData(self.data.clone(), target_format)
        
        return self._convert_format(target_format)
    
    def _convert_format(self, target_format: CameraDataFormat) -> 'UnifiedCameraData':
        """Internal format conversion logic"""
        
        # Get base components
        positions = self.positions  # [seq_len, 3]
        orientations = self.orientations  # [seq_len, 2 or 3]
        
        if target_format == CameraDataFormat.LEGACY_5:
            # Convert to [x, y, z, pitch, yaw]
            if orientations.shape[1] == 2:  # Already [pitch, yaw]
                converted_data = torch.cat([positions, orientations], dim=1)
            else:  # [pitch, yaw, roll] -> [pitch, yaw]
                converted_data = torch.cat([positions, orientations[:, :2]], dim=1)
                
        elif target_format == CameraDataFormat.POSITION_ORIENTATION_6:
            # Convert to [x, y, z, pitch, yaw, roll]
            if orientations.shape[1] == 2:  # [pitch, yaw] -> [pitch, yaw, roll]
                # Add zero roll if not present
                roll = torch.zeros(self.seq_len, 1, device=orientations.device)
                full_orientations = torch.cat([orientations, roll], dim=1)
            else:  # Already [pitch, yaw, roll]
                full_orientations = orientations
            converted_data = torch.cat([positions, full_orientations], dim=1)
            
        else:  # target_format == CameraDataFormat.FULL_12
            # Convert to [x, y, z, dx, dy, dz, pitch, yaw, roll, dpitch, dyaw, droll]
            
            # Calculate velocities if not available
            if self.velocities is not None:
                velocities = self.velocities
            else:
                velocities = self._calculate_velocities(positions)
            
            # Ensure full orientations [pitch, yaw, roll]
            if orientations.shape[1] == 2:  # [pitch, yaw] -> [pitch, yaw, roll]
                roll = torch.zeros(self.seq_len, 1, device=orientations.device)
                full_orientations = torch.cat([orientations, roll], dim=1)
            else:  # Already [pitch, yaw, roll]
                full_orientations = orientations
            
            # Calculate angular velocities if not available
            if self.angular_velocities is not None:
                angular_velocities = self.angular_velocities
            else:
                angular_velocities = self._calculate_angular_velocities(full_orientations)
            
            converted_data = torch.cat([
                positions,          # [x, y, z]
                velocities,         # [dx, dy, dz]
                full_orientations,  # [pitch, yaw, roll]
                angular_velocities  # [dpitch, dyaw, droll]
            ], dim=1)
        
        return UnifiedCameraData(converted_data, target_format)
    
    def _calculate_velocities(self, positions: torch.Tensor) -> torch.Tensor:
        """Calculate linear velocities from positions"""
        if len(positions) < 2:
            return torch.zeros_like(positions)
        
        velocities = torch.zeros_like(positions)
        velocities[1:] = positions[1:] - positions[:-1]
        velocities[0] = velocities[1]  # Copy first velocity
        
        return velocities
    
    def _calculate_angular_velocities(self, orientations: torch.Tensor) -> torch.Tensor:
        """Calculate angular velocities from orientations"""
        if len(orientations) < 2:
            return torch.zeros_like(orientations)
        
        angular_velocities = torch.zeros_like(orientations)
        angular_velocities[1:] = orientations[1:] - orientations[:-1]
        
        # Handle angle wrapping for yaw
        yaw_diff = angular_velocities[:, 1]
        yaw_diff = torch.where(yaw_diff > np.pi, yaw_diff - 2*np.pi, yaw_diff)
        yaw_diff = torch.where(yaw_diff < -np.pi, yaw_diff + 2*np.pi, yaw_diff)
        angular_velocities[:, 1] = yaw_diff
        
        angular_velocities[0] = angular_velocities[1]  # Copy first angular velocity
        
        return angular_velocities
    
    def get_momask_compatible_data(self) -> torch.Tensor:
        """
        Get data in format compatible with MoMask training
        Returns the raw tensor data
        """
        return self.data
    
    def get_feature_info(self) -> Dict[str, any]:
        """Get information about the current data format"""
        return {
            'format_type': self.format_type,
            'num_features': self.num_features,
            'seq_len': self.seq_len,
            'has_velocities': self.velocities is not None,
            'has_angular_velocities': self.angular_velocities is not None,
            'orientation_dims': self.orientations.shape[1]
        }

def detect_dataset_format(data_root: str, sample_file: str = None) -> CameraDataFormat:
    """
    Detect the format of a dataset by examining sample files
    
    Args:
        data_root: Path to dataset root
        sample_file: Optional specific file to check, otherwise checks first available
        
    Returns:
        Detected format type
    """
    from pathlib import Path
    
    motion_dir = Path(data_root) / "new_joint_vecs"
    if not motion_dir.exists():
        raise FileNotFoundError(f"Motion directory not found: {motion_dir}")
    
    # Find a sample file
    if sample_file is None:
        npy_files = list(motion_dir.glob("*.npy"))
        if not npy_files:
            raise FileNotFoundError(f"No .npy files found in {motion_dir}")
        sample_file = npy_files[0]
    else:
        sample_file = motion_dir / f"{sample_file}.npy"
    
    # Load and check dimensions
    try:
        data = np.load(sample_file)
        if len(data.shape) != 2:
            raise ValueError(f"Expected 2D data, got shape {data.shape}")
        
        num_features = data.shape[1]
        if num_features == 5:
            return CameraDataFormat.LEGACY_5
        elif num_features == 6:
            return CameraDataFormat.POSITION_ORIENTATION_6
        elif num_features == 12:
            return CameraDataFormat.FULL_12
        else:
            raise ValueError(f"Unsupported format with {num_features} features")
            
    except Exception as e:
        raise RuntimeError(f"Failed to detect format from {sample_file}: {e}")

def create_unified_data_from_file(file_path: str, format_type: Optional[CameraDataFormat] = None) -> UnifiedCameraData:
    """
    Create UnifiedCameraData from a file
    
    Args:
        file_path: Path to .npy file
        format_type: Optional explicit format type
        
    Returns:
        UnifiedCameraData instance
    """
    data = np.load(file_path)
    return UnifiedCameraData(data, format_type)

def batch_convert_dataset(input_dir: str, output_dir: str, 
                         source_format: CameraDataFormat, 
                         target_format: CameraDataFormat):
    """
    Convert an entire dataset from one format to another
    
    Args:
        input_dir: Input dataset directory
        output_dir: Output dataset directory
        source_format: Source data format
        target_format: Target data format
    """
    from pathlib import Path
    import shutil
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory structure
    output_motion_dir = output_path / "new_joint_vecs"
    output_motion_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy text directory if it exists
    input_text_dir = input_path / "texts"
    if input_text_dir.exists():
        output_text_dir = output_path / "texts"
        if output_text_dir.exists():
            shutil.rmtree(output_text_dir)
        shutil.copytree(input_text_dir, output_text_dir)
    
    # Convert motion files
    input_motion_dir = input_path / "new_joint_vecs"
    npy_files = list(input_motion_dir.glob("*.npy"))
    
    print(f"Converting {len(npy_files)} files from {source_format.name} to {target_format.name}")
    
    for npy_file in npy_files:
        # Load and convert
        unified_data = create_unified_data_from_file(str(npy_file), source_format)
        converted_data = unified_data.to_format(target_format)
        
        # Save converted data
        output_file = output_motion_dir / npy_file.name
        np.save(output_file, converted_data.get_momask_compatible_data().numpy())
    
    # Copy other files (train.txt, val.txt, etc.)
    for txt_file in input_path.glob("*.txt"):
        shutil.copy2(txt_file, output_path / txt_file.name)
    
    # Create format info file
    format_info = {
        'format_type': target_format.name,
        'num_features': target_format.value,
        'converted_from': source_format.name,
        'conversion_date': str(np.datetime64('now'))
    }
    
    with open(output_path / "format_info.json", 'w') as f:
        import json
        json.dump(format_info, f, indent=2)
    
    print(f"Dataset conversion completed. Output saved to: {output_path}")
