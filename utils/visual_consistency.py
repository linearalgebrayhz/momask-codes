"""
Visual Consistency Module for LPIPS-based Camera Trajectory Training

This module provides visual consistency loss using LPIPS metric to enhance
quantization and generation quality by comparing rendered views from camera trajectories.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Optional, Tuple, List, Dict
from pathlib import Path

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not available. Install with: pip install lpips")


class SimpleRenderer(nn.Module):
    """
    Simple differentiable renderer for camera trajectories.
    Renders basic views from camera positions and orientations.
    Can be extended or replaced with more sophisticated renderers like 3DGS.
    """
    
    def __init__(self, image_size: int = 256, device: str = 'cuda'):
        super().__init__()
        self.image_size = image_size
        self.device = device
        
        # Simple scene setup - can be made configurable
        self.scene_bounds = 10.0  # Scene extends from -10 to +10 in each axis
        
    def render_view(self, camera_pos: torch.Tensor, camera_ori: torch.Tensor) -> torch.Tensor:
        """
        Render a view from given camera position and orientation.
        
        Args:
            camera_pos: [3] - camera position [x, y, z]
            camera_ori: [2] - camera orientation [pitch, yaw]
            
        Returns:
            rendered_image: [3, H, W] - RGB image
        """
        batch_size = camera_pos.shape[0] if len(camera_pos.shape) > 1 else 1
        
        # Create a simple synthetic scene (for now)
        # This can be replaced with actual scene rendering
        image = torch.zeros(batch_size, 3, self.image_size, self.image_size, 
                          device=self.device, dtype=torch.float32)
        
        # Simple gradient based on camera position and orientation
        # This creates different views based on camera parameters
        for i in range(batch_size):
            pos = camera_pos[i] if batch_size > 1 else camera_pos
            ori = camera_ori[i] if batch_size > 1 else camera_ori
            
            # Create position-dependent color gradient
            x_grad = torch.linspace(-1, 1, self.image_size, device=self.device)
            y_grad = torch.linspace(-1, 1, self.image_size, device=self.device)
            xx, yy = torch.meshgrid(x_grad, y_grad)
            
            # Modulate by camera position and orientation
            pos_factor = (pos / self.scene_bounds).clamp(-1, 1)
            ori_factor = (ori / (2 * np.pi)).clamp(-1, 1)
            
            # Create RGB channels with different responses
            r_channel = (xx * pos_factor[0] + yy * ori_factor[0]).clamp(0, 1)
            g_channel = (yy * pos_factor[1] + xx * ori_factor[1]).clamp(0, 1)
            b_channel = ((xx + yy) * pos_factor[2] * 0.5 + 0.5).clamp(0, 1)
            
            image[i, 0] = r_channel
            image[i, 1] = g_channel
            image[i, 2] = b_channel
            
        return image.squeeze(0) if batch_size == 1 else image
    
    def forward(self, camera_trajectory: torch.Tensor, keyframe_indices: List[int]) -> torch.Tensor:
        """
        Render multiple keyframes from camera trajectory.
        
        Args:
            camera_trajectory: [T, 5] - trajectory with [x, y, z, pitch, yaw]
            keyframe_indices: List of frame indices to render
            
        Returns:
            rendered_frames: [K, 3, H, W] - rendered keyframes
        """
        keyframes = camera_trajectory[keyframe_indices]  # [K, 5]
        positions = keyframes[:, :3]  # [K, 3]
        orientations = keyframes[:, 3:]  # [K, 2]
        
        return self.render_view(positions, orientations)


class GroundTruthLoader:
    """
    Loads ground truth video frames corresponding to camera trajectories.
    Handles missing data gracefully.
    """
    
    def __init__(self, data_root: str, image_size: int = 256):
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.video_dir = self.data_root / "videos"  # Expected video frame directory
        
    def load_frames(self, data_id: str, keyframe_indices: List[int]) -> Optional[torch.Tensor]:
        """
        Load ground truth frames for given data ID and keyframe indices.
        
        Args:
            data_id: Data identifier (e.g., "00001")
            keyframe_indices: List of frame indices to load
            
        Returns:
            frames: [K, 3, H, W] or None if data not available
        """
        video_path = self.video_dir / f"{data_id}"
        
        if not video_path.exists():
            return None
            
        if not CV2_AVAILABLE:
            return None
            
        frames = []
        for frame_idx in keyframe_indices:
            frame_path = video_path / f"frame_{frame_idx:06d}.jpg"
            
            if not frame_path.exists():
                return None
                
            # Load and process frame
            frame = cv2.imread(str(frame_path))
            if frame is None:
                return None
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.image_size, self.image_size))
            frame = torch.from_numpy(frame).float() / 255.0
            frame = frame.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
            frames.append(frame)
            
        return torch.stack(frames) if frames else None


class KeyframeSelector:
    """
    Selects keyframes from camera trajectory for visual consistency evaluation.
    """
    
    @staticmethod
    def uniform_sampling(trajectory_length: int, num_keyframes: int = 4) -> List[int]:
        """
        Uniformly sample keyframes from trajectory.
        
        Args:
            trajectory_length: Total number of frames in trajectory
            num_keyframes: Number of keyframes to select
            
        Returns:
            keyframe_indices: List of selected frame indices
        """
        if trajectory_length <= num_keyframes:
            return list(range(trajectory_length))
            
        indices = np.linspace(0, trajectory_length - 1, num_keyframes, dtype=int)
        return indices.tolist()
    
    @staticmethod
    def motion_based_sampling(trajectory: torch.Tensor, num_keyframes: int = 4) -> List[int]:
        """
        Sample keyframes based on motion magnitude (future enhancement).
        
        Args:
            trajectory: [T, 5] - camera trajectory
            num_keyframes: Number of keyframes to select
            
        Returns:
            keyframe_indices: List of selected frame indices
        """
        # For now, fall back to uniform sampling
        # TODO: Implement motion-based selection
        return KeyframeSelector.uniform_sampling(len(trajectory), num_keyframes)


class VisualConsistencyModule(nn.Module):
    """
    Main visual consistency module using LPIPS metric.
    """
    
    def __init__(self, 
                 data_root: str,
                 image_size: int = 256,
                 lpips_net: str = 'alex',
                 num_keyframes: int = 4,
                 keyframe_strategy: str = 'uniform',
                 device: str = 'cuda',
                 enabled: bool = True):
        super().__init__()
        
        self.enabled = enabled
        self.num_keyframes = num_keyframes
        self.keyframe_strategy = keyframe_strategy
        self.device = device
        
        if not self.enabled:
            return
            
        if not LPIPS_AVAILABLE:
            print("Warning: LPIPS not available, disabling visual consistency module")
            self.enabled = False
            return
            
        # Initialize LPIPS metric
        self.lpips_metric = lpips.LPIPS(net=lpips_net, verbose=False).to(device)
        self.lpips_metric.eval()
        
        # Initialize renderer
        self.renderer = SimpleRenderer(image_size=image_size, device=device)
        
        # Initialize ground truth loader
        self.gt_loader = GroundTruthLoader(data_root, image_size=image_size)
        
        # Initialize keyframe selector
        self.keyframe_selector = KeyframeSelector()
        
    def compute_visual_loss(self, 
                          pred_trajectory: torch.Tensor,
                          gt_trajectory: torch.Tensor,
                          data_id: Optional[str] = None,
                          step: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Compute visual consistency loss using LPIPS.
        
        Args:
            pred_trajectory: [T, 5] - predicted camera trajectory
            gt_trajectory: [T, 5] - ground truth camera trajectory
            data_id: Data identifier for loading ground truth frames
            step: Current training step (for logging)
            
        Returns:
            loss_dict: Dictionary containing loss components
        """
        if not self.enabled:
            return {'lpips_loss': torch.tensor(0.0, device=self.device)}
        
        # Select keyframes
        trajectory_length = min(len(pred_trajectory), len(gt_trajectory))
        if self.keyframe_strategy == 'uniform':
            keyframe_indices = self.keyframe_selector.uniform_sampling(
                trajectory_length, self.num_keyframes)
        else:
            keyframe_indices = self.keyframe_selector.motion_based_sampling(
                gt_trajectory, self.num_keyframes)
        
        # Render predicted views
        pred_views = self.renderer(pred_trajectory, keyframe_indices)  # [K, 3, H, W]
        
        # Try to load ground truth frames, fall back to rendered GT if not available
        gt_views = None
        if data_id is not None:
            gt_views = self.gt_loader.load_frames(data_id, keyframe_indices)
        
        if gt_views is None:
            # Fall back to rendering ground truth trajectory
            gt_views = self.renderer(gt_trajectory, keyframe_indices)
        
        # Ensure both are on the correct device
        pred_views = pred_views.to(self.device)
        gt_views = gt_views.to(self.device)
        
        # Normalize to [-1, 1] for LPIPS
        pred_views = pred_views * 2.0 - 1.0
        gt_views = gt_views * 2.0 - 1.0
        
        # Compute LPIPS loss
        lpips_loss = 0.0
        for i in range(len(keyframe_indices)):
            loss = self.lpips_metric(pred_views[i:i+1], gt_views[i:i+1])
            lpips_loss += loss.mean()
        
        lpips_loss = lpips_loss / len(keyframe_indices)
        
        return {
            'lpips_loss': torch.tensor(lpips_loss, device=self.device),
            'num_keyframes': torch.tensor(len(keyframe_indices), device=self.device)
        }
    
    def set_enabled(self, enabled: bool):
        """Enable or disable the visual consistency module."""
        self.enabled = enabled and LPIPS_AVAILABLE


def create_visual_consistency_module(opt) -> VisualConsistencyModule:
    """
    Factory function to create visual consistency module from options.
    
    Args:
        opt: Training options object
        
    Returns:
        VisualConsistencyModule instance
    """
    # Check if visual consistency is enabled for this dataset
    enabled = getattr(opt, 'use_visual_consistency', False)
    
    # For camera datasets, try to enable by default if not explicitly disabled
    if opt.dataset_name == "cam" and not hasattr(opt, 'use_visual_consistency'):
        enabled = True
    
    # Disable if no video data available
    if enabled and hasattr(opt, 'no_video_data') and opt.no_video_data:
        enabled = False
        print("Visual consistency disabled: no video data available")
    
    return VisualConsistencyModule(
        data_root=opt.data_root,
        image_size=getattr(opt, 'visual_consistency_image_size', 256),
        lpips_net=getattr(opt, 'lpips_net', 'alex'),
        num_keyframes=getattr(opt, 'num_keyframes', 4),
        keyframe_strategy=getattr(opt, 'keyframe_strategy', 'uniform'),
        device=opt.device,
        enabled=enabled
    )
