"""
Unified Plotting Module for Camera Trajectory Visualization

This module provides flexible plotting functions that can handle different camera data formats
automatically by using the unified data format system.
"""

import numpy as np
import torch
from .unified_data_format import UnifiedCameraData, CameraDataFormat

def plot_camera_trajectory_unified(save_path, camera_data, title, fps=30, radius=4, figsize=(10, 10)):
    """
    Plot camera trajectory with automatic format detection and handling
    
    Args:
        save_path: Path to save the animation
        camera_data: Camera trajectory data (seq_len, features) - can be 5D, 6D, or 12D
        title: Title for the plot
        fps: Frames per second for animation
        radius: Plot radius
        figsize: Figure size tuple
    """
    
    # Import the corrected camera plotting functions
    from gen_camera import plot_camera_trajectory_animation
    
    # Use the corrected camera plotting function with proper coordinate system
    plot_camera_trajectory_animation(
        data=camera_data,
        save_path=save_path,
        title=title,
        fps=fps,
        figsize=figsize,
        show_trail=True,
        trail_length=20
    )
    
    # Create unified camera data object for format detection
    unified_data = UnifiedCameraData(camera_data)
    return unified_data.format_type

def plot_camera_with_orientation_info(save_path, camera_data, title, fps=30, radius=4, figsize=(10, 10)):
    """
    Enhanced camera plotting that includes orientation information in the title
    
    Args:
        save_path: Path to save the animation
        camera_data: Camera trajectory data (seq_len, features)
        title: Base title for the plot
        fps: Frames per second for animation
        radius: Plot radius
        figsize: Figure size tuple
    
    Returns:
        Detected format type
    """
    
    # Import the corrected camera plotting functions
    from gen_camera import plot_camera_trajectory_animation
    
    # Create unified camera data object
    unified_data = UnifiedCameraData(camera_data)
    
    # Enhance title with format information
    format_info = {
        CameraDataFormat.LEGACY_5: "5D (pos + pitch/yaw)",
        CameraDataFormat.POSITION_ORIENTATION_6: "6D (pos + full orientation)", 
        CameraDataFormat.FULL_12: "12D (pos + vel + orientation + angular vel)"
    }
    
    enhanced_title = f"{title} [{format_info[unified_data.format_type]}]"
    
    # Use the corrected camera plotting function with proper coordinate system
    plot_camera_trajectory_animation(
        data=camera_data,
        save_path=save_path,
        title=enhanced_title,
        fps=fps,
        figsize=figsize,
        show_trail=True,
        trail_length=20
    )
    
    return unified_data.format_type

def create_plotting_function_for_transformer(dataset_name):
    """
    Create a plotting function tailored for a specific dataset that can be used in transformers
    
    Args:
        dataset_name: Name of the dataset (e.g., 'realestate10k_6', 'cam', etc.)
        
    Returns:
        A plotting function that can be used in transformer training
    """
    
    def plot_function(data, save_dir, captions, m_lengths, **kwargs):
        """
        Plotting function for transformer training
        
        Args:
            data: Motion data after inv_transform
            save_dir: Directory to save plots
            captions: List of captions
            m_lengths: List of motion lengths
            **kwargs: Additional arguments (fps, radius, etc.)
        """
        from os.path import join as pjoin
        from utils.motion_process import recover_from_ric
        from utils.plot_script import plot_3d_motion
        
        # Get plotting parameters
        fps = kwargs.get('fps', 30)
        radius = kwargs.get('radius', 4)
        joints_num = kwargs.get('joints_num', 22)
        kinematic_chain = kwargs.get('kinematic_chain', None)
        
        # Determine if this is a camera dataset
        is_camera_dataset = any(name in dataset_name.lower() for name in ["cam", "estate", "realestate"])
        
        for i, (caption, joint_data) in enumerate(zip(captions, data)):
            joint_data = joint_data[:m_lengths[i]]
            save_path = pjoin(save_dir, '%02d.mp4' % i)
            
            if is_camera_dataset:
                # Use corrected camera plotting with proper coordinate system
                try:
                    detected_format = plot_camera_with_orientation_info(
                        save_path, joint_data, caption, fps=fps, radius=radius
                    )
                    # Log the detected format for debugging
                    if i == 0:  # Only log for first sample to avoid spam
                        print(f"Detected camera format: {detected_format.name} ({detected_format.value} features)")
                except Exception as e:
                    print(f"Warning: Camera plotting failed for sample {i}: {e}")
                    print(f"Data shape: {joint_data.shape}")
                    # Fallback to basic camera plotting using corrected functions
                    try:
                        from gen_camera import plot_camera_trajectory_animation
                        plot_camera_trajectory_animation(
                            data=joint_data,
                            save_path=save_path,
                            title=caption,
                            fps=fps,
                            show_trail=False  # Simplified for fallback
                        )
                    except Exception as e2:
                        print(f"Fallback plotting also failed: {e2}")
                        # Last resort: create a simple plot
                        import matplotlib.pyplot as plt
                        fig = plt.figure()
                        ax = fig.add_subplot(111, projection='3d')
                        positions = joint_data[:, :3]
                        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
                        ax.set_title(caption)
                        plt.savefig(save_path.replace('.mp4', '.png'))
                        plt.close()
            else:
                # For human motion, use the original plotting function
                joint = recover_from_ric(torch.from_numpy(joint_data).float(), joints_num).numpy()
                plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=fps, radius=radius)
    
    return plot_function

def get_camera_data_info(camera_data):
    """
    Get information about camera data format
    
    Args:
        camera_data: Camera trajectory data
        
    Returns:
        Dictionary with format information
    """
    unified_data = UnifiedCameraData(camera_data)
    
    info = {
        'format': unified_data.format_type,
        'features': unified_data.num_features,
        'sequence_length': unified_data.seq_len,
        'has_position': True,
        'has_orientation': unified_data.format_type in [CameraDataFormat.POSITION_ORIENTATION_6, CameraDataFormat.FULL_12],
        'has_velocity': unified_data.format_type == CameraDataFormat.FULL_12,
        'has_angular_velocity': unified_data.format_type == CameraDataFormat.FULL_12
    }
    
    return info
