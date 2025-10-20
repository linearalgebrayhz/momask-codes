import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from utils.camera_process import recover_from_camera_data
from utils.camera_format_handler import (
    recover_from_camera_data_universal, 
    detect_camera_format, 
    CameraFormat,
    get_format_info
)

def plot_camera_trajectory_3d(gt_data, pred_data, save_path, title="Camera Trajectory Comparison", seq_idx=0):
    """
    Plot 3D comparison of ground truth vs predicted camera trajectories
    Supports all camera formats: 5, 6, and 12 features
    
    Args:
        gt_data: Ground truth camera data (batch_size, seq_len, features)
        pred_data: Predicted camera data (batch_size, seq_len, features)
        save_path: Path to save the plot
        title: Title for the plot
        seq_idx: Index of sequence to plot (in case of batch)
    """
    
    # Take specific sequence from batch if needed
    if gt_data.ndim == 3:
        gt_seq = gt_data[seq_idx]
        pred_seq = pred_data[seq_idx]
    else:
        gt_seq = gt_data
        pred_seq = pred_data
    
    # Detect format and extract components
    gt_format = detect_camera_format(gt_seq)
    pred_format = detect_camera_format(pred_seq)
    
    if gt_format != pred_format:
        print(f"Warning: GT format ({gt_format.value}) != Pred format ({pred_format.value})")
    
    # Use universal recovery function
    gt_pos, gt_ori, gt_vel, gt_ang_vel = recover_from_camera_data_universal(gt_seq[None, ...])
    pred_pos, pred_ori, pred_vel, pred_ang_vel = recover_from_camera_data_universal(pred_seq[None, ...])
    
    # Remove batch dimension
    gt_pos = gt_pos[0]  # (seq_len, 3)
    gt_ori = gt_ori[0]  # (seq_len, 2 or 3)
    pred_pos = pred_pos[0]  # (seq_len, 3)
    pred_ori = pred_ori[0]  # (seq_len, 2 or 3)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax1.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], 'r--', linewidth=2, label='Predicted', alpha=0.8)
    
    # Mark start and end points
    ax1.scatter(gt_pos[0, 0], gt_pos[0, 1], gt_pos[0, 2], color='green', s=100, marker='o', label='Start')
    ax1.scatter(gt_pos[-1, 0], gt_pos[-1, 1], gt_pos[-1, 2], color='orange', s=100, marker='s', label='End')
    
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_zlabel('Z Position')
    ax1.set_title('3D Camera Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Top-down view (X-Y plane)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(gt_pos[:, 0], gt_pos[:, 1], 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax2.plot(pred_pos[:, 0], pred_pos[:, 1], 'r--', linewidth=2, label='Predicted', alpha=0.8)
    ax2.scatter(gt_pos[0, 0], gt_pos[0, 1], color='green', s=100, marker='o', label='Start')
    ax2.scatter(gt_pos[-1, 0], gt_pos[-1, 1], color='orange', s=100, marker='s', label='End')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_title('Top-down View (X-Y Plane)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Position error over time
    ax3 = fig.add_subplot(2, 2, 3)
    position_error = np.sqrt(np.sum((gt_pos - pred_pos) ** 2, axis=1))
    time_steps = np.arange(len(position_error))
    ax3.plot(time_steps, position_error, 'r-', linewidth=2)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Position Error (L2 norm)')
    ax3.set_title('Position Error Over Time')
    ax3.grid(True, alpha=0.3)
    
    # Orientation comparison - handle both 2D and 3D orientations
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(time_steps, gt_ori[:, 0], 'b-', linewidth=2, label='GT Pitch', alpha=0.8)
    ax4.plot(time_steps, pred_ori[:, 0], 'r--', linewidth=2, label='Pred Pitch', alpha=0.8)
    ax4.plot(time_steps, gt_ori[:, 1], 'g-', linewidth=2, label='GT Yaw', alpha=0.8)
    ax4.plot(time_steps, pred_ori[:, 1], 'm--', linewidth=2, label='Pred Yaw', alpha=0.8)
    
    # Add roll if available (6-feature or 12-feature format)
    if gt_ori.shape[-1] > 2:
        ax4.plot(time_steps, gt_ori[:, 2], 'c-', linewidth=2, label='GT Roll', alpha=0.8)
        ax4.plot(time_steps, pred_ori[:, 2], 'y--', linewidth=2, label='Pred Roll', alpha=0.8)
    
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Angle (radians)')
    ax4.set_title(f'Orientation Comparison ({gt_format.value}-feature)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Add format information to the plot
    format_info = get_format_info(gt_format)
    print(f"3D trajectory plot saved to: {save_path}")
    print(f"Format: {gt_format.value}-feature ({', '.join(format_info['features'])})")

def plot_camera_batch_comparison(gt_batch, pred_batch, save_dir, prefix="trajectory", max_plots=4):
    """
    Plot multiple camera trajectory comparisons from a batch
    
    Args:
        gt_batch: Ground truth camera data (batch_size, seq_len, 5)
        pred_batch: Predicted camera data (batch_size, seq_len, 5)
        save_dir: Directory to save plots
        prefix: Prefix for saved filenames
        max_plots: Maximum number of plots to generate
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    batch_size = gt_batch.shape[0]
    num_plots = min(batch_size, max_plots)
    
    for i in range(num_plots):
        save_path = os.path.join(save_dir, f"{prefix}_sample_{i:02d}.png")
        title = f"Camera Trajectory Comparison - Sample {i+1}"
        
        plot_camera_trajectory_3d(
            gt_batch[i:i+1], 
            pred_batch[i:i+1], 
            save_path, 
            title=title, 
            seq_idx=0
        )
    
    print(f"Generated {num_plots} trajectory plots in {save_dir}")

def create_trajectory_summary_plot(gt_batch, pred_batch, save_path, title="Camera Trajectory Summary"):
    """
    Create a summary plot showing multiple trajectories overlaid
    
    Args:
        gt_batch: Ground truth camera data (batch_size, seq_len, 5)
        pred_batch: Predicted camera data (batch_size, seq_len, 5)
        save_path: Path to save the plot
        title: Title for the plot
    """
    
    batch_size = gt_batch.shape[0]
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each trajectory with different colors
    from matplotlib import cm
    colors = cm.get_cmap('tab10')(np.linspace(0, 1, batch_size))
    
    for i in range(batch_size):
        # Recover positions
        gt_pos, _ = recover_from_camera_data(gt_batch[i:i+1])
        pred_pos, _ = recover_from_camera_data(pred_batch[i:i+1])
        
        gt_pos = gt_pos[0]
        pred_pos = pred_pos[0]
        
        # Plot with transparency
        ax.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], 
                color=colors[i], linewidth=2, alpha=0.7, linestyle='-')
        ax.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], 
                color=colors[i], linewidth=2, alpha=0.7, linestyle='--')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', linewidth=2, linestyle='-', label='Ground Truth'),
        Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='Predicted')
    ]
    ax.legend(handles=legend_elements)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Summary plot saved to: {save_path}")

def plot_camera_metrics_summary(metrics_dict, save_path, title="Camera Metrics Summary"):
    """
    Create a summary plot of camera metrics over time
    
    Args:
        metrics_dict: Dictionary containing metrics over epochs
        save_path: Path to save the plot
        title: Title for the plot
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = list(metrics_dict.keys())
    
    # Position error
    pos_errors = [metrics_dict[ep]['position_error'] for ep in epochs]
    axes[0, 0].plot(epochs, pos_errors, 'b-', linewidth=2, marker='o')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Position Error')
    axes[0, 0].set_title('Position Error vs Epoch')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Orientation error
    ori_errors = [metrics_dict[ep]['orientation_error'] for ep in epochs]
    axes[0, 1].plot(epochs, ori_errors, 'r-', linewidth=2, marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Orientation Error')
    axes[0, 1].set_title('Orientation Error vs Epoch')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Smoothness
    smoothness = [metrics_dict[ep]['smoothness'] for ep in epochs]
    axes[1, 0].plot(epochs, smoothness, 'g-', linewidth=2, marker='^')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Smoothness')
    axes[1, 0].set_title('Smoothness vs Epoch')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    recon_losses = [metrics_dict[ep]['recon_loss'] for ep in epochs]
    axes[1, 1].plot(epochs, recon_losses, 'm-', linewidth=2, marker='d')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Reconstruction Loss')
    axes[1, 1].set_title('Reconstruction Loss vs Epoch')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Metrics summary plot saved to: {save_path}")

def plot_camera_data_for_training(data, save_dir, epoch=None, prefix="camera_viz"):
    """
    Plot camera data during training - designed to be called from VQ trainer
    
    Args:
        data: Camera data (batch_size, seq_len, 5) - first half GT, second half pred
        save_dir: Directory to save plots
        epoch: Current epoch (if None, will be extracted from save_dir)
        prefix: Prefix for saved filenames
    """
    
    batch_size = data.shape[0]
    mid_idx = batch_size // 2
    
    # Split into ground truth and predicted
    gt_data = data[:mid_idx]  # First half is ground truth
    pred_data = data[mid_idx:]  # Second half is predicted
    
    # Extract epoch from save_dir if not provided
    if epoch is None:
        import re
        epoch_match = re.search(r'E(\d+)', save_dir)
        epoch = int(epoch_match.group(1)) if epoch_match else 0
    
    # Create epoch-specific directory
    epoch_dir = os.path.join(save_dir, f"epoch_{epoch:04d}")
    os.makedirs(epoch_dir, exist_ok=True)
    
    # Plot individual comparisons
    plot_camera_batch_comparison(
        gt_data, pred_data, epoch_dir, 
        prefix=f"{prefix}_ep{epoch:04d}", 
        max_plots=min(4, mid_idx)
    )
    
    # Create summary plot
    summary_path = os.path.join(epoch_dir, f"{prefix}_summary_ep{epoch:04d}.png")
    create_trajectory_summary_plot(
        gt_data, pred_data, summary_path, 
        title=f"Camera Trajectory Summary - Epoch {epoch}"
    )
    
    print(f"Camera visualization plots saved for epoch {epoch} in {epoch_dir}")

def create_camera_plot_function():
    """
    Create a plot function that can be used in the VQ trainer
    This function conforms to the expected signature for plot_eval
    Automatically chooses the right plotting function based on data format
    """
    def plot_camera_eval(data, save_dir):
        """
        Plot function that will be passed to VQ trainer for camera datasets
        Supports all camera formats: 5, 6, and 12 features
        
        Args:
            data: Camera data (batch_size, seq_len, features) - first half GT, second half pred
            save_dir: Directory to save plots
        """
        try:
            # Detect format and choose appropriate plotting function
            if data.ndim >= 2:
                format_type = detect_camera_format(data)
                
                if format_type == CameraFormat.FULL_12:
                    print(f"Using full dynamics plotting for {format_type.value}-feature data")
                    plot_camera_data_for_training_full_dynamics(data, save_dir)
                else:
                    print(f"Using standard plotting for {format_type.value}-feature data")
                    plot_camera_data_for_training(data, save_dir)
            else:
                print("Warning: Invalid data shape for camera plotting")
                np.save(os.path.join(save_dir, 'camera_data.npy'), data)
                
        except Exception as e:
            print(f"Error in camera plotting: {e}")
            # Fallback to save raw data
            np.save(os.path.join(save_dir, 'camera_data.npy'), data)
    
    return plot_camera_eval

def plot_camera_data_for_training_full_dynamics(data, save_dir, epoch=0):
    """
    Enhanced plotting function for 12-feature camera data during training
    Shows velocity and angular velocity information
    
    Args:
        data: Camera data (batch_size, seq_len, 12) - first half GT, second half pred
        save_dir: Directory to save plots
        epoch: Current epoch number
    """
    batch_size, seq_len, feature_dim = data.shape
    
    if feature_dim != 12:
        print(f"Warning: Expected 12 features for full dynamics, got {feature_dim}. Falling back to standard plotting.")
        plot_camera_data_for_training(data, save_dir, epoch)
        return
    
    # Split GT and predicted data (split batch, not sequence!)
    mid_batch = batch_size // 2
    gt_data = data[:mid_batch, :, :]    # First half of batch is GT
    pred_data = data[mid_batch:, :, :]  # Second half of batch is predicted
    
    # Create epoch-specific directory
    epoch_dir = os.path.join(save_dir, f'epoch_{epoch:03d}')
    os.makedirs(epoch_dir, exist_ok=True)
    
    # Plot first few samples from the batch
    num_plots = min(4, batch_size)
    
    for i in range(num_plots):
        save_path = os.path.join(epoch_dir, f'full_dynamics_sample_{i:02d}.png')
        plot_camera_trajectory_full_dynamics(
            gt_data[i:i+1], pred_data[i:i+1], save_path,
            title=f"Full Dynamics - Epoch {epoch}, Sample {i}",
            seq_idx=0
        )
    
    print(f"Full dynamics visualization plots saved for epoch {epoch} in {epoch_dir}")

def plot_camera_trajectory_full_dynamics(gt_data, pred_data, save_path, title="Camera Full Dynamics Comparison", seq_idx=0):
    """
    Plot comprehensive comparison for 12-feature camera data including velocities
    
    Args:
        gt_data: Ground truth camera data (batch_size, seq_len, 12)
        pred_data: Predicted camera data (batch_size, seq_len, 12)
        save_path: Path to save the plot
        title: Title for the plot
        seq_idx: Index of sequence to plot (in case of batch)
    """
    
    # Take specific sequence from batch if needed
    if gt_data.ndim == 3:
        gt_seq = gt_data[seq_idx]
        pred_seq = pred_data[seq_idx]
    else:
        gt_seq = gt_data
        pred_seq = pred_data
    
    # Detect format
    gt_format = detect_camera_format(gt_seq)
    pred_format = detect_camera_format(pred_seq)
    
    if gt_format != CameraFormat.FULL_12 or pred_format != CameraFormat.FULL_12:
        print(f"Warning: This function is designed for 12-feature data. Got GT: {gt_format.value}, Pred: {pred_format.value}")
        # Fall back to standard plotting
        plot_camera_trajectory_3d(gt_data, pred_data, save_path, title, seq_idx)
        return
    
    # Extract all components
    gt_pos, gt_ori, gt_vel, gt_ang_vel = recover_from_camera_data_universal(gt_seq[None, ...])
    pred_pos, pred_ori, pred_vel, pred_ang_vel = recover_from_camera_data_universal(pred_seq[None, ...])
    
    # Remove batch dimension
    gt_pos = gt_pos[0]  # (seq_len, 3)
    gt_ori = gt_ori[0]  # (seq_len, 3)
    gt_vel = gt_vel[0] if gt_vel is not None else None  # (seq_len, 3)
    gt_ang_vel = gt_ang_vel[0] if gt_ang_vel is not None else None  # (seq_len, 3)
    
    pred_pos = pred_pos[0]  # (seq_len, 3)
    pred_ori = pred_ori[0]  # (seq_len, 3)
    pred_vel = pred_vel[0] if pred_vel is not None else None  # (seq_len, 3)
    pred_ang_vel = pred_ang_vel[0] if pred_ang_vel is not None else None  # (seq_len, 3)
    
    # Create figure with 6 subplots (2x3)
    fig = plt.figure(figsize=(20, 12))
    time_steps = np.arange(len(gt_pos))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax1.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], 'r--', linewidth=2, label='Predicted', alpha=0.8)
    ax1.scatter(gt_pos[0, 0], gt_pos[0, 1], gt_pos[0, 2], color='green', s=100, marker='o', label='Start')
    ax1.scatter(gt_pos[-1, 0], gt_pos[-1, 1], gt_pos[-1, 2], color='orange', s=100, marker='s', label='End')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_zlabel('Z Position')
    ax1.set_title('3D Camera Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Position error over time
    ax2 = fig.add_subplot(2, 3, 2)
    position_error = np.sqrt(np.sum((gt_pos - pred_pos) ** 2, axis=1))
    ax2.plot(time_steps, position_error, 'r-', linewidth=2)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Position Error (L2 norm)')
    ax2.set_title('Position Error Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Orientation comparison (pitch, yaw, roll)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(time_steps, gt_ori[:, 0], 'b-', linewidth=2, label='GT Pitch', alpha=0.8)
    ax3.plot(time_steps, pred_ori[:, 0], 'r--', linewidth=2, label='Pred Pitch', alpha=0.8)
    ax3.plot(time_steps, gt_ori[:, 1], 'g-', linewidth=2, label='GT Yaw', alpha=0.8)
    ax3.plot(time_steps, pred_ori[:, 1], 'm--', linewidth=2, label='Pred Yaw', alpha=0.8)
    ax3.plot(time_steps, gt_ori[:, 2], 'c-', linewidth=2, label='GT Roll', alpha=0.8)
    ax3.plot(time_steps, pred_ori[:, 2], 'y--', linewidth=2, label='Pred Roll', alpha=0.8)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Angle (radians)')
    ax3.set_title('Orientation Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Velocity comparison
    if gt_vel is not None and pred_vel is not None:
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(time_steps, gt_vel[:, 0], 'b-', linewidth=2, label='GT Vx', alpha=0.8)
        ax4.plot(time_steps, pred_vel[:, 0], 'r--', linewidth=2, label='Pred Vx', alpha=0.8)
        ax4.plot(time_steps, gt_vel[:, 1], 'g-', linewidth=2, label='GT Vy', alpha=0.8)
        ax4.plot(time_steps, pred_vel[:, 1], 'm--', linewidth=2, label='Pred Vy', alpha=0.8)
        ax4.plot(time_steps, gt_vel[:, 2], 'c-', linewidth=2, label='GT Vz', alpha=0.8)
        ax4.plot(time_steps, pred_vel[:, 2], 'y--', linewidth=2, label='Pred Vz', alpha=0.8)
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Velocity')
        ax4.set_title('Linear Velocity Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Angular velocity comparison
    if gt_ang_vel is not None and pred_ang_vel is not None:
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.plot(time_steps, gt_ang_vel[:, 0], 'b-', linewidth=2, label='GT ωpitch', alpha=0.8)
        ax5.plot(time_steps, pred_ang_vel[:, 0], 'r--', linewidth=2, label='Pred ωpitch', alpha=0.8)
        ax5.plot(time_steps, gt_ang_vel[:, 1], 'g-', linewidth=2, label='GT ωyaw', alpha=0.8)
        ax5.plot(time_steps, pred_ang_vel[:, 1], 'm--', linewidth=2, label='Pred ωyaw', alpha=0.8)
        ax5.plot(time_steps, gt_ang_vel[:, 2], 'c-', linewidth=2, label='GT ωroll', alpha=0.8)
        ax5.plot(time_steps, pred_ang_vel[:, 2], 'y--', linewidth=2, label='Pred ωroll', alpha=0.8)
        ax5.set_xlabel('Time Step')
        ax5.set_ylabel('Angular Velocity (rad/s)')
        ax5.set_title('Angular Velocity Comparison')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # Combined error metrics
    ax6 = fig.add_subplot(2, 3, 6)
    orientation_error = np.sqrt(np.sum((gt_ori - pred_ori) ** 2, axis=1))
    ax6.plot(time_steps, position_error, 'r-', linewidth=2, label='Position Error', alpha=0.8)
    ax6.plot(time_steps, orientation_error, 'b-', linewidth=2, label='Orientation Error', alpha=0.8)
    
    if gt_vel is not None and pred_vel is not None:
        velocity_error = np.sqrt(np.sum((gt_vel - pred_vel) ** 2, axis=1))
        ax6.plot(time_steps, velocity_error, 'g-', linewidth=2, label='Velocity Error', alpha=0.8)
    
    if gt_ang_vel is not None and pred_ang_vel is not None:
        ang_vel_error = np.sqrt(np.sum((gt_ang_vel - pred_ang_vel) ** 2, axis=1))
        ax6.plot(time_steps, ang_vel_error, 'm-', linewidth=2, label='Angular Vel Error', alpha=0.8)
    
    ax6.set_xlabel('Time Step')
    ax6.set_ylabel('Error (L2 norm)')
    ax6.set_title('Combined Error Metrics')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Full dynamics plot saved to: {save_path}")
    print("12-feature format: [x, y, z, dx, dy, dz, pitch, yaw, roll, dpitch, dyaw, droll]") 