import os
from os.path import join as pjoin
from pathlib import Path

import torch
import torch.nn.functional as F

from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from models.vq.model import RVQVAE, LengthEstimator

from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt

from utils.fixseed import fixseed
from utils.plot_script import plot_3d_motion
from utils.unified_data_format import UnifiedCameraData, CameraDataFormat
from utils.dataset_config import get_unified_dataset_config
from utils.camera_geometry import (
    sixd_to_matrix, forward_from_sixd, to_mpl, matrix_to_sixd,
)

import numpy as np
from torch.distributions.categorical import Categorical
from PIL import Image
clip_version = 'ViT-B/32'

def load_vq_model(vq_opt):
    vq_model = RVQVAE(vq_opt,
                vq_opt.dim_pose,
                vq_opt.nb_code,
                vq_opt.code_dim,
                vq_opt.output_emb_width,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)
    
    # Choose checkpoint file based on dataset type
    is_camera_dataset = any(name in vq_opt.dataset_name.lower() for name in ["cam", "estate", "realestate"])
    if is_camera_dataset:
        # For camera datasets, try different checkpoint files in order of preference
        checkpoint_files = [
            'net_best_recon.tar',      # Best reconstruction loss
            'net_best_position.tar',   # Best position accuracy
            'net_best_smoothness.tar', # Best smoothness
            'latest.tar'               # Latest checkpoint
        ]
        
        checkpoint_loaded = False
        for checkpoint_file in checkpoint_files:
            checkpoint_path = pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', checkpoint_file)
            if os.path.exists(checkpoint_path):
                ckpt = torch.load(checkpoint_path, map_location='cpu')
                model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
                vq_model.load_state_dict(ckpt[model_key])
                print(f'Loading VQ Model {vq_opt.name} from {checkpoint_file} Completed!')
                checkpoint_loaded = True
                break
        
        if not checkpoint_loaded:
            raise FileNotFoundError(f"No VQ checkpoint found in {pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model')}")
    else:
        # For human motion datasets, use the original logic
        ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'net_best_fid.tar'),
                                map_location='cpu')
        model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
        vq_model.load_state_dict(ckpt[model_key])
        print(f'Loading VQ Model {vq_opt.name} Completed!')
    
    return vq_model, vq_opt

def load_trans_model(model_opt, opt, which_model):
    conditioning_mode = getattr(opt, 'conditioning_mode', 'clip')
    t2m_transformer = MaskTransformer(code_dim=model_opt.code_dim,
                                      cond_mode='text',
                                      latent_dim=model_opt.latent_dim,
                                      ff_size=model_opt.ff_size,
                                      num_layers=model_opt.n_layers,
                                      num_heads=model_opt.n_heads,
                                      dropout=model_opt.dropout,
                                      clip_dim=512,
                                      cond_drop_prob=model_opt.cond_drop_prob,
                                      clip_version=clip_version,
                                      conditioning_mode=conditioning_mode,
                                      num_id_samples=getattr(opt, 'num_id_samples', 50),
                                      t5_model_name=getattr(opt, 't5_model_name', 't5-base'),
                                      opt=model_opt)
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model),
                      map_location='cpu')
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Transformer {opt.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer

def load_res_model(res_opt, vq_opt, opt):
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens = vq_opt.nb_code
    conditioning_mode = getattr(opt, 'conditioning_mode', 'clip')
    res_transformer = ResidualTransformer(code_dim=vq_opt.code_dim,
                                            cond_mode='text',
                                            latent_dim=res_opt.latent_dim,
                                            ff_size=res_opt.ff_size,
                                            num_layers=res_opt.n_layers,
                                            num_heads=res_opt.n_heads,
                                            dropout=res_opt.dropout,
                                            clip_dim=512,
                                            shared_codebook=vq_opt.shared_codebook,
                                            cond_drop_prob=res_opt.cond_drop_prob,
                                            share_weight=res_opt.share_weight,
                                            clip_version=clip_version,
                                            conditioning_mode=conditioning_mode,
                                            num_id_samples=getattr(opt, 'num_id_samples', 50),
                                            t5_model_name=getattr(opt, 't5_model_name', 't5-base'),
                                            opt=res_opt)

    # Choose checkpoint file based on dataset type
    is_camera_dataset = any(name in res_opt.dataset_name.lower() for name in ["cam", "estate", "realestate"])
    if is_camera_dataset:
        # For camera datasets, try different checkpoint files in order of preference
        checkpoint_files = [
            'net_best_acc.tar',        # Best accuracy
            'net_best_loss.tar',       # Best loss
            'latest.tar'               # Latest checkpoint
        ]
        
        checkpoint_loaded = False
        for checkpoint_file in checkpoint_files:
            checkpoint_path = pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model', checkpoint_file)
            if os.path.exists(checkpoint_path):
                ckpt = torch.load(checkpoint_path, map_location=opt.device)
                checkpoint_loaded = True
                break
        
        if not checkpoint_loaded:
            raise FileNotFoundError(f"No residual transformer checkpoint found in {pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model')}")
    else:
        # For human motion datasets, use the original logic
        ckpt = torch.load(pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model', 'net_best_fid.tar'),
                          map_location=opt.device)
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer'], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Residual Transformer {res_opt.name} from epoch {ckpt["ep"]}!')
    return res_transformer

def load_len_estimator(opt):
    model = LengthEstimator(512, 50)
    
    # Try to load length estimator with smart checkpoint loading
    estimator_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, 'length_estimator', 'model')
    
    # Define checkpoint files to try in order of preference
    checkpoint_files = ['finest.tar', 'latest.tar']
    
    # For camera dataset, also try fallback to t2m length estimator
    is_camera_dataset = any(name in opt.dataset_name.lower() for name in ["cam", "estate", "realestate"])
    if is_camera_dataset:
        t2m_estimator_dir = pjoin(opt.checkpoints_dir, 't2m', 'length_estimator', 'model')
        checkpoint_files.extend([
            pjoin(t2m_estimator_dir, 'finest.tar'),
            pjoin(t2m_estimator_dir, 'latest.tar')
        ])
    
    ckpt = None
    loaded_file = None
    
    # Try each checkpoint file
    for i, filename in enumerate(checkpoint_files):
        if i < 2:  # First two are in dataset-specific directory
            filepath = pjoin(estimator_dir, filename)
        else:  # Fallback files are already full paths
            filepath = filename
            
        if os.path.exists(filepath):
            try:
                ckpt = torch.load(filepath, map_location=opt.device)
                loaded_file = filepath
                break
            except Exception as e:
                print(f'Failed to load {filepath}: {e}')
                continue
    
    if ckpt is None:
        raise FileNotFoundError(f'No valid length estimator checkpoint found. Tried: {checkpoint_files}')
    
    model.load_state_dict(ckpt['estimator'])
    epoch = ckpt.get('epoch', 'unknown')
    
    is_camera_fallback = any(name in opt.dataset_name.lower() for name in ["cam", "estate", "realestate"])
    if loaded_file and 't2m' in loaded_file and is_camera_fallback:
        print(f'Loading Length Estimator from t2m dataset (epoch {epoch}) as fallback for camera dataset!')
    else:
        print(f'Loading Length Estimator from epoch {epoch}!')
    
    return model

def load_keyframes_for_inference(keyframe_dir, keyframe_indices, target_length):
    """
    Load keyframe images and prepare them for inference with SparseKeyframeEncoder.
    
    Args:
        keyframe_dir: Directory containing keyframe images (jpg/png)
        keyframe_indices: List of frame indices where keyframes should be placed
        target_length: Total trajectory length (in raw frames, not downsampled)
    
    Returns:
        List of Path objects representing a sparse frame sequence with keyframes at specified indices
    """
    keyframe_dir = Path(keyframe_dir)
    
    # Get all image files
    image_files = sorted(list(keyframe_dir.glob('*.jpg')) + list(keyframe_dir.glob('*.png')))
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {keyframe_dir}")
    
    if len(image_files) != len(keyframe_indices):
        raise ValueError(f"Number of images ({len(image_files)}) must match number of indices ({len(keyframe_indices)})")
    
    # Validate indices
    for idx in keyframe_indices:
        if idx < 0 or idx >= target_length:
            raise ValueError(f"Keyframe index {idx} out of range [0, {target_length})")
    
    # Create sparse frame path list
    # We'll create a list where most entries are None, and keyframe positions have actual paths
    frame_paths = [None] * target_length
    
    for img_path, frame_idx in zip(image_files, keyframe_indices):
        frame_paths[frame_idx] = img_path
    
    print(f"✓ Loaded {len(image_files)} keyframes at indices: {keyframe_indices}")
    print(f"  Total trajectory length: {target_length} frames")
    
    # Convert to the format expected by dataset (list of Path objects, with placeholder paths for non-keyframes)
    # The encoder will handle None/missing paths appropriately
    return frame_paths

def plot_camera_trajectory_animation(data, save_path, title="Camera Trajectory", 
                                   fps=30, arrow_scale_factor=0.05, 
                                   min_arrow_length=0.01, max_arrow_length=0.2,
                                   show_trail=True, trail_length=30, figsize=(12, 10),
                                   format_type=None, stride=1, rotate_view=True):
    """
    Create an animated 3D visualization of camera trajectory with smooth movement
    Supports multiple camera data formats (5D, 6D, 12D) with automatic detection
    
    Args:
        data: Camera trajectory data (seq_len, features) - supports 5D, 6D, or 12D formats
        save_path: Path to save the animation (supports .gif, .mp4)
        title: Title for the animation
        fps: Frames per second for the animation
        arrow_scale_factor: Factor to scale arrows relative to trajectory extent
        min_arrow_length: Minimum arrow length to ensure visibility
        max_arrow_length: Maximum arrow length to prevent overly long arrows
        show_trail: Whether to show a trail behind the camera
        trail_length: Number of previous positions to show in trail
        figsize: Figure size tuple
        format_type: Explicit format type (CameraDataFormat), if None will auto-detect
        stride: Render every Nth frame (stride=2 halves render time, stride=3 thirds it, etc.)
        rotate_view: Whether to rotate camera view each frame (False = fixed view, much faster)
    """
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    
    # Use unified data format for automatic handling of different dimensions
    unified_data = UnifiedCameraData(data, format_type=format_type)
    raw_positions = unified_data.positions.numpy()  # Always [x, y, z]
    orientations = unified_data.orientations.numpy()  # Depends on format
    
    # OpenGL convention: X=right, Y=up, -Z=forward
    # Visualization: X=right, Y=depth, Z=up
    # Transform via camera_geometry.to_mpl: [x, y, z] -> [x, -z, y]
    positions = to_mpl(raw_positions)
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate dynamic arrow length based on trajectory extent
    pos_ranges = np.ptp(positions, axis=0)
    trajectory_extent = np.max(pos_ranges)
    
    if len(positions) > 1:
        step_distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        avg_step_size = np.mean(step_distances)
        scale_reference = max(trajectory_extent, avg_step_size * 10)
    else:
        scale_reference = trajectory_extent
    
    base_arrow_length = max(arrow_scale_factor * scale_reference, min_arrow_length)
    base_arrow_length = min(base_arrow_length, max_arrow_length)
    
    # Set up plot limits with some padding
    padding = trajectory_extent * 0.1
    ax.set_xlim(positions[:, 0].min() - padding, positions[:, 0].max() + padding)
    ax.set_ylim(positions[:, 1].min() - padding, positions[:, 1].max() + padding)
    ax.set_zlim(positions[:, 2].min() - padding, positions[:, 2].max() + padding)
    
    # Format title
    title_length = len(title)
    if title_length > 100:
        title_fontsize = 8
        title_wrap_width = 80
    elif title_length > 60:
        title_fontsize = 10
        title_wrap_width = 60
    elif title_length > 30:
        title_fontsize = 12
        title_wrap_width = 40
    else:
        title_fontsize = 14
        title_wrap_width = 30
    
    import textwrap
    wrapped_title = '\n'.join(textwrap.wrap(title, width=title_wrap_width))
    
    # Format info with descriptive names
    format_display_names = {
        "LEGACY_5": "5D Legacy",
        "POSITION_ORIENTATION_6": "6D Euler",
        "QUATERNION_10": "10D Quat",
        "FULL_12_EULER": "12D Euler",
        "FULL_12_ROTMAT": "12D RotMat"
    }
    format_name = format_display_names.get(unified_data.format_type.name, unified_data.format_type.name)
    format_info = f"[{format_name}: {unified_data.num_features}D]"
    display_title = f"{wrapped_title}\n{format_info}"
    
    ax.set_title(display_title, fontsize=title_fontsize, pad=20)
    ax.set_xlabel('X (Right)')
    ax.set_ylabel('Depth (Forward)')  # This is now -Z from original data
    ax.set_zlabel('Y (Up)')           # This is now Y from original data
    
    # Initialize empty line and point objects for animation
    trajectory_line, = ax.plot([], [], [], 'b-', linewidth=2, alpha=0.6, label='Full Path')
    trail_line, = ax.plot([], [], [], 'orange', linewidth=3, alpha=0.8, label='Recent Trail')
    current_point = ax.scatter([], [], [], c='red', s=200, label='Current Position')
    orientation_arrow = None
    
    # Pre-compute all orientation vectors (expensive operations done once)
    orientation_vectors = []
    for frame_idx in range(len(positions)):
        ori = orientations[frame_idx]
        
        # Compute forward vector in OpenGL world space, then map to MPL
        if unified_data.format_type == CameraDataFormat.QUATERNION_10:
            from common.quaternion import qrot
            quat = torch.tensor(ori, dtype=torch.float32)
            forward_local = torch.tensor([0.0, 0.0, -1.0])
            fwd_gl = qrot(quat.unsqueeze(0), forward_local.unsqueeze(0)).squeeze(0).numpy()
        elif unified_data.format_type == CameraDataFormat.FULL_12_ROTMAT:
            # Use camera_geometry: forward = -(col0 x col1)
            fwd_gl = forward_from_sixd(ori.reshape(1, 6)).squeeze(0)
        else:
            # Euler-based formats (5D, 6D, 12D Euler)
            pitch, yaw = ori[0], ori[1]
            # OpenGL forward from Euler: direction camera looks
            fwd_gl = np.array([
                np.cos(pitch) * np.sin(yaw),   # X
                -np.sin(pitch),                  # Y
                -np.cos(pitch) * np.cos(yaw)     # -Z (forward)
            ])

        # Deprecated. This may not be correct now.
        
        # Apply to_mpl mapping identically to position transform: [x,y,z] -> [x,-z,y]
        fwd_mpl = to_mpl(fwd_gl.reshape(1, 3)).squeeze(0)
        
        # Normalize
        norm = np.linalg.norm(fwd_mpl)
        if norm > 1e-6:
            fwd_mpl = fwd_mpl / norm
        
        orientation_vectors.append(tuple(fwd_mpl))
    
    # Add start and end markers
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
              c='green', s=150, label='Start', marker='^')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
              c='red', s=150, label='End', marker='v')
    
    ax.legend()
    
    def animate(frame):
        nonlocal orientation_arrow
        
        # Clear previous orientation arrow
        if orientation_arrow is not None:
            orientation_arrow.remove()
        
        # Update full trajectory (fade in effect)
        alpha = min(1.0, frame / 20)  # Fade in over first 20 frames
        trajectory_line.set_data_3d(positions[:frame+1, 0], 
                                   positions[:frame+1, 1], 
                                   positions[:frame+1, 2])
        trajectory_line.set_alpha(alpha * 0.6)
        
        # Update trail
        if show_trail and frame > 0:
            trail_start = max(0, frame - trail_length)
            trail_positions = positions[trail_start:frame+1]
            trail_line.set_data_3d(trail_positions[:, 0], 
                                  trail_positions[:, 1], 
                                  trail_positions[:, 2])
        
        # Update current position
        current_pos = positions[frame]
        current_point._offsets3d = ([current_pos[0]], [current_pos[1]], [current_pos[2]])
        
        # Get pre-computed orientation
        dx, dy, dz = orientation_vectors[frame]
        
        # Draw camera forward direction arrow (where camera looks)
        orientation_arrow = ax.quiver(current_pos[0], current_pos[1], current_pos[2], 
                                     dx, dy, dz, 
                                     length=1.5*base_arrow_length, 
                                     color='purple', alpha=0.8, 
                                     arrow_length_ratio=0.3,
                                     linewidth=2,
                                     label='Camera Forward' if frame == 0 else '')
        
        # Update view angle for dynamic perspective (optional, expensive)
        if rotate_view:
            ax.view_init(elev=20, azim=frame * 0.5 % 360)
        
        return trajectory_line, trail_line, current_point, orientation_arrow
    
    # Create animation with optional frame subsampling
    frame_indices = list(range(0, len(positions), stride))
    # Always include last frame
    if frame_indices[-1] != len(positions) - 1:
        frame_indices.append(len(positions) - 1)
    
    interval = 1000 / fps  # Convert fps to interval in milliseconds
    anim = FuncAnimation(fig, animate, frames=frame_indices, 
                        interval=interval, blit=False, repeat=True)
    
    # Save animation
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if save_path.endswith('.gif'):
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=100)
    elif save_path.endswith('.mp4'):
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='CamTraj'), bitrate=1800)
        anim.save(save_path, writer=writer, dpi=100)
    else:
        # Default to gif
        writer = PillowWriter(fps=fps)
        anim.save(save_path + '.gif', writer=writer, dpi=100)
    
    plt.close()
    print(f"Camera trajectory animation saved to {save_path}")

def plot_camera_trajectory_debug(data, save_path, title="Camera Trajectory Debug", 
                                fps=20, show_velocity=True, show_topdown=True,
                                show_statistics=True, figsize=(15, 12), text_prompt=None,
                                format_type=None, stride=1):
    """
    Create comprehensive debugging visualization with multiple views and analysis
    
    Args:
        data: Camera trajectory data (seq_len, features)
        save_path: Path to save the debug animation
        title: Title for the debug view
        fps: Frames per second
        show_velocity: Whether to show velocity vectors
        show_topdown: Whether to show top-down view (X-Depth plane)
        show_statistics: Whether to show statistical information
        figsize: Figure size tuple
        text_prompt: Optional text prompt to display in statistics panel for comparison
        format_type: Explicit format type (CameraDataFormat), if None will auto-detect
        stride: Render every Nth frame (stride=2 halves render time, etc.)
    """
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    
    # Use unified data format
    unified_data = UnifiedCameraData(data, format_type=format_type)
    raw_positions = unified_data.positions.numpy()
    orientations = unified_data.orientations.numpy()
    
    # OpenGL convention: X=right, Y=up, -Z=forward
    # Transform via camera_geometry.to_mpl: [x, y, z] -> [x, -z, y]
    positions = to_mpl(raw_positions)
    
    # Calculate derivatives for analysis
    velocities = np.zeros_like(positions)
    if len(positions) > 1:
        velocities[1:] = np.diff(positions, axis=0)
    
    # Calculate statistics
    velocity_magnitudes = np.linalg.norm(velocities, axis=1)
    
    # Pre-compute all orientation vectors to avoid repeated calculations
    orientation_vectors_debug = []
    for frame_idx in range(len(positions)):
        ori = orientations[frame_idx]
        
        # Compute forward vector in OpenGL world space
        if unified_data.format_type == CameraDataFormat.QUATERNION_10:
            from common.quaternion import qrot
            quat = torch.tensor(ori, dtype=torch.float32)
            forward_local = torch.tensor([0.0, 0.0, -1.0])
            fwd_gl = qrot(quat.unsqueeze(0), forward_local.unsqueeze(0)).squeeze(0).numpy()
        elif unified_data.format_type == CameraDataFormat.FULL_12_ROTMAT:
            fwd_gl = forward_from_sixd(ori.reshape(1, 6)).squeeze(0)
        else:
            # Euler-based formats
            pitch, yaw = ori[0], ori[1]
            fwd_gl = np.array([
                np.cos(pitch) * np.sin(yaw),
                -np.sin(pitch),
                -np.cos(pitch) * np.cos(yaw)
            ])
        
        # Apply to_mpl mapping: [x,y,z] -> [x,-z,y]
        fwd_mpl = to_mpl(fwd_gl.reshape(1, 3)).squeeze(0)
        
        norm = np.linalg.norm(fwd_mpl)
        if norm > 1e-6:
            fwd_mpl = fwd_mpl / norm
        
        orientation_vectors_debug.append((fwd_mpl[0], fwd_mpl[1], fwd_mpl[2], ori))
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # Main 3D view
    ax_3d = fig.add_subplot(2, 2, 1, projection='3d')
    
    # Velocity plot
    ax_vel = fig.add_subplot(2, 2, 2)
    
    # Top-down view (X-Depth plane)
    ax_topdown = fig.add_subplot(2, 2, 3)
    
    # Statistics text
    ax_stats = fig.add_subplot(2, 2, 4)
    ax_stats.axis('off')
    
    # Set up 3D plot
    padding = np.max(np.ptp(positions, axis=0)) * 0.1
    ax_3d.set_xlim(positions[:, 0].min() - padding, positions[:, 0].max() + padding)
    ax_3d.set_ylim(positions[:, 1].min() - padding, positions[:, 1].max() + padding)
    ax_3d.set_zlim(positions[:, 2].min() - padding, positions[:, 2].max() + padding)
    ax_3d.set_title('3D Camera Trajectory')
    ax_3d.set_xlabel('X (Right)')
    ax_3d.set_ylabel('Depth (Forward)')
    ax_3d.set_zlabel('Y (Up)')
    
    # Set up velocity plot
    ax_vel.set_title('Velocity Magnitude Over Time')
    ax_vel.set_xlabel('Frame')
    ax_vel.set_ylabel('Velocity')
    ax_vel.grid(True)
    
    # Set up top-down view (calculate limits once)
    x_range = positions[:, 0].max() - positions[:, 0].min()
    y_range = positions[:, 1].max() - positions[:, 1].min()
    max_range = max(x_range, y_range)
    center_x = (positions[:, 0].max() + positions[:, 0].min()) / 2
    center_y = (positions[:, 1].max() + positions[:, 1].min()) / 2
    margin = max_range * 0.1
    topdown_xlim = (center_x - max_range/2 - margin, center_x + max_range/2 + margin)
    topdown_ylim = (center_y - max_range/2 - margin, center_y + max_range/2 + margin)
    arrow_scale_3d = np.max(np.ptp(positions, axis=0)) * 0.1
    arrow_scale_2d = np.max(np.ptp(positions[:, :2], axis=0)) * 0.1
    
    # Initialize plot artists (create once, update data in animation)
    traj_3d_line, = ax_3d.plot([], [], [], 'b-', linewidth=2, alpha=0.7)
    curr_3d_point = ax_3d.scatter([], [], [], c='red', s=100)
    
    vel_line, = ax_vel.plot([], [], 'g-', linewidth=2)
    vel_point = ax_vel.scatter([], [], c='red', s=50, zorder=5)
    
    traj_topdown_line, = ax_topdown.plot([], [], 'b-', linewidth=2, alpha=0.7)
    curr_topdown_point = ax_topdown.scatter([], [], c='red', s=100, zorder=5)
    ax_topdown.set_xlim(topdown_xlim)
    ax_topdown.set_ylim(topdown_ylim)
    
    def animate_debug(frame):
        # Get current data
        current_pos = positions[frame]
        dx, dy, dz, ori = orientation_vectors_debug[frame]
        
        # Update 3D trajectory
        if frame > 0:
            traj_3d_line.set_data_3d(positions[:frame+1, 0], positions[:frame+1, 1], positions[:frame+1, 2])
        curr_3d_point._offsets3d = ([current_pos[0]], [current_pos[1]], [current_pos[2]])
        
        # Remove previous arrows if they exist (cannot reuse quiver/arrow objects)
        for artist in list(ax_3d.collections) + list(ax_topdown.patches):
            if artist not in [curr_3d_point, curr_topdown_point]:
                artist.remove()
        
        # Draw orientation arrow in 3D
        if np.sqrt(dx**2 + dy**2 + dz**2) > 1e-6:
            ax_3d.quiver(current_pos[0], current_pos[1], current_pos[2], 
                        dx, dy, dz, length=arrow_scale_3d, color='purple', alpha=0.8)
        
        # Draw velocity vector in 3D
        if show_velocity and frame > 0:
            vel = velocities[frame]
            vel_norm = np.linalg.norm(vel)
            if vel_norm > 1e-6:
                vel_normalized = vel / vel_norm
                vel_length = min(vel_norm * 10, arrow_scale_3d)
                ax_3d.quiver(current_pos[0], current_pos[1], current_pos[2], 
                           vel_normalized[0], vel_normalized[1], vel_normalized[2], 
                           length=vel_length, color='green', alpha=0.6)
        
        # Update velocity plot
        vel_line.set_data(range(frame + 1), velocity_magnitudes[:frame+1])
        vel_point.set_offsets([[frame, velocity_magnitudes[frame]]])
        if frame > 0:
            ax_vel.set_xlim(0, max(10, frame + 1))
            max_vel = max(velocity_magnitudes[:frame+1])
            ax_vel.set_ylim(0, max_vel * 1.1 if max_vel > 0 else 1)
        
        # Update top-down view
        if frame > 0:
            traj_topdown_line.set_data(positions[:frame+1, 0], positions[:frame+1, 1])
        curr_topdown_point.set_offsets([[current_pos[0], current_pos[1]]])
        
        # Draw orientation arrow in top-down view
        if np.sqrt(dx**2 + dy**2 + dz**2) > 1e-6:
            ax_topdown.arrow(current_pos[0], current_pos[1], 
                           dx * arrow_scale_2d, dy * arrow_scale_2d,
                           head_width=arrow_scale_2d*0.3, head_length=arrow_scale_2d*0.2,
                           fc='purple', ec='purple', alpha=0.8)
        
        # Update statistics text
        ax_stats.clear()
        ax_stats.axis('off')
        
        if show_statistics:
            stats_text = f"""Frame: {frame}/{len(positions)-1}

Current Position:
X: {current_pos[0]:.3f}
Y: {current_pos[1]:.3f}  
Z: {current_pos[2]:.3f}

Current Orientation:
Pitch: {ori[0]:.3f}
Yaw: {ori[1]:.3f}
"""
            
            if text_prompt:
                import textwrap
                wrapped_prompt = textwrap.fill(text_prompt, width=35)
                stats_text = f"""TEXT PROMPT:
"{wrapped_prompt}"
{'─' * 40}

{stats_text}"""
            if len(ori) > 2:
                stats_text += f"Roll: {ori[2]:.3f}\n"
                
            if frame > 0:
                stats_text += f"""
Current Velocity: {velocity_magnitudes[frame]:.3f}
Avg Velocity: {np.mean(velocity_magnitudes[1:frame+1]):.3f}
"""
            
            stats_text += f"""
Orientation Vector:
X-component: {dx:.3f}
Y-component: {dy:.3f}
Z-component: {dz:.3f}
"""
            
            if frame > 5:
                recent_positions = positions[max(0, frame-5):frame+1]
                path_length = np.sum(np.linalg.norm(np.diff(recent_positions, axis=0), axis=1))
                smoothness = 1.0 / (1.0 + np.std(velocity_magnitudes[max(1, frame-5):frame+1]))
                
                recent_displacement = recent_positions[-1] - recent_positions[0]
                dominant_axis = np.argmax(np.abs(recent_displacement))
                axis_names = ['X (Right)', 'Depth (Forward)', 'Y (Up)']
                direction = 'positive' if recent_displacement[dominant_axis] > 0 else 'negative'
                
                stats_text += f"""
MOTION ANALYSIS:
Recent Path Length: {path_length:.3f}
Smoothness: {smoothness:.3f}
Dominant Motion: {direction} {axis_names[dominant_axis]}
Speed: {velocity_magnitudes[frame]:.3f}
"""
            
            ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, 
                         fontsize=10, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Create animation with optional frame subsampling
    frame_indices = list(range(0, len(positions), stride))
    if frame_indices[-1] != len(positions) - 1:
        frame_indices.append(len(positions) - 1)
    
    interval = 1000 / fps
    anim = FuncAnimation(fig, animate_debug, frames=frame_indices, 
                        interval=interval, blit=False, repeat=True)
    
    # Save animation
    if save_path.endswith('.gif'):
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=80)
    elif save_path.endswith('.mp4'):
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='CamTraj'), bitrate=1800)
        anim.save(save_path, writer=writer, dpi=80)
    else:
        # Default to mp4 for better quality
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='CamTraj'), bitrate=1800)
        anim.save(save_path + '.mp4', writer=writer, dpi=100)
    
    plt.close()
    print(f"Debug animation saved to {save_path}")

def plot_camera_trajectory(data, save_path, title="Camera Trajectory", arrow_scale_factor=0.05, 
                          min_arrow_length=0.01, max_arrow_length=0.2, format_type=None):
    """
    Plot camera trajectory as a 3D path with dynamically scaled orientation arrows
    Supports multiple camera data formats (5D, 6D, 12D) with automatic detection
    
    Args:
        data: Camera trajectory data (seq_len, features) - supports 5D, 6D, or 12D formats
        save_path: Path to save the plot
        title: Title for the plot
        arrow_scale_factor: Factor to scale arrows relative to trajectory extent (default: 0.05 = 5%)
        min_arrow_length: Minimum arrow length to ensure visibility (default: 0.01)
        max_arrow_length: Maximum arrow length to prevent overly long arrows (default: 0.2)
        format_type: Explicit format type (CameraDataFormat), if None will auto-detect
    """
    # Use unified data format for automatic handling of different dimensions
    unified_data = UnifiedCameraData(data, format_type=format_type)
    raw_positions = unified_data.positions.numpy()
    orientations = unified_data.orientations.numpy()
    
    # OpenGL: X=right, Y=up, -Z=forward -> Viz: X=right, Y=depth, Z=up
    # Transform via camera_geometry.to_mpl: [x, y, z] -> [x, -z, y]
    positions = to_mpl(raw_positions)
    
    # Create a simple 3D plot
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Adaptive font size based on title length
    title_length = len(title)
    if title_length > 100:
        title_fontsize = 8
        title_wrap_width = 80
    elif title_length > 60:
        title_fontsize = 10
        title_wrap_width = 60
    elif title_length > 30:
        title_fontsize = 12
        title_wrap_width = 40
    else:
        title_fontsize = 14
        title_wrap_width = 30
    
    # Wrap long titles
    import textwrap
    wrapped_title = '\n'.join(textwrap.wrap(title, width=title_wrap_width))
    
    # Add format information to title with friendly display names
    format_display_names = {
        "LEGACY_5": "5D Legacy",
        "POSITION_ORIENTATION_6": "6D Euler",
        "QUATERNION_10": "10D Quat",
        "FULL_12_EULER": "12D Euler",
        "FULL_12_ROTMAT": "12D RotMat"
    }
    format_name = format_display_names.get(unified_data.format_type.name, unified_data.format_type.name)
    format_info = f"[{format_name}: {unified_data.num_features}D]"
    display_title = f"{wrapped_title}\n{format_info}"
    
    # Plot camera positions
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Camera Path')
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', s=100, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', s=100, label='End')
    
    # Calculate dynamic arrow length based on trajectory extent
    pos_ranges = np.ptp(positions, axis=0)  # Range (max - min) for each axis
    trajectory_extent = np.max(pos_ranges)  # Maximum extent across all axes
    
    # Also consider average step size for more refined scaling
    if len(positions) > 1:
        step_distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        avg_step_size = np.mean(step_distances)
        # Use the larger of trajectory extent or a multiple of average step size
        # This helps when trajectories are very dense or very sparse
        scale_reference = max(trajectory_extent, avg_step_size * 10)
    else:
        scale_reference = trajectory_extent
    
    # Adaptive arrow length: scale based on trajectory characteristics with user-configurable parameters
    base_arrow_length = max(arrow_scale_factor * scale_reference, min_arrow_length)
    base_arrow_length = min(base_arrow_length, max_arrow_length)
    
    # Plot camera orientations as arrows at key points
    step = max(1, len(positions) // 10)
    for i in range(0, len(positions), step):
        pos = positions[i]
        ori = orientations[i]
        
        # Compute forward vector in OpenGL world space
        if unified_data.format_type == CameraDataFormat.QUATERNION_10:
            from common.quaternion import qrot
            quat = torch.tensor(ori, dtype=torch.float32)
            forward_local = torch.tensor([0.0, 0.0, -1.0])
            fwd_gl = qrot(quat.unsqueeze(0), forward_local.unsqueeze(0)).squeeze(0).numpy()
        elif unified_data.format_type == CameraDataFormat.FULL_12_ROTMAT:
            fwd_gl = forward_from_sixd(ori.reshape(1, 6)).squeeze(0)
        else:
            # Euler-based formats (5D, 6D, 12D Euler)
            pitch, yaw = ori[0], ori[1]
            fwd_gl = np.array([
                np.cos(pitch) * np.sin(yaw),
                -np.sin(pitch),
                -np.cos(pitch) * np.cos(yaw)
            ])
        
        # Apply to_mpl mapping: [x,y,z] -> [x,-z,y]
        fwd_mpl = to_mpl(fwd_gl.reshape(1, 3)).squeeze(0)
        
        # Normalize
        norm = np.linalg.norm(fwd_mpl)
        if norm > 1e-6:
            fwd_mpl = fwd_mpl / norm
        
        dx, dy, dz = fwd_mpl[0], fwd_mpl[1], fwd_mpl[2]
        
        # Draw arrow with adaptive length (orange for camera forward direction)
        ax.quiver(pos[0], pos[1], pos[2], dx, dy, dz, 
                 length=1.5*base_arrow_length, color='orange', alpha=0.7, 
                 arrow_length_ratio=0.3)  # Make arrowhead proportional
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(display_title, fontsize=title_fontsize, pad=20)
    ax.legend()
    
    # Add text annotation showing the arrow scale for reference
    # info_text = f'Arrow scale: {base_arrow_length:.3f}\nTrajectory extent: {trajectory_extent:.3f}'
    # if len(positions) > 1:
    #     info_text += f'\nAvg step size: {avg_step_size:.3f}'
    # ax.text2D(0.02, 0.98, info_text, 
    #           transform=ax.transAxes, fontsize=8, verticalalignment='top',
    #           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    parser = EvalT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    # Get dataset configuration with automatic format detection
    dataset_config = get_unified_dataset_config(opt)
    dim_pose = dataset_config['dim_pose']
    detected_format = dataset_config.get('detected_format', 'Unknown')
    
    # Get format type for correct visualization
    from utils.unified_data_format import detect_format_from_dataset_name
    viz_format_type = detect_format_from_dataset_name(opt.dataset_name)
    
    print(f"Dataset: {opt.dataset_name}")
    print(f"Detected camera format: {detected_format}")
    print(f"Feature dimensions: {dim_pose}")
    print(f"Visualization format: {viz_format_type}")

    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_dir = pjoin(root_dir, 'model')
    result_dir = pjoin('./generation', opt.ext)
    joints_dir = pjoin(result_dir, 'joints')
    animation_dir = pjoin(result_dir, 'animations')
    os.makedirs(joints_dir, exist_ok=True)
    os.makedirs(animation_dir, exist_ok=True)

    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)

    #######################
    ######Loading RVQ######
    #######################
    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_opt.dim_pose = dim_pose
    vq_model, vq_opt = load_vq_model(vq_opt)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    #################################
    ######Loading R-Transformer######
    #################################
    res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, 'opt.txt')
    res_opt = get_opt(res_opt_path, device=opt.device)
    res_model = load_res_model(res_opt, vq_opt, opt)

    assert res_opt.vq_name == model_opt.vq_name

    #################################
    ######Loading M-Transformer######
    #################################
    t2m_transformer = load_trans_model(model_opt, opt, 'latest.tar')

    ##################################
    #####Loading Length Predictor#####
    ##################################
    is_camera_dataset = any(name in opt.dataset_name.lower() for name in ["cam", "estate", "realestate"])
    if is_camera_dataset:
        length_estimator = None
        print("Camera dataset: skipping length estimator (not trained on camera data, uses fixed length).")
    else:
        try:
            length_estimator = load_len_estimator(model_opt)
            length_estimator.eval()
            length_estimator.to(opt.device)
        except:
            length_estimator = None
            print("Length estimator not found, length prediction disabled.")


    t2m_transformer.eval()
    vq_model.eval()
    res_model.eval()


    res_model.to(opt.device)
    t2m_transformer.to(opt.device)
    vq_model.to(opt.device)

    ##### ---- Dataloader ---- #####
    opt.nb_joints = 1  # Camera has no joints

    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'std.npy'))
    def inv_transform(data):
        return data * std + mean

    prompt_list = []
    length_list = []
    conditioning_mode = getattr(opt, 'conditioning_mode', 'clip')

    # ── id_embedding mode: conditions are sample IDs, not text ──
    if conditioning_mode == 'id_embedding':
        sample_ids_str = getattr(opt, 'sample_ids', '')
        num_id_samples = getattr(opt, 'num_id_samples', 50)
        
        if sample_ids_str:
            sample_id_list = [int(x.strip()) for x in sample_ids_str.split(',')]
        else:
            # Generate all IDs
            sample_id_list = list(range(num_id_samples))
        
        # Validate IDs
        for sid in sample_id_list:
            if sid < 0 or sid >= num_id_samples:
                raise ValueError(f"Sample ID {sid} out of range [0, {num_id_samples})")
        
        # Create caption labels for display
        captions = [f"Sample ID: {sid}" for sid in sample_id_list]
        # Create condition tensor (LongTensor of sample IDs)
        cond_ids = torch.LongTensor(sample_id_list).to(opt.device)
        
        # Set lengths 
        if opt.motion_length > 0:
            token_lens = torch.LongTensor([opt.motion_length // 4] * len(sample_id_list)).to(opt.device)
        else:
            default_camera_length = getattr(opt, 'default_camera_length', 200)
            print(f"id_embedding mode: Using FIXED length of {default_camera_length} frames")
            token_lens = torch.LongTensor([default_camera_length // 4] * len(sample_id_list)).to(opt.device)
        
        m_length = token_lens * 4
        print(f"id_embedding mode: generating {len(sample_id_list)} samples (IDs: {sample_id_list})")
        
        # Load ground truth data and text descriptions for comparison
        gt_data_list = []
        gt_lengths = []
        gt_text_descriptions = []  # Actual text prompts for each sample
        train_split_file = pjoin(opt.data_root, 'train.txt')
        if os.path.exists(train_split_file):
            with open(train_split_file, 'r') as f:
                all_sample_names = [line.strip() for line in f.readlines()]
            
            motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
            texts_dir = pjoin(opt.data_root, 'texts')
            for sid in sample_id_list:
                if sid < len(all_sample_names):
                    sample_name = all_sample_names[sid]
                    gt_path = pjoin(motion_dir, f"{sample_name}.npy")
                    if os.path.exists(gt_path):
                        gt_motion = np.load(gt_path)
                        gt_data_list.append(gt_motion)
                        gt_lengths.append(len(gt_motion))
                    else:
                        gt_data_list.append(None)
                        gt_lengths.append(0)
                        print(f"  Warning: GT not found for sample ID {sid} at {gt_path}")
                    
                    # Load text description
                    text_path = pjoin(texts_dir, f"{sample_name}.txt")
                    if os.path.exists(text_path):
                        with open(text_path, 'r') as tf:
                            raw_text = tf.readline().strip()
                            # Text format: "description#POS-tagged version" — take first part
                            gt_text_descriptions.append(raw_text.split('#')[0].strip())
                    else:
                        gt_text_descriptions.append(f"Sample ID: {sid}")
                else:
                    gt_data_list.append(None)
                    gt_lengths.append(0)
                    gt_text_descriptions.append(f"Sample ID: {sid}")
            print(f"  Loaded {sum(1 for g in gt_data_list if g is not None)}/{len(sample_id_list)} ground truth trajectories")
            print(f"  Loaded {sum(1 for t in gt_text_descriptions if not t.startswith('Sample ID'))} text descriptions")
        else:
            print(f"  Warning: train.txt not found at {train_split_file}, GT comparison disabled")
            gt_data_list = [None] * len(sample_id_list)
            gt_text_descriptions = [f"Sample ID: {sid}" for sid in sample_id_list]

    # ── text-based modes (clip / t5) ──
    else:
        est_length = False
        if opt.text_prompt != "":
            prompt_list.append(opt.text_prompt)
            if opt.motion_length == 0:
                est_length = True
            else:
                length_list.append(opt.motion_length)
        elif opt.text_path != "":
            with open(opt.text_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    infos = line.split('#')
                    prompt_list.append(infos[0])
                    if len(infos) == 1 or (not infos[1].isdigit()):
                        est_length = True
                        length_list = []
                    else:
                        length_list.append(int(infos[-1]))
        else:
            raise ValueError("A text prompt, or a file of text prompts are required!!!")

        if est_length:
            default_camera_length = getattr(opt, 'default_camera_length', 200)
            print(f"Camera dataset: Using FIXED length of {default_camera_length} frames (~{default_camera_length/30:.1f}s)")
            print(f"  (Length estimator disabled - not trained on camera data)")
            token_lens = torch.LongTensor([default_camera_length // 4] * len(prompt_list))
            token_lens = token_lens.to(opt.device).long()
        else:
            token_lens = torch.LongTensor(length_list) // 4
            token_lens = token_lens.to(opt.device).long()

        m_length = token_lens * 4
        captions = prompt_list
        cond_ids = None  # Not used in text modes
        gt_data_list = None  # No GT comparison for text modes
        gt_text_descriptions = None
    
    # Load keyframes if specified
    keyframe_paths_list = None
    if opt.use_keyframes and opt.keyframe_dir and opt.keyframe_indices:
        print(f"\n{'='*70}")
        print("Keyframe conditioning enabled for inference")
        print(f"{'='*70}")
        
        # Parse keyframe indices
        keyframe_indices = [int(x.strip()) for x in opt.keyframe_indices.split(',')]
        
        # Load keyframes for each sample (assuming same keyframes for all samples)
        keyframe_paths_list = []
        for i, traj_length in enumerate(m_length.cpu().numpy()):
            frame_paths = load_keyframes_for_inference(opt.keyframe_dir, keyframe_indices, traj_length)
            keyframe_paths_list.append(frame_paths)
        
        # Initialize sparse keyframe encoder (same as in trainer)
        from models.sparse_keyframe_encoder import SparseKeyframeEncoder
        keyframe_arch = getattr(model_opt, 'keyframe_arch', 'resnet18')
        latent_dim = model_opt.latent_dim
        
        sparse_keyframe_encoder = SparseKeyframeEncoder(
            resnet_arch=keyframe_arch,
            latent_dim=latent_dim,
            pretrained=True
        ).to(opt.device)
        sparse_keyframe_encoder.eval()
        
        print(f"✓ SparseKeyframeEncoder loaded (arch={keyframe_arch}, latent_dim={latent_dim})")
        print(f"  Keyframes will condition trajectory generation")
        print(f"{'='*70}\n")
    else:
        sparse_keyframe_encoder = None

    sample = 0

    for r in range(opt.repeat_times):
        print("-->Repeat %d"%r)
        with torch.no_grad():
            # Encode keyframes if provided
            frame_emb_batch = None
            if keyframe_paths_list is not None:
                # Encode frames for this batch
                m_lens_tensor = token_lens * 4  # Original lengths
                frame_emb_batch, has_frames = sparse_keyframe_encoder(
                    keyframe_paths_list, 
                    m_lens_tensor,
                    deterministic=True  # Use deterministic sampling for inference
                )
                
                # Pad to match token sequence length
                target_seq_len = token_lens[0].item()  # Token length (T//4)
                if frame_emb_batch.shape[0] < target_seq_len:
                    pad_len = target_seq_len - frame_emb_batch.shape[0]
                    padding = torch.zeros(pad_len, frame_emb_batch.shape[1], frame_emb_batch.shape[2], 
                                        device=frame_emb_batch.device)
                    frame_emb_batch = torch.cat([frame_emb_batch, padding], dim=0)
                
                print(f"  Frame embeddings: {frame_emb_batch.shape}, has_frames={has_frames}")
            
            # Generate with optional keyframe conditioning
            # Note: We need to modify transformer.generate() to accept frame_emb
            # For now, we'll use a workaround through the forward pass
            
            # Choose conditions based on mode
            gen_conds = cond_ids if conditioning_mode == 'id_embedding' else captions
            
            mids = t2m_transformer.generate(gen_conds, token_lens,
                                            timesteps=opt.time_steps,
                                            cond_scale=opt.cond_scale,
                                            temperature=opt.temperature,
                                            topk_filter_thres=opt.topkr,
                                            gsample=opt.gumbel_sample)
            mids = res_model.generate(mids, gen_conds, token_lens, temperature=1, cond_scale=5)
            pred_motions = vq_model.forward_decoder(mids)

            pred_motions = pred_motions.detach().cpu().numpy()

            data = inv_transform(pred_motions)

        for k, (caption, joint_data) in enumerate(zip(captions, data)):
            print("---->Sample %d: %s %d"%(k, caption, m_length[k]))
            animation_path = pjoin(animation_dir, str(k))
            joint_path = pjoin(joints_dir, str(k))

            os.makedirs(animation_path, exist_ok=True)
            os.makedirs(joint_path, exist_ok=True)

            joint_data = joint_data[:m_length[k]]
            
            # Save raw camera data
            np.save(pjoin(joint_path, "sample%d_repeat%d_len%d_pred.npy"%(k, r, m_length[k])), joint_data)
            
            # Create camera trajectory visualization (both static and animated)
            # Pass viz_format_type so rotmat datasets are not misdetected as euler
            plot_path = pjoin(animation_path, "sample%d_repeat%d_len%d_pred_trajectory.png"%(k, r, m_length[k]))
            plot_camera_trajectory(joint_data, plot_path, title=f"[Pred] {caption}",
                                  format_type=viz_format_type)
            
            # Create animated version for better debugging (MP4 for better quality & smaller size)
            anim_path = pjoin(animation_path, "sample%d_repeat%d_len%d_pred_trajectory.mp4"%(k, r, m_length[k]))
            plot_camera_trajectory_animation(joint_data, anim_path, title=f"[Pred] {caption}",
                                           fps=30, show_trail=True, trail_length=20,
                                           format_type=viz_format_type)
            
            # Create comprehensive debug animation with analysis
            debug_path = pjoin(animation_path, "sample%d_repeat%d_len%d_pred_debug.mp4"%(k, r, m_length[k]))
            plot_camera_trajectory_debug(joint_data, debug_path, title=f"[Pred] {caption}",
                                       fps=30, show_velocity=True, show_topdown=True, 
                                       text_prompt=caption, format_type=viz_format_type)
            
            # ── Ground truth comparison for id_embedding mode ──
            if conditioning_mode == 'id_embedding' and gt_data_list is not None and gt_data_list[k] is not None:
                gt_raw = gt_data_list[k]  # Already in raw (unnormalized) feature space
                gt_len = min(len(gt_raw), m_length[k].item())
                gt_trimmed = gt_raw[:gt_len]
                
                # Save GT raw data
                np.save(pjoin(joint_path, "sample%d_repeat%d_len%d_gt.npy"%(k, r, gt_len)), gt_trimmed)
                
                # GT static trajectory plot
                gt_plot_path = pjoin(animation_path, "sample%d_repeat%d_len%d_gt_trajectory.png"%(k, r, gt_len))
                plot_camera_trajectory(gt_trimmed, gt_plot_path, title=f"[GT] {caption}",
                                      format_type=viz_format_type)
                
                # GT animated trajectory
                gt_anim_path = pjoin(animation_path, "sample%d_repeat%d_len%d_gt_trajectory.mp4"%(k, r, gt_len))
                plot_camera_trajectory_animation(gt_trimmed, gt_anim_path, title=f"[GT] {caption}",
                                               fps=30, show_trail=True, trail_length=20,
                                               format_type=viz_format_type)
                
                # GT debug animation (same comprehensive view as predicted)
                # Use actual text description in sidebar instead of numeric ID
                gt_text = gt_text_descriptions[k] if gt_text_descriptions else caption
                gt_debug_path = pjoin(animation_path, "sample%d_repeat%d_len%d_gt_debug.mp4"%(k, r, gt_len))
                plot_camera_trajectory_debug(gt_trimmed, gt_debug_path, title=f"[GT] {caption}",
                                           fps=30, show_velocity=True, show_topdown=True,
                                           text_prompt=gt_text, format_type=viz_format_type)
                
                print(f"  GT trajectory saved ({gt_len} frames) with trajectory + debug views")
            
            # Save camera data as text file for easy inspection with format-aware headers
            unified_data = UnifiedCameraData(joint_data, format_type=viz_format_type)
            positions = unified_data.positions.numpy()
            orientations = unified_data.orientations.numpy()
            
            # Create format-appropriate header
            if unified_data.format_type == CameraDataFormat.LEGACY_5:
                header = 'x y z pitch yaw'
            elif unified_data.format_type == CameraDataFormat.POSITION_ORIENTATION_6:
                header = 'x y z pitch yaw roll'
            else:  # FULL_12
                header = 'x y z pitch yaw roll'  # Only save position + orientation for readability
            
            camera_data = np.column_stack([positions, orientations])
            np.savetxt(pjoin(joint_path, "sample%d_repeat%d_len%d_pred.txt"%(k, r, m_length[k])), 
                      camera_data, fmt='%.6f', 
                      header=header, comments='')
            
            # Also save format information
            format_info_path = pjoin(joint_path, "sample%d_repeat%d_len%d_pred_format.txt"%(k, r, m_length[k]))
            with open(format_info_path, 'w') as f:
                f.write(f"Original format: {unified_data.format_type.name}\n")
                f.write(f"Original dimensions: {unified_data.num_features}\n")
                f.write(f"Raw data shape: {joint_data.shape}\n")
                f.write(f"Position shape: {positions.shape}\n")
                f.write(f"Orientation shape: {orientations.shape}\n")
                f.write(f"Caption: {caption}\n")

            print(f"Camera trajectory saved to {plot_path}")
            print(f"Camera trajectory animation saved to {anim_path}")
            print(f"Debug animation saved to {debug_path}")
            print(f"Raw data saved to {pjoin(joint_path, 'sample%d_repeat%d_len%d.npy'%(k, r, m_length[k]))}") 

"""
Enhanced Camera Trajectory Generation Script with 3D Animation Support

Supports multiple camera data formats:
- 5D: [x, y, z, pitch, yaw] (legacy cam dataset)
- 6D: [x, y, z, pitch, yaw, roll] (realestate10k_6 dataset)
- 12D: [x, y, z, dx, dy, dz, pitch, yaw, roll, dpitch, dyaw, droll] (full format)

NEW FEATURES:
- 3D animated trajectory visualization for better debugging
- Comprehensive debug animations with velocity/acceleration analysis
- Interactive trail visualization showing camera movement history
- Multi-format support with automatic detection

Output Files Generated Per Sample:
- sample_X_repeat_Y_len_Z_trajectory.png (static 3D plot)
- sample_X_repeat_Y_len_Z_trajectory.mp4 (smooth 3D animation video)
- sample_X_repeat_Y_len_Z_debug.mp4 (comprehensive debug animation video)
- sample_X_repeat_Y_len_Z.npy (raw trajectory data)
- sample_X_repeat_Y_len_Z.txt (human-readable trajectory)
- sample_X_repeat_Y_len_Z_format.txt (format information)

Usage Examples:

CUDA_VISIBLE_DEVICES=3 python gen_camera.py \
    --dataset_name realestate10k_rotmat \
    --name mtrans_overfit50_idcond \
    --res_name rtrans_overfit50_idcond \
    --conditioning_mode id_embedding \
    --gpu_id 0 \
    --sample_ids "0,1,2,3,4,5,6,7,8,9" \
    --repeat_times 2 \
    --time_steps 10 \
    --cond_scale 3 \
    --temperature 1.0 \
    --topkr 0.9 \
    --ext camera_overfit50_idcond

python gen_camera.py \
    --dataset_name realestate10k_quat \
    --name mtrans_reduce_data \
    --res_name rtrans_reduce_data \
    --gpu_id 0 \
    --text_path camera_prompts.txt \
    --repeat_times 3 \
    --time_steps 10 \
    --cond_scale 3 \
    --temperature 1.0 \
    --topkr 0.9 \
    --ext camera_quat_baseline

python gen_camera.py \
  --dataset_name realestate10k_quat \
  --name mtrans_reduce_mid \
  --text_prompt "A camera slowly pans left then zooms in"

# Demo the animation features
python demo_camera_animation.py --demo-type full

# Quick animation demo
python demo_camera_animation.py --demo-type quick

Animation Features:
- Smooth camera movement with orientation arrows
- Dynamic trail showing recent camera positions
- Real-time velocity and acceleration analysis
- Format-aware visualization (5D/6D/12D)
- Statistical debugging information
- Multi-panel debug view with comprehensive analysis

echo "Batch Camera Trajectory Generation with Animations completed!"
echo "Results saved in ./generation/camera_batch_generation/"
echo "Generated static plots, animations, and debug visualizations"
echo "Check the animations folder for .mp4 files with smooth 3D movement"
nvidia-smi 
"""