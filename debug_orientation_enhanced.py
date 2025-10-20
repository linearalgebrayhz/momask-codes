#!/usr/bin/env python3
"""
Enhanced orientation debug with magnified arrows and angle analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
from utils.unified_data_format import UnifiedCameraData, CameraDataFormat

def plot_enhanced_orientation_debug(data, save_path, title="Enhanced Orientation Debug", 
                                  fps=10, arrow_scale=0.2, magnify_factor=5.0, figsize=(16, 10)):
    """
    Enhanced orientation debug with magnified arrows and detailed angle analysis
    
    Args:
        data: Camera trajectory data
        save_path: Path to save animation
        title: Animation title
        fps: Frames per second
        arrow_scale: Base arrow scale
        magnify_factor: Factor to magnify small angle changes
        figsize: Figure size
    """
    
    # Use unified data format
    unified_data = UnifiedCameraData(data)
    raw_positions = unified_data.positions.numpy()
    orientations = unified_data.orientations.numpy()
    
    # Apply coordinate remapping
    positions = np.column_stack([
        raw_positions[:, 0],   # X stays the same (right)
        -raw_positions[:, 2],  # Z becomes Y (forward becomes depth, with sign flip)
        raw_positions[:, 1]    # Y becomes Z (up becomes up)
    ])
    
    # Extract coordinates for top-down view
    x_coords = positions[:, 0]
    y_coords = positions[:, 1]
    
    # Calculate orientation arrows with magnification
    arrow_dx = []
    arrow_dy = []
    raw_angles = []
    
    for i in range(len(orientations)):
        ori = orientations[i]
        pitch, yaw = ori[0], ori[1]
        roll = ori[2] if len(ori) > 2 else 0.0
        
        raw_angles.append([pitch, yaw, roll])
        
        # Calculate original direction
        dx_orig = np.cos(pitch) * np.sin(yaw)
        dy_orig = -np.sin(pitch)
        dz_orig = np.cos(pitch) * np.cos(yaw)
        
        # Apply remapping (no sign flip for direction)
        dx = dx_orig
        dy = dz_orig
        dz = dy_orig
        
        # For small angles, magnify the deviation from forward direction
        # Forward direction is [0, 1, 0] in our coordinate system
        deviation_x = dx
        deviation_y = dy - 1.0  # Subtract the "forward" component
        
        # Magnify small deviations
        if abs(deviation_x) < 0.1:  # Small yaw changes
            deviation_x *= magnify_factor
        if abs(deviation_y) < 0.1:  # Small pitch changes  
            deviation_y *= magnify_factor
        
        # Reconstruct magnified direction
        magnified_dx = deviation_x
        magnified_dy = 1.0 + deviation_y  # Add back the forward component
        
        # Normalize to unit length
        mag = np.sqrt(magnified_dx**2 + magnified_dy**2)
        if mag > 1e-6:
            magnified_dx /= mag
            magnified_dy /= mag
        
        arrow_dx.append(magnified_dx)
        arrow_dy.append(magnified_dy)
    
    arrow_dx = np.array(arrow_dx)
    arrow_dy = np.array(arrow_dy)
    raw_angles = np.array(raw_angles)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=figsize)
    
    # Main trajectory plot (top-left)
    ax1 = plt.subplot(2, 3, (1, 2))
    ax1.plot(x_coords, y_coords, 'b-', alpha=0.6, linewidth=2, label='Trajectory')
    ax1.scatter(x_coords[0], y_coords[0], c='green', s=100, marker='^', label='Start', zorder=5)
    ax1.scatter(x_coords[-1], y_coords[-1], c='red', s=100, marker='v', label='End', zorder=5)
    
    # Add magnified arrows for every few frames
    step = max(1, len(x_coords) // 15)
    for i in range(0, len(x_coords), step):
        ax1.arrow(x_coords[i], y_coords[i], 
                 arrow_dx[i] * arrow_scale, arrow_dy[i] * arrow_scale,
                 head_width=arrow_scale*0.2, head_length=arrow_scale*0.15, 
                 fc='purple', ec='purple', alpha=0.7)
    
    ax1.set_xlabel('X (Right →)')
    ax1.set_ylabel('Depth (Forward →)')
    ax1.set_title(f'Trajectory with Magnified Arrows (×{magnify_factor})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axis('equal')
    
    # Angle plots (right side)
    ax2 = plt.subplot(2, 3, 3)  # Pitch
    ax2.plot(np.degrees(raw_angles[:, 0]), 'r-', linewidth=2)
    ax2.set_title('Pitch Angle')
    ax2.set_ylabel('Degrees')
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(2, 3, 6)  # Yaw
    ax3.plot(np.degrees(raw_angles[:, 1]), 'g-', linewidth=2)
    ax3.set_title('Yaw Angle')
    ax3.set_ylabel('Degrees')
    ax3.set_xlabel('Frame')
    ax3.grid(True, alpha=0.3)
    
    # Animated view (bottom-left)
    ax4 = plt.subplot(2, 3, (4, 5))
    ax4.set_xlabel('X (Right →)')
    ax4.set_ylabel('Depth (Forward →)')
    ax4.set_title('Animated View - Current Position')
    ax4.grid(True, alpha=0.3)
    
    # Set limits for animated view
    margin = 0.15
    x_range = x_coords.max() - x_coords.min()
    y_range = y_coords.max() - y_coords.min()
    if x_range == 0: x_range = 1
    if y_range == 0: y_range = 1
    
    ax4.set_xlim(x_coords.min() - margin * x_range, x_coords.max() + margin * x_range)
    ax4.set_ylim(y_coords.min() - margin * y_range, y_coords.max() + margin * y_range)
    
    # Initialize animated elements
    trail_line, = ax4.plot([], [], 'orange', linewidth=3, alpha=0.8, label='Trail')
    current_point = ax4.scatter([], [], c='red', s=200, label='Current', zorder=5)
    current_arrow = None
    
    # Angle indicators
    pitch_indicator = ax2.axvline(0, color='blue', linewidth=2, alpha=0.7)
    yaw_indicator = ax3.axvline(0, color='blue', linewidth=2, alpha=0.7)
    
    # Debug text
    debug_text = ax4.text(0.02, 0.98, '', transform=ax4.transAxes, 
                         verticalalignment='top', fontfamily='monospace', fontsize=10,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    ax4.legend()
    
    def animate(frame):
        nonlocal current_arrow
        
        # Clear previous arrow
        if current_arrow is not None:
            current_arrow.remove()
        
        # Update trail
        trail_start = max(0, frame - 15)
        trail_x = x_coords[trail_start:frame+1]
        trail_y = y_coords[trail_start:frame+1]
        trail_line.set_data(trail_x, trail_y)
        
        # Update current position
        current_point.set_offsets([[x_coords[frame], y_coords[frame]]])
        
        # Update orientation arrow (magnified)
        current_arrow = ax4.arrow(x_coords[frame], y_coords[frame], 
                                arrow_dx[frame] * arrow_scale, arrow_dy[frame] * arrow_scale,
                                head_width=arrow_scale*0.2, head_length=arrow_scale*0.15, 
                                fc='purple', ec='purple', alpha=0.9, linewidth=2)
        
        # Update angle indicators
        pitch_indicator.set_xdata([frame, frame])
        yaw_indicator.set_xdata([frame, frame])
        
        # Calculate angle from forward direction
        angle_from_forward = np.degrees(np.arctan2(arrow_dx[frame], arrow_dy[frame]))
        
        # Update debug text
        ori = raw_angles[frame]
        debug_info = f"Frame: {frame}/{len(raw_angles)-1}\n"
        debug_info += f"Position: [{x_coords[frame]:.3f}, {y_coords[frame]:.3f}]\n"
        debug_info += f"Raw Angles:\n"
        debug_info += f"  Pitch: {ori[0]:.6f} rad ({np.degrees(ori[0]):6.2f}°)\n"
        debug_info += f"  Yaw:   {ori[1]:.6f} rad ({np.degrees(ori[1]):6.2f}°)\n"
        debug_info += f"  Roll:  {ori[2]:.6f} rad ({np.degrees(ori[2]):6.2f}°)\n"
        debug_info += f"Arrow Direction:\n"
        debug_info += f"  [dx, dy]: [{arrow_dx[frame]:.6f}, {arrow_dy[frame]:.6f}]\n"
        debug_info += f"  Angle from forward: {angle_from_forward:6.2f}°\n"
        
        # Interpretation
        if abs(angle_from_forward) < 5:
            debug_info += "→ FORWARD"
        elif angle_from_forward > 45:
            debug_info += "→ RIGHT"
        elif angle_from_forward < -45:
            debug_info += "→ LEFT"
        elif angle_from_forward > 0:
            debug_info += "→ FORWARD-RIGHT"
        else:
            debug_info += "→ FORWARD-LEFT"
        
        debug_text.set_text(debug_info)
        
        return trail_line, current_point, current_arrow, pitch_indicator, yaw_indicator, debug_text
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(x_coords), 
                        interval=1000//fps, blit=False, repeat=True)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Save animation
    from matplotlib.animation import FFMpegWriter
    if save_path.endswith('.gif'):
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
    elif save_path.endswith('.mp4'):
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='CamTraj'), bitrate=1800)
        anim.save(save_path, writer=writer)
    else:
        # Default to MP4
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='CamTraj'), bitrate=1800)
        anim.save(save_path + '.mp4', writer=writer)
    
    plt.close()

def test_enhanced_debug(sample_id="000025"):
    """Test the enhanced debug visualization"""
    
    DATA_ROOT = "/home/haozhe/CamTraj/momask-codes/dataset/RealEstate10K_6feat_motion"
    
    data_file = f"{DATA_ROOT}/new_joint_vecs/{sample_id}.npy"
    text_file = f"{DATA_ROOT}/untagged_text/{sample_id}.txt"
    
    if not os.path.exists(data_file):
        print(f"Error: Data file not found: {data_file}")
        return
    
    data = np.load(data_file)
    
    try:
        with open(text_file, 'r') as f:
            caption = f.read().strip()
    except:
        caption = "No caption available"
    
    print(f"Sample: {sample_id}")
    print(f"Caption: {caption}")
    print(f"Data shape: {data.shape}")
    
    # Analyze angle ranges
    if data.shape[1] >= 6:
        pitch_range = data[:, 3].max() - data[:, 3].min()
        yaw_range = data[:, 4].max() - data[:, 4].min()
        roll_range = data[:, 5].max() - data[:, 5].min()
        
        print(f"Angle ranges:")
        print(f"  Pitch: {pitch_range:.6f} rad ({np.degrees(pitch_range):.3f}°)")
        print(f"  Yaw:   {yaw_range:.6f} rad ({np.degrees(yaw_range):.3f}°)")
        print(f"  Roll:  {roll_range:.6f} rad ({np.degrees(roll_range):.3f}°)")
        
        # Determine magnification factor based on yaw range
        if yaw_range < 0.1:  # Less than ~6 degrees
            magnify_factor = 10.0
            print(f"  → Using high magnification (×{magnify_factor}) for small yaw changes")
        elif yaw_range < 0.3:  # Less than ~17 degrees
            magnify_factor = 5.0
            print(f"  → Using medium magnification (×{magnify_factor}) for moderate yaw changes")
        else:
            magnify_factor = 2.0
            print(f"  → Using low magnification (×{magnify_factor}) for large yaw changes")
    else:
        magnify_factor = 5.0
    
    # Generate enhanced visualization
    output_path = f"./dataset_vis/RealEstate10K_6feat_motion/{sample_id}_enhanced_debug.mp4"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plot_enhanced_orientation_debug(
        data=data,
        save_path=output_path,
        title=f"Enhanced Debug: {sample_id} - {caption}",
        fps=8,
        magnify_factor=magnify_factor
    )
    
    print(f"✓ Enhanced debug visualization saved to: {output_path}")

if __name__ == "__main__":
    # Test with a sample that has small yaw changes
    test_enhanced_debug("000025")
