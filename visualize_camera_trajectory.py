#!/usr/bin/env python3
"""
Camera Trajectory Visualization Tool

This tool visualizes camera trajectories in 3D space with orientation arrows
and displays text prompts for manual validation of text-motion alignment.

Usage:
    python visualize_camera_trajectory.py --data_id 00006
    python visualize_camera_trajectory.py --data_id 00006 --save_validated
    python visualize_camera_trajectory.py --batch_mode --start_id 0 --end_id 100
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
from pathlib import Path
import json
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, simpledialog
import matplotlib.backends.backend_tkagg as tkagg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class CameraTrajectoryVisualizer:
    def __init__(self, dataset_path="./dataset/CameraTraj"):
        self.dataset_path = Path(dataset_path)
        self.motion_dir = self.dataset_path / "new_joint_vecs"
        self.text_dir = self.dataset_path / "texts"
        # self.validated_dir = self.dataset_path / "validated"
        # self.validated_dir.mkdir(exist_ok=True)
        
        # # Load validation log
        # self.validation_log_file = self.validated_dir / "validation_log.json"
        # self.validation_log = self.load_validation_log()
        
        # GUI components
        self.root = None
        self.fig = None
        self.canvas = None
        self.current_data_id = None
        self.current_text_lines = []
        self.current_motion = None
        
    def load_validation_log(self):
        """Load existing validation log or create new one"""
        if self.validation_log_file.exists():
            with open(self.validation_log_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_validation_log(self):
        """Save validation log to file"""
        with open(self.validation_log_file, 'w') as f:
            json.dump(self.validation_log, f, indent=2)
    
    def load_motion_data(self, data_id):
        """Load motion data for given ID"""
        motion_file = self.motion_dir / f"{data_id}.npy"
        if not motion_file.exists():
            raise FileNotFoundError(f"Motion file not found: {motion_file}")
        
        motion = np.load(motion_file)
        print(f"Loaded motion {data_id}: shape {motion.shape}")
        return motion
    
    def load_text_data(self, data_id):
        """Load text descriptions for given ID"""
        text_file = self.text_dir / f"{data_id}.txt"
        if not text_file.exists():
            raise FileNotFoundError(f"Text file not found: {text_file}")
        
        text_lines = []
        with open(text_file, 'r') as f:
            for i, line in enumerate(f.readlines()):
                line = line.strip()
                if line:
                    # Extract caption (before #)
                    caption = line.split('#')[0].strip()
                    text_lines.append({
                        'line_num': i,
                        'caption': caption,
                        'full_line': line
                    })
        
        print(f"Loaded {len(text_lines)} text descriptions for {data_id}")
        return text_lines
    
    def analyze_motion(self, motion):
        """Analyze motion characteristics"""
        positions = motion[:, :3]  # x, y, z
        orientations = motion[:, 3:]  # pitch, yaw
        
        # Position analysis
        pos_range = {
            'x': (positions[:, 0].min(), positions[:, 0].max()),
            'y': (positions[:, 1].min(), positions[:, 1].max()),
            'z': (positions[:, 2].min(), positions[:, 2].max())
        }
        
        pos_movement = {
            'x': positions[:, 0].max() - positions[:, 0].min(),
            'y': positions[:, 1].max() - positions[:, 1].min(),
            'z': positions[:, 2].max() - positions[:, 2].min()
        }
        
        # Orientation analysis
        ori_range = {
            'pitch': (orientations[:, 0].min(), orientations[:, 0].max()),
            'yaw': (orientations[:, 1].min(), orientations[:, 1].max())
        }
        
        ori_movement = {
            'pitch': orientations[:, 0].max() - orientations[:, 0].min(),
            'yaw': orientations[:, 1].max() - orientations[:, 1].min()
        }
        
        # Motion characteristics
        total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        
        analysis = {
            'frames': len(motion),
            'duration_sec': len(motion) / 30.0,  # Assuming 30 FPS
            'position_range': pos_range,
            'position_movement': pos_movement,
            'orientation_range': ori_range,
            'orientation_movement': ori_movement,
            'total_distance': total_distance,
            'is_static': total_distance < 0.01
        }
        
        return analysis
    
    def plot_trajectory_3d(self, motion, data_id=0, text_lines=None, title="Camera Trajectory", figsize=(16, 8)):
        """Plot 3D camera trajectory with orientation arrows and text descriptions"""
        fig = Figure(figsize=figsize)
        
        # Create subplot layout: 3D plot on left, text on right
        if text_lines:
            ax = fig.add_subplot(121, projection='3d')  # Left subplot for 3D plot
        else:
            ax = fig.add_subplot(111, projection='3d')  # Full plot if no text
        
        positions = motion[:, :3]
        orientations = motion[:, 3:]
        
        # Plot trajectory path
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                'b-', linewidth=2, label='Camera Path', alpha=0.8)
        
        # Mark start and end points
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                  c='green', s=100, label='Start', marker='o')
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                  c='red', s=100, label='End', marker='s')
        
        # Add orientation arrows at key points
        step = max(1, len(positions) // 15)  # Show ~15 arrows
        for i in range(0, len(positions), step):
            pos = positions[i]
            pitch, yaw = orientations[i, 0], orientations[i, 1]
            
            # Convert pitch, yaw to direction vector
            dx = np.cos(pitch) * np.sin(yaw)
            dy = -np.sin(pitch)
            dz = np.cos(pitch) * np.cos(yaw)
            
            # Scale arrow based on movement amount
            arrow_scale = 0.05
            arrow_norm = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            ax.quiver(pos[0], pos[1], pos[2],
                     dx / arrow_norm * arrow_scale, dy / arrow_norm * arrow_scale, dz / arrow_norm * arrow_scale,
                     color='orange', alpha=0.7, arrow_length_ratio=0.3)
        
        # Set labels and title
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.set_title(title + str(data_id))
        ax.legend()
        
        # Equal aspect ratio
        max_range = np.array([positions[:, 0].max() - positions[:, 0].min(),
                             positions[:, 1].max() - positions[:, 1].min(),
                             positions[:, 2].max() - positions[:, 2].min()]).max() / 2.0
        mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
        mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
        mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Add text descriptions in right subplot if provided
        if text_lines:
            # Create text subplot
            ax_text = fig.add_subplot(122)
            ax_text.axis('off')  # Remove axes for text display
            
            # Title for text section
            ax_text.text(0.05, 0.95, "Text Descriptions:", fontsize=14, fontweight='bold', 
                        transform=ax_text.transAxes, verticalalignment='top')
            
            # Calculate available space and line spacing
            available_height = 0.85  # From y=0.9 to y=0.05
            line_height = 0.08  # Increased spacing for better readability
            max_displayable_lines = int(available_height / line_height)
            
            # Determine how many lines to show
            num_lines_to_show = min(len(text_lines), max_displayable_lines)
            
            # Display each text description
            for i in range(num_lines_to_show):
                text_line = text_lines[i]
                y_pos = 0.9 - (i * line_height)
                
                # Extract caption (handle both dict and string formats)
                if isinstance(text_line, dict):
                    caption = text_line.get('caption', str(text_line))
                else:
                    caption = str(text_line)
                
                # Smart text wrapping for longer captions
                max_chars_per_line = 60  # Adjust based on subplot width
                if len(caption) > max_chars_per_line:
                    # Split into multiple lines if too long
                    words = caption.split()
                    lines = []
                    current_line = ""
                    
                    for word in words:
                        if len(current_line + " " + word) <= max_chars_per_line:
                            current_line += (" " + word) if current_line else word
                        else:
                            if current_line:
                                lines.append(current_line)
                            current_line = word
                    
                    if current_line:
                        lines.append(current_line)
                    
                    # Display first line with number, subsequent lines indented
                    ax_text.text(0.05, y_pos, f"{i+1}. {lines[0]}", fontsize=9, 
                                transform=ax_text.transAxes, verticalalignment='top')
                    
                    # Display additional lines if space allows
                    for j, line in enumerate(lines[1:], 1):
                        line_y = y_pos - (j * 0.03)
                        if line_y > 0.05:  # Check if within bounds
                            ax_text.text(0.08, line_y, line, fontsize=9, 
                                        transform=ax_text.transAxes, verticalalignment='top')
                else:
                    # Short caption, display normally
                    ax_text.text(0.05, y_pos, f"{i+1}. {caption}", fontsize=9, 
                                transform=ax_text.transAxes, verticalalignment='top')
            
            # Show pagination info if there are more lines
            if len(text_lines) > num_lines_to_show:
                remaining = len(text_lines) - num_lines_to_show
                ax_text.text(0.05, 0.02, f"... and {remaining} more descriptions", 
                            fontsize=9, style='italic', color='gray',
                            transform=ax_text.transAxes, verticalalignment='bottom')
                
                # Add note about scrolling in GUI mode
                ax_text.text(0.95, 0.02, "(Use GUI for full list)", 
                            fontsize=8, style='italic', color='gray',
                            transform=ax_text.transAxes, verticalalignment='bottom',
                            horizontalalignment='right')
        
        return fig
    
    def create_gui(self):
        """Create interactive GUI for validation"""
        self.root = tk.Tk()
        self.root.title("Camera Trajectory Validator")
        self.root.geometry("1400x900")
        
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls and text
        left_panel = tk.Frame(main_frame, width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Data ID input
        id_frame = tk.Frame(left_panel)
        id_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(id_frame, text="Data ID:").pack(side=tk.LEFT)
        self.id_entry = tk.Entry(id_frame, width=10)
        self.id_entry.pack(side=tk.LEFT, padx=(5, 5))
        self.id_entry.bind('<Return>', self.load_data_callback)
        
        tk.Button(id_frame, text="Load", command=self.load_data_callback).pack(side=tk.LEFT)
        tk.Button(id_frame, text="Random", command=self.load_random_data).pack(side=tk.LEFT, padx=(5, 0))
        
        # Motion analysis display
        self.analysis_text = tk.Text(left_panel, height=8, width=50)
        self.analysis_text.pack(fill=tk.X, pady=(0, 10))
        
        # Text descriptions
        tk.Label(left_panel, text="Text Descriptions:", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        
        # Text list with checkboxes
        self.text_frame = tk.Frame(left_panel)
        self.text_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 10))
        
        # Validation buttons
        button_frame = tk.Frame(left_panel)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Button(button_frame, text="Save Valid Descriptions", 
                 command=self.save_valid_descriptions, bg='lightgreen').pack(fill=tk.X, pady=(0, 5))
        tk.Button(button_frame, text="Mark All Invalid", 
                 command=self.mark_all_invalid, bg='lightcoral').pack(fill=tk.X, pady=(0, 5))
        tk.Button(button_frame, text="Skip This Sample", 
                 command=self.skip_sample, bg='lightgray').pack(fill=tk.X)
        
        # Right panel for plot
        right_panel = tk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Matplotlib canvas
        self.fig = Figure(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                             relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def load_data_callback(self, event=None):
        """Load data based on ID entry"""
        data_id = self.id_entry.get().strip()
        if data_id:
            self.load_and_display_data(data_id)
    
    def load_random_data(self):
        """Load a random data sample"""
        # Get all available motion files
        motion_files = list(self.motion_dir.glob("*.npy"))
        if motion_files:
            random_file = np.random.choice(motion_files)
            data_id = random_file.stem
            self.id_entry.delete(0, tk.END)
            self.id_entry.insert(0, data_id)
            self.load_and_display_data(data_id)
    
    def load_and_display_data(self, data_id):
        """Load and display motion and text data"""
        try:
            self.current_data_id = data_id
            self.current_motion = self.load_motion_data(data_id)
            self.current_text_lines = self.load_text_data(data_id)
            
            # Analyze motion
            analysis = self.analyze_motion(self.current_motion)
            
            # Update analysis display
            self.update_analysis_display(analysis)
            
            # Update text descriptions display
            self.update_text_display()
            
            # Plot trajectory
            self.plot_current_trajectory()
            
            self.status_var.set(f"Loaded data {data_id}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data {data_id}: {str(e)}")
            self.status_var.set(f"Error loading {data_id}")
    
    def update_analysis_display(self, analysis):
        """Update motion analysis text display"""
        self.analysis_text.delete(1.0, tk.END)
        
        text = f"""Motion Analysis:
Frames: {analysis['frames']} ({analysis['duration_sec']:.1f}s)
Total Distance: {analysis['total_distance']:.3f}

Position Movement:
  X: {analysis['position_movement']['x']:.3f}
  Y: {analysis['position_movement']['y']:.3f}
  Z: {analysis['position_movement']['z']:.3f}

Orientation Movement:
  Pitch: {analysis['orientation_movement']['pitch']:.3f} rad
  Yaw: {analysis['orientation_movement']['yaw']:.3f} rad

Status: {'STATIC' if analysis['is_static'] else 'MOVING'}
"""
        self.analysis_text.insert(1.0, text)
    
    def update_text_display(self):
        """Update text descriptions display with checkboxes"""
        # Clear existing widgets
        for widget in self.text_frame.winfo_children():
            widget.destroy()
        
        # Create scrollable frame
        canvas = tk.Canvas(self.text_frame)
        scrollbar = tk.Scrollbar(self.text_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Add text lines with checkboxes
        self.text_vars = []
        for i, text_line in enumerate(self.current_text_lines):
            var = tk.BooleanVar()
            self.text_vars.append(var)
            
            frame = tk.Frame(scrollable_frame)
            frame.pack(fill=tk.X, pady=2)
            
            cb = tk.Checkbutton(frame, variable=var, text=f"{i+1}.")
            cb.pack(side=tk.LEFT)
            
            # Text with word wrap
            text_widget = tk.Text(frame, height=2, wrap=tk.WORD)
            text_widget.insert(1.0, text_line['caption'])
            text_widget.config(state=tk.DISABLED)
            text_widget.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def plot_current_trajectory(self):
        """Plot current trajectory in the GUI"""
        self.fig.clear()
        ax = self.fig.add_subplot(111, projection='3d')
        
        positions = self.current_motion[:, :3]
        orientations = self.current_motion[:, 3:]
        
        # Plot trajectory
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                'b-', linewidth=2, label='Camera Path')
        
        # Start and end points
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                  c='green', s=100, label='Start')
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                  c='red', s=100, label='End')
        
        # Orientation arrows
        step = max(1, len(positions) // 10)
        for i in range(0, len(positions), step):
            pos = positions[i]
            pitch, yaw = orientations[i, 0], orientations[i, 1]
            
            dx = np.cos(pitch) * np.sin(yaw)
            dy = -np.sin(pitch)
            dz = np.cos(pitch) * np.cos(yaw)
            
            ax.quiver(pos[0], pos[1], pos[2], dx*0.1, dy*0.1, dz*0.1,
                     color='orange', alpha=0.7)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Camera Trajectory: {self.current_data_id}')
        ax.legend()
        
        self.canvas.draw()
    
    def save_valid_descriptions(self):
        """Save selected valid descriptions"""
        if not self.current_data_id:
            return
        
        valid_indices = []
        for i, var in enumerate(self.text_vars):
            if var.get():
                valid_indices.append(i)
        
        if not valid_indices:
            messagebox.showwarning("Warning", "No descriptions selected!")
            return
        
        # Save to validation log
        self.validation_log[self.current_data_id] = {
            'timestamp': datetime.now().isoformat(),
            'valid_indices': valid_indices,
            'total_descriptions': len(self.current_text_lines),
            'status': 'validated'
        }
        
        # Save valid descriptions to separate file
        valid_file = self.validated_dir / f"{self.current_data_id}.txt"
        with open(valid_file, 'w') as f:
            for idx in valid_indices:
                text_line = self.current_text_lines[idx]
                f.write(text_line['full_line'] + '\n')
        
        self.save_validation_log()
        
        messagebox.showinfo("Success", 
                           f"Saved {len(valid_indices)} valid descriptions for {self.current_data_id}")
        self.status_var.set(f"Validated {self.current_data_id}")
        
        # Auto-advance to next sample
        self.load_next_sample()
    
    def mark_all_invalid(self):
        """Mark all descriptions as invalid"""
        if not self.current_data_id:
            return
        
        self.validation_log[self.current_data_id] = {
            'timestamp': datetime.now().isoformat(),
            'valid_indices': [],
            'total_descriptions': len(self.current_text_lines),
            'status': 'all_invalid'
        }
        
        self.save_validation_log()
        messagebox.showinfo("Marked", f"Marked all descriptions as invalid for {self.current_data_id}")
        self.load_next_sample()
    
    def skip_sample(self):
        """Skip current sample"""
        if not self.current_data_id:
            return
        
        self.validation_log[self.current_data_id] = {
            'timestamp': datetime.now().isoformat(),
            'status': 'skipped'
        }
        
        self.save_validation_log()
        self.load_next_sample()
    
    def load_next_sample(self):
        """Load next unvalidated sample"""
        # Get all motion files
        motion_files = sorted([f.stem for f in self.motion_dir.glob("*.npy")])
        
        # Find next unvalidated sample
        current_idx = motion_files.index(self.current_data_id) if self.current_data_id in motion_files else -1
        
        for i in range(current_idx + 1, len(motion_files)):
            data_id = motion_files[i]
            if data_id not in self.validation_log:
                self.id_entry.delete(0, tk.END)
                self.id_entry.insert(0, data_id)
                self.load_and_display_data(data_id)
                return
        
        messagebox.showinfo("Complete", "No more unvalidated samples!")
    
    def run_gui(self):
        """Run the GUI application"""
        self.create_gui()
        self.root.mainloop()
    
    def visualize_single(self, data_id, save_path=None):
        """Visualize single trajectory (non-GUI mode)"""
        motion = self.load_motion_data(data_id)
        text_lines = self.load_text_data(data_id)
        analysis = self.analyze_motion(motion)
        
        # Create plot with text descriptions
        fig = self.plot_trajectory_3d(motion, data_id, text_lines, f"Camera Trajectory: {data_id}")
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()
        
        return fig, analysis

def main():
    parser = argparse.ArgumentParser(description="Camera Trajectory Visualization Tool")
    parser.add_argument('--data_id', type=str, help='Specific data ID to visualize')
    parser.add_argument('--dataset_path', type=str, default='./dataset/CameraTraj',
                       help='Path to camera trajectory dataset')
    parser.add_argument('--save_path', type=str, help='Path to save visualization')
    parser.add_argument('--gui', action='store_true', help='Launch interactive GUI')
    parser.add_argument('--batch_mode', action='store_true', help='Batch visualization mode')
    parser.add_argument('--start_id', type=int, default=0, help='Start ID for batch mode')
    parser.add_argument('--end_id', type=int, default=100, help='End ID for batch mode')
    
    args = parser.parse_args()
    
    visualizer = CameraTrajectoryVisualizer(args.dataset_path)
    
    if args.gui:
        visualizer.run_gui()
    elif args.data_id:
        visualizer.visualize_single(args.data_id, args.save_path)
    elif args.batch_mode:
        # Batch visualization
        output_dir = Path("trajectory_visualizations")
        output_dir.mkdir(exist_ok=True)
        
        for i in range(args.start_id, args.end_id):
            data_id = f"{i:05d}"
            try:
                save_path = output_dir / f"{data_id}_trajectory.png"
                visualizer.visualize_single(data_id, save_path)
                print(f"Processed {data_id}")
            except Exception as e:
                print(f"Error processing {data_id}: {e}")
    else:
        print("Please specify --data_id, --gui, or --batch_mode")
        print("Examples:")
        print("  python visualize_camera_trajectory.py --gui")
        print("  python visualize_camera_trajectory.py --data_id 00006")
        print("  python visualize_camera_trajectory.py --batch_mode --start_id 0 --end_id 50")

if __name__ == "__main__":
    main() 