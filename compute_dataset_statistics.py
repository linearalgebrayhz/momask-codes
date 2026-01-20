#!/usr/bin/env python3
"""
Comprehensive Dataset Statistics Calculator for RealEstate10K Camera Trajectory Dataset

Computes:
1. Number of scenes and frames
2. Average motion length (seconds @ 24fps)
3. Vocabulary statistics and richness metrics
4. Motion distribution (translation and rotation)
5. Caption quality metrics
"""

import numpy as np
import json
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import re
from typing import Dict, List, Tuple, Set


class DatasetStatistics:
    """Compute comprehensive statistics for camera trajectory dataset"""
    
    # Motion detection thresholds (matching realestate10k_processor.py)
    THRESHOLD_TRANSLATION_ABS = 0.10
    THRESHOLD_YAW = 0.12
    THRESHOLD_PITCH = 0.10
    THRESHOLD_ROLL = 0.20
    # Note: Angular speed threshold adjusted for per-frame analysis (not cumulative)
    THRESHOLD_MIN_ANGULAR_SPEED = 0.001  # ~0.06°/frame, allows detection of slow rotations
    THRESHOLD_MIN_TRANSLATION_SPEED = 0.01
    
    def __init__(self, dataset_root: Path, frames_roots: List[Path] = None, fps: float = 24.0):
        self.dataset_root = dataset_root
        self.frames_roots = frames_roots if frames_roots else []
        self.fps = fps
        
        self.motion_dir = dataset_root / "new_joint_vecs"
        self.text_dir = dataset_root / "untagged_text"
        self.metadata_dir = dataset_root / "metadata"
        
        # Statistics containers
        self.stats = {}
        self.motion_distribution = defaultdict(int)
        self.vocabulary = Counter()
        self.caption_lengths = []
        self.sequence_lengths = []
        self.sequence_durations = []
    
    def count_frames_in_directory(self, scene_id: str) -> int:
        """Count frames for a scene in the frames directories"""
        if not self.frames_roots:
            return 0
        
        # Try all provided frames directories
        for frames_root in self.frames_roots:
            # Try multiple possible scene ID formats
            possible_dirs = [
                frames_root / scene_id,
                frames_root / f"{scene_id}",
            ]
            
            for frames_dir in possible_dirs:
                if frames_dir.exists() and frames_dir.is_dir():
                    # Count image files
                    image_files = (
                        list(frames_dir.glob("*.jpg")) +
                        list(frames_dir.glob("*.jpeg")) +
                        list(frames_dir.glob("*.png"))
                    )
                    if image_files:
                        return len(image_files)
        
        return 0
    
    def analyze_vocabulary(self, captions: List[str]) -> Dict:
        """Analyze vocabulary richness and diversity"""
        all_words = []
        
        for caption in captions:
            # Clean and tokenize
            caption_lower = caption.lower()
            # Remove punctuation and split
            words = re.findall(r'\b[a-z]+\b', caption_lower)
            all_words.extend(words)
            self.vocabulary.update(words)
            self.caption_lengths.append(len(words))
        
        # Calculate vocabulary metrics
        total_tokens = len(all_words)
        unique_tokens = len(self.vocabulary)
        
        # Type-Token Ratio (TTR)
        ttr = unique_tokens / total_tokens if total_tokens > 0 else 0
        
        # Root TTR (RTTR) - more stable for large corpora
        rttr = unique_tokens / np.sqrt(total_tokens) if total_tokens > 0 else 0
        
        # Moving-Average Type-Token Ratio (MATTR) - using 100-word windows
        mattr_scores = []
        window_size = 100
        if len(all_words) >= window_size:
            for i in range(len(all_words) - window_size + 1):
                window = all_words[i:i + window_size]
                window_ttr = len(set(window)) / window_size
                mattr_scores.append(window_ttr)
            mattr = np.mean(mattr_scores)
        else:
            mattr = ttr
        
        # Most common words
        most_common_words = self.vocabulary.most_common(20)
        
        vocab_stats = {
            'total_tokens': total_tokens,
            'unique_tokens': unique_tokens,
            'type_token_ratio': ttr,
            'root_ttr': rttr,
            'mattr': mattr,
            'avg_caption_length': np.mean(self.caption_lengths) if self.caption_lengths else 0,
            'median_caption_length': np.median(self.caption_lengths) if self.caption_lengths else 0,
            'most_common_words': most_common_words
        }
        
        return vocab_stats
    
    def classify_motion(self, trajectory: np.ndarray) -> Dict[str, bool]:
        """
        Classify motion types in a trajectory
        Returns dict with motion flags
        """
        positions = trajectory[:, :3]  # [x, y, z]
        orientations = trajectory[:, 3:6]  # [pitch, yaw, roll]
        
        # Calculate total changes
        pos_delta = positions[-1] - positions[0]
        ori_delta = orientations[-1] - orientations[0]
        
        # Calculate velocities
        pos_diffs = np.diff(positions, axis=0)
        ori_diffs = np.diff(orientations, axis=0)
        
        translation_speed = np.mean(np.linalg.norm(pos_diffs, axis=1))
        angular_speed = np.mean(np.linalg.norm(ori_diffs, axis=1))
        
        # Translation detection
        abs_trans = np.abs(pos_delta)
        max_trans = np.max(abs_trans) if abs_trans.size > 0 else 0
        
        motion_flags = {
            'static': False,
            'forward': False,
            'backward': False,
            'left': False,
            'right': False,
            'up': False,
            'down': False,
            'pan_left': False,
            'pan_right': False,
            'tilt_up': False,
            'tilt_down': False,
            'roll_cw': False,
            'roll_ccw': False
        }
        
        # Check if static
        if max_trans < self.THRESHOLD_TRANSLATION_ABS and angular_speed < self.THRESHOLD_MIN_ANGULAR_SPEED:
            motion_flags['static'] = True
            return motion_flags
        
        # Translation motions
        if translation_speed > self.THRESHOLD_MIN_TRANSLATION_SPEED:
            # X-axis: right/left
            if abs_trans[0] >= self.THRESHOLD_TRANSLATION_ABS:
                if pos_delta[0] > 0:
                    motion_flags['right'] = True
                else:
                    motion_flags['left'] = True
            
            # Y-axis: up/down
            if abs_trans[1] >= self.THRESHOLD_TRANSLATION_ABS:
                if pos_delta[1] > 0:
                    motion_flags['up'] = True
                else:
                    motion_flags['down'] = True
            
            # Z-axis: forward/backward (forward = negative Z in OpenGL)
            if abs_trans[2] >= self.THRESHOLD_TRANSLATION_ABS:
                if pos_delta[2] < 0:
                    motion_flags['forward'] = True
                else:
                    motion_flags['backward'] = True
        
        # Rotational motions
        if angular_speed > self.THRESHOLD_MIN_ANGULAR_SPEED:
            # Yaw: pan left/right
            if np.abs(ori_delta[1]) >= self.THRESHOLD_YAW:
                if ori_delta[1] > 0:
                    motion_flags['pan_right'] = True
                else:
                    motion_flags['pan_left'] = True
            
            # Pitch: tilt up/down
            if np.abs(ori_delta[0]) >= self.THRESHOLD_PITCH:
                if ori_delta[0] > 0:
                    motion_flags['tilt_up'] = True
                else:
                    motion_flags['tilt_down'] = True
            
            # Roll: clockwise/counterclockwise
            if np.abs(ori_delta[2]) >= self.THRESHOLD_ROLL:
                if ori_delta[2] > 0:
                    motion_flags['roll_cw'] = True
                else:
                    motion_flags['roll_ccw'] = True
        
        return motion_flags
    
    def compute_statistics(self):
        """Compute all statistics for the dataset"""
        print("Computing dataset statistics...")
        print(f"Dataset root: {self.dataset_root}")
        
        # Get all scene files
        npy_files = sorted(self.motion_dir.glob("*.npy"))
        txt_files = sorted(self.text_dir.glob("*.txt"))
        
        num_scenes = len(npy_files)
        print(f"\nFound {num_scenes} scenes")
        
        # Load scene_id_mapping if available
        mapping_file = self.dataset_root / "scene_id_mapping.json"
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                scene_mapping = json.load(f)
        else:
            scene_mapping = {}
        
        # Process each scene
        captions = []
        total_frames_in_dirs = 0
        scenes_with_frames = 0
        
        print("\nProcessing scenes...")
        for i, npy_file in enumerate(npy_files):
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i+1}/{num_scenes} scenes...")
            
            scene_id = npy_file.stem
            
            # Load trajectory
            trajectory = np.load(npy_file)
            seq_len = len(trajectory)
            duration = seq_len / self.fps
            
            self.sequence_lengths.append(seq_len)
            self.sequence_durations.append(duration)
            
            # Classify motion
            motion_flags = self.classify_motion(trajectory)
            
            # Count motion types
            for motion_type, is_active in motion_flags.items():
                if is_active:
                    self.motion_distribution[motion_type] += 1
            
            # Count motion combinations
            active_motions = [k for k, v in motion_flags.items() if v]
            if len(active_motions) > 1:
                combo_key = '+'.join(sorted(active_motions))
                self.motion_distribution[f"combo:{combo_key}"] += 1
            
            # Load caption
            txt_file = self.text_dir / f"{scene_id}.txt"
            if txt_file.exists():
                with open(txt_file, 'r') as f:
                    caption = f.read().strip()
                    captions.append(caption)
            
            # Count frames in directories if provided
            if self.frames_roots:
                original_id = scene_mapping.get(scene_id, scene_id)
                frame_count = self.count_frames_in_directory(original_id)
                if frame_count > 0:
                    total_frames_in_dirs += frame_count
                    scenes_with_frames += 1
        
        print(f"  Completed processing {num_scenes} scenes")
        
        # Compute aggregate statistics
        self.stats = {
            'num_scenes': num_scenes,
            'num_captions': len(captions),
            'total_trajectory_frames': int(np.sum(self.sequence_lengths)),
            'avg_trajectory_length_frames': float(np.mean(self.sequence_lengths)),
            'median_trajectory_length_frames': float(np.median(self.sequence_lengths)),
            'min_trajectory_length_frames': int(np.min(self.sequence_lengths)),
            'max_trajectory_length_frames': int(np.max(self.sequence_lengths)),
            'avg_duration_seconds': float(np.mean(self.sequence_durations)),
            'median_duration_seconds': float(np.median(self.sequence_durations)),
            'min_duration_seconds': float(np.min(self.sequence_durations)),
            'max_duration_seconds': float(np.max(self.sequence_durations)),
            'fps': self.fps,
        }
        
        # Frame directory statistics
        if self.frames_roots:
            self.stats['frames_directory'] = {
                'total_frames': total_frames_in_dirs,
                'scenes_with_frames': scenes_with_frames,
                'avg_frames_per_scene': total_frames_in_dirs / scenes_with_frames if scenes_with_frames > 0 else 0
            }
        
        # Vocabulary analysis
        print("\nAnalyzing vocabulary...")
        self.stats['vocabulary'] = self.analyze_vocabulary(captions)
        
        # Motion distribution (convert to percentages)
        print("\nComputing motion distribution...")
        motion_dist_counts = dict(self.motion_distribution)
        motion_dist_percentages = {
            k: (v / num_scenes * 100) for k, v in motion_dist_counts.items()
        }
        
        # Separate basic motions from combinations
        basic_motions = {k: v for k, v in motion_dist_percentages.items() if not k.startswith('combo:')}
        combo_motions = {k.replace('combo:', ''): v for k, v in motion_dist_percentages.items() if k.startswith('combo:')}
        
        self.stats['motion_distribution'] = {
            'basic_motions': basic_motions,
            'top_combinations': dict(sorted(combo_motions.items(), key=lambda x: x[1], reverse=True)[:20])
        }
        
        return self.stats
    
    def print_report(self):
        """Print comprehensive statistics report"""
        print("\n" + "=" * 80)
        print("DATASET STATISTICS REPORT")
        print("=" * 80)
        
        # Basic statistics
        print(f"\n{'BASIC STATISTICS':^80}")
        print("-" * 80)
        print(f"Total scenes:                {self.stats['num_scenes']:,}")
        print(f"Total captions:              {self.stats['num_captions']:,}")
        print(f"Total trajectory frames:     {self.stats['total_trajectory_frames']:,}")
        
        # Sequence length statistics
        print(f"\n{'SEQUENCE LENGTH STATISTICS':^80}")
        print("-" * 80)
        print(f"Average length:              {self.stats['avg_trajectory_length_frames']:.1f} frames ({self.stats['avg_duration_seconds']:.2f}s)")
        print(f"Median length:               {self.stats['median_trajectory_length_frames']:.0f} frames ({self.stats['median_duration_seconds']:.2f}s)")
        print(f"Min length:                  {self.stats['min_trajectory_length_frames']} frames ({self.stats['min_duration_seconds']:.2f}s)")
        print(f"Max length:                  {self.stats['max_trajectory_length_frames']} frames ({self.stats['max_duration_seconds']:.2f}s)")
        print(f"Frame rate:                  {self.stats['fps']} fps")
        
        # Frame directory statistics
        if 'frames_directory' in self.stats:
            print(f"\n{'FRAME DIRECTORY STATISTICS':^80}")
            print("-" * 80)
            fd = self.stats['frames_directory']
            print(f"Total frames in directories: {fd['total_frames']:,}")
            print(f"Scenes with frames:          {fd['scenes_with_frames']:,}")
            print(f"Avg frames per scene:        {fd['avg_frames_per_scene']:.1f}")
        
        # Vocabulary statistics
        print(f"\n{'VOCABULARY STATISTICS':^80}")
        print("-" * 80)
        vocab = self.stats['vocabulary']
        print(f"Total tokens:                {vocab['total_tokens']:,}")
        print(f"Unique tokens:               {vocab['unique_tokens']:,}")
        print(f"Type-Token Ratio (TTR):      {vocab['type_token_ratio']:.4f}")
        print(f"Root TTR:                    {vocab['root_ttr']:.4f}")
        print(f"MATTR:                       {vocab['mattr']:.4f}")
        print(f"Avg caption length:          {vocab['avg_caption_length']:.1f} words")
        print(f"Median caption length:       {vocab['median_caption_length']:.1f} words")
        
        print(f"\nMost common words:")
        for word, count in vocab['most_common_words'][:15]:
            print(f"  {word:20s} {count:6,} ({count/vocab['total_tokens']*100:.2f}%)")
        
        # Motion distribution
        print(f"\n{'MOTION DISTRIBUTION (Basic Motions)':^80}")
        print("-" * 80)
        motion_dist = self.stats['motion_distribution']['basic_motions']
        
        # Group by category
        translation = {k: v for k, v in motion_dist.items() if k in ['forward', 'backward', 'left', 'right', 'up', 'down']}
        rotation = {k: v for k, v in motion_dist.items() if k.startswith('pan_') or k.startswith('tilt_') or k.startswith('roll_')}
        other = {k: v for k, v in motion_dist.items() if k not in translation and k not in rotation}
        
        if translation:
            print("\nTranslation motions:")
            for motion, pct in sorted(translation.items(), key=lambda x: x[1], reverse=True):
                print(f"  {motion:20s} {pct:6.2f}%")
        
        if rotation:
            print("\nRotation motions:")
            for motion, pct in sorted(rotation.items(), key=lambda x: x[1], reverse=True):
                print(f"  {motion:20s} {pct:6.2f}%")
        
        if other:
            print("\nOther:")
            for motion, pct in sorted(other.items(), key=lambda x: x[1], reverse=True):
                print(f"  {motion:20s} {pct:6.2f}%")
        
        # Top motion combinations
        print(f"\n{'TOP MOTION COMBINATIONS':^80}")
        print("-" * 80)
        combo_dist = self.stats['motion_distribution']['top_combinations']
        for i, (combo, pct) in enumerate(list(combo_dist.items())[:10], 1):
            print(f"{i:2}. {combo:50s} {pct:6.2f}%")
        
        print("\n" + "=" * 80)
    
    def save_report(self, output_file: Path):
        """Save statistics to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"\nStatistics saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Compute comprehensive dataset statistics")
    parser.add_argument(
        "dataset_root",
        help="Root directory of the dataset (contains new_joint_vecs/, untagged_text/, etc.)"
    )
    parser.add_argument(
        "--frames-root",
        nargs='+',
        help="Root directory/directories containing frame subdirectories (optional, for frame counting). Can specify multiple directories."
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Frame rate for duration calculations (default: 24.0)"
    )
    parser.add_argument(
        "--output",
        help="Output JSON file for statistics (optional)"
    )
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    frames_roots = [Path(p) for p in args.frames_root] if args.frames_root else None
    
    if not dataset_root.exists():
        print(f"Error: Dataset root does not exist: {dataset_root}")
        return 1
    
    # Validate frames directories
    if frames_roots:
        print(f"Frame directories to search:")
        for fr in frames_roots:
            exists = fr.exists()
            print(f"  - {fr} {'✓' if exists else '✗ (not found)'}")
    
    # Compute statistics
    analyzer = DatasetStatistics(dataset_root, frames_roots, args.fps)
    analyzer.compute_statistics()
    
    # Print report
    analyzer.print_report()
    
    # Save to file if requested
    if args.output:
        analyzer.save_report(Path(args.output))
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

