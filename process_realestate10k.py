#!/usr/bin/env python3
"""
Real Estate 10K Dataset Processing Script

This script processes Real Estate 10K camera trajectory data directly from the provided
camera parameters to create MoMask-compatible datasets.

Usage:
    # Process with 6-feature format
    python process_realestate10k.py /path/to/camera/files ./dataset/RealEstate10K_6feat --format 6

    # Process with 12-feature format  
    python process_realestate10k.py /path/to/camera/files ./dataset/RealEstate10K_12feat --format 12

    # Process only specific scenes
    python process_realestate10k.py /path/to/camera/files ./dataset/RealEstate10K --scene-list scene_list.txt
"""

import argparse
import os
import sys
from pathlib import Path

# Add the project root to path
sys.path.append(str(Path(__file__).parent))

from utils.realestate10k_processor import RealEstate10KProcessor

def main():
    parser = argparse.ArgumentParser(
        description="Process Real Estate 10K camera trajectories for MoMask training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "input_dir",
        help="Directory containing Real Estate 10K camera parameter files (.txt files)"
    )
    parser.add_argument(
        "output_dir",
        help="Output directory for processed MoMask dataset"
    )
    parser.add_argument(
        "--format", 
        choices=["6", "12"], 
        default="6",
        help="Output format: '6' for [x,y,z,pitch,yaw,roll] or '12' for full dynamics (default: 6)"
    )
    parser.add_argument(
        "--scene-list", 
        help="Text file containing list of scene IDs to process (one per line)"
    )
    parser.add_argument(
        "--max-scenes", 
        type=int,
        help="Maximum number of scenes to process (for testing)"
    )
    parser.add_argument(
        "--min-length", 
        type=int, 
        default=30,
        help="Minimum sequence length to keep (default: 30)"
    )
    parser.add_argument(
        "--max-length", 
        type=int, 
        default=300,
        help="Maximum sequence length to keep (default: 300)"
    )
    parser.add_argument(
        "--no-smoothing", 
        action="store_true",
        help="Disable Gaussian smoothing of trajectories"
    )
    parser.add_argument(
        "--no-numeric-ids",
        action="store_true", 
        help="Keep original scene IDs instead of converting to numeric format (0XXXX)"
    )
    parser.add_argument(
        "--existing-captions",
        type=str,
        default=None,
        help="Path to existing processed_estate captions directory (e.g., /data5/haozhe/CamTraj/data/processed_estate/train_video_captions)"
    )
    parser.add_argument(
        "--caption-only",
        action="store_true",
        help="Only process scenes that have existing technical captions (for small-scale testing)"
    )
    parser.add_argument(
        "--transform",
        type=str,
        default="relative",
        help="Camera Extrinsic type: 'relative' or 'absolute'"
    )

    parser.add_argument(
        "--caption-motion",
        action="store_true",
        help="Generate caption using relative motion statistics"
    )
    parser.add_argument(
        "--use-ai-captioning",
        action="store_true",
        help="Use Qwen VL to refine deterministic captions with AI-based analysis"
    )
    parser.add_argument(
        "--ai-model",
        type=str,
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="Qwen model to use for AI captioning (default: Qwen/Qwen3-VL-4B-Instruct)"
    )
    parser.add_argument(
        "--ai-max-frames",
        type=int,
        default=32,
        help="Maximum number of frames to use for AI caption generation (default: 32)"
    )
    parser.add_argument(
        "--video-frames-dir",
        type=str,
        default=None,
        help="Directory containing extracted video frames (required for AI captioning)"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use multi-GPU parallel processing for AI captioning (requires multiple GPUs)"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use for parallel processing (default: auto-detect optimal number)"
    )
    parser.add_argument(
        "--ai-batch-size",
        type=int,
        default=2,
        help="Number of scenes to process per inference batch (default: 2, recommended: 2-4)"
    )
    parser.add_argument(
        "--resume-captioning",
        action="store_true",
        help="Resume captioning from previous run - skip scenes that already have text files"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_path}")
        sys.exit(1)
    
    # Count available camera files
    camera_files = list(input_path.glob("*.txt"))
    if not camera_files:
        print(f"Error: No .txt files found in input directory: {input_path}")
        sys.exit(1)
    
    print(f"Found {len(camera_files)} camera parameter files in {input_path}")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Validate AI captioning requirements
    if args.use_ai_captioning and not args.video_frames_dir:
        print("Warning: --use-ai-captioning enabled but --video-frames-dir not specified.")
        print("AI captioning will fall back to deterministic captions if frames are not found.")
    
    # Initialize processor
    processor = RealEstate10KProcessor(
        output_format=args.format,
        min_sequence_length=args.min_length,
        max_sequence_length=args.max_length,
        existing_captions_path=args.existing_captions,
        caption_only=args.caption_only,
        transform=args.transform,
        caption_motion=args.caption_motion,
        use_ai_captioning=args.use_ai_captioning,
        ai_model_name=args.ai_model,
        ai_max_frames=args.ai_max_frames,
        video_source_dir=args.video_frames_dir,
        use_parallel=args.parallel,
        num_gpus=args.num_gpus,
        ai_batch_size=args.ai_batch_size,
        resume_captioning=args.resume_captioning
    )
    
    print(f"Processing Real Estate 10K dataset...")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Format: {args.format}-feature")
    print(f"  Sequence length: {args.min_length}-{args.max_length} frames")
    if args.caption_only:
        print(f"  Caption filtering: Only scenes with existing captions")
    print(f"  Smoothing: {'disabled' if args.no_smoothing else 'enabled'}")
    print(f"  Scene IDs: {'original' if args.no_numeric_ids else 'numeric (XXXXXX)'}")
    print(f"  Transform: {args.transform}")
    print(f"  Caption Motion: {'enabled' if args.caption_motion else 'disabled'}")
    print(f"  AI Captioning: {'enabled' if args.use_ai_captioning else 'disabled'}")
    if args.use_ai_captioning:
        print(f"    Model: {args.ai_model}")
        print(f"    Max frames: {args.ai_max_frames}")
        print(f"    Frames dir: {args.video_frames_dir if args.video_frames_dir else 'Not specified'}")
        print(f"    Parallel: {'enabled' if args.parallel else 'disabled'}")
        print(f"    Resume mode: {'enabled' if args.resume_captioning else 'disabled'}")
        if args.parallel:
            import torch
            available_gpus = torch.cuda.device_count()
            print(f"    Available GPUs: {available_gpus}")
            if args.num_gpus:
                print(f"    Using GPUs: {args.num_gpus}")

    if args.scene_list:
        print(f"  Scene list: {args.scene_list}")
    if args.max_scenes:
        print(f"  Max scenes: {args.max_scenes}")
    if args.existing_captions:
        print(f"  Existing captions: {args.existing_captions}")
    
    print("-" * 60)
    
    # Process dataset
    try:
        stats = processor.process_dataset(
            str(input_path),
            str(output_path),
            args.scene_list,
            args.max_scenes,
            use_numeric_ids=not args.no_numeric_ids,
            apply_smoothing=not args.no_smoothing
        )
        
        print("-" * 60)
        print("   Dataset processing completed successfully!")
        print(f"   Statistics:")
        print(f"   ✓ Processed: {stats['processed']} scenes")
        print(f"   ✗ Failed: {stats['failed']} scenes")
        print(f"   Output directory: {output_path}")
        
        # Show dataset structure
        print(f"\n Generated dataset structure:")
        print(f"   {output_path}/")
        print(f"   ├── new_joint_vecs/     # Camera trajectory data (.npy files)")
        print(f"   ├── texts/              # Text descriptions (.txt files)")
        print(f"   ├── untagged_text/      # Motion-generated captions (.txt files)")
        print(f"   ├── metadata/           # Scene metadata (.json files)")
        print(f"   ├── train.txt           # Training scene IDs")
        print(f"   ├── val.txt             # Validation scene IDs")
        print(f"   ├── test.txt            # Test scene IDs")
        print(f"   ├── Mean.npy            # Dataset mean statistics")
        print(f"   └── Std.npy             # Dataset std statistics")
        
        print(f"\n Ready for MoMask training!")
        print(f"   Use dataset name: 'realestate10k'")
        print(f"   Use data root: '{output_path}'")
        
        # Show example training commands
        print(f"\n Example training commands:")
        print(f"   # VQ training:")
        print(f"   python train_vq.py --dataset_name realestate10k --data_root {output_path} --batch_size 256")
        print(f"   ")
        print(f"   # Transformer training:")
        print(f"   python train_t2m_transformer.py --dataset_name realestate10k --data_root {output_path} --batch_size 64")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

"""
# Example commands:

# Process with existing captions (vision language model generated)
python process_realestate10k.py \
    /data5/haozhe/CamTraj/data/real-state-10k/RealEstate10K/train \
    ./dataset/RealEstate10K_12feat \
    --format 12 \
    --existing-captions /data5/haozhe/CamTraj/data/processed_estate/train_video_captions \
    --caption-only


python process_realestate10k.py \
    /data5/haozhe/CamTraj/data/real-state-10k/RealEstate10K/train \
    ./dataset/RealEstate10K_6feat_motion_test \
    --format 6 \
    --transform relative \
    --caption-motion \
    --use-ai-captioning \
    --video-frames-dir /data4/haozhe/CamTraj/data/processed_estate/train_frames \
    --scene-list test_scenes.txt

# Process with absolute transform and existing captions
python process_realestate10k.py \
    /data5/haozhe/CamTraj/data/real-state-10k/RealEstate10K/train \
    ./dataset/RealEstate10K_6feat_absolute \
    --format 6 \
    --transform absolute \
    --existing-captions /data5/haozhe/CamTraj/data/processed_estate/train_video_captions \
    --caption-only

CUDA_VISIBLE_DEVICES=7 python process_realestate10k.py \0
    /data5/haozhe/CamTraj/data/real-state-10k/RealEstate10K/train \
    ./dataset/RealEstate10K_6feat_qwen \
    --format 6 \
    --transform relative \
    --caption-motion \
    --use-ai-captioning \
    --video-frames-dir /data4/haozhe/CamTraj/data/processed_estate/train_frames \
    --ai-model Qwen/Qwen3-VL-4B-Instruct \
    --ai-batch-size 4 \
    --max-scenes 45000
"""