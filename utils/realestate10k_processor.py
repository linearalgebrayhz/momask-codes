#!/usr/bin/env python3
"""
Real Estate 10K Data Processor for MoMask

This module processes Real Estate 10K camera trajectory data directly from the provided
camera parameters, bypassing VGGT reconstruction for more accurate ground truth data.

Real Estate 10K format:
- Line 1: YouTube video URL
- Following lines: 19 columns per frame:
  1. timestamp (microseconds)
  2-5. camera intrinsics (fx, fy, cx, cy)
  6-7: unused
  8-19. camera pose (3x4 matrix in row-major order)

Output formats:
- 6-feature: [x, y, z, pitch, yaw, roll]
- 12-feature: [x, y, z, dx, dy, dz, pitch, yaw, roll, dpitch, dyaw, droll]
"""

import numpy as np
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import argparse
import torch
from PIL import Image
import glob
import cv2

from .unified_data_format import UnifiedCameraData, CameraDataFormat


class QwenVideoCaptioner:
    """Qwen 2.5 VL-based video captioning for camera motion analysis"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        use_flash_attention: bool = True,
        max_frames: int = 32,
        device: Optional[str] = None
    ):
        """
        Initialize Qwen VL model for video captioning
        
        Supported models:
        - Qwen/Qwen2.5-VL-7B-Instruct (original)
        - Qwen/Qwen2.5-VL-32B-Instruct (larger, higher quality)
        - Qwen/Qwen3-VL-8B-Instruct (newer, improved)
        
        Args:
            model_name: HuggingFace model identifier
            use_flash_attention: Whether to use flash attention for efficiency
            max_frames: Maximum number of frames to use for inference
            device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
        """
        self.model_name = model_name
        self.max_frames = max_frames
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
        self.model_loaded = False
        
        # Detect model generation for compatibility
        self.is_qwen3 = "Qwen3" in model_name or "qwen3" in model_name
        
        print(f"QwenVideoCaptioner initialized (device: {self.device})")
        print(f"Model will be loaded on first use: {model_name}")
        if self.is_qwen3:
            print(f"  Detected Qwen3 model - using updated API")
    
    def load_model(self, use_flash_attention: bool = True):
        """Load the Qwen VL model (lazy loading on first use)"""
        if self.model_loaded:
            return
        
        try:
            from transformers import AutoProcessor
            
            print(f"Loading Qwen VL model: {self.model_name}...")
            
            # Import the correct model class based on Qwen version
            if self.is_qwen3:
                from transformers import Qwen3VLForConditionalGeneration
                ModelClass = Qwen3VLForConditionalGeneration
                print("  Using Qwen3VLForConditionalGeneration")
            else:
                from transformers import Qwen2_5_VLForConditionalGeneration
                ModelClass = Qwen2_5_VLForConditionalGeneration
                print("  Using Qwen2_5_VLForConditionalGeneration")
            
            # Load model with appropriate settings
            if use_flash_attention and self.device == 'cuda':
                self.model = ModelClass.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="auto"
                )
            else:
                self.model = ModelClass.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                    device_map="auto" if self.device == 'cuda' else None
                )
                if self.device != 'cuda':
                    self.model = self.model.to(self.device)
            
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model_loaded = True
            
            # Import qwen_vl_utils for Qwen2.5 (Qwen3 may not need it)
            try:
                from qwen_vl_utils import process_vision_info
                self.process_vision_info = process_vision_info
            except ImportError:
                print("  Note: qwen_vl_utils not available (may not be needed for Qwen3)")
                self.process_vision_info = None
            
            print(f"Model loaded successfully on {self.device}!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            if use_flash_attention:
                print("Retrying without flash attention...")
                self.load_model(use_flash_attention=False)
            else:
                raise e
    
    def extract_frames_from_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None
    ) -> List[str]:
        """
        Extract frames uniformly from video file
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract (uses self.max_frames if None)
            
        Returns:
            List of paths to extracted frame images
        """
        if max_frames is None:
            max_frames = self.max_frames
        
        # Create temporary directory for frames
        video_dir = Path(video_path).parent
        video_stem = Path(video_path).stem
        frames_dir = video_dir / f"temp_frames_{video_stem}"
        frames_dir.mkdir(exist_ok=True)
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                print(f"Warning: Could not read video {video_path}")
                return []
            
            # Calculate frame indices to extract (uniform sampling)
            if total_frames <= max_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()
            
            extracted_frames = []
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    frame_path = frames_dir / f"frame_{idx:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    extracted_frames.append(str(frame_path))
            
            cap.release()
            
            return extracted_frames
            
        except Exception as e:
            print(f"Error extracting frames: {e}")
            return []
    
    def sample_existing_frames(
        self,
        frames_dir: str,
        max_frames: Optional[int] = None
    ) -> List[str]:
        """
        Sample frames from a directory of extracted frames
        
        Args:
            frames_dir: Directory containing frame images
            max_frames: Maximum number of frames to sample
            
        Returns:
            List of paths to sampled frames
        """
        if max_frames is None:
            max_frames = self.max_frames
        
        # Find all image files
        frame_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        frame_files = []
        
        for pattern in frame_patterns:
            frame_files.extend(glob.glob(os.path.join(frames_dir, pattern)))
        
        frame_files = sorted(frame_files)
        
        if len(frame_files) == 0:
            return []
        
        # Uniform sampling
        if len(frame_files) <= max_frames:
            return frame_files
        
        indices = np.linspace(0, len(frame_files) - 1, max_frames, dtype=int)
        return [frame_files[i] for i in indices]
    
    def generate_captions_batch(
        self,
        batch_data: List[Tuple[List[str], str, str, int]]
    ) -> List[Dict[str, any]]:
        """
        Generate captions for multiple scenes in a single batch (more efficient)
        
        Note: Vision-language models with variable image inputs are difficult to batch.
        Falls back to one-by-one processing if batching fails.
        
        Args:
            batch_data: List of (frame_paths, deterministic_description, scene_id, total_frames) tuples
            
        Returns:
            List of result dictionaries
        """
        # Lazy load model on first use
        if not self.model_loaded:
            self.load_model()
        
        if len(batch_data) == 0:
            return []
        
        # For vision-language models, batching is complex due to variable image sizes
        # Process one-by-one for stability
        if len(batch_data) > 1:
            results = []
            for frame_paths, deterministic_description, scene_id, total_frames in batch_data:
                result = self.generate_caption_with_context(
                    frame_paths, deterministic_description, scene_id, total_frames
                )
                results.append(result)
            return results
        
        results = []
        batch_messages = []
        valid_indices = []
        
        # Prepare all messages for the batch
        for idx, (frame_paths, deterministic_description, scene_id, total_frames) in enumerate(batch_data):
            if len(frame_paths) == 0:
                results.append({
                    "success": False,
                    "error": "No frames provided",
                    "caption": deterministic_description
                })
                continue
            
            # Design prompt for camera motion refinement
            prompt = f"""You are analyzing camera motion from a video sequence. 

DETERMINISTIC ANALYSIS:
{deterministic_description}

Based on the video frames provided, please:
1. **Verify and refine** the motion description above
2. **Assess smoothness**: Is the camera movement smooth, jittery, or shaky?
3. **Identify intention**: What is the cinematographer trying to achieve? (e.g., reveal scene, follow subject, establish shot, emphasize architecture)
4. **Correct errors**: Fix any inaccuracies in the deterministic analysis if you are confident about the analysis or if there the trajectory has different moving directions with time.

Provide a single, concise, natural sentence describing the camera motion.
Focus on: primary motion type, direction, speed/pace, smoothness quality, and cinematographic purpose.

Format: "The camera [motion] [direction] at [pace], [smoothness/quality], [purpose]."

Example responses:
- "The camera pans right at a slow pace, smoothly revealing the building's facade."
- "The camera dollies forward at a medium pace with slight shake, approaching the entrance."
- "The camera tilts up slowly and steadily, emphasizing the building's height."

Your concise camera motion description:"""
            
            # Prepare message content
            content = []
            for frame_path in frame_paths:
                content.append({
                    "type": "image",
                    "image": frame_path
                })
            content.append({
                "type": "text",
                "text": prompt
            })
            
            batch_messages.append({
                "role": "user",
                "content": content
            })
            valid_indices.append(idx)
            results.append(None)  # Placeholder
        
        if len(batch_messages) == 0:
            return results
        
        try:
            # Process batch
            image_inputs = None
            video_inputs = None
            all_inputs = []
            
            # Prepare inputs for each scene in the batch
            for messages in batch_messages:
                if self.is_qwen3:
                    # Qwen3 API
                    inputs = self.processor.apply_chat_template(
                        [messages],
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt"
                    )
                else:
                    # Qwen2.5 API
                    text = self.processor.apply_chat_template(
                        [messages], tokenize=False, add_generation_prompt=True
                    )
                    img_inputs, vid_inputs = self.process_vision_info([messages])
                    inputs = self.processor(
                        text=[text],
                        images=img_inputs,
                        videos=vid_inputs,
                        padding=True,
                        return_tensors="pt"
                    )
                all_inputs.append(inputs)
            
            # Pad sequences to the same length for batching
            # Find max length in the batch
            max_length = max(inp['input_ids'].shape[1] for inp in all_inputs)
            
            # Pad each input to max_length
            padded_inputs = []
            for inputs in all_inputs:
                padded = {}
                for key, value in inputs.items():
                    if key == 'input_ids':
                        # Pad input_ids with pad_token_id
                        pad_length = max_length - value.shape[1]
                        if pad_length > 0:
                            padding = torch.full((value.shape[0], pad_length), 
                                               self.processor.tokenizer.pad_token_id,
                                               dtype=value.dtype, device=value.device)
                            padded[key] = torch.cat([value, padding], dim=1)
                        else:
                            padded[key] = value
                    elif key == 'attention_mask':
                        # Pad attention_mask with 0s
                        pad_length = max_length - value.shape[1]
                        if pad_length > 0:
                            padding = torch.zeros((value.shape[0], pad_length),
                                                dtype=value.dtype, device=value.device)
                            padded[key] = torch.cat([value, padding], dim=1)
                        else:
                            padded[key] = value
                    elif len(value.shape) >= 2 and value.shape[1] != max_length:
                        # For other sequence-length dependent tensors, pad appropriately
                        pad_length = max_length - value.shape[1]
                        if pad_length > 0:
                            pad_shape = list(value.shape)
                            pad_shape[1] = pad_length
                            padding = torch.zeros(pad_shape, dtype=value.dtype, device=value.device)
                            padded[key] = torch.cat([value, padding], dim=1)
                        else:
                            padded[key] = value
                    else:
                        # Keep as is for non-sequence tensors
                        padded[key] = value
                padded_inputs.append(padded)
            
            # Concatenate padded batch inputs
            batch_inputs = {
                key: torch.cat([inp[key] for inp in padded_inputs], dim=0)
                for key in padded_inputs[0].keys()
            }
            
            batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
            
            # Generate captions for the batch
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **batch_inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.8,
                    top_k=20
                )
            
            # Decode outputs for each scene
            for i, valid_idx in enumerate(valid_indices):
                input_len = all_inputs[i]['input_ids'].shape[1]
                output_ids = generated_ids[i][input_len:]
                caption = self.processor.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
                
                frame_paths, deterministic_description, scene_id, total_frames = batch_data[valid_idx]
                results[valid_idx] = {
                    "success": True,
                    "caption": caption,
                    "num_frames_used": len(frame_paths),
                    "deterministic_input": deterministic_description,
                    "total_frames": total_frames,
                    "scene_id": scene_id
                }
            
            # Clean up GPU memory
            del batch_inputs
            del generated_ids
            del all_inputs
            if image_inputs is not None:
                del image_inputs
            if video_inputs is not None:
                del video_inputs
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(self.model, 'clear_cache'):
                    self.model.clear_cache()
            
            return results
            
        except Exception as e:
            print(f"Error in batch caption generation: {e}")
            import traceback
            traceback.print_exc()
            
            # Clean up on error
            if 'batch_inputs' in locals():
                del batch_inputs
            if 'generated_ids' in locals():
                del generated_ids
            if 'all_inputs' in locals():
                del all_inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Return deterministic captions as fallback
            for idx in valid_indices:
                if results[idx] is None:
                    frame_paths, deterministic_description, scene_id, total_frames = batch_data[idx]
                    results[idx] = {
                        "success": False,
                        "error": str(e),
                        "caption": deterministic_description,
                        "scene_id": scene_id
                    }
            
            return results
    
    def generate_caption_with_context(
        self,
        frame_paths: List[str],
        deterministic_description: str,
        scene_id: str = "",
        total_frames: int = None
    ) -> Dict[str, any]:
        """
        Generate enhanced caption using Qwen VL with deterministic context
        
        Args:
            frame_paths: List of paths to frame images
            deterministic_description: Deterministic motion description as context
            scene_id: Optional scene identifier for logging
            total_frames: Total number of frames in the original sequence
            
        Returns:
            Dictionary with generated caption and metadata
        """
        # Lazy load model on first use
        if not self.model_loaded:
            self.load_model()
        
        if len(frame_paths) == 0:
            return {
                "success": False,
                "error": "No frames provided",
                "caption": deterministic_description
            }
        
        # Design prompt for camera motion refinement
        prompt = f"""You are analyzing camera motion from a video sequence. 

DETERMINISTIC ANALYSIS:
{deterministic_description}

Based on the video frames provided, please:
1. **Verify and refine** the motion description above
2. **Assess smoothness**: Is the camera movement smooth, jittery, or shaky?
3. **Identify intention**: What is the cinematographer trying to achieve? (e.g., reveal scene, follow subject, establish shot, emphasize architecture)
4. **Correct errors**: Fix any inaccuracies in the deterministic analysis if you are confident about the analysis.

Provide a single, concise, natural sentence describing the camera motion.
Focus on: primary motion type, direction, speed/pace, smoothness quality, and cinematographic purpose.

Format: "The camera [motion] [direction] at [pace], [smoothness/quality], [purpose]."

Example responses:
- "The camera pans right at a slow pace, smoothly revealing the building's facade."
- "The camera dollies forward at a medium pace with slight shake, approaching the entrance."
- "The camera tilts up slowly and steadily, emphasizing the building's height."

Your concise camera motion description:"""

        # Prepare messages for the model
        content = []
        
        # Add frames
        for frame_path in frame_paths:
            content.append({
                "type": "image",
                "image": frame_path
            })
        
        # Add text prompt
        content.append({
            "type": "text",
            "text": prompt
        })
        
        messages = [{
            "role": "user",
            "content": content
        }]
        
        try:
            # Prepare inputs differently for Qwen3 vs Qwen2.5
            image_inputs = None
            video_inputs = None
            
            if self.is_qwen3:
                # Qwen3 API: use apply_chat_template with tokenize=True
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                )
                inputs = inputs.to(self.device)
            else:
                # Qwen2.5 API: use process_vision_info + processor
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = self.process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                )
                inputs = inputs.to(self.device)
            
            # Generate caption
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.8,
                    top_k=20
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            caption = output_text[0].strip()
            
            # Aggressive cleanup to prevent memory accumulation
            # Delete all intermediate tensors
            del inputs
            del generated_ids
            del generated_ids_trimmed
            del output_text
            if image_inputs is not None:
                del image_inputs
            if video_inputs is not None:
                del video_inputs
            
            # Clear message content to free image references
            del messages
            del content
            
            # Force Python garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache and KV cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Also clear any cached computations in the model
                if hasattr(self.model, 'clear_cache'):
                    self.model.clear_cache()
            
            return {
                "success": True,
                "caption": caption,
                "num_frames_used": len(frame_paths),
                "deterministic_input": deterministic_description,
                "total_frames": total_frames,
                "scene_id": scene_id
            }
            
        except Exception as e:
            print(f"Error generating caption for scene {scene_id}: {e}")
            # Clean up on error too
            if 'inputs' in locals():
                del inputs
            if 'generated_ids' in locals():
                del generated_ids
            if 'generated_ids_trimmed' in locals():
                del generated_ids_trimmed
            if 'image_inputs' in locals() and image_inputs is not None:
                del image_inputs
            if 'video_inputs' in locals() and video_inputs is not None:
                del video_inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                "success": False,
                "error": str(e),
                "caption": deterministic_description,  # Fallback to deterministic
                "scene_id": scene_id
            }
    
    def cleanup_temp_frames(self, frames_dir: str):
        """Clean up temporary frame directory"""
        try:
            import shutil
            if os.path.exists(frames_dir):
                shutil.rmtree(frames_dir)
        except Exception as e:
            print(f"Warning: Could not cleanup temp frames: {e}")


class RealEstate10KProcessor:
    """Process Real Estate 10K camera trajectory data"""
    
    def __init__(self, output_format: str = "6", min_sequence_length: int = 30, max_sequence_length: int = 300, 
                 existing_captions_path: Optional[str] = None, caption_only: bool = False, transform: str = "relative",
                 caption_motion: bool = False, use_ai_captioning: bool = False, ai_model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
                 ai_max_frames: int = 32, video_source_dir: Optional[str] = None, use_parallel: bool = False, num_gpus: int = None,
                 ai_batch_size: int = 2, resume_captioning: bool = False):
        """
        Initialize processor
        
        Args:
            output_format: "6" for 6-feature or "12" for 12-feature output
            min_sequence_length: Minimum sequence length to keep
            max_sequence_length: Maximum sequence length to keep
            existing_captions_path: Path to existing processed_estate captions directory
            caption_only: If True, only process scenes that have existing captions
            transform: Camera Extrinsic type: 'relative' or 'absolute'
            caption_motion: If True, generate captions based on relative motion analysis
            use_ai_captioning: If True, use Qwen VL to refine deterministic captions
            ai_model_name: Qwen model to use for AI captioning
            ai_max_frames: Maximum frames to use for AI captioning
            video_source_dir: Directory containing source videos (for frame extraction)
            use_parallel: If True, use multi-GPU parallel processing for AI captioning
            num_gpus: Number of GPUs to use for parallel processing (None = auto-detect)
            ai_batch_size: Number of scenes to process per inference batch (default: 2)
            resume_captioning: If True, skip scenes that already have text captions
        """
        self.output_format = output_format
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.existing_captions_path = existing_captions_path
        self.resume_captioning = resume_captioning
        self.caption_only = caption_only
        self.scene_counter = 0  # For sequential numbering
        self.scene_id_mapping = {}  # Maps sequential ID to original hex ID
        self.transform = transform
        self.caption_motion = caption_motion
        self.use_ai_captioning = use_ai_captioning
        self.video_source_dir = video_source_dir
        self.use_parallel = use_parallel
        self.num_gpus = num_gpus
        self.ai_batch_size = ai_batch_size
        
        # Store relative motion data for caption generation
        self.relative_motion_data = None
        
        # Initialize AI captioner based on mode
        self.ai_captioner = None
        self.parallel_captioner = None
        
        if self.use_ai_captioning:
            if self.use_parallel and torch.cuda.device_count() > 1:
                # Use parallel processing
                from .parallel_captioning import ParallelQwenCaptioner, estimate_optimal_gpu_count
                
                # Estimate optimal GPU count
                available_gpus = torch.cuda.device_count()
                if num_gpus is None:
                    recommended_gpus, explanation = estimate_optimal_gpu_count(
                        ai_model_name, available_gpus
                    )
                    num_gpus = recommended_gpus
                    print(f"GPU Recommendation: {explanation}")
                
                print(f"Initializing parallel AI captioning with {num_gpus} GPUs...")
                self.parallel_captioner = ParallelQwenCaptioner(
                    model_name=ai_model_name,
                    num_gpus=num_gpus,
                    max_frames=ai_max_frames
                )
            else:
                # Use single-GPU processing
                if self.use_parallel:
                    print("Warning: Parallel processing requested but only 1 GPU available")
                    print("Falling back to single-GPU mode")
                print("Initializing AI captioning with Qwen VL (single-GPU)...")
                self.ai_captioner = QwenVideoCaptioner(
                    model_name=ai_model_name,
                    max_frames=ai_max_frames
                )
        
        if output_format == "6":
            self.target_format = CameraDataFormat.POSITION_ORIENTATION_6
        elif output_format == "12":
            self.target_format = CameraDataFormat.FULL_12
        else:
            raise ValueError(f"Unsupported output format: {output_format}. Use '6' or '12'")
    
    def parse_camera_file(self, file_path: str) -> Tuple[str, List[Dict]]:
        """
        Parse Real Estate 10K camera parameter file
        RealEstate10K format:
        ...
        
        Args:
            file_path: Path to the camera parameter file
            
        Returns:
            Tuple of (video_url, list of camera_data dictionaries)

        """
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        if not lines:
            raise ValueError(f"Camera file is empty: {file_path}")
        
        video_url = lines[0]
        
        # Parse camera data
        camera_data = []
        relative_motions = []  # Store relative motion data for caption generation
        
        if self.transform == "relative":
            # compute first frame's pose matrix
            values = lines[1].split()
            pose_values = [float(v) for v in values[7:]]
            pose_matrix = np.array(pose_values).reshape(3, 4)
            first_frame_pose_matrix = pose_matrix

            # add first frame as identity
            camera_data.append({
                'timestamp_us': int(float(lines[1].split()[0])),
                'timestamp_s': int(float(lines[1].split()[0])) / 1_000_000,
                'intrinsics': {'fx': float(lines[1].split()[1]), 'fy': float(lines[1].split()[2]), 'cx': float(lines[1].split()[3]), 'cy': float(lines[1].split()[4])},
                'pose_matrix': np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1),
                'frame_index': 1
            })
            
            # Store identity as first relative motion
            relative_motions.append({
                'relative_rotation': np.eye(3),
                'relative_translation': np.zeros(3),
                'timestamp_s': int(float(lines[1].split()[0])) / 1_000_000
            })

        for i, line in enumerate(lines[1:], start= 2 if self.transform == "relative" else 1):
            try:
                values = line.split()
                if len(values) < 19:  # Need exactly 19 values according to official format
                    print(f"Warning: Line {i+1} has only {len(values)} values, expected 19, skipping...")
                    continue
                
                # Parse timestamp (microseconds -> seconds)
                timestamp_us = int(float(values[0]))
                timestamp_s = timestamp_us / 1_000_000
                
                # Parse camera intrinsics (columns 2-6, but we only use 2-5)
                fx = float(values[1])    # Column 2: focal_length_x
                fy = float(values[2])    # Column 3: focal_length_y  
                cx = float(values[3])    # Column 4: principal_point_x
                cy = float(values[4])    # Column 5: principal_point_y
                # values[5] is column 6, which is unused according to format
                
                # Parse camera pose (3x4 matrix in row-major order)
                # Columns 7-19 contain the 12 values of the 3x4 pose matrix
                pose_values = [float(v) for v in values[7:]]
                pose_matrix = np.array(pose_values).reshape(3, 4)

                if self.transform == "relative":
                    abs_rot = pose_matrix[:, :3]
                    abs_trans = pose_matrix[:, 3]
                    relative_rot = first_frame_pose_matrix[:, :3].T @ abs_rot
                    relative_trans = first_frame_pose_matrix[:, :3].T @ (abs_trans - first_frame_pose_matrix[:, 3])
                    pose_matrix = np.concatenate([relative_rot, relative_trans.reshape(-1, 1)], axis=1)
                    
                    # Store relative motion data for caption generation
                    relative_motions.append({
                        'relative_rotation': relative_rot,
                        'relative_translation': relative_trans,
                        'timestamp_s': timestamp_s
                    })
                
                camera_data.append({
                    'timestamp_us': timestamp_us,
                    'timestamp_s': timestamp_s,
                    'intrinsics': {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy},
                    'pose_matrix': pose_matrix,
                    'frame_index': i
                })
                
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse line {i+1}: {line[:50]}... Error: {e}")
                continue
        
        # Store relative motion data for caption generation
        if self.transform == "relative" and self.caption_motion:
            self.relative_motion_data = relative_motions
        
        return video_url, camera_data
    
    def extract_camera_trajectory(self, camera_data: List[Dict]) -> np.ndarray:
        """
        Extract camera trajectory from parsed camera data
        
        Args:
            camera_data: List of camera data dictionaries
            
        Returns:
            Camera trajectory array of shape (seq_len, 6) with [x, y, z, pitch, yaw, roll]
        """
        if not camera_data:
            raise ValueError("No camera data provided")
        
        # Extract positions and rotations
        positions = []
        rotations = []
        
        for frame_data in camera_data:
            pose_matrix = frame_data['pose_matrix']  # 3x4 matrix
            
            # Extract position (translation vector)
            position = pose_matrix[:, 3]  # Last column
            positions.append(position)
            
            # Extract rotation matrix (first 3x3 part)
            rotation_matrix = pose_matrix[:, :3]
            rotations.append(rotation_matrix)
        
        positions = np.array(positions)  # (seq_len, 3)
        rotations = np.array(rotations)   # (seq_len, 3, 3)
        
        # Convert rotation matrices to Euler angles
        euler_angles = self._rotation_matrices_to_euler(rotations)  # (seq_len, 3)
        
        # Combine positions and orientations
        trajectory = np.concatenate([positions, euler_angles], axis=1)  # (seq_len, 6)
        
        return trajectory
    
    def _rotation_matrices_to_euler(self, rotation_matrices: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrices to Euler angles (pitch, yaw, roll)
        
        Args:
            rotation_matrices: Array of shape (seq_len, 3, 3)
            
        Returns:
            euler_angles: Array of shape (seq_len, 3) with [pitch, yaw, roll]
        """
        seq_len = rotation_matrices.shape[0]
        euler_angles = np.zeros((seq_len, 3))
        
        for i in range(seq_len):
            try:
                rotation_matrix = rotation_matrices[i]
                
                # Validate rotation matrix
                det = np.linalg.det(rotation_matrix)
                if det < 0:
                    print(f"Warning: Negative determinant ({det:.6f}) in rotation matrix at frame {i}")
                    print(f"Matrix:\n{rotation_matrix}")
                    # Try to fix by negating the matrix (this might be a reflection)
                    rotation_matrix = -rotation_matrix
                    det = np.linalg.det(rotation_matrix)
                    print(f"After negation, determinant: {det:.6f}")
                
                # Check if it's close to orthogonal
                should_be_identity = rotation_matrix @ rotation_matrix.T
                identity_error = np.linalg.norm(should_be_identity - np.eye(3))
                if identity_error > 0.01:  # Stricter threshold
                    print(f"Warning: Rotation matrix at frame {i} is not orthogonal (error: {identity_error:.6f})")
                    
                    if identity_error > 2.0:
                        print(f"  SEVERE corruption detected! This matrix is far from being a rotation.")
                        print(f"  Original matrix:\n{rotation_matrix}")
                    
                    # Use more robust orthogonalization methods
                    if identity_error < 1.0:
                        # For moderate errors, use Polar decomposition (more stable)
                        try:
                            U, s, Vt = np.linalg.svd(rotation_matrix)
                            # Ensure proper rotation (det = 1, not -1)
                            if np.linalg.det(U @ Vt) < 0:
                                Vt[-1, :] *= -1
                            rotation_matrix = U @ Vt
                            print(f"  Matrix orthogonalized using Polar decomposition")
                        except Exception as e:
                            print(f"  Failed Polar decomposition: {e}, falling back to SVD")
                            U, _, Vt = np.linalg.svd(rotation_matrix)
                            rotation_matrix = U @ Vt
                    else:
                        # For severe errors, this might not be a rotation matrix at all
                        print(f"Matrix too corrupted for reliable recovery, using identity")
                        rotation_matrix = np.eye(3)
                        if i > 0:
                            # Try to use previous frame's rotation instead  
                            # We'll use identity for now, but could store previous rotations
                            rotation_matrix = np.eye(3)
                            print(f"  Using identity matrix as fallback")
                    
                    # Verify the fix worked
                    new_error = np.linalg.norm(rotation_matrix @ rotation_matrix.T - np.eye(3))
                    print(f"  After correction, orthogonality error: {new_error:.6f}")
                
                # Use scipy to convert rotation matrix to Euler angles
                # 'xyz' order gives us [roll, pitch, yaw], so we need to reorder
                r = R.from_matrix(rotation_matrix)
                roll, pitch, yaw = r.as_euler('xyz', degrees=False)
                
                # Store as [pitch, yaw, roll] for consistency with existing format
                euler_angles[i] = [pitch, yaw, roll]
                
            except Exception as e:
                print(f"Warning: Could not convert rotation matrix at frame {i}: {e}")
                print(f"Matrix:\n{rotation_matrices[i]}")
                # Use previous frame's rotation or zeros
                if i > 0:
                    euler_angles[i] = euler_angles[i-1]
                else:
                    euler_angles[i] = [0, 0, 0]
        
        return euler_angles
    
    def smooth_trajectory(self, trajectory: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        Apply Gaussian smoothing to trajectory to reduce noise
        
        Args:
            trajectory: Input trajectory array
            sigma: Gaussian smoothing parameter
            
        Returns:
            Smoothed trajectory
        """
        from scipy.ndimage import gaussian_filter1d
        
        smoothed = np.zeros_like(trajectory)
        for i in range(trajectory.shape[1]):
            smoothed[:, i] = gaussian_filter1d(trajectory[:, i], sigma=sigma)
        
        return smoothed
    
    def _convert_to_numeric_id(self, scene_id: str) -> str:
        """
        Convert scene ID to sequential numeric format (000000-999999)
        
        Args:
            scene_id: Original scene identifier (hex ID from filename)
            
        Returns:
            Sequential numeric scene ID in format XXXXXX (6 digits)
        """
        # Use sequential counter for ordered IDs
        numeric_id = f"{self.scene_counter:06d}"
        
        # Store mapping from numeric ID to original hex ID for caption lookup
        self.scene_id_mapping[numeric_id] = scene_id
        
        self.scene_counter += 1
        return numeric_id
    
    def _process_scene_trajectory(self, camera_file: str, output_dir: str, scene_id: str,
                                 apply_smoothing: bool = True, use_numeric_id: bool = True) -> Tuple[bool, Dict]:
        """
        Process scene trajectory and generate deterministic caption (without AI refinement)
        Used for parallel processing workflow.
        
        Returns:
            Tuple of (success: bool, metadata: Dict)
        """
        try:
            # Convert scene_id to numeric format if requested
            if use_numeric_id:
                numeric_scene_id = self._convert_to_numeric_id(scene_id)
                print(f"Processing scene: {scene_id} -> {numeric_scene_id}")
            else:
                numeric_scene_id = scene_id
                print(f"Processing scene: {scene_id}")
            
            # Parse camera file
            video_url, camera_data = self.parse_camera_file(camera_file)
            
            if len(camera_data) < self.min_sequence_length:
                print(f"Scene {scene_id} too short ({len(camera_data)} frames), skipping...")
                return False, {}
            
            if len(camera_data) > self.max_sequence_length:
                print(f"Scene {scene_id} too long ({len(camera_data)} frames), truncating...")
                camera_data = camera_data[:self.max_sequence_length]
            
            # Extract camera trajectory
            trajectory_6d = self.extract_camera_trajectory(camera_data)
            
            # Apply smoothing if requested
            if apply_smoothing:
                trajectory_6d = self.smooth_trajectory(trajectory_6d, sigma=1.0)
            
            # Convert to target format using unified data format
            unified_data = UnifiedCameraData(trajectory_6d, CameraDataFormat.POSITION_ORIENTATION_6)
            final_trajectory = unified_data.to_format(self.target_format)
            
            # Create output directories
            output_path = Path(output_dir)
            motion_dir = output_path / "new_joint_vecs"
            motion_dir.mkdir(parents=True, exist_ok=True)
            
            # Save trajectory data
            motion_file = motion_dir / f"{numeric_scene_id}.npy"
            np.save(motion_file, final_trajectory.get_momask_compatible_data().numpy())
            
            # Generate deterministic caption
            original_scene_id = self.scene_id_mapping.get(numeric_scene_id, scene_id)
            deterministic_caption = self.generate_caption_motion()
            
            # Return metadata for AI processing
            metadata = {
                'scene_id': scene_id,
                'numeric_scene_id': numeric_scene_id,
                'original_scene_id': original_scene_id,
                'deterministic_caption': deterministic_caption,
                'total_frames': len(camera_data),
                'video_url': video_url
            }
            
            return True, metadata
            
        except Exception as e:
            print(f"Error processing scene {scene_id}: {e}")
            import traceback
            traceback.print_exc()
            return False, {}
    
    def process_scene(self, camera_file: str, output_dir: str, scene_id: str, 
                     apply_smoothing: bool = True, generate_captions: bool = True,
                     use_numeric_id: bool = True) -> bool:
        """
        Process a single scene from Real Estate 10K data
        
        Args:
            camera_file: Path to camera parameter file
            output_dir: Output directory for processed data
            scene_id: Scene identifier (will be converted to numeric format if use_numeric_id=True)
            apply_smoothing: Whether to apply Gaussian smoothing
            generate_captions: Whether to generate text captions
            use_numeric_id: Whether to convert scene_id to numeric format (0XXXX)
            
        Returns:
            True if processing succeeded, False otherwise
        """
        try:
            # Convert scene_id to numeric format if requested
            if use_numeric_id:
                numeric_scene_id = self._convert_to_numeric_id(scene_id)
                print(f"Processing scene: {scene_id} -> {numeric_scene_id}")
            else:
                numeric_scene_id = scene_id
                print(f"Processing scene: {scene_id}")
            
            # Parse camera file
            video_url, camera_data = self.parse_camera_file(camera_file)
            
            if len(camera_data) < self.min_sequence_length:
                print(f"Scene {scene_id} too short ({len(camera_data)} frames), skipping...")
                return False
            
            if len(camera_data) > self.max_sequence_length:
                print(f"Scene {scene_id} too long ({len(camera_data)} frames), truncating...")
                camera_data = camera_data[:self.max_sequence_length]
            
            # Extract camera trajectory
            trajectory_6d = self.extract_camera_trajectory(camera_data)
            
            # Apply smoothing if requested
            if apply_smoothing:
                trajectory_6d = self.smooth_trajectory(trajectory_6d, sigma=1.0)
            
            # Convert to target format using unified data format
            unified_data = UnifiedCameraData(trajectory_6d, CameraDataFormat.POSITION_ORIENTATION_6)
            final_trajectory = unified_data.to_format(self.target_format)
            
            # Create output directories
            output_path = Path(output_dir)
            motion_dir = output_path / "new_joint_vecs"
            text_dir = output_path / "texts"
            untagged_text_dir = output_path / "untagged_text"  # For motion-generated captions
            motion_dir.mkdir(parents=True, exist_ok=True)
            text_dir.mkdir(parents=True, exist_ok=True)
            untagged_text_dir.mkdir(parents=True, exist_ok=True)
            
            # Save trajectory data
            motion_file = motion_dir / f"{numeric_scene_id}.npy"
            np.save(motion_file, final_trajectory.get_momask_compatible_data().numpy())
            
            # Generate and save text caption
            if generate_captions:
                # For caption lookup, use the original hex scene_id
                original_scene_id = self.scene_id_mapping.get(numeric_scene_id, scene_id)
                
                if self.caption_motion:
                    # Generate motion-based caption (overrides existing captions)
                    deterministic_caption = self.generate_caption_motion()
                    print(f"  Generated deterministic caption for {numeric_scene_id}")
                    
                    # If AI captioning is enabled, refine the deterministic caption
                    if self.use_ai_captioning and (self.ai_captioner is not None or self.parallel_captioner is not None):
                        caption = self._generate_ai_caption(
                            deterministic_caption,
                            scene_id,
                            numeric_scene_id,
                            original_scene_id,
                            len(camera_data)
                        )
                    else:
                        caption = deterministic_caption
                    
                    # Save motion-generated caption to untagged_text directory
                    text_file = untagged_text_dir / f"{numeric_scene_id}.txt"
                    with open(text_file, 'w') as f:
                        f.write(caption)
                else:
                    # Use existing caption or fallback to basic generated caption
                    caption = self._get_existing_caption(original_scene_id, numeric_scene_id)
                
                if caption is None:
                        # Fallback to basic generated caption
                    caption = self._generate_caption(final_trajectory, video_url)
                    print(f"  Generated fallback caption for {numeric_scene_id} (original: {original_scene_id})")
                else:
                    print(f"  Using existing caption for {numeric_scene_id} (original: {original_scene_id})")
                
                    # Save to regular texts directory
                text_file = text_dir / f"{numeric_scene_id}.txt"
                with open(text_file, 'w') as f:
                    f.write(caption)
            
            # Save metadata
            metadata = {
                'scene_id': numeric_scene_id,
                'original_scene_id': scene_id,
                'video_url': video_url,
                'num_frames': len(camera_data),
                'format': self.target_format.name,
                'num_features': self.target_format.value,
                'smoothed': apply_smoothing,
                'trajectory_stats': {
                    'position_range': {
                        'x': [float(final_trajectory.positions[:, 0].min()), float(final_trajectory.positions[:, 0].max())],
                        'y': [float(final_trajectory.positions[:, 1].min()), float(final_trajectory.positions[:, 1].max())],
                        'z': [float(final_trajectory.positions[:, 2].min()), float(final_trajectory.positions[:, 2].max())]
                    },
                    'orientation_range': {
                        'pitch': [float(final_trajectory.orientations[:, 0].min()), float(final_trajectory.orientations[:, 0].max())],
                        'yaw': [float(final_trajectory.orientations[:, 1].min()), float(final_trajectory.orientations[:, 1].max())],
                        'roll': [float(final_trajectory.orientations[:, 2].min()), float(final_trajectory.orientations[:, 2].max())]
                    }
                }
            }
            
            metadata_file = output_path / "metadata" / f"{numeric_scene_id}.json"
            metadata_file.parent.mkdir(exist_ok=True)
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✓ Scene {scene_id} -> {numeric_scene_id} processed successfully ({len(camera_data)} frames)")
            return True
            
        except Exception as e:
            print(f"✗ Failed to process scene {scene_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _generate_caption(self, trajectory: UnifiedCameraData, video_url: str) -> str:
        """
        Generate a text caption describing the camera trajectory
        
        Args:
            trajectory: Unified camera trajectory data
            video_url: Source video URL
            
        Returns:
            Text caption describing the motion
        """
        positions = trajectory.positions.numpy()
        orientations = trajectory.orientations.numpy()
        
        # Calculate motion statistics
        pos_range = np.ptp(positions, axis=0)  # Range for each axis
        ori_range = np.ptp(orientations, axis=0)  # Range for each rotation
        
        # Determine dominant motion types
        motion_types = []
        
        # Position motion
        if pos_range[0] > 0.5:  # X movement
            motion_types.append("horizontal movement")
        if pos_range[1] > 0.5:  # Y movement  
            motion_types.append("vertical movement")
        if pos_range[2] > 0.5:  # Z movement
            motion_types.append("forward/backward movement")
        
        # Rotation motion
        if ori_range[0] > 0.2:  # Pitch
            motion_types.append("tilting")
        if ori_range[1] > 0.2:  # Yaw
            motion_types.append("panning")
        if ori_range[2] > 0.2:  # Roll
            motion_types.append("rolling")
        
        # Generate caption
        if not motion_types:
            caption = "camera remains relatively static"
        elif len(motion_types) == 1:
            caption = f"camera performs {motion_types[0]}"
        elif len(motion_types) == 2:
            caption = f"camera performs {motion_types[0]} and {motion_types[1]}"
        else:
            caption = f"camera performs {', '.join(motion_types[:-1])}, and {motion_types[-1]}"
        
        return caption

    def _process_scenes_batched(
        self,
        scenes_for_ai: List[Tuple[str, str, str, int]],
        scene_metadata: Dict[str, Dict],
        output_path: Path = None
    ) -> Dict[str, Dict]:
        """
        Process scenes in batches for single-GPU mode
        
        Args:
            scenes_for_ai: List of (scene_id, numeric_scene_id, deterministic_caption, total_frames)
            scene_metadata: Metadata for each scene
            output_path: Output directory (for immediate text writing)
            
        Returns:
            Dictionary mapping numeric_scene_id to result dict
        """
        from tqdm import tqdm
        
        ai_results = {}
        num_scenes = len(scenes_for_ai)
        
        # Create untagged_text directory if writing immediately
        if output_path:
            untagged_text_dir = output_path / "untagged_text"
            untagged_text_dir.mkdir(parents=True, exist_ok=True)
        
        # Process in batches (note: VLMs process one-by-one due to variable image sizes)
        pbar = tqdm(range(0, num_scenes, self.ai_batch_size), desc=f"AI Captioning (Qwen3-VL-4B)")
        
        import gc
        batch_count = 0
        
        for batch_start in pbar:
            batch_end = min(batch_start + self.ai_batch_size, num_scenes)
            batch = scenes_for_ai[batch_start:batch_end]
            batch_count += 1
            
            # Prepare batch data for AI captioning
            batch_data = []
            for scene_id, numeric_scene_id, deterministic_caption, total_frames in batch:
                # Find frames for this scene
                frame_paths = []
                if self.video_source_dir:
                    frames_dir_candidates = [
                        Path(self.video_source_dir) / scene_id,
                        Path(self.video_source_dir) / numeric_scene_id,
                    ]
                    
                    for frames_dir in frames_dir_candidates:
                        if frames_dir.exists() and frames_dir.is_dir():
                            frame_paths = self.ai_captioner.sample_existing_frames(str(frames_dir))
                            if frame_paths:
                                break
                
                batch_data.append((frame_paths, deterministic_caption, numeric_scene_id, total_frames))
            
            # Generate captions for the batch
            batch_results = self.ai_captioner.generate_captions_batch(batch_data)
            
            # Store results and write immediately if output_path provided
            for i, (scene_id, numeric_scene_id, deterministic_caption, total_frames) in enumerate(batch):
                result = batch_results[i]
                ai_results[numeric_scene_id] = {
                    'scene_id': scene_id,
                    'numeric_scene_id': numeric_scene_id,
                    'caption': result['caption'],
                    'success': result['success'],
                    'error': result.get('error', None)
                }
                
                # Write text immediately for this batch
                if output_path:
                    text_file = untagged_text_dir / f"{numeric_scene_id}.txt"
                    with open(text_file, 'w') as f:
                        f.write(result['caption'])
                    
                    # Update progress bar with status
                    status = "✓" if result['success'] else "✗"
                    pbar.set_postfix({'last': f"{numeric_scene_id} {status}"})
            
            # Aggressive cleanup every 50 scenes to prevent slowdown
            if batch_count % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # Log memory status
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / 1024**3
                        reserved = torch.cuda.memory_reserved() / 1024**3
                        pbar.write(f"[Cleanup] GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        return ai_results
    
    def _generate_ai_caption(
        self, 
        deterministic_caption: str,
        scene_id: str,
        numeric_scene_id: str,
        original_scene_id: str,
        total_frames: int
    ) -> str:
        """
        Generate AI-refined caption using Qwen VL
        
        Args:
            deterministic_caption: The deterministic motion description
            scene_id: Original scene ID (hex format)
            numeric_scene_id: Numeric scene ID
            original_scene_id: Mapped original scene ID
            total_frames: Number of frames in the sequence
            
        Returns:
            AI-refined caption or deterministic caption as fallback
        """
        try:
            # Select the appropriate captioner (single or parallel)
            captioner = self.parallel_captioner if self.parallel_captioner is not None else self.ai_captioner
            
            if captioner is None:
                print(f"    Warning: No captioner available, using deterministic caption")
                return deterministic_caption
            
            # Try to find video frames for this scene
            frame_paths = []
            
            # Option 1: Look for pre-extracted frames in video_source_dir
            if self.video_source_dir:
                frames_dir_candidates = [
                    Path(self.video_source_dir) / scene_id,
                    Path(self.video_source_dir) / numeric_scene_id,
                    Path(self.video_source_dir) / original_scene_id,
                ]
                
                for frames_dir in frames_dir_candidates:
                    if frames_dir.exists() and frames_dir.is_dir():
                        frame_paths = captioner.sample_existing_frames(str(frames_dir))
                        if frame_paths:
                            print(f"    Found {len(frame_paths)} frames in {frames_dir}")
                            break
            
            # If no frames found, fall back to deterministic caption
            if not frame_paths:
                print(f"    No video frames found for {scene_id}, using deterministic caption")
                return deterministic_caption
            
            # Generate AI-refined caption
            print(f"    Generating AI caption using {len(frame_paths)} frames...")
            result = captioner.generate_caption_with_context(
                frame_paths=frame_paths,
                deterministic_description=deterministic_caption,
                scene_id=numeric_scene_id,
                total_frames=total_frames
            )
            
            if result["success"]:
                ai_caption = result["caption"]
                print(f"    ✓ AI caption: {ai_caption}")
                return ai_caption
            else:
                print(f"    AI captioning failed: {result.get('error', 'Unknown error')}")
                return deterministic_caption
                
        except Exception as e:
            print(f"    Error in AI captioning for {scene_id}: {e}")
            return deterministic_caption

    def generate_caption_motion(self) -> str:
        """
        Generate caption using relative motion statistics.
        Analyzes speed, direction, and camera angle changes from relative transformations.
        
        Returns:
            Text caption describing the camera motion
        """
        if not self.relative_motion_data or len(self.relative_motion_data) < 2:
            return "camera remains static"
        
        # Extract motion data
        translations = np.array([motion['relative_translation'] for motion in self.relative_motion_data])
        rotations = np.array([motion['relative_rotation'] for motion in self.relative_motion_data])
        timestamps = np.array([motion['timestamp_s'] for motion in self.relative_motion_data])
        
        # Calculate time differences for velocity computation
        dt = np.diff(timestamps)
        dt[dt == 0] = 1e-6  # Avoid division by zero
        
        # Calculate translational velocities (speed and direction)
        translation_diffs = np.diff(translations, axis=0)
        translation_speeds = np.linalg.norm(translation_diffs, axis=1) / dt
        
        # Calculate dominant translation direction
        total_translation = translations[-1] - translations[0]
        total_distance = np.linalg.norm(total_translation)
        
        # Calculate rotational changes (convert rotation matrices to euler angles)
        euler_angles = []
        for rot_matrix in rotations:
            try:
                r = R.from_matrix(rot_matrix)
                roll, pitch, yaw = r.as_euler('xyz', degrees=False)
                euler_angles.append([pitch, yaw, roll])
            except Exception:
                euler_angles.append([0, 0, 0])
        
        euler_angles = np.array(euler_angles)
        euler_diffs = np.diff(euler_angles, axis=0)
        
        # Calculate angular velocities
        angular_speeds = np.linalg.norm(euler_diffs, axis=1) / dt
        
        # Analyze motion patterns
        motion_description = []
        
        # Analyze translation motion
        avg_translation_speed = np.mean(translation_speeds)
        if avg_translation_speed > 0.01:  # Threshold for significant translation
            # Determine dominant direction
            if abs(total_translation[0]) > abs(total_translation[1]) and abs(total_translation[0]) > abs(total_translation[2]):
                direction = "right" if total_translation[0] > 0 else "left"
                motion_description.append(f"moves {direction}")
            elif abs(total_translation[1]) > abs(total_translation[2]):
                direction = "up" if total_translation[1] > 0 else "down"
                motion_description.append(f"moves {direction}")
            elif abs(total_translation[2]) > 0.01:
                # In our coordinate system: negative Z = forward motion, positive Z = backward motion
                direction = "forward" if total_translation[2] < 0 else "backward"
                motion_description.append(f"moves {direction}")
            
            # Add speed qualifier
            if avg_translation_speed > 0.1:
                motion_description[-1] = motion_description[-1].replace("moves", "moves quickly")
            elif avg_translation_speed < 0.03:
                motion_description[-1] = motion_description[-1].replace("moves", "moves slowly")
        
        # Analyze rotational motion
        avg_angular_speed = np.mean(angular_speeds)
        if avg_angular_speed > 0.05:  # Threshold for significant rotation
            # Determine dominant rotation
            total_euler_change = euler_angles[-1] - euler_angles[0]
            
            if abs(total_euler_change[1]) > abs(total_euler_change[0]) and abs(total_euler_change[1]) > abs(total_euler_change[2]):
                # Yaw dominant
                direction = "right" if total_euler_change[1] > 0 else "left"
                motion_description.append(f"pans {direction}")
            elif abs(total_euler_change[0]) > abs(total_euler_change[2]):
                # Pitch dominant
                direction = "up" if total_euler_change[0] > 0 else "down"
                motion_description.append(f"tilts {direction}")
            elif abs(total_euler_change[2]) > 0.05:
                # Roll dominant
                direction = "clockwise" if total_euler_change[2] > 0 else "counterclockwise"
                motion_description.append(f"rolls {direction}")
            
            # Add speed qualifier for rotation
            if avg_angular_speed > 0.2:
                motion_description[-1] = motion_description[-1].replace("pans", "pans quickly").replace("tilts", "tilts quickly").replace("rolls", "rolls quickly")
            elif avg_angular_speed < 0.08:
                motion_description[-1] = motion_description[-1].replace("pans", "pans slowly").replace("tilts", "tilts slowly").replace("rolls", "rolls slowly")
        
        # Generate final caption
        if not motion_description:
            return "camera remains relatively static"
        elif len(motion_description) == 1:
            return f"camera {motion_description[0]}"
        elif len(motion_description) == 2:
            return f"camera {motion_description[0]} and {motion_description[1]}"
        else:
            return f"camera {', '.join(motion_description[:-1])}, and {motion_description[-1]}"
    
    def _get_existing_caption(self, scene_id: str, numeric_scene_id: str) -> Optional[str]:
        """
        Try to get existing caption from processed_estate directory
        
        Args:
            scene_id: Original scene ID (filename without extension)
            numeric_scene_id: Numeric scene ID for output
            
        Returns:
            Caption text if found, None otherwise
        """
        if not self.existing_captions_path:
            return None
        
        captions_path = Path(self.existing_captions_path)
        if not captions_path.exists():
            print(f"Warning: Existing captions path does not exist: {captions_path}")
            return None
        
        # The scene_id should be the hex filename (like "00703cbf7531ef11")
        # Try different possible scene ID formats
        possible_scene_ids = [
            scene_id,  # Original scene ID (hex filename)
            scene_id.replace('.txt', ''),  # Remove .txt if present
        ]
        
        for potential_id in possible_scene_ids:
            caption_file = captions_path / potential_id / "caption_motion_v2.txt"
            if caption_file.exists():
                try:
                    with open(caption_file, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                    if caption:  # Only return non-empty captions
                        return caption
                except Exception as e:
                    print(f"Warning: Failed to read caption file {caption_file}: {e}")
                    continue
        
        # Debug: List available caption directories to help diagnose issues
        if not any(caption_file.exists() for potential_id in possible_scene_ids 
                  for caption_file in [captions_path / potential_id / "caption_motion_v2.txt"]):
            print(f"Debug: No caption found for scene '{scene_id}'. Tried:")
            for potential_id in possible_scene_ids:
                caption_dir = captions_path / potential_id
                print(f"  - {caption_dir} (exists: {caption_dir.exists()})")
        
        return None
    
    def _check_caption_exists(self, output_path: Path, numeric_scene_id: str) -> bool:
        """
        Check if a caption text file already exists for a scene
        
        Args:
            output_path: Output directory path
            numeric_scene_id: The numeric scene ID (e.g., "000153")
            
        Returns:
            True if caption exists, False otherwise
        """
        untagged_text_dir = output_path / "untagged_text"
        text_file = untagged_text_dir / f"{numeric_scene_id}.txt"
        return text_file.exists() and text_file.stat().st_size > 0
    
    def _has_existing_caption(self, scene_id: str) -> bool:
        """
        Check if a scene has an existing technical caption
        
        Args:
            scene_id: Original scene ID (hex filename)
            
        Returns:
            True if caption exists, False otherwise
        """
        if not self.existing_captions_path:
            return False
        
        captions_path = Path(self.existing_captions_path)
        if not captions_path.exists():
            return False
        
        # Try different possible scene ID formats
        possible_scene_ids = [
            scene_id,  # Original scene ID (hex filename)
            scene_id.replace('.txt', ''),  # Remove .txt if present
        ]
        
        for potential_id in possible_scene_ids:
            caption_file = captions_path / potential_id / "caption_motion_v2.txt"
            if caption_file.exists():
                # Check if the caption file has content
                try:
                    with open(caption_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    return bool(content)  # Return True only if content exists
                except Exception:
                    continue
        
        return False
    
    def process_dataset(self, input_dir: str, output_dir: str, 
                       scene_list_file: Optional[str] = None,
                       max_scenes: Optional[int] = None,
                       use_numeric_ids: bool = True,
                       apply_smoothing: bool = True) -> Dict[str, int]:
        """
        Process entire Real Estate 10K dataset
        
        Args:
            input_dir: Directory containing camera parameter files
            output_dir: Output directory for processed dataset
            scene_list_file: Optional file containing list of scenes to process
            max_scenes: Optional limit on number of scenes to process
            use_numeric_ids: Whether to convert scene IDs to numeric format
            
        Returns:
            Dictionary with processing statistics
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Find camera parameter files
        if scene_list_file and os.path.exists(scene_list_file):
            with open(scene_list_file, 'r') as f:
                scene_ids = [line.strip() for line in f.readlines() if line.strip()]
            camera_files = [input_path / f"{scene_id}.txt" for scene_id in scene_ids]
            camera_files = [f for f in camera_files if f.exists()]
        else:
            camera_files = list(input_path.glob("*.txt"))
        
        if max_scenes:
            camera_files = camera_files[:max_scenes]
        
        print(f"Found {len(camera_files)} camera parameter files to process")
        
        # Load or reset scene counter and mapping
        scene_mapping_file = output_path / "scene_id_mapping.json"
        if self.resume_captioning and scene_mapping_file.exists():
            # Load existing scene mapping
            with open(scene_mapping_file, 'r') as f:
                self.scene_id_mapping = json.load(f)
            self.scene_counter = len(self.scene_id_mapping)
            print(f"Loaded existing scene mapping: {self.scene_counter} scenes already processed")
            
            # Check which scenes already have captions
            untagged_text_dir = output_path / "untagged_text"
            if untagged_text_dir.exists():
                existing_captions = set()
                for text_file in untagged_text_dir.glob("*.txt"):
                    if text_file.stat().st_size > 0:
                        existing_captions.add(text_file.stem)
                
                print(f"Found {len(existing_captions)} existing caption files")
                print(f"Will skip scenes with existing captions and process remaining scenes")
            else:
                existing_captions = set()
                print(f"No untagged_text directory found, will process all scenes")
        else:
        # Reset scene counter and mapping for consistent numbering
            self.scene_counter = 0
            self.scene_id_mapping = {}
            existing_captions = set()
            if self.resume_captioning:
                print(f"⚠️  Resume requested but no existing scene_id_mapping.json found, starting fresh")
        
        # Process scenes
        stats = {'processed': 0, 'failed': 0, 'skipped': 0}
        scene_ids = []
        
        # Sort camera files for consistent ordering
        camera_files = sorted(camera_files, key=lambda x: x.stem)
        
        # Filter for caption-only processing if requested
        if self.caption_only:
            print(f"Filtering scenes to only those with existing captions...")
            original_count = len(camera_files)
            camera_files = [f for f in camera_files if self._has_existing_caption(f.stem)]
            print(f"Filtered from {original_count} to {len(camera_files)} scenes with captions")
        
        # Check if we should use batched/parallel AI captioning
        use_batched_ai = (self.use_ai_captioning and 
                         self.caption_motion and 
                         len(camera_files) > 1)
        
        if use_batched_ai:
            if self.parallel_captioner is not None:
                print(f"\nUsing parallel AI captioning with {self.parallel_captioner.num_gpus} GPUs")
                print(f"   Processing {len(camera_files)} scenes in parallel...")
            else:
                print(f"\nUsing AI captioning with Qwen3-VL-4B (efficient model)")
                print(f"   Processing {len(camera_files)} scenes...")
            
            # Phase 1: Process trajectories and generate deterministic captions (sequential, fast)
            print("\nPhase 1: Processing trajectories and generating deterministic captions...")
            scenes_for_ai = []
            scene_metadata = {}
            skipped_count = 0
            
            for camera_file in tqdm(camera_files, desc="Phase 1: Trajectories"):
                scene_id = camera_file.stem
                
                # Check if scene already has a caption (for resume mode)
                # First, determine what the numeric_scene_id would be
                if use_numeric_ids:
                    if scene_id in self.scene_id_mapping.values():
                        # Scene already mapped, get its numeric ID
                        numeric_scene_id = next(k for k, v in self.scene_id_mapping.items() if v == scene_id)
                    else:
                        # New scene, would get next counter value
                        numeric_scene_id = f"{self.scene_counter:06d}"
                else:
                    numeric_scene_id = scene_id
                
                # Skip if caption exists and resume mode is enabled
                if self.resume_captioning and numeric_scene_id in existing_captions:
                    skipped_count += 1
                    stats['skipped'] += 1
                    scene_ids.append(numeric_scene_id)
                    continue
                
                # Process trajectory without AI captioning
                success, metadata = self._process_scene_trajectory(
                    str(camera_file), str(output_path), scene_id,
                    apply_smoothing=apply_smoothing, use_numeric_id=use_numeric_ids
                )
                
                if success:
                    scenes_for_ai.append((
                        metadata['scene_id'],
                        metadata['numeric_scene_id'],
                        metadata['deterministic_caption'],
                        metadata['total_frames']
                    ))
                    scene_metadata[metadata['numeric_scene_id']] = metadata
                    stats['processed'] += 1
                    scene_ids.append(metadata['numeric_scene_id'])
                else:
                    stats['failed'] += 1
            
            if skipped_count > 0:
                print(f"Skipped {skipped_count} scenes with existing captions")
            
            # Phase 2: AI captioning across all scenes (parallel or sequential with efficient model)
            if scenes_for_ai:
                print(f"\nPhase 2: AI captioning for {len(scenes_for_ai)} scenes...")
                
                if self.parallel_captioner is not None:
                    # Multi-GPU parallel processing
                    ai_results = self.parallel_captioner.process_scenes_parallel(
                        scenes_for_ai,
                        self.video_source_dir,
                        progress_callback=None
                    )
                else:
                    # Single-GPU batched processing (writes texts immediately)
                    ai_results = self._process_scenes_batched(scenes_for_ai, scene_metadata, output_path)
                    print(f"\n✓ Batched AI captioning complete: {len(ai_results)} scenes processed")
                
                # Phase 3: Save AI-refined captions (only for multi-GPU, single-GPU already wrote)
                if self.parallel_captioner is not None:
                    print("\nPhase 3: Saving AI-refined captions...")
                    untagged_text_dir = output_path / "untagged_text"
                    untagged_text_dir.mkdir(parents=True, exist_ok=True)
                    
                    for numeric_scene_id, result in ai_results.items():
                        text_file = untagged_text_dir / f"{numeric_scene_id}.txt"
                        with open(text_file, 'w') as f:
                            f.write(result['caption'])
                        
                        if result['success']:
                            print(f"  ✓ {numeric_scene_id}: AI caption saved")
                        else:
                            print(f"  ⚠ {numeric_scene_id}: Using deterministic (AI failed: {result.get('error', 'unknown')})")
        else:
            # Sequential processing (original behavior)
            if use_batched_ai is False and self.parallel_captioner is not None:
                print("\nParallel captioning available but not used (need --caption-motion and multiple scenes)")
        
        for camera_file in tqdm(camera_files, desc="Processing scenes"):
            scene_id = camera_file.stem
            
            success = self.process_scene(
                str(camera_file), str(output_path), scene_id,
                    apply_smoothing=apply_smoothing, generate_captions=True, use_numeric_id=use_numeric_ids
            )
            
            if success:
                stats['processed'] += 1
                # Store the numeric scene ID for dataset splits
                if use_numeric_ids:
                    # The numeric ID was already generated in process_scene
                    numeric_scene_id = f"{self.scene_counter - 1:06d}"  # scene_counter was incremented
                    scene_ids.append(numeric_scene_id)
                else:
                    scene_ids.append(scene_id)
            else:
                stats['failed'] += 1
        
        # Create dataset split files
        self._create_dataset_splits(output_path, scene_ids)
        
        # Calculate and save dataset statistics
        self._calculate_dataset_statistics(output_path, scene_ids)
        
        print(f"\nDataset processing completed:")
        print(f"  Processed: {stats['processed']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Total: {len(camera_files)}")
        
        return stats
    
    def _create_dataset_splits(self, output_path: Path, scene_ids: List[str]):
        """Create train/val/test splits"""
        np.random.seed(42)  # For reproducible splits
        np.random.shuffle(scene_ids)
        
        # 70% train, 15% val, 15% test
        n_total = len(scene_ids)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        
        train_ids = scene_ids[:n_train]
        val_ids = scene_ids[n_train:n_train + n_val]
        test_ids = scene_ids[n_train + n_val:]
        
        # Save split files
        with open(output_path / "train.txt", 'w') as f:
            for scene_id in train_ids:
                f.write(f"{scene_id}\n")
        
        with open(output_path / "val.txt", 'w') as f:
            for scene_id in val_ids:
                f.write(f"{scene_id}\n")
        
        with open(output_path / "test.txt", 'w') as f:
            for scene_id in test_ids:
                f.write(f"{scene_id}\n")
        
        print(f"Dataset splits created: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")
    
    def _calculate_dataset_statistics(self, output_path: Path, scene_ids: List[str]):
        """Calculate and save dataset statistics"""
        motion_dir = output_path / "new_joint_vecs"
        
        all_data = []
        for scene_id in scene_ids:
            motion_file = motion_dir / f"{scene_id}.npy"
            if motion_file.exists():
                data = np.load(motion_file)
                all_data.append(data)
        
        if all_data:
            # Concatenate all data
            combined_data = np.concatenate(all_data, axis=0)
            
            # Calculate mean and std
            mean = np.mean(combined_data, axis=0)
            std = np.std(combined_data, axis=0)
            
            # Save statistics
            np.save(output_path / "Mean.npy", mean)
            np.save(output_path / "Std.npy", std)
            
            print(f"Dataset statistics calculated and saved")
            print(f"  Mean: {mean}")
            print(f"  Std: {std}")
        
        # Save scene ID mapping for reference
        if self.scene_id_mapping:
            import json
            mapping_file = output_path / "scene_id_mapping.json"
            with open(mapping_file, 'w') as f:
                json.dump(self.scene_id_mapping, f, indent=2)
            print(f"Scene ID mapping saved to: {mapping_file}")

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Process Real Estate 10K camera trajectories for MoMask")
    parser.add_argument("input_dir", help="Directory containing camera parameter files")
    parser.add_argument("output_dir", help="Output directory for processed dataset")
    parser.add_argument("--format", choices=["6", "12"], default="6", 
                       help="Output format: 6-feature or 12-feature")
    parser.add_argument("--scene-list", help="File containing list of scenes to process")
    parser.add_argument("--max-scenes", type=int, help="Maximum number of scenes to process")
    parser.add_argument("--min-length", type=int, default=30, 
                       help="Minimum sequence length")
    parser.add_argument("--max-length", type=int, default=300,
                       help="Maximum sequence length")
    
    args = parser.parse_args()
    
    processor = RealEstate10KProcessor(
        output_format=args.format,
        min_sequence_length=args.min_length,
        max_sequence_length=args.max_length
    )
    
    stats = processor.process_dataset(
        args.input_dir,
        args.output_dir,
        args.scene_list,
        args.max_scenes
    )
    
    print(f"\nProcessing completed: {stats}")

if __name__ == "__main__":
    main()
