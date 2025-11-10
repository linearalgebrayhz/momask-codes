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
from typing import List, Dict, Tuple, Optional, Union, Any
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import argparse
import torch
import glob
import tempfile
from PIL import Image

from .unified_data_format import UnifiedCameraData, CameraDataFormat


class QwenVideoCaptioner:
    """Qwen 2.5 VL-based video captioning for camera motion analysis"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        use_flash_attention: bool = True,
        max_frames: int = 8,  # Reduced from 16 to 8 for 2x speed
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
            import importlib
            transformers = importlib.import_module('transformers')
            
            print(f"Loading Qwen VL model: {self.model_name}...")
            
            # Import the correct model class based on Qwen version
            if self.is_qwen3:
                ModelClass = getattr(transformers, 'Qwen3VLForConditionalGeneration')
                print("  Using Qwen3VLForConditionalGeneration")
            else:
                ModelClass = getattr(transformers, 'Qwen2_5_VLForConditionalGeneration')
                print("  Using Qwen2_5_VLForConditionalGeneration")
            
            # Load model with appropriate settings
            if use_flash_attention and self.device == 'cuda':
                self.model = ModelClass.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="auto"
                )
                print(f"  ✓ Flash Attention 2 enabled")
            else:
                self.model = ModelClass.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                    device_map="auto" if self.device == 'cuda' else None
                )
                if self.device != 'cuda':
                    self.model = self.model.to(self.device)
                print(f"  ⚠️  Flash Attention NOT enabled")
            
            AutoProcessor = getattr(transformers, 'AutoProcessor')
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model_loaded = True
            
            # Import qwen_vl_utils for Qwen2.5 (Qwen3 may not need it)
            try:
                qwen_utils = importlib.import_module('qwen_vl_utils')
                self.process_vision_info = getattr(qwen_utils, 'process_vision_info', None)
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
    
    def sample_existing_frames(
        self,
        frames_dir: str,
        max_frames: Optional[int] = None,
        max_resolution: Tuple[int, int] = (640, 360)  # Back to 360p for comparison
    ) -> List[str]:
        """
        Sample frames from a directory with automatic resizing for large images
        
        CRITICAL: Large resolution images (4K, 8K) can cause OOM!
        This method automatically resizes oversized images to 360p.
        
        Args:
            frames_dir: Directory containing frame images
            max_frames: Maximum number of frames to sample
            max_resolution: Maximum (width, height), default 360p (640×360)
            
        Returns:
            List of paths to sampled frames (may include temp resized images)
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
            sampled_files = frame_files
        else:
            indices = np.linspace(0, len(frame_files) - 1, max_frames, dtype=int)
            sampled_files = [frame_files[i] for i in indices]
        
        # CRITICAL: Check and resize if needed to prevent OOM
        processed_files = []
        temp_dir = None
        resized_count = 0
        
        for img_path in sampled_files:
            try:
                img = Image.open(img_path)
                width, height = img.size
                
                # If image is oversized, resize to max_resolution
                if width > max_resolution[0] or height > max_resolution[1]:
                    if temp_dir is None:
                        temp_dir = tempfile.mkdtemp(prefix='qwen_resized_')
                    
                    # Keep aspect ratio and resize
                    img.thumbnail(max_resolution, Image.Resampling.LANCZOS)
                    
                    # Save to temporary file (keep original format for speed)
                    temp_path = os.path.join(temp_dir, os.path.basename(img_path))
                    img.save(temp_path, quality=90)
                    processed_files.append(temp_path)
                    resized_count += 1
                    
                    if resized_count == 1:  # Only print once per batch
                        print(f"     ⚠️  Auto-resizing to 360p: {width}×{height} → {img.size[0]}×{img.size[1]}")
                else:
                    processed_files.append(img_path)
                
                img.close()
                
            except Exception as e:
                print(f"     Warning: Could not process image {img_path}: {e}")
                # Use original if resize fails
                processed_files.append(img_path)
        
        # Track temp directories for cleanup
        if temp_dir:
            if not hasattr(self, '_temp_dirs'):
                self._temp_dirs = []
            self._temp_dirs.append(temp_dir)
        
        return processed_files
    
    def cleanup_temp_images(self):
        """Clean up temporary resized images"""
        if hasattr(self, '_temp_dirs'):
            import shutil
            for temp_dir in self._temp_dirs:
                try:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Warning: Could not cleanup temp dir {temp_dir}: {e}")
            self._temp_dirs = []
    
    def generate_captions_batch(
        self,
        batch_data: List[Tuple[List[str], str, str, int]]
    ) -> List[Dict[str, Any]]:
        """
        Generate captions for multiple scenes using batch inference
        
        Implements proper Qwen3-VL batch processing with padding:
        - Sets padding_side='left' for generation tasks
        - Uses padding=True in apply_chat_template
        - Normalizes frame counts across batch (pads to max_frames)
        - Auto-resizes large images to 720p to prevent OOM
        - Immediate cleanup of massive image embeddings after generation
        - Processes multiple scenes in parallel on GPU
        
        Args:
            batch_data: List of (frame_paths, deterministic_description, scene_id, total_frames) tuples
            
        Returns:
            List of result dictionaries (one per scene, batched inference)
        """
        # Lazy load model on first use
        if not self.model_loaded:
            self.load_model()

        # If model/processor failed to load (optional dependency missing), fall back immediately
        if self.model is None or self.processor is None:
            fallback = []
            for frame_paths, deterministic_description, scene_id, total_frames in batch_data:
                fallback.append({
                    "success": False,
                    "error": "AI captioning unavailable (transformers not installed)",
                    "caption": deterministic_description,
                    "scene_id": scene_id,
                    "batched": False
                })
            return fallback
        
        if len(batch_data) == 0:
            return []
        
        # Single scene - no need for batching
        if len(batch_data) == 1:
            frame_paths, deterministic_description, scene_id, total_frames = batch_data[0]
            result = self.generate_caption_with_context(
                frame_paths, deterministic_description, scene_id, total_frames
            )
            return [result]
        
        # CRITICAL: Set padding side to LEFT for generation tasks
        # This is required for proper batch generation with Qwen models
        original_padding_side = None
        if hasattr(self.processor, 'tokenizer') and hasattr(self.processor.tokenizer, 'padding_side'):
            original_padding_side = self.processor.tokenizer.padding_side
            self.processor.tokenizer.padding_side = 'left'
        
        try:
            import time
            t_start = time.time()
            
            results = []
            batch_messages = []
            batch_metadata = []
            
            # Step 1: Normalize frame counts and prepare messages
            t_prep_start = time.time()
            for idx, (frame_paths, deterministic_description, scene_id, total_frames) in enumerate(batch_data):
                if len(frame_paths) == 0:
                    results.append({
                        "success": False,
                        "error": "No frames provided",
                        "caption": deterministic_description,
                        "scene_id": scene_id
                    })
                    continue
                
                # Normalize frame count: ensure all scenes have same number of frames for batching
                # If we have fewer frames than max_frames, pad by repeating last frame
                # Images are auto-resized to 360p in sample_existing_frames()
                if len(frame_paths) < self.max_frames:
                    padded_frames = frame_paths + [frame_paths[-1]] * (self.max_frames - len(frame_paths))
                else:
                    padded_frames = frame_paths[:self.max_frames]
                
                # Concise prompt for natural one-line descriptions
                prompt = (
                    f"Analyze this camera trajectory ({len(frame_paths)} frames from {total_frames}, uniformly sampled).\n\n"
                    f"Reference: {deterministic_description}\n\n"
                    "Describe the complete motion sequence in ONE sentence. If direction changes, describe in order: \"first does X, then does Y\".\n\n"
                    "Include: movement type (pan/tilt/dolly/track/arc), direction, pace (slow/medium/fast), quality (smooth/shaky), and purpose.\n\n"
                    "Note: Always moving, never static.\n\n"
                    "Examples:\n"
                    "- \"The camera pans left slowly, then reverses right, smoothly revealing the building.\"\n"
                    "- \"The camera dollies forward while tilting up, emphasizing the building's height.\"\n"
                    "- \"The camera pans right steadily then shifts left with a slight arc, exploring the architecture.\"\n\n"
                    "Description:"
                )
                
                # Prepare message content with normalized frames
                content = []
                for frame_path in padded_frames:
                    content.append({
                        "type": "image",
                        "image": frame_path
                    })
                content.append({
                    "type": "text",
                    "text": prompt
                })
                
                batch_messages.append([{
                    "role": "user",
                    "content": content
                }])
                
                batch_metadata.append({
                    'idx': idx,
                    'scene_id': scene_id,
                    'original_frame_count': len(frame_paths),
                    'deterministic_description': deterministic_description,
                    'total_frames': total_frames
                })
            
            if len(batch_messages) == 0:
                return results
            
            t_prep_end = time.time()
            print(f"    ⏱️  Prep: {t_prep_end - t_prep_start:.2f}s")
            
            # Step 2: Apply chat template with PADDING enabled (key for batching!)
            t_template_start = time.time()
            if self.is_qwen3:
                # Qwen3 API: process entire batch at once
                inputs = self.processor.apply_chat_template(
                    batch_messages,  # List of message lists
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True  # CRITICAL: Enable padding for batch processing!
                )
                inputs = inputs.to(self.device)
            else:
                # Qwen2.5 API: requires process_vision_info
                # Process each message's vision info, then batch
                all_texts = []
                all_image_inputs = []
                all_video_inputs = []
                
                for messages in batch_messages:
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    if self.process_vision_info is None:
                        raise RuntimeError("qwen_vl_utils.process_vision_info not available for Qwen2.5")
                    img_inputs, vid_inputs = self.process_vision_info(messages)
                    all_texts.append(text)
                    all_image_inputs.extend(img_inputs if img_inputs else [])
                    all_video_inputs.extend(vid_inputs if vid_inputs else [])
                
                inputs = self.processor(
                    text=all_texts,
                    images=all_image_inputs if all_image_inputs else None,
                    videos=all_video_inputs if all_video_inputs else None,
                    padding=True,  # CRITICAL: Enable padding!
                    return_tensors="pt"
                )
                inputs = inputs.to(self.device)
                
                # Cleanup intermediate variables for Qwen2.5
                del all_texts
                del all_image_inputs
                del all_video_inputs
            
            t_template_end = time.time()
            print(f"    ⏱️  Template+Encoding: {t_template_end - t_template_start:.2f}s")
            
            # Step 3: Generate captions for entire batch in parallel!
            t_gen_start = time.time()
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50
                )
            
            t_gen_end = time.time()
            print(f"    ⏱️  Generation: {t_gen_end - t_gen_start:.2f}s")
            
            # CRITICAL: Extract input lengths BEFORE deleting inputs
            # inputs contains MASSIVE image embeddings that must be freed ASAP
            input_lengths = [len(in_ids) for in_ids in inputs.input_ids]
            
            # Step 4: Delete inputs IMMEDIATELY - image embeddings are HUGE (2-3GB per batch!)
            del inputs
            
            # Synchronize and clear cache immediately
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
            # Now decode using saved lengths
            generated_ids_trimmed = [
                generated_ids[i][input_lengths[i]:] for i in range(len(input_lengths))
            ]
            
            # Delete generated_ids immediately
            del generated_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            output_texts = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            # Step 5: Package results
            for i, metadata in enumerate(batch_metadata):
                caption = output_texts[i].strip()
                results.append({
                    "success": True,
                    "caption": caption,
                    "num_frames_used": metadata['original_frame_count'],
                    "deterministic_input": metadata['deterministic_description'],
                    "total_frames": metadata['total_frames'],
                    "scene_id": metadata['scene_id'],
                    "batched": True  # Flag to indicate this was batched
                })
            
            # Step 6: Final cleanup
            # (inputs and generated_ids already deleted above)
            del generated_ids_trimmed
            del output_texts
            del batch_messages
            del batch_metadata
            
            # Final cache clear
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            t_end = time.time()
            print(f"    ⏱️  TOTAL batch time: {t_end - t_start:.2f}s ({len(batch_data)} scenes, {(t_end-t_start)/len(batch_data):.2f}s/scene)")
            
            return results
            
        except Exception as e:
            print(f"Error in batch caption generation: {e}")
            print(f"Batch size: {len(batch_data)} scenes")
            import traceback
            traceback.print_exc()
            
            # Cleanup on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Fallback: Return deterministic captions
            fallback_results = []
            for frame_paths, deterministic_description, scene_id, total_frames in batch_data:
                fallback_results.append({
                    "success": False,
                    "error": str(e),
                    "caption": deterministic_description,
                    "scene_id": scene_id,
                    "batched": False
                })
            
            return fallback_results
        
        finally:
            # Always restore original padding side if available
            if original_padding_side is not None and hasattr(self.processor, 'tokenizer'):
                self.processor.tokenizer.padding_side = original_padding_side
    
    def generate_caption_with_context(
        self,
        frame_paths: List[str],
        deterministic_description: str,
        scene_id: str = "",
        total_frames: Optional[int] = None
    ) -> Dict[str, Any]:
        # Note: previously had a debug print/exit that terminated single-scene captioning.
        # Removed to allow normal execution.
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

        # If model/processor failed to load (optional dependency missing), fall back
        if self.model is None or self.processor is None:
            return {
                "success": False,
                "error": "AI captioning unavailable (transformers not installed)",
                "caption": deterministic_description,
                "scene_id": scene_id,
                "total_frames": total_frames,
            }
        
        if len(frame_paths) == 0:
            return {
                "success": False,
                "error": "No frames provided",
                "caption": deterministic_description
            }
        
        # Concise prompt for natural one-line descriptions
        prompt = (
            f"Analyze this camera trajectory ({len(frame_paths)} frames from {total_frames}, uniformly sampled).\n\n"
            f"Reference: {deterministic_description}\n\n"
            "Describe the complete motion sequence in ONE sentence. If direction changes, describe in order: \"first does X, then does Y\".\n\n"
            "Include: movement type (pan/tilt/dolly/track/arc), direction, pace (slow/medium/fast), quality (smooth/shaky), and purpose.\n\n"
            "Note: Always moving, never static.\n\n"
            "Examples:\n"
            "- \"The camera pans left slowly, then reverses right, smoothly revealing the building.\"\n"
            "- \"The camera dollies forward while tilting up, emphasizing the building's height.\"\n"
            "- \"The camera pans right steadily then shifts left with a slight arc, exploring the architecture.\"\n\n"
            "Description:"
        )

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
                if self.process_vision_info is None:
                    raise RuntimeError("qwen_vl_utils.process_vision_info not available for Qwen2.5")
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
                    max_new_tokens=64,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50
                )
            
            # CRITICAL: Save input length and delete inputs immediately
            # inputs contains huge image embeddings that must be freed ASAP
            input_length = len(inputs.input_ids[0])
            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            generated_ids_trimmed = [
                generated_ids[0][input_length:]
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            caption = output_text[0].strip()
            
            # Cleanup remaining tensors
            # (inputs already deleted above)
            del generated_ids
            del generated_ids_trimmed
            del output_text
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
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
            # Clean up on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                "success": False,
                "error": str(e),
                "caption": deterministic_description,  # Fallback to deterministic
                "scene_id": scene_id
            }
    

class RealEstate10KProcessor:
    """Process Real Estate 10K camera trajectory data"""
    
    # Motion detection thresholds (radians for rotations, dataset units for translations)
    THRESHOLD_TRANSLATION_ABS = 0.10    # Minimum per-axis translation to report
    THRESHOLD_TRANSLATION_RATIO = 0.6   # Include axis if >= 60% of max axis
    THRESHOLD_YAW = 0.12                # ~6.9° - left/right panning
    THRESHOLD_PITCH = 0.10              # ~5.7° - up/down tilting  
    THRESHOLD_ROLL = 0.20               # ~11.5° - clockwise/CCW roll (stricter)
    THRESHOLD_ROTATION_RATIO = 0.6      # Include axis if >= 60% of max axis
    THRESHOLD_MIN_ANGULAR_SPEED = 0.05  # Minimum angular velocity to report rotation
    THRESHOLD_MIN_TRANSLATION_SPEED = 0.01  # Minimum translation speed
    
    def __init__(self, output_format: str = "6", min_sequence_length: int = 30, max_sequence_length: int = 300, 
                 existing_captions_path: Optional[str] = None, caption_only: bool = False, transform: str = "relative",
                 caption_motion: bool = False, use_ai_captioning: bool = False, ai_model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
                 ai_max_frames: int = 32, video_source_dir: Optional[str] = None, use_parallel: bool = False, num_gpus: Optional[int] = None,
                 ai_batch_size: int = 2, resume_captioning: bool = False, pose_semantics: str = "twc"):
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
            pose_semantics: Interpretation of input 3x4 pose. One of {"auto", "twc", "tcw"}.
                - "twc": matrix maps camera->world (Twc)
                - "tcw": matrix maps world->camera (Tcw) (will be inverted to Twc)
                - "auto": try both and pick the one that best matches OpenGL forward (-Z)
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
        # Pose semantics handling
        self.pose_semantics = pose_semantics.lower()
        
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

        def _convert_block_to_twc(block: np.ndarray, semantics: str) -> np.ndarray:
            """Ensure pose block is Twc (camera->world) for downstream usage.
            block: 3x4 matrix from file
            semantics: 'twc' (as-is) or 'tcw' (invert)
            Returns 3x4 Twc
            """
            Rb = block[:, :3]
            tb = block[:, 3]
            if semantics == 'twc':
                return np.concatenate([Rb, tb.reshape(3, 1)], axis=1)
            elif semantics == 'tcw':
                # Invert [R|t]: Twc = [R^T | -R^T t]
                Rinv = Rb.T
                tinv = -Rinv @ tb
                return np.concatenate([Rinv, tinv.reshape(3, 1)], axis=1)
            else:
                raise ValueError(f"Unknown semantics: {semantics}")

        def _compute_relative(first_block_twc: np.ndarray, block_twc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Compute relative rotation and translation wrt first frame (both Twc)."""
            R0 = first_block_twc[:, :3]
            t0 = first_block_twc[:, 3]
            R_abs = block_twc[:, :3]
            t_abs = block_twc[:, 3]
            R_rel = R0.T @ R_abs
            t_rel = R0.T @ (t_abs - t0)
            return R_rel, t_rel

        # (Auto-scoring function removed; we standardize on Twc unless explicitly set to 'tcw')
        
        # Prepare semantics determination for relative transform
        if self.transform == "relative":
            # Read first pose and convert to Twc according to chosen semantics (default Twc)
            values = lines[1].split()
            pose_values = [float(v) for v in values[7:]]
            raw_block = np.array(pose_values).reshape(3, 4)
            effective_semantics = 'tcw' if self.pose_semantics == 'tcw' else 'twc'
            first_frame_pose_matrix = _convert_block_to_twc(raw_block, effective_semantics)

            # add first frame as identity in relative coordinates
            camera_data.append({
                'timestamp_us': int(float(lines[1].split()[0])),
                'timestamp_s': int(float(lines[1].split()[0])) / 1_000_000,
                'intrinsics': {'fx': float(lines[1].split()[1]), 'fy': float(lines[1].split()[2]), 'cx': float(lines[1].split()[3]), 'cy': float(lines[1].split()[4])},
                'pose_matrix': np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1),
                'frame_index': 1
            })
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
                raw_block = np.array(pose_values).reshape(3, 4)

                if self.transform == "relative":
                    # Ensure we use Twc representation per chosen semantics
                    block_twc = _convert_block_to_twc(raw_block, effective_semantics)
                    relative_rot, relative_trans = _compute_relative(first_frame_pose_matrix, block_twc)
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
        Convert rotation matrices to Euler angles (pitch, yaw, roll).
        Yaw/pitch are derived from the forward vector to avoid gimbal/ordering issues; roll is recovered from up vector.
        OpenGL convention: +X right, +Y up, -Z forward.
        
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
                
                # Derive yaw/pitch from forward vector; roll from up vector
                # Camera local axes in initial frame
                fwd = rotation_matrix @ np.array([0.0, 0.0, -1.0])
                up = rotation_matrix @ np.array([0.0, 1.0, 0.0])
                # Normalize
                fwd = fwd / (np.linalg.norm(fwd) + 1e-8)
                up = up / (np.linalg.norm(up) + 1e-8)

                # Yaw: left/right around +Y; forward -Z
                yaw = np.arctan2(fwd[0], -fwd[2])
                # Pitch: up/down around +X
                pitch = np.arctan2(fwd[1], np.sqrt(fwd[0]**2 + fwd[2]**2))

                # Reference basis from yaw/pitch
                cos_y = np.cos(yaw); sin_y = np.sin(yaw)
                cos_p = np.cos(pitch); sin_p = np.sin(pitch)
                f_ref = np.array([sin_y * cos_p, sin_p, -cos_y * cos_p])
                # Use global up to derive right and up_ref
                up_global = np.array([0.0, 1.0, 0.0])
                right_ref = np.cross(up_global, f_ref)
                right_ref /= (np.linalg.norm(right_ref) + 1e-8)
                up_ref = np.cross(f_ref, right_ref)
                up_ref /= (np.linalg.norm(up_ref) + 1e-8)

                # Roll: signed angle between up_ref and actual up around forward axis
                cross_ur = np.cross(up_ref, up)
                sin_roll = np.dot(cross_ur, f_ref)
                cos_roll = float(np.clip(np.dot(up_ref, up), -1.0, 1.0))
                roll = np.arctan2(sin_roll, cos_roll)

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

    def _segment_motion_sequence(self, translations: np.ndarray, ypr: np.ndarray, min_segment_ratio: float = 0.1, max_segments: int = 3) -> List[Tuple[int, int, str]]:
        """
        Segment the motion into up to max_segments based on dominant per-frame motion labels.
        Returns list of (start_idx, end_idx, phrase).
        """
        n = translations.shape[0]
        if n < 3:
            return [(0, n - 1, "moves slightly")]

        # Per-frame diffs
        v = np.diff(translations, axis=0)  # (n-1, 3)
        dypr = np.diff(ypr, axis=0)        # (n-1, 3)
        # Magnitudes
        v_mag = np.linalg.norm(v, axis=1) + 1e-8
        yaw_rate = np.abs(dypr[:, 1])  # yaw dominates panning
        pitch_rate = np.abs(dypr[:, 0])

        # Build a label per frame (1..n-1)
        labels = []
        for i in range(n - 1):
            # Dominant translation axis
            ax = int(np.argmax(np.abs(v[i])))
            phrase = None
            if v_mag[i] > 1e-6:
                if ax == 0:
                    phrase = "tracks right" if v[i, 0] > 0 else "tracks left"
                elif ax == 1:
                    phrase = "moves up" if v[i, 1] > 0 else "moves down"
                else:
                    phrase = "dollies forward" if v[i, 2] < 0 else "dollies backward"
            # Consider rotation if stronger than translation
            if max(yaw_rate[i], pitch_rate[i]) > 0.02 and (phrase is None or max(yaw_rate[i], pitch_rate[i]) > 0.5 * v_mag[i]):
                if yaw_rate[i] >= pitch_rate[i]:
                    phrase = "pans right" if dypr[i, 1] > 0 else "pans left"
                else:
                    phrase = "tilts up" if dypr[i, 0] > 0 else "tilts down"
            if phrase is None:
                phrase = "moves slightly"
            labels.append(phrase)

        # Collapse consecutive identical labels
        segments = []
        start = 0
        for i in range(1, len(labels)):
            if labels[i] != labels[i - 1]:
                segments.append((start, i, labels[start]))
                start = i
        segments.append((start, len(labels), labels[start]))

        # Merge tiny segments
        min_len = max(1, int(min_segment_ratio * (n - 1)))
        merged = []
        for seg in segments:
            if not merged:
                merged.append(seg)
            else:
                prev = merged[-1]
                if seg[1] - seg[0] < min_len:
                    # Merge into previous
                    merged[-1] = (prev[0], seg[1], prev[2])
                else:
                    merged.append(seg)

        # Limit to max_segments by merging smallest middle ones
        while len(merged) > max_segments:
            # Find shortest interior segment to merge with neighbor
            lengths = [s[1] - s[0] for s in merged]
            # Exclude first and last if possible
            candidates = list(range(1, len(merged) - 1)) or list(range(len(merged)))
            k = min(candidates, key=lambda idx: lengths[idx])
            if k == 0:
                merged[1] = (merged[0][0], merged[1][1], merged[1][2])
                merged.pop(0)
            else:
                merged[k - 1] = (merged[k - 1][0], merged[k][1], merged[k - 1][2])
                merged.pop(k)

        # Convert to (start_idx, end_idx, phrase) in original frame index
        out = []
        for s, e, phr in merged:
            out.append((s, e + 1, phr))  # map diff-index to frame index
        return out
    
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
            
            # Save metadata JSON file (same format as process_scene)
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
            
            # Write metadata file
            metadata_file = output_path / "metadata" / f"{numeric_scene_id}.json"
            metadata_file.parent.mkdir(exist_ok=True)
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Return simplified metadata for AI processing
            return True, {
                'scene_id': scene_id,
                'numeric_scene_id': numeric_scene_id,
                'original_scene_id': original_scene_id,
                'deterministic_caption': deterministic_caption,
                'total_frames': len(camera_data),
                'video_url': video_url
            }
            
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
                    text_file = (output_path / "untagged_text") / f"{numeric_scene_id}.txt"
                    with open(text_file, 'w') as f:
                        f.write(caption)
                else:
                    # Use existing caption or fallback to basic generated caption
                    caption = self._get_existing_caption(original_scene_id, numeric_scene_id)
                
                if caption is None:
                    # Fallback to basic generated caption (non-motion path only)
                    caption = self._generate_caption(final_trajectory, video_url)

                # IMPORTANT: Only write to `texts/` when NOT using motion-based captioning.
                # Motion-based captions are exclusively stored in `untagged_text/` as per spec.
                if not self.caption_motion:
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

        # Basic stats (net change)
        pos_delta = positions[-1] - positions[0]            # [dx, dy, dz]
        ori_delta = orientations[-1] - orientations[0]      # [dpitch, dyaw, droll]

        # Use class-level thresholds for consistency
        t_trans_abs = self.THRESHOLD_TRANSLATION_ABS
        trans_ratio = self.THRESHOLD_TRANSLATION_RATIO
        t_yaw = self.THRESHOLD_YAW
        t_pitch = self.THRESHOLD_PITCH
        t_roll = self.THRESHOLD_ROLL
        rot_ratio = self.THRESHOLD_ROTATION_RATIO

        # Select dominant translation axes
        abs_trans = np.abs(pos_delta)
        max_t = float(np.max(abs_trans)) if np.any(abs_trans > 0) else 0.0
        translations: List[str] = []
        if max_t >= t_trans_abs:
            # Include axes meeting both absolute and relative thresholds (limit to top-2 for readability)
            candidates = []
            # X: right/left
            if abs_trans[0] >= max(t_trans_abs, trans_ratio * max_t):
                candidates.append((abs_trans[0], "moves right" if pos_delta[0] > 0 else "moves left"))
            # Y: up/down
            if abs_trans[1] >= max(t_trans_abs, trans_ratio * max_t):
                candidates.append((abs_trans[1], "moves up" if pos_delta[1] > 0 else "moves down"))
            # Z: forward/backward (forward = negative Z)
            if abs_trans[2] >= max(t_trans_abs, trans_ratio * max_t):
                candidates.append((abs_trans[2], "moves forward" if pos_delta[2] < 0 else "moves backward"))
            # Sort by magnitude and take up to 2
            candidates.sort(key=lambda x: x[0], reverse=True)
            translations = [c[1] for c in candidates[:2]]

        # Select dominant rotation axes (prefer yaw/pitch; roll requires higher threshold)
        abs_rot = np.abs(ori_delta)
        # Apply per-axis absolute thresholds first
        rot_candidates: List[Tuple[float, str, int]] = []  # (magnitude, phrase, axis_index)
        # Yaw (index 1)
        if abs_rot[1] >= t_yaw:
            rot_candidates.append((abs_rot[1], "pans right" if ori_delta[1] > 0 else "pans left", 1))
        # Pitch (index 0)
        if abs_rot[0] >= t_pitch:
            rot_candidates.append((abs_rot[0], "tilts up" if ori_delta[0] > 0 else "tilts down", 0))
        # Roll (index 2)
        if abs_rot[2] >= t_roll:
            rot_candidates.append((abs_rot[2], "rolls clockwise" if ori_delta[2] > 0 else "rolls counterclockwise", 2))

        rotations: List[str] = []
        if rot_candidates:
            # Enforce dominance ratio relative to the strongest axis
            rot_candidates.sort(key=lambda x: x[0], reverse=True)
            max_r = rot_candidates[0][0]
            filtered = [c for c in rot_candidates if c[0] >= rot_ratio * max_r]
            # Prefer yaw/pitch in tie, de-prioritize roll if mixed
            filtered.sort(key=lambda x: (x[2] == 2, -x[0]))  # roll last
            rotations = [c[1] for c in filtered[:2]]

        if not translations and not rotations:
            return "camera is mostly static"

        def join_list(parts: List[str]) -> str:
            if len(parts) == 1:
                return parts[0]
            if len(parts) == 2:
                return f"{parts[0]} and {parts[1]}"
            return f"{', '.join(parts[:-1])}, and {parts[-1]}"

        # Build caption with emphasis on dominant components only
        if translations and rotations:
            return f"camera {join_list(translations)} while it {join_list(rotations)}"
        elif translations:
            return f"camera {join_list(translations)}"
        else:
            return f"camera {join_list(rotations)}"

    def _process_scenes_batched(
        self,
        scenes_for_ai: List[Tuple[str, str, str, int]],
        scene_metadata: Dict[str, Dict],
    output_path: Optional[Path] = None
    ) -> Dict[str, Dict]:
        """
        Process scenes using TRUE batch inference for single-GPU mode
        
        Implements Qwen3-VL batch processing with:
        - Frame normalization (pad to max_frames for uniform batch)
        - Left-side padding for generation
        - Parallel GPU processing of batch_size scenes at once
        - Memory cleanup between batches
        
        Args:
            scenes_for_ai: List of (scene_id, numeric_scene_id, deterministic_caption, total_frames)
            scene_metadata: Metadata for each scene
            output_path: Output directory (for immediate text writing)
            
        Returns:
            Dictionary mapping numeric_scene_id to result dict
        """
        from tqdm import tqdm

        # Try to load model if not already loaded
        if self.ai_captioner is not None and not self.ai_captioner.model_loaded:
            try:
                print("Loading AI captioning model...")
                self.ai_captioner.load_model()
            except Exception as e:
                print(f"⚠️  Failed to load AI model: {e}")
                print(f"   Falling back to deterministic captions for all scenes")
                self.ai_captioner = None

        # Quick fallback if AI unavailable
        if self.ai_captioner is None:
            print(f"\n⚠️  AI captioner unavailable, writing deterministic captions")
            fallback_results: Dict[str, Dict] = {}
            if output_path:
                untagged_text_dir = output_path / "untagged_text"
                untagged_text_dir.mkdir(parents=True, exist_ok=True)
            for scene_id, numeric_scene_id, deterministic_caption, total_frames in scenes_for_ai:
                if output_path:
                    text_file = (output_path / "untagged_text") / f"{numeric_scene_id}.txt"
                    with open(text_file, 'w') as f:
                        f.write(deterministic_caption)
                fallback_results[numeric_scene_id] = {"success": False, "error": "AI unavailable", "caption": deterministic_caption}
            return fallback_results
        
        ai_results = {}
        num_scenes = len(scenes_for_ai)
        
        # Create untagged_text directory if writing immediately
        if output_path:
            untagged_text_dir = output_path / "untagged_text"
            untagged_text_dir.mkdir(parents=True, exist_ok=True)
        
        # Smarter batching: accumulate scenes with frames until batch is full
        pbar = tqdm(total=num_scenes, desc=f"AI Captioning (adaptive batching, target BS={self.ai_batch_size})")
        
        import gc
        batch_count = 0
        scene_idx = 0
        
        while scene_idx < num_scenes:
            # Accumulate scenes with frames until we have a full batch or run out
            batch_data_with_frames = []
            batch_original_data = []  # Track original data for all processed scenes
            
            # Keep searching until batch is full or we've checked all remaining scenes
            search_idx = scene_idx
            while len(batch_data_with_frames) < self.ai_batch_size and search_idx < num_scenes:
                scene_id, numeric_scene_id, deterministic_caption, total_frames = scenes_for_ai[search_idx]
                frame_paths = []
                
                # Quick frame lookup
                if self.video_source_dir and self.ai_captioner is not None:
                    frames_dir_candidates = [
                        Path(self.video_source_dir) / scene_id,
                        Path(self.video_source_dir) / numeric_scene_id,
                    ]
                    
                    for frames_dir in frames_dir_candidates:
                        if frames_dir.exists() and frames_dir.is_dir():
                            frame_paths = self.ai_captioner.sample_existing_frames(str(frames_dir))
                            if frame_paths:
                                break
                
                # Handle based on whether frames were found
                if frame_paths:
                    batch_data_with_frames.append((frame_paths, deterministic_caption, numeric_scene_id, total_frames))
                    batch_original_data.append((search_idx, numeric_scene_id, True))
                else:
                    # Write deterministic caption immediately for scenes without frames
                    if output_path:
                        text_file = (output_path / "untagged_text") / f"{numeric_scene_id}.txt"
                        with open(text_file, 'w') as f:
                            f.write(deterministic_caption)
                    ai_results[numeric_scene_id] = {'success': False, 'error': 'No frames'}
                    batch_original_data.append((search_idx, numeric_scene_id, False))
                
                search_idx += 1
            
            # Update scene_idx to continue from where we left off
            scene_idx = search_idx
            scenes_processed = len(batch_original_data)
            pbar.update(scenes_processed)
            
            # Generate AI captions for scenes with frames (if any)
            if batch_data_with_frames and self.ai_captioner is not None:
                batch_count += 1
            # Generate AI captions for scenes with frames (if any)
            if batch_data_with_frames and self.ai_captioner is not None:
                batch_count += 1
                batch_results = self.ai_captioner.generate_captions_batch(batch_data_with_frames)
                
                # Write AI results immediately
                for i, (_, _, numeric_scene_id, _) in enumerate(batch_data_with_frames):
                    result = batch_results[i]
                    if output_path:
                        text_file = (output_path / "untagged_text") / f"{numeric_scene_id}.txt"
                        with open(text_file, 'w') as f:
                            f.write(result['caption'])
                        ai_results[numeric_scene_id] = {
                            'success': result['success'],
                            'error': result.get('error', None)
                        }
                    else:
                        ai_results[numeric_scene_id] = result
                    
                    status = "✓" if result['success'] else "✗"
                    pbar.set_postfix({'batch': batch_count, 'size': len(batch_data_with_frames), 'last': f"{numeric_scene_id} {status}"})
                
                # Cleanup batch
                del batch_results
                del batch_data_with_frames
            
            # IMMEDIATE GPU cleanup after generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Cleanup batch and temp images
            if hasattr(self, 'ai_captioner') and self.ai_captioner is not None:
                try:
                    self.ai_captioner.cleanup_temp_images()
                except Exception:
                    pass
            
            # GPU cleanup every batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Log memory status every 5 batches for monitoring
            if batch_count % 5 == 0 and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                pbar.write(f"[Batch {batch_count}] GPU: {allocated:.2f}GB alloc, {reserved:.2f}GB reserved")
        
        pbar.close()
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
        
        # Calculate dominant translation direction (OpenGL: +X right, +Y up, -Z forward)
        total_translation = translations[-1] - translations[0]
        
        # Calculate rotational changes (yaw/pitch from forward vector, roll from up)
        euler_angles = self._rotation_matrices_to_euler(rotations)
        euler_diffs = np.diff(euler_angles, axis=0)
        
        # Calculate angular velocities
        angular_speeds = np.linalg.norm(euler_diffs, axis=1) / dt
        
        # Temporal segmentation into up to 3 segments (>=10% frames each)
        segments = self._segment_motion_sequence(translations, euler_angles, min_segment_ratio=0.1, max_segments=3)
        
        # Compose segment-wise descriptions with pace qualifiers
        motion_description = []
        total_frames = len(translations)
        for (s, e, phr) in segments:
            seg_speed = np.mean(translation_speeds[max(0, s-1):max(0, e-1)]) if len(translation_speeds) > 0 else 0.0
            qualifier = ""
            if seg_speed > 0.1:
                qualifier = " quickly"
            elif seg_speed < 0.03:
                qualifier = " slowly"
            motion_description.append(f"{phr}{qualifier}")
        
        # Analyze translation motion (use max to find truly dominant direction)
        avg_translation_speed = np.mean(translation_speeds) if len(translation_speeds) else 0.0
        abs_translations = np.abs(total_translation)
        max_t = float(np.max(abs_translations)) if np.any(abs_translations > 0) else 0.0
        
        if max_t >= self.THRESHOLD_TRANSLATION_ABS and avg_translation_speed > self.THRESHOLD_MIN_TRANSLATION_SPEED:
            dominant_axis = int(np.argmax(abs_translations))
            # Ensure dominance over other axes
            sorted_axes = np.sort(abs_translations)
            second = float(sorted_axes[-2]) if len(sorted_axes) >= 2 else 0.0
            if second <= self.THRESHOLD_TRANSLATION_RATIO * max_t:
                if dominant_axis == 0:
                    direction = "right" if total_translation[0] > 0 else "left"
                    motion_description.append(f"tracks {direction}")
                elif dominant_axis == 1:
                    direction = "up" if total_translation[1] > 0 else "down"
                    motion_description.append(f"moves {direction}")
                else:
                    direction = "forward" if total_translation[2] < 0 else "backward"
                    motion_description.append(f"dollies {direction}")
                # Speed qualifier
                if avg_translation_speed > 0.12:
                    motion_description[-1] = motion_description[-1].replace("tracks", "tracks quickly").replace("moves", "moves quickly").replace("dollies", "dollies quickly")
                elif avg_translation_speed < 0.03:
                    motion_description[-1] = motion_description[-1].replace("tracks", "tracks slowly").replace("moves", "moves slowly").replace("dollies", "dollies slowly")
        
        # Analyze rotational motion (use max to find truly dominant rotation)
        avg_angular_speed = np.mean(angular_speeds) if len(angular_speeds) else 0.0
        total_euler_change = euler_angles[-1] - euler_angles[0]
        abs_rotations = np.abs(total_euler_change)
        
        if avg_angular_speed > self.THRESHOLD_MIN_ANGULAR_SPEED:
            # Candidates above per-axis thresholds
            cand = []
            if abs_rotations[1] >= self.THRESHOLD_YAW:
                cand.append((abs_rotations[1], "pans right" if total_euler_change[1] > 0 else "pans left", 1))
            if abs_rotations[0] >= self.THRESHOLD_PITCH:
                cand.append((abs_rotations[0], "tilts up" if total_euler_change[0] > 0 else "tilts down", 0))
            if abs_rotations[2] >= self.THRESHOLD_ROLL:
                cand.append((abs_rotations[2], "rolls clockwise" if total_euler_change[2] > 0 else "rolls counterclockwise", 2))
            if cand:
                cand.sort(key=lambda x: x[0], reverse=True)
                max_r = cand[0][0]
                # Enforce dominance: others must be close to max; de-prioritize roll
                dom = [c for c in cand if c[0] >= self.THRESHOLD_ROTATION_RATIO * max_r]
                dom.sort(key=lambda x: (x[2] == 2, -x[0]))
                # Pick the top dominant one only for summary
                phrase = dom[0][1]
                motion_description.append(phrase)
                # Speed qualifier
                if avg_angular_speed > 0.22:
                    motion_description[-1] = motion_description[-1].replace("pans", "pans quickly").replace("tilts", "tilts quickly").replace("rolls", "rolls quickly")
                elif avg_angular_speed < 0.08:
                    motion_description[-1] = motion_description[-1].replace("pans", "pans slowly").replace("tilts", "tilts slowly").replace("rolls", "rolls slowly")
        
        # Generate final caption
        if not motion_description:
            return "camera remains relatively static"
        elif len(motion_description) == 1:
            return f"camera {motion_description[0]}"
        elif len(motion_description) == 2:
            return f"camera {motion_description[0]} then {motion_description[1]}"
        else:
            return f"camera {motion_description[0]} then {motion_description[1]} then {motion_description[2]}"
    
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
        
        # Sort camera files for consistent ordering
        camera_files = sorted(camera_files, key=lambda x: x.stem)
        
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
                print(f"Resume requested but no existing scene_id_mapping.json found, starting fresh")
        
        # Process scenes
        stats = {'processed': 0, 'failed': 0, 'skipped': 0}
        scene_ids = []
        
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
            
            # Save scene_id_mapping after Phase 1 for resume functionality
            if self.scene_id_mapping:
                mapping_file = output_path / "scene_id_mapping.json"
                with open(mapping_file, 'w') as f:
                    json.dump(self.scene_id_mapping, f, indent=2)
                print(f"Scene ID mapping saved: {len(self.scene_id_mapping)} scenes")
            
            # Phase 2: AI captioning across all scenes (parallel or sequential with efficient model)
            if scenes_for_ai:
                print(f"\nPhase 2: AI captioning for {len(scenes_for_ai)} scenes...")
                
                if self.parallel_captioner is not None:
                    # Multi-GPU parallel processing
                    ai_results = self.parallel_captioner.process_scenes_parallel(
                        scenes_for_ai,
                        self.video_source_dir or "",
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
