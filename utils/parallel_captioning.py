#!/usr/bin/env python3
"""
Parallel Video Captioning with Multi-GPU Support

This module provides parallel processing capabilities for AI captioning,
allowing efficient utilization of multiple GPUs for faster dataset processing.
"""

import torch
import torch.multiprocessing as mp
from multiprocessing import Queue, Process
from queue import Empty
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import time


class ParallelQwenCaptioner:
    """
    Multi-GPU parallel captioning system for efficient processing
    
    Features:
    - Distributes scenes across multiple GPUs
    - Load balancing with work queue
    - Progress tracking across workers
    - Robust error handling per worker
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        num_gpus: int = None,
        max_frames: int = 32,
        batch_size: int = 1  # Future: support batch processing
    ):
        """
        Initialize parallel captioner
        
        Args:
            model_name: Qwen model to use
            num_gpus: Number of GPUs to use (None = auto-detect)
            max_frames: Max frames per video
            batch_size: Batch size per GPU (currently 1)
        """
        self.model_name = model_name
        self.max_frames = max_frames
        self.batch_size = batch_size
        
        # Auto-detect available GPUs
        if num_gpus is None:
            self.num_gpus = torch.cuda.device_count()
        else:
            self.num_gpus = min(num_gpus, torch.cuda.device_count())
        
        if self.num_gpus == 0:
            raise RuntimeError("No CUDA GPUs available for parallel processing")
        
        print(f"Parallel captioner initialized with {self.num_gpus} GPUs")
        print(f"Model: {model_name}")
        print(f"Max frames: {max_frames}")
        
        # Create a single captioner instance for compatibility methods
        # (used for single-scene calls like sample_existing_frames)
        from utils.realestate10k_processor import QwenVideoCaptioner
        self._single_captioner = QwenVideoCaptioner(
            model_name=model_name,
            max_frames=max_frames,
            device='cuda:0'  # Use first GPU for single calls
        )
    
    def sample_existing_frames(self, frames_dir: str) -> List[str]:
        """
        Compatibility method: Sample frames from directory
        Delegates to single captioner instance
        """
        return self._single_captioner.sample_existing_frames(frames_dir)
    
    def generate_caption_with_context(
        self,
        frame_paths: List[str],
        deterministic_description: str,
        scene_id: str,
        total_frames: int
    ) -> Dict:
        """
        Compatibility method: Generate caption for a single scene
        Delegates to single captioner instance (loads model on first use)
        
        Note: For batch processing, use process_scenes_parallel() instead for better performance
        """
        # Ensure model is loaded
        if not self._single_captioner.model_loaded:
            print("Loading Qwen model on first caption request...")
            self._single_captioner.load_model()
        
        return self._single_captioner.generate_caption_with_context(
            frame_paths=frame_paths,
            deterministic_description=deterministic_description,
            scene_id=scene_id,
            total_frames=total_frames
        )
    
    @staticmethod
    def worker_process(
        gpu_id: int,
        work_queue: Queue,
        result_queue: Queue,
        model_name: str,
        max_frames: int,
        video_frames_dir: str
    ):
        """
        Worker process that runs on a single GPU
        
        Args:
            gpu_id: GPU device ID
            work_queue: Queue of work items (scene_info tuples)
            result_queue: Queue for results
            model_name: Qwen model name
            max_frames: Maximum frames to process
            video_frames_dir: Directory containing video frames
        """
        # Import here to avoid issues with multiprocessing
        from utils.realestate10k_processor import QwenVideoCaptioner
        
        try:
            # Use the GPU ID directly without CUDA_VISIBLE_DEVICES manipulation
            # torch.multiprocessing with 'spawn' handles this correctly
            device = f'cuda:{gpu_id}'
            
            # Initialize captioner for this worker
            captioner = QwenVideoCaptioner(
                model_name=model_name,
                max_frames=max_frames,
                device=device
            )
            
            # Load model once for this worker
            captioner.load_model(use_flash_attention=True)
            
            print(f"Worker GPU {gpu_id}: Model loaded, ready to process")
            
            processed = 0
            
            # Process work items until queue is empty
            while True:
                try:
                    # Get work item with timeout
                    work_item = work_queue.get(timeout=5)
                    
                    if work_item is None:  # Poison pill to stop worker
                        print(f"Worker GPU {gpu_id}: Received stop signal, processed {processed} scenes")
                        break
                    
                    # Unpack work item
                    scene_id, numeric_scene_id, deterministic_caption, total_frames = work_item
                    
                    # Find frames for this scene
                    frame_paths = []
                    frames_dir_candidates = [
                        Path(video_frames_dir) / scene_id,
                        Path(video_frames_dir) / numeric_scene_id,
                    ]
                    
                    for frames_dir in frames_dir_candidates:
                        if frames_dir.exists() and frames_dir.is_dir():
                            frame_paths = captioner.sample_existing_frames(str(frames_dir))
                            if frame_paths:
                                break
                    
                    # Generate caption
                    if frame_paths:
                        result = captioner.generate_caption_with_context(
                            frame_paths=frame_paths,
                            deterministic_description=deterministic_caption,
                            scene_id=numeric_scene_id,
                            total_frames=total_frames
                        )
                        
                        if result["success"]:
                            caption = result["caption"]
                            success = True
                            error = None
                        else:
                            caption = deterministic_caption
                            success = False
                            error = result.get("error", "Unknown error")
                    else:
                        # No frames found, use deterministic
                        caption = deterministic_caption
                        success = False
                        error = "No frames found"
                    
                    # Send result back
                    result_queue.put({
                        'scene_id': scene_id,
                        'numeric_scene_id': numeric_scene_id,
                        'caption': caption,
                        'success': success,
                        'error': error,
                        'gpu_id': gpu_id
                    })
                    
                    processed += 1
                    
                except Empty:
                    # No more work items, exit gracefully
                    print(f"Worker GPU {gpu_id}: Queue empty, processed {processed} scenes")
                    break
                    
                except Exception as e:
                    # Log error but continue processing
                    print(f"Worker GPU {gpu_id}: Error processing scene: {e}")
                    result_queue.put({
                        'scene_id': scene_id if 'scene_id' in locals() else 'unknown',
                        'numeric_scene_id': numeric_scene_id if 'numeric_scene_id' in locals() else 'unknown',
                        'caption': deterministic_caption if 'deterministic_caption' in locals() else '',
                        'success': False,
                        'error': str(e),
                        'gpu_id': gpu_id
                    })
        
        except Exception as e:
            print(f"Worker GPU {gpu_id}: Fatal error: {e}")
            import traceback
            traceback.print_exc()
    
    def process_scenes_parallel(
        self,
        scenes_data: List[Tuple[str, str, str, int]],
        video_frames_dir: str,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Dict]:
        """
        Process multiple scenes in parallel across GPUs
        
        Args:
            scenes_data: List of (scene_id, numeric_scene_id, deterministic_caption, total_frames)
            video_frames_dir: Directory containing video frames
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary mapping numeric_scene_id to result dict
        """
        # Set multiprocessing start method to 'spawn' for CUDA compatibility
        # This creates fresh Python processes without inheriting CUDA state
        mp_context = mp.get_context('spawn')
        
        # Create queues using spawn context
        work_queue = mp_context.Queue()
        result_queue = mp_context.Queue()
        
        # Fill work queue
        for scene_data in scenes_data:
            work_queue.put(scene_data)
        
        # Add poison pills (one per worker)
        for _ in range(self.num_gpus):
            work_queue.put(None)
        
        # Start worker processes using spawn context
        workers = []
        for gpu_id in range(self.num_gpus):
            worker = mp_context.Process(
                target=self.worker_process,
                args=(
                    gpu_id,
                    work_queue,
                    result_queue,
                    self.model_name,
                    self.max_frames,
                    video_frames_dir
                )
            )
            worker.start()
            workers.append(worker)
        
        # Collect results with progress bar
        results = {}
        pbar = tqdm(total=len(scenes_data), desc="AI Captioning (Multi-GPU)")
        
        completed = 0
        while completed < len(scenes_data):
            try:
                result = result_queue.get(timeout=1)
                results[result['numeric_scene_id']] = result
                completed += 1
                pbar.update(1)
                
                # Update progress bar description with GPU stats
                if result['success']:
                    pbar.set_postfix({'GPU': result['gpu_id'], 'status': '✓'})
                else:
                    pbar.set_postfix({'GPU': result['gpu_id'], 'status': '✗'})
                
                if progress_callback:
                    progress_callback(result)
                    
            except Empty:
                continue
        
        pbar.close()
        
        # Wait for all workers to finish
        for worker in workers:
            worker.join()
        
        # Print summary
        successful = sum(1 for r in results.values() if r['success'])
        print(f"\nParallel processing complete:")
        print(f"  Total: {len(results)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {len(results) - successful}")
        
        return results


def estimate_optimal_gpu_count(
    model_name: str,
    available_gpus: int,
    gpu_memory_gb: int = 80
) -> Tuple[int, str]:
    """
    Estimate optimal number of GPUs to use based on model size
    
    Args:
        model_name: Qwen model name
        available_gpus: Number of available GPUs
        gpu_memory_gb: Memory per GPU in GB
        
    Returns:
        Tuple of (recommended_gpu_count, explanation)
    """
    # Estimate model memory requirements
    if "32B" in model_name or "72B" in model_name:
        model_memory = 40  # GB
        recommended = min(2, available_gpus)
        explanation = f"Large model (~{model_memory}GB), recommend {recommended} GPUs"
    elif "7B" in model_name or "8B" in model_name:
        model_memory = 15  # GB
        # Can fit ~5 models per 80GB GPU
        models_per_gpu = gpu_memory_gb // (model_memory + 5)  # +5GB overhead
        recommended = min(available_gpus, max(1, available_gpus))
        explanation = f"Small model (~{model_memory}GB), can use all {recommended} GPUs"
    else:
        model_memory = 20  # GB (default estimate)
        recommended = min(3, available_gpus)
        explanation = f"Medium model (~{model_memory}GB), recommend {recommended} GPUs"
    
    return recommended, explanation


if __name__ == "__main__":
    # Example usage
    print("Parallel Captioning Module")
    print("=" * 50)
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    if num_gpus > 0:
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            print(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")
        
        # Estimate optimal configuration
        model = "Qwen/Qwen2.5-VL-7B-Instruct"
        recommended, explanation = estimate_optimal_gpu_count(model, num_gpus)
        print(f"\nRecommendation for {model}:")
        print(f"  {explanation}")
    else:
        print("No CUDA GPUs available")

