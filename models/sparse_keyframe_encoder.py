"""
Sparse Keyframe Encoder for MoMask Camera Trajectory Generation

Randomly samples N: [1,4] keyframes from T frames, encodes with ResNet (trainable),
creates sparse tensor, and downsamples to match motion token sequence length.

Architecture:
    1. Random sparse sampling: N keyframes from T frames
    2. ResNet18/34 encoding: (N frames) → (N, 512)
    3. Sparse tensor creation: (N, 512) → (T, 512) with zeros
    4. Temporal conv downsampling: (T, 512) → (T//4, latent_dim)
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from typing import List
from pathlib import Path


class SparseKeyframeEncoder(nn.Module):
    """
    Encodes sparse keyframes using ResNet + Temporal Convolution.
    
    Args:
        resnet_arch: 'resnet18' or 'resnet34'
        latent_dim: Output dimension (should match motion embedding dim, e.g., 384)
        pretrained: Use ImageNet pretrained weights
    """
    
    def __init__(self, resnet_arch='resnet18', latent_dim=384, pretrained=True):
        super().__init__()
        
        self.resnet_arch = resnet_arch
        self.latent_dim = latent_dim
        
        # Load ResNet backbone (trainable)
        if resnet_arch == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            self.resnet_dim = 512
        elif resnet_arch == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            self.resnet_dim = 512
        else:
            raise ValueError(f"Unsupported ResNet architecture: {resnet_arch}")
        
        # Remove final FC layer, keep feature extractor
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Temporal downsampling: (T, 512) -> (T//4, latent_dim)
        # Use 2 conv layers with stride=2 each: T -> T/2 -> T/4
        self.temporal_downsample = nn.Sequential(
            nn.Conv1d(self.resnet_dim, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True)
        )
        
        print(f"SparseKeyframeEncoder: {resnet_arch} (trainable) + Temporal Conv → {latent_dim}D")
    
    def encode_frames(self, images):
        """
        Encode a batch of images with ResNet.
        
        Args:
            images: (N_total, C, H, W) - all keyframes from batch stacked
        
        Returns:
            features: (N_total, resnet_dim) - ResNet features
        """
        # ResNet forward
        features = self.resnet_features(images)  # (N_total, resnet_dim, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (N_total, resnet_dim)
        return features
    
    def forward(self, frame_paths_batch, m_lens, deterministic=False):
        """
        Main forward pass: load frames → ResNet → sparse tensor → temporal downsample.
        
        Args:
            frame_paths_batch: List of List[Path], length B
                Each element is a list of ALL frame paths for that scene (T_i frames, varies 100-280)
            m_lens: Tensor (B,) - original motion lengths (before VQ downsampling)
            deterministic: If True, always sample N=2 frames at fixed positions (for validation)
        
        Returns:
            frame_embeddings: Tensor (T_motion, B, latent_dim)
                Where T_motion = T_original // 4 (matches motion token sequence length)
        """
        from PIL import Image
        import torchvision.transforms as transforms
        import random
        import numpy as np
        
        device = next(self.parameters()).device
        batch_size = len(frame_paths_batch)
        
        # Image preprocessing (ImageNet normalization)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # ===== PHASE 1: Collect all images to batch encode =====
        all_images_list = []  # Will collect all sampled images
        batch_metadata = []   # Track which images belong to which sample
        
        for b_idx, frame_paths in enumerate(frame_paths_batch):
            # Use the actual motion length, not frame path count
            T = int(m_lens[b_idx].item())  # Motion trajectory length (100-280)
            num_available_frames = len(frame_paths)  # Available frame images
            
            if num_available_frames == 0 or T == 0:
                # No frames available - will create dummy zeros later
                batch_metadata.append({
                    'b_idx': b_idx,
                    'T': T,
                    'N': 0,
                    'sampled_indices': [],
                    'image_start_idx': len(all_images_list),
                    'image_end_idx': len(all_images_list)
                })
                continue
            
            # Sample keyframes
            if deterministic:
                # Validation mode: fixed N=2 frames at 1/3 and 2/3 positions
                N = 2
                max_frame_idx = min(num_available_frames, T) - 1
                if max_frame_idx < 1:
                    sampled_indices = [0]
                    N = 1
                else:
                    idx1 = max_frame_idx // 3
                    idx2 = (max_frame_idx * 2) // 3
                    sampled_indices = sorted([idx1, idx2])
            else:
                # Training mode: random N ∈ [0, 1, 2, 3, 4]
                N = random.randint(0, 4)
                
                if N == 0:
                    # No keyframes - will create all-zeros sparse tensor later
                    batch_metadata.append({
                        'b_idx': b_idx,
                        'T': T,
                        'N': 0,
                        'sampled_indices': [],
                        'image_start_idx': len(all_images_list),
                        'image_end_idx': len(all_images_list)
                    })
                    continue
                
                # Sample from available frames, but respect motion length T
                max_frame_idx = min(num_available_frames, T) - 1
                if max_frame_idx < 0:
                    max_frame_idx = 0
                
                # Non-uniformly sample N unique indices from [0, max_frame_idx]
                perm = torch.randperm(max_frame_idx + 1)[:N]
                sampled_indices = sorted(perm.tolist())
            
            # Load sampled frames
            image_start_idx = len(all_images_list)
            for idx in sampled_indices:
                img = Image.open(frame_paths[idx]).convert('RGB')
                all_images_list.append(transform(img))
            
            batch_metadata.append({
                'b_idx': b_idx,
                'T': T,
                'N': N,
                'sampled_indices': sampled_indices,
                'image_start_idx': image_start_idx,
                'image_end_idx': len(all_images_list)
            })
        
        # ===== PHASE 2: Batch encode all images with ResNet =====
        if len(all_images_list) > 0:
            # Stack all images into single batch
            all_images_tensor = torch.stack(all_images_list).to(device)  # (total_N, 3, 224, 224)
            
            # Single ResNet forward pass for ALL images
            with torch.set_grad_enabled(self.training):
                all_features = self.encode_frames(all_images_tensor)  # (total_N, resnet_dim)
        else:
            all_features = None
        
        # ===== PHASE 3: Reconstruct sparse tensors per sample =====
        sparse_features_list = []
        has_any_frames = False  # Track if any sample in batch has frames
        
        for meta in batch_metadata:
            T = meta['T']
            N = meta['N']
            
            if N > 0:
                has_any_frames = True
            
            if N == 0:
                # No frames - create zeros
                sparse_features = torch.zeros(T, self.resnet_dim, device=device)
            else:
                # Extract this sample's features from batched output
                sample_features = all_features[meta['image_start_idx']:meta['image_end_idx']]
                
                # Create sparse tensor: (T, resnet_dim) with zeros everywhere except sampled positions
                sparse_features = torch.zeros(T, self.resnet_dim, device=device)
                for i, idx in enumerate(meta['sampled_indices']):
                    sparse_features[idx] = sample_features[i]
            
            sparse_features_list.append(sparse_features)
        
        # Pad to same length (max T in batch)
        max_T = max(sf.shape[0] for sf in sparse_features_list)
        
        padded_features = []
        for sf in sparse_features_list:
            T = sf.shape[0]
            if T < max_T:
                padding = torch.zeros(max_T - T, self.resnet_dim, device=device)
                sf = torch.cat([sf, padding], dim=0)
            padded_features.append(sf)
        
        # Stack: (B, T, resnet_dim)
        sparse_batch = torch.stack(padded_features)
        
        # Temporal downsampling: (B, T, resnet_dim) → (B, T//4, latent_dim)
        # Conv1d expects (B, C, T), so permute
        sparse_batch = sparse_batch.permute(0, 2, 1)  # (B, resnet_dim, T)
        
        downsampled = self.temporal_downsample(sparse_batch)  # (B, latent_dim, T//4)
        
        # Permute back to (T//4, B, latent_dim) for transformer
        downsampled = downsampled.permute(2, 0, 1)
        
        return downsampled, has_any_frames


class SparseKeyframeEncoderWithCaching(SparseKeyframeEncoder):
    """
    Version with optional frame caching for faster training.
    Can cache ResNet features to avoid repeated encoding.
    """
    
    def __init__(self, resnet_arch='resnet18', latent_dim=384, pretrained=True, use_cache=False):
        super().__init__(resnet_arch, latent_dim, pretrained)
        self.use_cache = use_cache
        self.feature_cache = {}  # hash_id -> (T, resnet_dim)
        print(f"Frame caching: {'enabled' if use_cache else 'disabled'}")
    
    def cache_scene_features(self, hash_id, features):
        """Cache ResNet features for a scene."""
        if self.use_cache:
            self.feature_cache[hash_id] = features.detach().cpu()
    
    def get_cached_features(self, hash_id):
        """Retrieve cached features if available."""
        if self.use_cache and hash_id in self.feature_cache:
            return self.feature_cache[hash_id].to(next(self.parameters()).device)
        return None
