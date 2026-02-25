"""
CLaTr Evaluator Wrapper for TKCAM (migrated from MoMask)
Integrates trained CLaTr models to evaluate camera trajectory generation
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Add CLaTr to Python path
CLATR_PATH = "/home/haozhe/CamTraj/CLaTr"
if CLATR_PATH not in sys.path:
    sys.path.insert(0, CLATR_PATH)

from src.models.clatr import CLaTr
from src.models.clatr_frames import CLaTrFrames
from src.models.actor import ACTORStyleEncoder, ACTORStyleDecoder
from src.models.video_encoder import VideoEncoder
from src.training.metrics import all_contrastive_metrics


def euler_to_rotation_6d(euler_angles):
    """
    Convert Euler angles (pitch, yaw, roll) to 6D rotation representation
    
    Args:
        euler_angles: [B, N, 3] tensor with (pitch, yaw, roll) in radians
    
    Returns:
        rot6d: [B, N, 6] tensor with first two columns of rotation matrix
    """
    pitch, yaw, roll = euler_angles[..., 0], euler_angles[..., 1], euler_angles[..., 2]
    
    # Compute rotation matrix from Euler angles (ZYX convention)
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    cos_r, sin_r = torch.cos(roll), torch.sin(roll)
    
    # Build rotation matrix
    R = torch.zeros((*euler_angles.shape[:-1], 3, 3), device=euler_angles.device)
    
    R[..., 0, 0] = cos_y * cos_p
    R[..., 0, 1] = cos_y * sin_p * sin_r - sin_y * cos_r
    R[..., 0, 2] = cos_y * sin_p * cos_r + sin_y * sin_r
    
    R[..., 1, 0] = sin_y * cos_p
    R[..., 1, 1] = sin_y * sin_p * sin_r + cos_y * cos_r
    R[..., 1, 2] = sin_y * sin_p * cos_r - cos_y * sin_r
    
    R[..., 2, 0] = -sin_p
    R[..., 2, 1] = cos_p * sin_r
    R[..., 2, 2] = cos_p * cos_r
    
    # Extract first two columns for 6D representation
    rot6d = R[..., :, :2].reshape(*euler_angles.shape[:-1], 6)
    
    return rot6d


def tkcam_to_clatr_format(tkcam_trajectories):
    """
    Convert TKCAM 6D format (x,y,z,pitch,yaw,roll) to CLaTr 9D format (rot6d[6], trans[3])
    
    Args:
        tkcam_trajectories: [B, N, 6] tensor with (x, y, z, pitch, yaw, roll)
    
    Returns:
        clatr_trajectories: [B, N, 9] tensor with (rot6d[6], x, y, z)
    """
    # Split position and rotation
    position = tkcam_trajectories[..., :3]  # [B, N, 3]
    euler = tkcam_trajectories[..., 3:]     # [B, N, 3]
    
    # Convert Euler angles to 6D rotation
    rot6d = euler_to_rotation_6d(euler)  # [B, N, 6]
    
    # Concatenate: [rot6d (6), translation (3)]
    clatr_trajectories = torch.cat([rot6d, position], dim=-1)  # [B, N, 9]
    
    return clatr_trajectories


class CLaTrEvaluator:
    """Wrapper to use trained CLaTr models for TKCAM evaluation"""
    
    def __init__(
        self,
        text_ckpt_path=None,
        frame_ckpt_path=None,
        device='cuda'
    ):
        """
        Initialize CLaTr evaluators
        
        Args:
            text_ckpt_path: Path to text-trajectory CLaTr checkpoint
            frame_ckpt_path: Path to frame-trajectory CLaTr checkpoint
            device: Device to run models on
        """
        self.device = device
        self.text_model = None
        self.frame_model = None
        
        if text_ckpt_path is not None:
            self.load_text_model(text_ckpt_path)
        
        if frame_ckpt_path is not None:
            self.load_frame_model(frame_ckpt_path)
    
    def load_text_model(self, ckpt_path):
        """Load text-trajectory CLaTr model from checkpoint"""
        print(f"Loading text-trajectory CLaTr model from {ckpt_path}")
        
        # Model architecture parameters (should match training config)
        latent_dim = 256
        ff_size = 1024
        num_layers = 6  # Number of transformer layers (version_27 uses 6 layers)
        num_heads = 4
        dropout = 0.1
        activation = "gelu"
        
        # Create text encoder
        text_encoder = ACTORStyleEncoder(
            num_feats=512,  # CLIP text features
            vae=True,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            activation=activation
        )
        
        # Create trajectory encoder
        traj_encoder = ACTORStyleEncoder(
            num_feats=9,  # 6D rotation + 3D translation
            vae=True,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            activation=activation
        )
        
        # Create trajectory decoder
        traj_decoder = ACTORStyleDecoder(
            num_feats=9,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            activation=activation
        )
        
        # Create CLaTr model
        model = CLaTr(
            traj_encoder=traj_encoder,
            text_encoder=text_encoder,
            traj_decoder=traj_decoder,
            vae=True,
            fact=1.0,
            sample_mean=False,
            lmd={"recons": 1.0, "latent": 1.0e-5, "kl": 1.0e-5, "contrastive": 0.1},
            lr=1e-5,
            temperature=0.1,
            threshold_selfsim=0.995,
            threshold_selfsim_metrics=0.995,
            log_wandb=False,
            name="clatr_text_traj"
        )
        
        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = ckpt['state_dict']
        
        # Load state dict (strict=False to ignore optimizer states)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if len(missing_keys) > 0:
            print(f"  Warning: {len(missing_keys)} missing keys")
            if len(missing_keys) <= 10:
                for key in missing_keys:
                    print(f"    Missing: {key}")
            else:
                print("  First 10 missing keys:")
                for key in list(missing_keys)[:10]:
                    print(f"    Missing: {key}")
        
        if len(unexpected_keys) > 0:
            print(f"  Warning: {len(unexpected_keys)} unexpected keys")
            if len(unexpected_keys) <= 10:
                for key in unexpected_keys:
                    print(f"    Unexpected: {key}")
        
        model.eval()
        model.to(self.device)
        self.text_model = model
        print("  Text-trajectory CLaTr model loaded successfully!")
    
    def load_frame_model(self, ckpt_path):
        """Load frame-trajectory CLaTr model from checkpoint"""
        print(f"Loading frame-trajectory CLaTr model from {ckpt_path}")
        
        # Model architecture parameters (should match training config)
        latent_dim = 256
        ff_size = 1024
        num_layers = 6  # Number of transformer layers (version_28 uses 6 layers)
        num_heads = 4
        dropout = 0.1
        activation = "gelu"
        
        # Create trajectory encoder
        traj_encoder = ACTORStyleEncoder(
            num_feats=9,  # 6D rotation + 3D translation
            vae=True,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            activation=activation
        )
        
        # Create trajectory decoder
        traj_decoder = ACTORStyleDecoder(
            num_feats=9,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            activation=activation
        )
        
        # Create frame encoder
        frame_encoder = VideoEncoder(
            encoder_type="simple",
            hidden_dim=512,
            output_dim=latent_dim,
            num_frames=8,
            image_size=224
        )
        
        # Create CLaTrFrames model
        model = CLaTrFrames(
            traj_encoder=traj_encoder,
            frame_encoder=frame_encoder,
            traj_decoder=traj_decoder,
            vae=True,
            fact=1.0,
            sample_mean=False,
            lmd={"recons": 1.0, "latent": 1.0e-5, "kl": 1.0e-5, "contrastive": 0.1},
            lr=1e-5,
            temperature=0.1,
            threshold_selfsim=0.995,
            threshold_selfsim_metrics=0.995,
            log_wandb=False,
            name="clatr_frame_traj"
        )
        
        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = ckpt['state_dict']
        
        # Load state dict (strict=False to ignore optimizer states)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if len(missing_keys) > 0:
            print(f"  Warning: {len(missing_keys)} missing keys")
            if len(missing_keys) <= 10:
                for key in missing_keys:
                    print(f"    Missing: {key}")
            else:
                print("  First 10 missing keys:")
                for key in list(missing_keys)[:10]:
                    print(f"    Missing: {key}")
        
        if len(unexpected_keys) > 0:
            print(f"  Warning: {len(unexpected_keys)} unexpected keys")
            if len(unexpected_keys) <= 10:
                for key in unexpected_keys:
                    print(f"    Unexpected: {key}")
        
        model.eval()
        model.to(self.device)
        self.frame_model = model
        print("  Frame-trajectory CLaTr model loaded successfully!")
    
    def convert_tkcam_to_clatr_format(self, trajectories):
        """
        Convert TKCAM trajectory format to CLaTr expected format
        
        Args:
            trajectories: (batch, seq_len, 6) - TKCAM format [x,y,z,pitch,yaw,roll]
        
        Returns:
            Formatted trajectory tensor for CLaTr (batch, seq_len, 9) - [rot6d[6], x, y, z]
        """
        if isinstance(trajectories, np.ndarray):
            trajectories = torch.from_numpy(trajectories).float()
        
        trajectories = trajectories.to(self.device)
        
        # Ensure shape is correct (batch, seq_len, 6)
        if len(trajectories.shape) == 2:
            trajectories = trajectories.unsqueeze(0)
        
        # Convert from TKCAM 6D (x,y,z,pitch,yaw,roll) to CLaTr 9D (rot6d[6], trans[3])
        trajectories = tkcam_to_clatr_format(trajectories)
        
        return trajectories
    
    @torch.no_grad()
    def compute_text_trajectory_metrics(
        self,
        trajectories,
        text_features,
        masks=None,
        batch_size=64  # Process in smaller batches to avoid OOM
    ):
        """
        Compute text-to-trajectory retrieval metrics
        
        Args:
            trajectories: (N, seq_len, feat_dim) camera trajectories
            text_features: (N, text_dim) pre-computed CLIP text features  
            masks: (N, seq_len) padding masks (optional)
            batch_size: Batch size for encoding (to avoid OOM)
        
        Returns:
            dict with metrics: R@1, R@2, R@3, R@5, R@10, MedR for both t2m and m2t
        """
        if self.text_model is None:
            raise ValueError("Text model not loaded. Provide text_ckpt_path during init.")
        
        trajectories = self.convert_tkcam_to_clatr_format(trajectories)
        
        # Convert to tensors
        if isinstance(text_features, np.ndarray):
            text_features = torch.from_numpy(text_features).float()
        if isinstance(trajectories, np.ndarray):
            trajectories = torch.from_numpy(trajectories).float()
        
        # Create masks if not provided
        if masks is None:
            total_samples, seq_len = trajectories.shape[:2]
            masks = torch.ones(total_samples, seq_len, dtype=torch.bool)
        elif isinstance(masks, np.ndarray):
            masks = torch.from_numpy(masks)
        
        total_samples = trajectories.shape[0]
        text_seq_len = text_features.shape[1] if len(text_features.shape) == 3 else 1
        
        # Process in batches to avoid OOM
        print(f"  Encoding {total_samples} samples in batches of {batch_size}...")
        
        traj_latents_list = []
        text_latents_list = []
        
        for i in range(0, total_samples, batch_size):
            end_idx = min(i + batch_size, total_samples)
            
            # Get batch
            traj_batch = trajectories[i:end_idx].to(self.device)
            text_batch = text_features[i:end_idx].to(self.device)
            mask_batch = masks[i:end_idx].to(self.device)
            
            # Create text masks for this batch
            batch_len = end_idx - i
            text_masks = torch.ones(batch_len, text_seq_len, device=self.device, dtype=torch.bool)
            
            # Encode trajectories
            traj_inputs = {"x": traj_batch, "mask": mask_batch}
            traj_latents = self.text_model.encode(traj_inputs, modality="traj")
            
            # Encode text
            text_inputs = {"x": text_batch, "mask": text_masks}
            text_latents = self.text_model.encode(text_inputs, modality="text")
            
            # Move to CPU to save GPU memory
            traj_latents_list.append(traj_latents.cpu())
            text_latents_list.append(text_latents.cpu())
            
            if (i // batch_size + 1) % 20 == 0:
                print(f"    Processed {end_idx}/{total_samples} samples...")
        
        # Concatenate all latents
        traj_latents = torch.cat(traj_latents_list, dim=0)
        text_latents = torch.cat(text_latents_list, dim=0)
        
        print(f"  Computing similarity matrix ({total_samples}×{total_samples})...")
        
        # Compute similarity matrix on CPU for large datasets
        sim_matrix = self._get_sim_matrix_cpu(text_latents, traj_latents)
        sim_matrix_np = sim_matrix.numpy()
        
        print("  Computing retrieval metrics...")
        
        # Compute retrieval metrics
        metrics = all_contrastive_metrics(
            sim_matrix_np,
            emb=None,
            threshold=None
        )
        
        # Return metrics and latents for advanced metric computation
        return metrics, {
            'traj_latents': traj_latents.numpy(),
            'text_latents': text_latents.numpy()
        }
    
    @torch.no_grad()
    def compute_frame_trajectory_metrics(
        self,
        trajectories,
        frame_features,
        masks=None,
        batch_size=32  # Process in smaller batches to avoid OOM
    ):
        """
        Compute frame-to-trajectory retrieval metrics
        
        Args:
            trajectories: (N, seq_len, 6) camera trajectories
            frame_features: (N, num_frames, C, H, W) video frames
            masks: (N, seq_len) padding masks (optional)
            batch_size: Batch size for encoding (to avoid OOM)
        
        Returns:
            dict with metrics: R@1, R@2, R@3, R@5, R@10, MedR for both f2m and m2f
        """
        if self.frame_model is None:
            raise ValueError("Frame model not loaded. Provide frame_ckpt_path during init.")
        
        # Convert trajectories to CLaTr format
        trajectories = self.convert_tkcam_to_clatr_format(trajectories)
        
        # Convert to tensors
        if isinstance(trajectories, np.ndarray):
            trajectories = torch.from_numpy(trajectories).float()
        if isinstance(frame_features, np.ndarray):
            frame_features = torch.from_numpy(frame_features).float()
        
        # Create masks if not provided
        if masks is None:
            total_samples, seq_len = trajectories.shape[:2]
            masks = torch.ones(total_samples, seq_len, dtype=torch.bool)
        elif isinstance(masks, np.ndarray):
            masks = torch.from_numpy(masks)
        
        total_samples = trajectories.shape[0]
        num_frames = frame_features.shape[1]
        
        # Process in batches to avoid OOM
        print(f"  Encoding {total_samples} samples in batches of {batch_size}...")
        
        traj_latents_list = []
        frame_latents_list = []
        
        for i in range(0, total_samples, batch_size):
            end_idx = min(i + batch_size, total_samples)
            
            # Get batch
            traj_batch = trajectories[i:end_idx].to(self.device)
            frame_batch = frame_features[i:end_idx].to(self.device)
            mask_batch = masks[i:end_idx].to(self.device)
            
            # Encode trajectories
            traj_inputs = {"x": traj_batch, "mask": mask_batch}
            traj_latents = self.frame_model.encode(traj_inputs, modality="traj")
            
            # Encode frames
            frame_latents = self.frame_model.frame_encoder(frame_batch, mask_batch[:, :num_frames])
            
            # If VAE, extract mean
            if self.frame_model.vae and frame_latents.shape[1] == 2:
                frame_latents = frame_latents[:, 0]  # Take mu
            elif frame_latents.dim() == 3 and frame_latents.shape[1] == 1:
                frame_latents = frame_latents.squeeze(1)
            
            # Move to CPU to save GPU memory
            traj_latents_list.append(traj_latents.cpu())
            frame_latents_list.append(frame_latents.cpu())
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"    Processed {end_idx}/{total_samples} samples...")
        
        # Concatenate all latents
        traj_latents = torch.cat(traj_latents_list, dim=0)
        frame_latents = torch.cat(frame_latents_list, dim=0)
        
        print(f"  Computing similarity matrix ({total_samples}×{total_samples})...")
        
        # Compute similarity matrix on CPU to avoid GPU OOM
        # For very large datasets, we could also chunk this
        sim_matrix = self._get_sim_matrix_cpu(frame_latents, traj_latents)
        sim_matrix_np = sim_matrix.numpy()
        
        print("  Computing retrieval metrics...")
        
        # Compute retrieval metrics
        metrics = all_contrastive_metrics(
            sim_matrix_np,
            emb=None,
            threshold=None
        )
        
        return metrics
    
    def _get_sim_matrix(self, emb1, emb2):
        """Compute cosine similarity matrix between two sets of embeddings (GPU)"""
        # Normalize embeddings
        emb1 = F.normalize(emb1, dim=-1)
        emb2 = F.normalize(emb2, dim=-1)
        
        # Compute similarity
        sim_matrix = torch.matmul(emb1, emb2.T)
        return sim_matrix
    
    def _get_sim_matrix_cpu(self, emb1, emb2, chunk_size=512):
        """
        Compute cosine similarity matrix on CPU in chunks to handle large datasets
        
        Args:
            emb1: (N, D) embeddings
            emb2: (M, D) embeddings
            chunk_size: Process in chunks to avoid memory issues
        
        Returns:
            sim_matrix: (N, M) similarity matrix
        """
        # Normalize embeddings
        emb1 = F.normalize(emb1, dim=-1)
        emb2 = F.normalize(emb2, dim=-1)
        
        N, M = emb1.shape[0], emb2.shape[0]
        sim_matrix = torch.zeros(N, M)
        
        # Compute in chunks to avoid large memory allocation
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            sim_matrix[i:end_i] = torch.matmul(emb1[i:end_i], emb2.T)
        
        return sim_matrix
    
    def print_metrics(self, metrics, prefix=""):
        """Pretty print metrics"""
        print(f"\n{prefix}CLaTr Evaluation Metrics:")
        print("=" * 80)
        
        # Text-to-Motion metrics
        if 't2m/R01' in metrics:
            print(f"Text → Trajectory:")
            print(f"  R@1:  {metrics['t2m/R01']:.2f}%")
            print(f"  R@2:  {metrics['t2m/R02']:.2f}%")
            print(f"  R@3:  {metrics['t2m/R03']:.2f}%")
            print(f"  R@5:  {metrics['t2m/R05']:.2f}%")
            print(f"  R@10: {metrics['t2m/R10']:.2f}%")
            print(f"  MedR: {metrics['t2m/MedR']:.1f}")
        
        # Motion-to-Text metrics
        if 'm2t/R01' in metrics:
            print(f"\nTrajectory → Text:")
            print(f"  R@1:  {metrics['m2t/R01']:.2f}%")
            print(f"  R@2:  {metrics['m2t/R02']:.2f}%")
            print(f"  R@3:  {metrics['m2t/R03']:.2f}%")
            print(f"  R@5:  {metrics['m2t/R05']:.2f}%")
            print(f"  R@10: {metrics['m2t/R10']:.2f}%")
            print(f"  MedR: {metrics['m2t/MedR']:.1f}")
        
        print("=" * 80)


def evaluate_tkcam_with_clatr(
    val_loader,
    vq_model,
    trans_model,
    res_model,
    clatr_evaluator,
    num_samples=None,
    dataset_type='realestate10k',
    time_steps=18,
    cond_scale=4,
    temperature=1,
    sparse_keyframe_encoder=None
):
    """
    Evaluate TKCAM model using CLaTr metrics
    
    Args:
        val_loader: Validation data loader
        vq_model: VQ-VAE model
        trans_model: Transformer model
        res_model: Residual transformer model (optional)
        clatr_evaluator: CLaTrEvaluator instance
        num_samples: Number of samples to evaluate (None = all)
        dataset_type: Type of dataset for caption handling
        time_steps: Number of diffusion steps for generation
        cond_scale: Conditional guidance scale
        temperature: Sampling temperature
        sparse_keyframe_encoder: SparseKeyframeEncoder for frame-conditioned models (optional)
    
    Returns:
        Dictionary with all CLaTr metrics
    """
    vq_model.eval()
    trans_model.eval()
    if res_model is not None:
        res_model.eval()
    if sparse_keyframe_encoder is not None:
        sparse_keyframe_encoder.eval()
    
    all_pred_trajectories = []
    all_gt_trajectories = []
    all_text_features = []
    all_masks = []
    all_keyframes = []  # For frame-trajectory evaluation
    
    sample_count = 0
    
    print(f"Generating trajectories for evaluation...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Unpack batch - check if it includes frames
            has_frames = len(batch) == 8
            
            if has_frames:
                word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, keyframes = batch
                keyframes = keyframes.cuda().float()  # (bs, max_frames, 3, 224, 224)
            else:
                word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
                keyframes = None
            
            m_length = m_length.cuda()
            pose = pose.cuda().float()
            bs = pose.shape[0]
            
            # Clean text: remove POS tagging for CLaTr evaluation
            # TKCAM format: "text#word/POS word/POS..." -> CLaTr format: "text"
            cleaned_text = []
            for text in clip_text:
                if '#' in text:
                    # Split by '#' and take only the first part (the actual caption)
                    text = text.split('#')[0].strip()
                cleaned_text.append(text)
            
            # Encode cleaned text to CLIP features for CLaTr evaluation
            with torch.no_grad():
                clip_text_features = trans_model.encode_text(cleaned_text)  # [bs, 512]
            
            # Generate trajectories using TKCAM pipeline (also use cleaned text)
            # Step 1: Generate codes from transformer
            mids = trans_model.generate(
                cleaned_text,  # Use cleaned text (without POS tagging)
                torch.div(m_length, 4, rounding_mode='floor'),
                timesteps=time_steps,  # Note: parameter name is 'timesteps' not 'time_steps'
                cond_scale=cond_scale,
                temperature=temperature
            )
            
            # Step 2: Decode with VQ model
            mids.unsqueeze_(-1)
            pred_trajectories = vq_model.forward_decoder(mids)
            
            # Step 3: Optionally refine with residual model
            if res_model is not None:
                # Get initial codes
                code_indices, _ = vq_model.encode(pred_trajectories)
                # Refine with residual model (also use cleaned text)
                refined_codes = res_model.generate(
                    code_indices[..., 0],  # Input: first quantizer [batch, seq]
                    cleaned_text,  # Use cleaned text (without POS tagging)
                    torch.div(m_length, 4, rounding_mode='floor'),
                    temperature=temperature,
                    cond_scale=cond_scale
                )
                # refined_codes shape: [batch, seq, num_quantizers]
                # forward_decoder expects [batch, seq, num_quantizers]
                pred_trajectories = vq_model.forward_decoder(refined_codes)
            
            # Create padding masks
            masks = torch.arange(pose.shape[1], device=pose.device)[None, :] < m_length[:, None]
            
            # Expand CLIP text features to match trajectory sequence length
            # ACTORStyleEncoder expects [batch, seq_len, feature_dim]
            # We use a single text token repeated across the sequence
            text_seq_len = 1  # Use single token for text
            clip_text_expanded = clip_text_features.unsqueeze(1)  # [batch, 1, 512]
            
            # Store results
            all_pred_trajectories.append(pred_trajectories.cpu())
            all_gt_trajectories.append(pose.cpu())
            all_text_features.append(clip_text_expanded.cpu())  # [batch, 1, 512]
            all_masks.append(masks.cpu())
            
            # Store keyframes if available
            if has_frames and keyframes is not None:
                all_keyframes.append(keyframes.cpu())
            
            sample_count += bs
            
            if batch_idx % 10 == 0:
                print(f"  Processed {sample_count} samples...")
            
            if num_samples is not None and sample_count >= num_samples:
                break
    
    # Find maximum sequence length for padding
    max_seq_len = max(traj.shape[1] for traj in all_pred_trajectories)
    
    # Pad all trajectories to the same length
    def pad_to_max_len(trajectories, max_len):
        padded = []
        for traj in trajectories:
            if traj.shape[1] < max_len:
                pad_len = max_len - traj.shape[1]
                # Pad with zeros: (batch, seq, feat) -> (batch, max_seq, feat)
                padding = torch.zeros(traj.shape[0], pad_len, traj.shape[2], device=traj.device)
                traj = torch.cat([traj, padding], dim=1)
            padded.append(traj)
        return padded
    
    all_pred_trajectories = pad_to_max_len(all_pred_trajectories, max_seq_len)
    all_gt_trajectories = pad_to_max_len(all_gt_trajectories, max_seq_len)
    
    # Update masks to match padded length
    padded_masks = []
    for mask in all_masks:
        if mask.shape[1] < max_seq_len:
            pad_len = max_seq_len - mask.shape[1]
            padding = torch.zeros(mask.shape[0], pad_len, dtype=mask.dtype, device=mask.device)
            mask = torch.cat([mask, padding], dim=1)
        padded_masks.append(mask)
    all_masks = padded_masks
    
    # Concatenate all results
    all_pred_trajectories = torch.cat(all_pred_trajectories, dim=0)
    all_gt_trajectories = torch.cat(all_gt_trajectories, dim=0)
    all_text_features = torch.cat(all_text_features, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    print(f"\nTotal samples generated: {sample_count}")
    print(f"Padded sequence length: {max_seq_len}")

    
    # Compute CLaTr metrics on generated trajectories
    print("\n" + "="*80)
    print("Evaluating GENERATED trajectories with CLaTr")
    print("="*80)
    
    gen_metrics, gen_latents = clatr_evaluator.compute_text_trajectory_metrics(
        all_pred_trajectories.numpy(),
        all_text_features.numpy(),
        masks=all_masks.numpy()
    )
    clatr_evaluator.print_metrics(gen_metrics, prefix="Generated - ")
    
    # Compute CLaTr metrics on ground truth trajectories
    print("\n" + "="*80)
    print("Evaluating GROUND TRUTH trajectories with CLaTr")
    print("="*80)
    
    gt_metrics, gt_latents = clatr_evaluator.compute_text_trajectory_metrics(
        all_gt_trajectories.numpy(),
        all_text_features.numpy(),
        masks=all_masks.numpy()
    )
    clatr_evaluator.print_metrics(gt_metrics, prefix="Ground Truth - ")
    
    # Compute advanced metrics (CLaTr-FID, Coverage, etc.)
    print("\n" + "="*80)
    print("Computing Advanced CLaTr Metrics")
    print("="*80)
    
    from utils.clatr_advanced_metrics import compute_all_advanced_metrics, print_advanced_metrics
    
    advanced_metrics = compute_all_advanced_metrics(
        gt_traj_latents=gt_latents['traj_latents'],
        gen_traj_latents=gen_latents['traj_latents'],
        text_latents_gt=gt_latents['text_latents'],
        text_latents_gen=gen_latents['text_latents'],  # Same text for both
        max_samples_for_fid=5000  # Subsample for numerical stability
    )
    
    print_advanced_metrics(advanced_metrics)
    
    # Frame-Trajectory Evaluation (if frames are available)
    frame_metrics = None
    if len(all_keyframes) > 0 and clatr_evaluator.frame_model is not None:
        print("\n" + "="*80)
        print("Evaluating Frame-Trajectory Alignment with CLaTr")
        print("="*80)
        
        # Concatenate all keyframes
        all_keyframes_tensor = torch.cat(all_keyframes, dim=0)  # (N, T, C, H, W)
        
        print(f"Frame data shape: {all_keyframes_tensor.shape}")
        print(f"Generated trajectories: {all_pred_trajectories.shape}")
        print(f"Ground truth trajectories: {all_gt_trajectories.shape}")
        
        try:
            # Evaluate Generated trajectories with frames
            print("\nEvaluating GENERATED trajectories with frames...")
            gen_frame_metrics = clatr_evaluator.compute_frame_trajectory_metrics(
                all_pred_trajectories.numpy(),
                all_keyframes_tensor.numpy(),
                masks=all_masks.numpy()
            )
            
            # Evaluate GT trajectories with frames
            print("\nEvaluating GROUND TRUTH trajectories with frames...")
            gt_frame_metrics = clatr_evaluator.compute_frame_trajectory_metrics(
                all_gt_trajectories.numpy(),
                all_keyframes_tensor.numpy(),
                masks=all_masks.numpy()
            )
            
            # Print results
            print("\n" + "="*80)
            print("Frame-Trajectory Metrics")
            print("="*80)
            print("\nGenerated:")
            clatr_evaluator.print_metrics(gen_frame_metrics, prefix="Gen Frame - ")
            print("\nGround Truth:")
            clatr_evaluator.print_metrics(gt_frame_metrics, prefix="GT Frame - ")
            
            frame_metrics = {
                'generated': gen_frame_metrics,
                'ground_truth': gt_frame_metrics
            }
        except Exception as e:
            print(f"⚠️  Warning: Frame-trajectory evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    elif len(all_keyframes) == 0:
        print("\n⚠️  No frames loaded - skipping frame-trajectory evaluation")
        print("   Use --load_frames to enable frame evaluation")
    elif clatr_evaluator.frame_model is None:
        print("\n⚠️  Frame model not loaded - skipping frame-trajectory evaluation")
        print("   Use --clatr_frame_ckpt to provide frame model checkpoint")
    
    # Save latents for diagnostic analysis
    import numpy as np
    np.savez('clatr_eval_results.npz',
             gt_traj_latents=gt_latents['traj_latents'],
             gen_traj_latents=gen_latents['traj_latents'],
             gt_text_latents=gt_latents['text_latents'],
             gen_text_latents=gen_latents['text_latents'])
    print("\n✅ Saved latents to clatr_eval_results.npz for diagnostic analysis")
    
    result = {
        'generated': gen_metrics,
        'ground_truth': gt_metrics,
        'advanced': advanced_metrics,
        'num_samples': sample_count,
        'latents': {
            'gt_traj': gt_latents['traj_latents'],
            'gen_traj': gen_latents['traj_latents'],
            'text': gt_latents['text_latents']
        }
    }
    
    # Add frame metrics if available
    if frame_metrics is not None:
        result['frame_metrics'] = frame_metrics
    
    return result

