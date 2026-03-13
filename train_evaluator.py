#!/usr/bin/env python3
"""
Train a CLaTr-style Contrastive Evaluator for 12D Camera Trajectories.

This script trains a trajectory encoder + text projection to align both
modalities in a shared 512-D space using InfoNCE contrastive loss.

The trained evaluator is then used as a frozen metrics module during the
main camera trajectory generation training (FID, R-Precision, Matching Score).

Usage
-----
  # Default (RealEstate10K_rotmat_3k, 12D)
  python train_evaluator.py --data_root ./dataset/RealEstate10K_rotmat_3k \\
        --dataset_name realestate10k_rotmat --epochs 200 --batch_size 64

  # With a larger dataset
  python train_evaluator.py --data_root ./dataset/RealEstate10K_rotmat_5k \\
        --dataset_name realestate10k_rotmat --epochs 300 --batch_size 128

  # Resume training
  python train_evaluator.py --data_root ./dataset/RealEstate10K_rotmat_3k \\
        --dataset_name realestate10k_rotmat --resume checkpoints/evaluator/latest.pt

The checkpoint is saved to ``checkpoints/evaluator/{run_name}/``.
"""

import argparse
import os
import sys
import time
import math
import json
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    import clip as clip_module
except ImportError:
    print("ERROR: `clip` package not found. Install via: pip install git+https://github.com/openai/CLIP.git")
    sys.exit(1)

from models.evaluator.clatr_models import (
    TrajectoryEncoder,
    CLIPTextEncoder,
    InfoNCELoss,
    compute_fid,
    compute_r_precision,
    compute_matching_score,
    compute_diversity,
)

# ---------------------------------------------------------------------------
#  Dataset — loads (trajectory, caption) pairs for contrastive training
# ---------------------------------------------------------------------------

class TrajectoryTextDataset(Dataset):
    """
    Simple dataset returning (trajectory, caption) pairs.
    Trajectories are Z-normalized with the provided mean/std.
    Supports optional augmentation during training.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        max_motion_length: int = 300,
        min_motion_length: int = 24,
        input_dim: int = 12,
        augment: bool = False,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.motion_dir = self.data_root / "new_joint_vecs"
        self.text_dir = self.data_root / "texts"
        self.max_motion_length = max_motion_length
        self.input_dim = input_dim
        self.augment = augment

        # Load normalization stats
        self.mean = np.load(str(self.data_root / "Mean.npy")).astype(np.float32)
        self.std = np.load(str(self.data_root / "Std.npy")).astype(np.float32)

        # For rotmat: override rotation dims mean/std to identity (0, 1)
        if input_dim == 12 and self.mean.shape[-1] == 12:
            self.mean[6:] = 0.0
            self.std[6:] = 1.0

        # Avoid division by zero
        self.std = np.where(self.std < 1e-8, 1.0, self.std)

        # Load split
        split_file = self.data_root / f"{split}.txt"
        with open(split_file, "r") as f:
            id_list = [line.strip() for line in f.readlines() if line.strip()]

        self.samples = []
        skipped = 0
        for name in id_list:
            motion_path = self.motion_dir / f"{name}.npy"
            text_path = self.text_dir / f"{name}.txt"
            if not motion_path.exists() or not text_path.exists():
                skipped += 1
                continue
            motion = np.load(str(motion_path))
            if motion.shape[0] < min_motion_length or motion.shape[0] >= max_motion_length:
                skipped += 1
                continue
            # Parse captions (format: caption#tokens)
            captions = []
            with open(text_path, "r") as f:
                for line in f.readlines():
                    cap = line.strip().split("#")[0]
                    if cap:
                        captions.append(cap)
            if not captions:
                skipped += 1
                continue
            self.samples.append({"name": name, "motion": motion, "captions": captions})

        print(f"[TrajectoryTextDataset] split={split}: loaded {len(self.samples)} "
              f"samples, skipped {skipped}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        motion = sample["motion"].astype(np.float32).copy()
        m_length = motion.shape[0]

        # ── Augmentation (training only) ──
        if self.augment and m_length > 24:
            # Random temporal cropping: keep 80-100% of frames
            crop_ratio = np.random.uniform(0.8, 1.0)
            crop_len = max(24, int(m_length * crop_ratio))
            if crop_len < m_length:
                start = np.random.randint(0, m_length - crop_len)
                motion = motion[start:start + crop_len]
                m_length = crop_len

            # Gaussian noise on position (0:3) and velocity (3:6) dims
            noise_std = 0.01
            motion[:, :6] += np.random.randn(m_length, 6).astype(np.float32) * noise_std

        # Z-normalize
        motion = (motion - self.mean) / self.std

        # Pad to max_motion_length
        if motion.shape[0] < self.max_motion_length:
            pad = np.zeros(
                (self.max_motion_length - motion.shape[0], motion.shape[1]),
                dtype=np.float32,
            )
            motion = np.concatenate([motion, pad], axis=0)

        # Randomly select one caption
        caption = np.random.choice(sample["captions"])

        return {
            "motion": motion,      # (max_T, D)
            "m_length": m_length,   # int
            "caption": caption,     # str
        }


def collate_evaluator(batch):
    """Custom collate: stack tensors, keep captions as list."""
    motions = torch.from_numpy(np.stack([b["motion"] for b in batch]))
    lengths = torch.tensor([b["m_length"] for b in batch], dtype=torch.long)
    captions = [b["caption"] for b in batch]
    return motions, lengths, captions


# ---------------------------------------------------------------------------
#  Training Loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    traj_encoder,
    text_encoder,
    criterion,
    optimizer,
    dataloader,
    device,
    epoch,
    clip_tokenize,
    emb_dropout: float = 0.1,
):
    traj_encoder.train()
    text_encoder.train()

    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for motions, lengths, captions in pbar:
        motions = motions.to(device)
        lengths = lengths.to(device)

        # Tokenize text
        text_tokens = clip_tokenize(captions, truncate=True).to(device)

        # Forward
        traj_emb = traj_encoder(motions, lengths)       # (B, D)
        text_emb = text_encoder(text_tokens)             # (B, D)

        # L2-normalize for contrastive loss
        traj_emb = F.normalize(traj_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)

        # Embedding dropout: randomly zero dimensions to regularize
        if emb_dropout > 0:
            traj_emb = F.dropout(traj_emb, p=emb_dropout, training=True)
            text_emb = F.dropout(text_emb, p=emb_dropout, training=True)

        loss = criterion(text_emb, traj_emb)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(traj_encoder.parameters()) + list(text_encoder.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        bs = motions.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        pbar.set_postfix(loss=loss.item())

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def validate(
    traj_encoder,
    text_encoder,
    dataloader,
    device,
    clip_tokenize,
):
    traj_encoder.eval()
    text_encoder.eval()

    all_traj_emb = []
    all_text_emb = []
    all_captions = []

    for motions, lengths, captions in dataloader:
        motions = motions.to(device)
        lengths = lengths.to(device)
        text_tokens = clip_tokenize(captions, truncate=True).to(device)

        traj_emb = F.normalize(traj_encoder(motions, lengths), dim=-1)
        text_emb = F.normalize(text_encoder(text_tokens), dim=-1)

        all_traj_emb.append(traj_emb.cpu().numpy())
        all_text_emb.append(text_emb.cpu().numpy())
        all_captions.extend(captions)

    traj_emb_all = np.concatenate(all_traj_emb, axis=0)
    text_emb_all = np.concatenate(all_text_emb, axis=0)

    # R-Precision
    r_prec = compute_r_precision(text_emb_all, traj_emb_all, top_k=3)

    # Matching Score
    match_score = compute_matching_score(text_emb_all, traj_emb_all)

    # Diversity
    diversity = compute_diversity(traj_emb_all)

    # Retrieval accuracy (full-set, not batched)
    sim = text_emb_all @ traj_emb_all.T
    ranks_t2m = []
    for i in range(sim.shape[0]):
        sorted_idx = np.argsort(-sim[i])
        rank = np.where(sorted_idx == i)[0][0]
        ranks_t2m.append(rank)
    med_rank = float(np.median(ranks_t2m))
    recall_at_1 = float(np.mean([r < 1 for r in ranks_t2m]))
    recall_at_5 = float(np.mean([r < 5 for r in ranks_t2m]))

    return {
        "r_prec_top1": r_prec[0],
        "r_prec_top2": r_prec[1],
        "r_prec_top3": r_prec[2],
        "matching_score": match_score,
        "diversity": diversity,
        "recall_at_1": recall_at_1,
        "recall_at_5": recall_at_5,
        "median_rank": med_rank,
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train CLaTr-style evaluator")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to dataset root (e.g., ./dataset/RealEstate10K_rotmat_3k)")
    parser.add_argument("--dataset_name", type=str, default="realestate10k_rotmat",
                        help="Dataset name for dim detection")
    parser.add_argument("--input_dim", type=int, default=12,
                        help="Trajectory feature dimension (12 for rotmat)")
    parser.add_argument("--output_dim", type=int, default=512,
                        help="Shared latent space dimension")
    parser.add_argument("--latent_dim", type=int, default=256,
                        help="Transformer hidden dim")
    parser.add_argument("--num_layers", type=int, default=6,
                        help="Number of Transformer encoder layers")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--ff_size", type=int, default=1024,
                        help="Feed-forward hidden size")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="InfoNCE temperature")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--emb_dropout", type=float, default=0.1,
                        help="Dropout applied to embeddings before contrastive loss")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience (in eval intervals)")
    parser.add_argument("--min_lr_ratio", type=float, default=0.01,
                        help="Minimum LR as fraction of base LR")
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="Linear warmup epochs")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/evaluator",
                        help="Checkpoint save directory")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name (auto-generated if not set)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--eval_every", type=int, default=5,
                        help="Evaluate every N epochs")
    parser.add_argument("--save_every", type=int, default=20,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--max_motion_length", type=int, default=300,
                        help="Maximum trajectory length")
    parser.add_argument("--ortho_normalize", action="store_true", default=True,
                        help="Apply Gram-Schmidt to rot6d during encoding")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Run name
    if args.run_name is None:
        ds = Path(args.data_root).name
        args.run_name = f"clatr_eval_{ds}_{args.input_dim}d_ld{args.latent_dim}_od{args.output_dim}"
    save_dir = Path(args.save_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {save_dir}")

    # Save config
    with open(save_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Datasets ──────────────────────────────────────────
    train_dataset = TrajectoryTextDataset(
        data_root=args.data_root,
        split="train",
        max_motion_length=args.max_motion_length,
        min_motion_length=24,
        input_dim=args.input_dim,
        augment=True,
    )
    val_dataset = TrajectoryTextDataset(
        data_root=args.data_root,
        split="val",
        max_motion_length=args.max_motion_length,
        min_motion_length=24,
        input_dim=args.input_dim,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_evaluator,
        drop_last=True,  # Important for contrastive learning
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_evaluator,
        drop_last=False,
        pin_memory=True,
    )

    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_dataset)} samples, {len(val_loader)} batches")

    # ── Models ────────────────────────────────────────────
    traj_encoder = TrajectoryEncoder(
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_size=args.ff_size,
        dropout=args.dropout,
        ortho_normalize=args.ortho_normalize,
    ).to(device)

    text_encoder = CLIPTextEncoder(
        output_dim=args.output_dim,
    ).to(device)

    criterion = InfoNCELoss(temperature=args.temperature)

    # Only optimize trainable params (CLIP backbone is frozen)
    trainable_params = (
        list(traj_encoder.parameters())
        + list(text_encoder.projection.parameters())
    )
    optimizer = optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99)
    )

    # Cosine annealing LR scheduler with warmup and minimum LR floor
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return max(cosine, args.min_lr_ratio)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Resume ────────────────────────────────────────────
    start_epoch = 0
    best_r1 = 0.0
    patience_counter = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        traj_encoder.load_state_dict(ckpt["traj_encoder"])
        text_encoder.load_state_dict(ckpt["text_encoder"], strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_r1 = ckpt.get("best_r1", 0.0)
        print(f"Resumed from epoch {start_epoch}, best R@1={best_r1:.4f}")

    # Store normalization stats for downstream use
    mean_tensor = torch.from_numpy(train_dataset.mean)
    std_tensor = torch.from_numpy(train_dataset.std)

    clip_tokenize = clip_module.tokenize

    # ── Print model info ──────────────────────────────────
    traj_params = sum(p.numel() for p in traj_encoder.parameters() if p.requires_grad)
    text_proj_params = sum(p.numel() for p in text_encoder.projection.parameters())
    print(f"Trajectory encoder: {traj_params:,} trainable params")
    print(f"Text projection:    {text_proj_params:,} trainable params")
    print(f"Total trainable:    {traj_params + text_proj_params:,} params")

    # ── Training ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(
            traj_encoder, text_encoder, criterion, optimizer,
            train_loader, device, epoch + 1, clip_tokenize,
            emb_dropout=args.emb_dropout,
        )
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(f"Epoch {epoch + 1}/{args.epochs} | loss={train_loss:.4f} | "
              f"lr={lr:.2e} | time={elapsed:.1f}s")

        # ── Validation ──
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            metrics = validate(
                traj_encoder, text_encoder, val_loader, device, clip_tokenize,
            )
            print(f"  Val | R@1={metrics['recall_at_1']:.4f} R@5={metrics['recall_at_5']:.4f} "
                  f"MedR={metrics['median_rank']:.1f} | "
                  f"R-Prec [1/2/3]={metrics['r_prec_top1']:.4f}/{metrics['r_prec_top2']:.4f}/{metrics['r_prec_top3']:.4f} | "
                  f"Match={metrics['matching_score']:.4f} Div={metrics['diversity']:.4f}")

            # Save best model
            if metrics["recall_at_1"] > best_r1:
                best_r1 = metrics["recall_at_1"]
                patience_counter = 0
                ckpt = {
                    "traj_encoder": traj_encoder.state_dict(),
                    "text_encoder": text_encoder.state_dict(),
                    "epoch": epoch,
                    "best_r1": best_r1,
                    "metrics": metrics,
                    "mean": mean_tensor,
                    "std": std_tensor,
                    "config": vars(args),
                }
                torch.save(ckpt, save_dir / "best.pt")
                print(f"  ** New best R@1={best_r1:.4f} → saved best.pt")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping: no improvement for {args.patience} eval intervals")
                    break

        # ── Save checkpoint ──
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt = {
                "traj_encoder": traj_encoder.state_dict(),
                "text_encoder": text_encoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_r1": best_r1,
                "mean": mean_tensor,
                "std": std_tensor,
                "config": vars(args),
            }
            torch.save(ckpt, save_dir / "latest.pt")
            if (epoch + 1) % (args.save_every * 5) == 0:
                torch.save(ckpt, save_dir / f"epoch_{epoch + 1:04d}.pt")
            print(f"  Saved checkpoint → {save_dir / 'latest.pt'}")

    print(f"\n{'='*60}")
    print(f"Training complete. Best R@1 = {best_r1:.4f}")
    print(f"Best checkpoint: {save_dir / 'best.pt'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
