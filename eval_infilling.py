"""
Oracle Infilling Evaluation Script

Evaluates the MaskTransformer's ability to generate consistent camera trajectories
when provided with sparse Ground Truth (GT) anchor tokens.

Supports two anchor modes:
  - head_tail: Fix the first N and last N tokens.
  - sparse:    Fix tokens at regular intervals (every K frames).

Generates 3-panel side-by-side visualisations (GT / Unconstrained / Infilled) and
prints L2 position error + cosine rotation similarity metrics.

Usage Example:
  python eval_infilling.py \
      --dataset_name realestate10k_rotmat \
      --name mtrans_3k_clip_crossattn_reduce \
      --res_name rtrans_3k_clip_crossattn_reduce \
      --conditioning_mode clip \
      --gpu_id 0 \
      --anchor_mode head_tail \
      --anchor_n 5 \
      --num_samples 5 \
      --time_steps 10 \
      --cond_scale 3 \
      --temperature 0.3 \
      --topkr 0.9 \
      --ext infilling_eval
"""

import os
import sys
import textwrap
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from models.vq.model import RVQVAE

from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt
from utils.fixseed import fixseed
from utils.unified_data_format import (
    UnifiedCameraData,
    CameraDataFormat,
    detect_format_from_dataset_name,
    gram_schmidt_orthogonalize,
)
from utils.dataset_config import get_unified_dataset_config
from utils.camera_geometry import sixd_to_matrix, forward_from_sixd, to_mpl

# ---------------------------------------------------------------------------
# Model loading — reuse from gen_camera.py
# ---------------------------------------------------------------------------
from gen_camera import load_vq_model, load_trans_model, load_res_model

clip_version = "ViT-B/32"


# ===================================================================
#  Anchor mask construction
# ===================================================================

def build_anchor_mask_head_tail(token_len: int, n: int) -> np.ndarray:
    """Return bool array (token_len,) with first & last *n* positions True."""
    mask = np.zeros(token_len, dtype=bool)
    n = min(n, token_len // 2)
    mask[:n] = True
    mask[-n:] = True
    return mask


def build_anchor_mask_sparse(token_len: int, every_k: int) -> np.ndarray:
    """Return bool array (token_len,) with every *every_k*-th position True."""
    mask = np.zeros(token_len, dtype=bool)
    for i in range(0, token_len, every_k):
        mask[i] = True
    # Always include the last token
    mask[-1] = True
    return mask


# ===================================================================
#  Post-processing helpers
# ===================================================================

def smooth_positions(data: np.ndarray, sigma: float = 1.0,
                     format_type=None) -> np.ndarray:
    """Apply Gaussian smoothing to positions; ortho-normalise rotmat if 12D rotmat."""
    data = data.copy()
    # Positions are always columns 0-2
    for d in range(3):
        data[:, d] = gaussian_filter1d(data[:, d], sigma=sigma)

    # Gram-Schmidt on rotation columns for FULL_12_ROTMAT
    if format_type == CameraDataFormat.FULL_12_ROTMAT and data.shape[1] >= 12:
        data[:, 6:12] = gram_schmidt_orthogonalize(data[:, 6:12])

    return data


# ===================================================================
#  Metrics
# ===================================================================

def compute_l2_position_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean L2 distance between predicted and GT positions (cols 0-2)."""
    return float(np.mean(np.linalg.norm(pred[:, :3] - gt[:, :3], axis=-1)))


def compute_cosine_rotation_similarity(pred: np.ndarray, gt: np.ndarray,
                                       format_type) -> float:
    """Mean cosine similarity between predicted and GT forward vectors."""
    pred_ucd = UnifiedCameraData(pred, format_type=format_type)
    gt_ucd = UnifiedCameraData(gt, format_type=format_type)

    def _forward_vectors(ucd):
        ori = ucd.orientations.numpy()
        fwd_list = []
        for i in range(len(ori)):
            o = ori[i]
            if ucd.format_type == CameraDataFormat.FULL_12_ROTMAT:
                fwd = forward_from_sixd(o.reshape(1, 6)).squeeze(0)
            elif ucd.format_type == CameraDataFormat.QUATERNION_10:
                from common.quaternion import qrot
                quat = torch.tensor(o, dtype=torch.float32)
                fwd = (
                    torch.tensor(
                        [0.0, 0.0, -1.0], dtype=torch.float32
                    ).unsqueeze(0)
                )
                fwd = (
                    __import__("common.quaternion", fromlist=["qrot"])
                    .qrot(quat.unsqueeze(0), fwd)
                    .squeeze(0)
                    .numpy()
                )
            else:
                pitch, yaw = o[0], o[1]
                fwd = np.array([
                    np.cos(pitch) * np.sin(yaw),
                    -np.sin(pitch),
                    -np.cos(pitch) * np.cos(yaw),
                ])
            norm = np.linalg.norm(fwd)
            if norm > 1e-8:
                fwd = fwd / norm
            fwd_list.append(fwd)
        return np.stack(fwd_list, axis=0)

    fwd_pred = _forward_vectors(pred_ucd)
    fwd_gt = _forward_vectors(gt_ucd)

    # Cosine similarity per frame, then average
    cos_sim = np.sum(fwd_pred * fwd_gt, axis=-1)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return float(np.mean(cos_sim))


# ===================================================================
#  3-Panel Visualisation
# ===================================================================

def plot_three_panel(gt_data, uncond_data, infill_data,
                     anchor_frame_mask,  # bool array over *frames* (not tokens)
                     save_path, caption,
                     format_type=None, figsize=(24, 8)):
    """Side-by-side 3D trajectory: GT | Unconstrained | Infilled.

    Anchor frames are highlighted with distinct markers in the Infilled panel.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _draw_panel(ax, data, title, anchor_mask=None):
        ucd = UnifiedCameraData(data, format_type=format_type)
        raw_pos = ucd.positions.numpy()
        ori = ucd.orientations.numpy()
        pos = to_mpl(raw_pos)

        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                "b-", linewidth=1.5, alpha=0.7, label="Path")
        ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2],
                   c="green", s=80, marker="^", label="Start")
        ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2],
                   c="red", s=80, marker="v", label="End")

        # Draw orientation arrows at every 10th frame
        step = max(1, len(pos) // 10)
        extent = max(np.ptp(pos, axis=0)) if len(pos) > 1 else 1.0
        arrow_len = max(0.05 * extent, 0.01)
        for i in range(0, len(pos), step):
            o = ori[i]
            if format_type == CameraDataFormat.FULL_12_ROTMAT:
                fwd_gl = forward_from_sixd(o.reshape(1, 6)).squeeze(0)
            else:
                pitch, yaw = o[0], o[1]
                fwd_gl = np.array([
                    np.cos(pitch) * np.sin(yaw),
                    -np.sin(pitch),
                    -np.cos(pitch) * np.cos(yaw),
                ])
            fwd = to_mpl(fwd_gl.reshape(1, 3)).squeeze(0)
            n = np.linalg.norm(fwd)
            if n > 1e-6:
                fwd /= n
            ax.quiver(pos[i, 0], pos[i, 1], pos[i, 2],
                      fwd[0], fwd[1], fwd[2],
                      length=arrow_len, color="orange", alpha=0.7,
                      arrow_length_ratio=0.3)

        # Highlight anchor frames
        if anchor_mask is not None and np.any(anchor_mask):
            anc_pos = pos[anchor_mask]
            ax.scatter(anc_pos[:, 0], anc_pos[:, 1], anc_pos[:, 2],
                       c="magenta", s=60, marker="D", zorder=5,
                       label="Anchor", edgecolors="black", linewidths=0.5)

        ax.set_xlabel("X (Right)")
        ax.set_ylabel("Depth (Forward)")
        ax.set_zlabel("Y (Up)")
        ax.set_title(title, fontsize=10, pad=12)
        ax.legend(fontsize=7)

    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")

    _draw_panel(ax1, gt_data, "[GT] Ground Truth")
    _draw_panel(ax2, uncond_data, "[Unconstrained] Text Only")
    _draw_panel(ax3, infill_data, "[Infilled] Oracle Anchors",
                anchor_mask=anchor_frame_mask)

    wrapped = "\n".join(textwrap.wrap(caption, width=90))
    fig.suptitle(wrapped, fontsize=11, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def animate_three_panel(gt_data, uncond_data, infill_data,
                        anchor_frame_mask,
                        save_path, caption,
                        format_type=None, fps=20, stride=2,
                        figsize=(24, 8)):
    """Animated 3-panel comparison (MP4)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

    min_len = min(len(gt_data), len(uncond_data), len(infill_data))

    datasets = [
        (gt_data[:min_len], "[GT]", None),
        (uncond_data[:min_len], "[Unconstrained]", None),
        (infill_data[:min_len], "[Infilled]", anchor_frame_mask[:min_len]
         if anchor_frame_mask is not None else None),
    ]

    fig = plt.figure(figsize=(figsize[0], figsize[1] + 1.2))
    # Leave room at top for the caption text
    fig.subplots_adjust(top=0.88)
    axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]

    # Add wrapped caption as suptitle
    wrapped_caption = "\n".join(textwrap.wrap(caption, width=100))
    fig.suptitle(wrapped_caption, fontsize=10, y=0.97,
                 fontstyle="italic", color="0.2")

    # Pre-compute positions, forward vectors
    all_pos = []
    all_fwd = []
    for data, _, _ in datasets:
        ucd = UnifiedCameraData(data, format_type=format_type)
        raw = ucd.positions.numpy()
        ori = ucd.orientations.numpy()
        pos = to_mpl(raw)
        all_pos.append(pos)

        fwds = []
        for i in range(len(pos)):
            o = ori[i]
            if format_type == CameraDataFormat.FULL_12_ROTMAT:
                fgl = forward_from_sixd(o.reshape(1, 6)).squeeze(0)
            else:
                pitch, yaw = o[0], o[1]
                fgl = np.array([
                    np.cos(pitch) * np.sin(yaw),
                    -np.sin(pitch),
                    -np.cos(pitch) * np.cos(yaw),
                ])
            fm = to_mpl(fgl.reshape(1, 3)).squeeze(0)
            n = np.linalg.norm(fm)
            if n > 1e-6:
                fm /= n
            fwds.append(fm)
        all_fwd.append(np.stack(fwds))

    # Compute shared axis limits
    all_pts = np.concatenate(all_pos, axis=0)
    pad = max(np.ptp(all_pts, axis=0)) * 0.15
    lims = [(all_pts[:, d].min() - pad, all_pts[:, d].max() + pad) for d in range(3)]
    extent = max(np.ptp(all_pts, axis=0))
    arrow_len = max(0.05 * extent, 0.01)

    frame_indices = list(range(0, min_len, stride))
    if frame_indices[-1] != min_len - 1:
        frame_indices.append(min_len - 1)

    def animate(frame):
        for idx, (ax, (_, title, anc_mask)) in enumerate(zip(axes, datasets)):
            ax.cla()
            pos = all_pos[idx]
            fwd = all_fwd[idx]

            ax.set_xlim(*lims[0])
            ax.set_ylim(*lims[1])
            ax.set_zlim(*lims[2])

            # Trail
            ax.plot(pos[:frame + 1, 0], pos[:frame + 1, 1], pos[:frame + 1, 2],
                    "b-", linewidth=1.5, alpha=0.6)
            # Current position
            ax.scatter([pos[frame, 0]], [pos[frame, 1]], [pos[frame, 2]],
                       c="red", s=100, zorder=5)
            # Orientation arrow
            ax.quiver(pos[frame, 0], pos[frame, 1], pos[frame, 2],
                      fwd[frame, 0], fwd[frame, 1], fwd[frame, 2],
                      length=arrow_len * 1.5, color="purple", alpha=0.8,
                      arrow_length_ratio=0.3, linewidth=2)
            # Anchors
            if anc_mask is not None:
                visible = anc_mask[:frame + 1]
                if np.any(visible):
                    anc_pos = pos[:frame + 1][visible]
                    ax.scatter(anc_pos[:, 0], anc_pos[:, 1], anc_pos[:, 2],
                               c="magenta", s=50, marker="D", zorder=4,
                               edgecolors="black", linewidths=0.5)

            ax.set_xlabel("X")
            ax.set_ylabel("Depth")
            ax.set_zlabel("Y")
            ax.set_title(f"{title} (frame {frame}/{min_len - 1})", fontsize=9)

    anim = FuncAnimation(fig, animate, frames=frame_indices,
                         interval=1000 / fps, blit=False, repeat=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if save_path.endswith(".gif"):
        writer = PillowWriter(fps=fps)
    else:
        writer = FFMpegWriter(fps=fps, bitrate=1800)
    anim.save(save_path, writer=writer, dpi=100)
    plt.close()
    print(f"  3-panel animation saved to {save_path}")


# ===================================================================
#  Main
# ===================================================================

def add_infilling_args(parser):
    """Add infilling-specific CLI arguments."""
    parser.add_argument("--anchor_mode", type=str, default="head_tail",
                        choices=["head_tail", "sparse"],
                        help="Anchor selection strategy.")
    parser.add_argument("--anchor_n", type=int, default=5,
                        help="For head_tail: number of tokens to fix at each end. "
                             "For sparse: fix every anchor_n-th token.")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of validation samples to evaluate.")
    parser.add_argument("--smooth_sigma", type=float, default=1.0,
                        help="Gaussian smoothing sigma for position post-processing.")
    parser.add_argument("--split", type=str, default="val",
                        help="Which split to load GT samples from (val / test / train).")
    return parser


if __name__ == "__main__":
    # ── Parse options ──────────────────────────────────────────────────
    parser = EvalT2MOptions()
    parser.initialize()
    add_infilling_args(parser.parser)
    opt = parser.parser.parse_args()
    opt.is_train = False
    fixseed(opt.seed)

    opt.device = torch.device(
        "cpu" if opt.gpu_id == -1 else f"cuda:{opt.gpu_id}"
    )

    # ── Dataset config ─────────────────────────────────────────────────
    dataset_config = get_unified_dataset_config(opt)
    dim_pose = dataset_config["dim_pose"]
    viz_format_type = detect_format_from_dataset_name(opt.dataset_name)

    print(f"Dataset: {opt.dataset_name}")
    print(f"Feature dims: {dim_pose}")
    print(f"Viz format: {viz_format_type}")
    print(f"Anchor mode: {opt.anchor_mode}, anchor_n={opt.anchor_n}")
    print(f"Smooth sigma: {opt.smooth_sigma}")

    # ── Output dirs ────────────────────────────────────────────────────
    result_dir = pjoin("./generation", opt.ext)
    viz_dir = pjoin(result_dir, "infilling_viz")
    npy_dir = pjoin(result_dir, "infilling_npy")
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)

    # ── Load models ────────────────────────────────────────────────────
    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_opt_path = pjoin(root_dir, "opt.txt")
    model_opt = get_opt(model_opt_path, device=opt.device)

    # VQ
    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name,
                        model_opt.vq_name, "opt.txt")
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    print(f"VQ dim_pose = {vq_opt.dim_pose}")
    vq_model, vq_opt = load_vq_model(vq_opt)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    # Residual Transformer
    res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name,
                         opt.res_name, "opt.txt")
    res_opt = get_opt(res_opt_path, device=opt.device)
    res_model = load_res_model(res_opt, vq_opt, opt)

    # Mask Transformer
    conditioning_mode = getattr(opt, "conditioning_mode", "clip")
    t2m_transformer = load_trans_model(model_opt, opt, "latest.tar")

    # Move to device & eval
    for m in (t2m_transformer, vq_model, res_model):
        m.eval()
        m.to(opt.device)

    # ── Normalization stats ────────────────────────────────────────────
    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name,
                         model_opt.vq_name, "meta", "mean.npy"))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name,
                        model_opt.vq_name, "meta", "std.npy"))

    def inv_transform(data):
        return data * std + mean

    def fwd_transform(data):
        return (data - mean) / std

    # ── Load GT validation samples ────────────────────────────────────
    data_root = getattr(opt, "data_root", dataset_config.get("data_root", None))
    if data_root is None:
        raise ValueError("Cannot determine data_root; pass --data_root or check dataset config.")

    split_file = pjoin(data_root, f"{opt.split}.txt")
    if not os.path.exists(split_file):
        # Fallback to val.txt
        split_file = pjoin(data_root, "val.txt")
    if not os.path.exists(split_file):
        split_file = pjoin(data_root, "train.txt")
    print(f"Loading split from: {split_file}")

    with open(split_file) as f:
        sample_names = [l.strip() for l in f if l.strip()]

    num_samples = min(opt.num_samples, len(sample_names))
    sample_names = sample_names[:num_samples]

    motion_dir = pjoin(data_root, "new_joint_vecs")
    texts_dir = pjoin(data_root, "texts")

    # Collect GT motions and captions
    gt_motions_raw = []   # un-normalised numpy arrays
    gt_captions = []
    for sname in sample_names:
        m = np.load(pjoin(motion_dir, f"{sname}.npy"))
        gt_motions_raw.append(m)
        txt_path = pjoin(texts_dir, f"{sname}.txt")
        if os.path.exists(txt_path):
            with open(txt_path) as tf:
                raw = tf.readline().strip()
                gt_captions.append(raw.split("#")[0].strip())
        else:
            gt_captions.append(f"Sample {sname}")

    print(f"Loaded {len(gt_motions_raw)} GT samples for infilling evaluation")
    print("=" * 70)

    # ── Per-sample evaluation ──────────────────────────────────────────
    all_metrics = []

    for idx in range(num_samples):
        gt_raw = gt_motions_raw[idx]  # (T, dim_pose)
        caption = gt_captions[idx]
        T_frames = len(gt_raw)
        unit_length = getattr(vq_opt, "unit_length", 4)

        # Trim to multiple of unit_length
        T_frames = (T_frames // unit_length) * unit_length
        if T_frames == 0:
            print(f"[{idx}] Skipping — too short ({len(gt_motions_raw[idx])} frames)")
            continue
        gt_raw = gt_raw[:T_frames]
        T_tokens = T_frames // unit_length

        print(f"\n[{idx}] caption: \"{caption}\"")
        print(f"     frames={T_frames}, tokens={T_tokens}")

        # ── 1. Encode GT to VQ tokens ─────────────────────────────────
        gt_normed = fwd_transform(gt_raw)  # (T, dim)
        gt_tensor = torch.tensor(gt_normed, dtype=torch.float32).unsqueeze(0).to(opt.device)
        gt_codes, _ = vq_model.encode(gt_tensor)  # (1, T_tokens, Q) — all quantizer layers
        gt_base_tokens = gt_codes[:, :, 0]  # (1, T_tokens) — base quantizer only

        # ── 2. Build anchor mask (token-level) ────────────────────────
        if opt.anchor_mode == "head_tail":
            anc_np = build_anchor_mask_head_tail(T_tokens, opt.anchor_n)
        else:
            anc_np = build_anchor_mask_sparse(T_tokens, opt.anchor_n)

        anchor_mask = torch.tensor(anc_np, dtype=torch.bool, device=opt.device).unsqueeze(0)
        num_anchors = int(anc_np.sum())
        print(f"     anchors: {num_anchors}/{T_tokens} tokens "
              f"({100 * num_anchors / T_tokens:.1f}%)")

        # ── 3a. Unconstrained generation (text only) ──────────────────
        token_lens = torch.LongTensor([T_tokens]).to(opt.device)
        with torch.no_grad():
            uncond_mids = t2m_transformer.generate(
                [caption], token_lens,
                timesteps=opt.time_steps,
                cond_scale=opt.cond_scale,
                temperature=opt.temperature,
                topk_filter_thres=opt.topkr,
                gsample=opt.gumbel_sample,
            )
            uncond_mids = res_model.generate(
                uncond_mids, [caption], token_lens,
                temperature=1, cond_scale=5,
            )
            uncond_pred = vq_model.forward_decoder(uncond_mids)
            uncond_pred = inv_transform(uncond_pred.detach().cpu().numpy()[0])

        # ── 3b. Oracle infilling (GT anchors + text) ──────────────────
        with torch.no_grad():
            infill_mids = t2m_transformer.generate_infill(
                [caption], token_lens,
                timesteps=opt.time_steps,
                cond_scale=opt.cond_scale,
                gt_tokens=gt_base_tokens,
                anchor_mask=anchor_mask,
                temperature=opt.temperature,
                topk_filter_thres=opt.topkr,
                gsample=opt.gumbel_sample,
            )
            infill_mids = res_model.generate(
                infill_mids, [caption], token_lens,
                temperature=1, cond_scale=5,
            )
            # ResidualTransformer freely predicts residual codes (layers 1..Q-1)
            # for ALL positions, including anchors.  Overwrite anchor positions
            # with GT codes across ALL quantizer layers so the decoded anchor
            # frames exactly match the ground truth.
            infill_mids[:, anchor_mask[0], :] = gt_codes[:, anchor_mask[0], :]
            infill_pred = vq_model.forward_decoder(infill_mids)
            infill_pred = inv_transform(infill_pred.detach().cpu().numpy()[0])

        # Trim to same length
        L = min(T_frames, len(uncond_pred), len(infill_pred))
        gt_trimmed = gt_raw[:L]
        uncond_trimmed = uncond_pred[:L]
        infill_trimmed = infill_pred[:L]

        # ── 4. Post-processing (smooth + ortho-normalise) ─────────────
        gt_smooth = smooth_positions(gt_trimmed, sigma=opt.smooth_sigma,
                                     format_type=viz_format_type)
        uncond_smooth = smooth_positions(uncond_trimmed, sigma=opt.smooth_sigma,
                                         format_type=viz_format_type)
        infill_smooth = smooth_positions(infill_trimmed, sigma=opt.smooth_sigma,
                                          format_type=viz_format_type)

        # ── 5. Metrics ────────────────────────────────────────────────
        l2_uncond = compute_l2_position_error(uncond_smooth, gt_smooth)
        l2_infill = compute_l2_position_error(infill_smooth, gt_smooth)
        cos_uncond = compute_cosine_rotation_similarity(
            uncond_smooth, gt_smooth, viz_format_type)
        cos_infill = compute_cosine_rotation_similarity(
            infill_smooth, gt_smooth, viz_format_type)

        gain_l2 = l2_uncond - l2_infill
        gain_cos = cos_infill - cos_uncond

        print(f"     L2 pos error   — Uncond: {l2_uncond:.4f}  Infill: {l2_infill:.4f}  "
              f"(gain: {gain_l2:+.4f})")
        print(f"     Cos rot sim    — Uncond: {cos_uncond:.4f}  Infill: {cos_infill:.4f}  "
              f"(gain: {gain_cos:+.4f})")

        all_metrics.append({
            "idx": idx,
            "caption": caption,
            "frames": L,
            "tokens": T_tokens,
            "num_anchors": num_anchors,
            "l2_uncond": l2_uncond,
            "l2_infill": l2_infill,
            "cos_uncond": cos_uncond,
            "cos_infill": cos_infill,
        })

        # ── 6. Anchor mask expanded to frames ─────────────────────────
        # Each token covers unit_length frames
        anchor_frame_mask = np.repeat(anc_np, unit_length)[:L]

        # ── 7. Save visualisations ────────────────────────────────────
        tag = f"sample{idx}_{opt.anchor_mode}_n{opt.anchor_n}"

        # Static 3-panel PNG
        png_path = pjoin(viz_dir, f"{tag}_comparison.png")
        plot_three_panel(gt_smooth, uncond_smooth, infill_smooth,
                         anchor_frame_mask, png_path, caption,
                         format_type=viz_format_type)
        print(f"     Static comparison saved: {png_path}")

        # Animated 3-panel MP4
        mp4_path = pjoin(viz_dir, f"{tag}_comparison.mp4")
        animate_three_panel(gt_smooth, uncond_smooth, infill_smooth,
                            anchor_frame_mask, mp4_path, caption,
                            format_type=viz_format_type, fps=20, stride=2)

        # Save raw numpy
        np.save(pjoin(npy_dir, f"{tag}_gt.npy"), gt_smooth)
        np.save(pjoin(npy_dir, f"{tag}_uncond.npy"), uncond_smooth)
        np.save(pjoin(npy_dir, f"{tag}_infill.npy"), infill_smooth)

    # ── Aggregate metrics ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("AGGREGATE METRICS")
    print("=" * 70)

    if all_metrics:
        avg_l2_u = np.mean([m["l2_uncond"] for m in all_metrics])
        avg_l2_i = np.mean([m["l2_infill"] for m in all_metrics])
        avg_cos_u = np.mean([m["cos_uncond"] for m in all_metrics])
        avg_cos_i = np.mean([m["cos_infill"] for m in all_metrics])

        print(f"  Samples evaluated: {len(all_metrics)}")
        print(f"  Anchor mode: {opt.anchor_mode}, N={opt.anchor_n}")
        print()
        print(f"  Mean L2 Position Error:")
        print(f"    Unconstrained : {avg_l2_u:.4f}")
        print(f"    Infilled      : {avg_l2_i:.4f}")
        print(f"    Gain          : {avg_l2_u - avg_l2_i:+.4f}")
        print()
        print(f"  Mean Cosine Rotation Similarity:")
        print(f"    Unconstrained : {avg_cos_u:.4f}")
        print(f"    Infilled      : {avg_cos_i:.4f}")
        print(f"    Gain          : {avg_cos_i - avg_cos_u:+.4f}")
        print()

        # Save summary to file
        summary_path = pjoin(result_dir, "infilling_metrics.txt")
        with open(summary_path, "w") as sf:
            sf.write("Oracle Infilling Evaluation Summary\n")
            sf.write("=" * 50 + "\n")
            sf.write(f"Dataset: {opt.dataset_name}\n")
            sf.write(f"Model: {opt.name}\n")
            sf.write(f"Anchor mode: {opt.anchor_mode}, N={opt.anchor_n}\n")
            sf.write(f"Smooth sigma: {opt.smooth_sigma}\n")
            sf.write(f"Timesteps: {opt.time_steps}, Cond scale: {opt.cond_scale}\n")
            sf.write(f"Temperature: {opt.temperature}, TopKr: {opt.topkr}\n\n")
            sf.write(f"Samples evaluated: {len(all_metrics)}\n\n")
            sf.write(f"Mean L2 Position Error:\n")
            sf.write(f"  Unconstrained: {avg_l2_u:.6f}\n")
            sf.write(f"  Infilled:      {avg_l2_i:.6f}\n")
            sf.write(f"  Gain:          {avg_l2_u - avg_l2_i:+.6f}\n\n")
            sf.write(f"Mean Cosine Rotation Similarity:\n")
            sf.write(f"  Unconstrained: {avg_cos_u:.6f}\n")
            sf.write(f"  Infilled:      {avg_cos_i:.6f}\n")
            sf.write(f"  Gain:          {avg_cos_i - avg_cos_u:+.6f}\n\n")
            sf.write("-" * 50 + "\n")
            sf.write("Per-sample details:\n")
            for m in all_metrics:
                sf.write(f"\n  [{m['idx']}] \"{m['caption']}\"\n")
                sf.write(f"      frames={m['frames']}, tokens={m['tokens']}, "
                         f"anchors={m['num_anchors']}\n")
                sf.write(f"      L2  uncond={m['l2_uncond']:.6f}  "
                         f"infill={m['l2_infill']:.6f}\n")
                sf.write(f"      Cos uncond={m['cos_uncond']:.6f}  "
                         f"infill={m['cos_infill']:.6f}\n")
        print(f"  Metrics summary saved to: {summary_path}")
    else:
        print("  No samples were evaluated.")

    print("\nDone.")
