"""
Ablation Study Visualization: 2×2 Panel Comparison of RVQ + Transformer Contributions

Generates a side-by-side (2×2 grid) animated comparison of four trajectory paths:

  ┌──────────────────────────┬──────────────────────────┐
  │  Path A: GT Base-Only    │  Path B: Full GT Recon   │
  │  (Level-0 GT tokens)     │  (All GT RVQ levels)     │
  ├──────────────────────────┼──────────────────────────┤
  │  Path C: Pred Base-Only  │  Path D: Full Model Pred │
  │  (MTransformer only)     │  (MTrans + RTrans)       │
  └──────────────────────────┴──────────────────────────┘

Path A: Theoretical upper bound of the MTransformer — GT tokens, only Level 0.
Path B: RVQ reconstruction ceiling — GT tokens, all quantizer levels.
Path C: MTransformer predicted base token only (Level 0), decoded through RVQ.
Path D: Standard full-model inference (MTransformer base + RTransformer residuals).

All panels share axis limits, camera viewing angle, and OpenGL coordinate convention.
Optional 1D Gaussian smoothing (--smooth / --sigma) applied uniformly to all paths.
L2 position error (vs Path B) overlaid on Path A and Path C panels.

Usage:
    python vis_ablation_comparison.py \\
        --dataset_name realestate10k_rotmat \\
        --name mtrans_3k_clip_crossattn_reduce \\
        --res_name rtrans_3k_clip_crossattn_reduce \\
        --conditioning_mode clip \\
        --gpu_id 0 \\
        --text_path camera_prompts.txt \\
        --ext ablation_study \\
        --time_steps 10 --cond_scale 3 --temperature 0.3 --topkr 0.9 \\
        --smooth --sigma 1.0 \\
        --repeat_times 1
"""

import os
import sys
import textwrap
from os.path import join as pjoin

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.ndimage import gaussian_filter1d

from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from models.vq.model import RVQVAE, LengthEstimator

from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt
from utils.fixseed import fixseed
from utils.unified_data_format import (
    UnifiedCameraData,
    CameraDataFormat,
    detect_format_from_dataset_name,
)
from utils.camera_geometry import (
    sixd_to_matrix,
    forward_from_sixd,
    to_mpl,
    matrix_to_sixd,
)

clip_version = 'ViT-B/32'

# ──────────────────────────────────────────────────────────────────────────────
# Model loaders (reused from gen_camera.py)
# ──────────────────────────────────────────────────────────────────────────────

def load_vq_model(vq_opt):
    vq_model = RVQVAE(
        vq_opt, vq_opt.dim_pose, vq_opt.nb_code, vq_opt.code_dim,
        vq_opt.output_emb_width, vq_opt.down_t, vq_opt.stride_t,
        vq_opt.width, vq_opt.depth, vq_opt.dilation_growth_rate,
        vq_opt.vq_act, vq_opt.vq_norm,
    )
    is_camera = any(n in vq_opt.dataset_name.lower() for n in ("cam", "estate", "realestate"))
    if is_camera:
        ckpt_files = ['net_best_recon.tar', 'net_best_position.tar',
                      'net_best_smoothness.tar', 'latest.tar']
        loaded = False
        for cf in ckpt_files:
            path = pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', cf)
            if os.path.exists(path):
                ckpt = torch.load(path, map_location='cpu')
                key = 'vq_model' if 'vq_model' in ckpt else 'net'
                vq_model.load_state_dict(ckpt[key])
                print(f'Loading VQ Model {vq_opt.name} from {cf}')
                loaded = True
                break
        if not loaded:
            raise FileNotFoundError(f"No VQ checkpoint found for {vq_opt.name}")
    else:
        ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name,
                                'model', 'net_best_fid.tar'), map_location='cpu')
        key = 'vq_model' if 'vq_model' in ckpt else 'net'
        vq_model.load_state_dict(ckpt[key])
        print(f'Loading VQ Model {vq_opt.name} Completed!')
    return vq_model, vq_opt


def load_trans_model(model_opt, opt, which_model):
    conditioning_mode = getattr(opt, 'conditioning_mode', 'clip')
    t2m_transformer = MaskTransformer(
        code_dim=model_opt.code_dim, cond_mode='text',
        latent_dim=model_opt.latent_dim, ff_size=model_opt.ff_size,
        num_layers=model_opt.n_layers, num_heads=model_opt.n_heads,
        dropout=model_opt.dropout, clip_dim=512,
        cond_drop_prob=model_opt.cond_drop_prob, clip_version=clip_version,
        conditioning_mode=conditioning_mode,
        num_id_samples=getattr(opt, 'num_id_samples', 50),
        t5_model_name=getattr(opt, 't5_model_name', 't5-base'),
        opt=model_opt,
    )
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name,
                            model_opt.name, 'model', which_model), map_location='cpu')
    key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    missing, unexpected = t2m_transformer.load_state_dict(ckpt[key], strict=False)
    assert len(unexpected) == 0
    assert all(k.startswith('clip_model.') or k.startswith('cond_provider.') for k in missing)
    print(f'Loading Transformer {opt.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer


def load_res_model(res_opt, vq_opt, opt):
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens = vq_opt.nb_code
    conditioning_mode = getattr(opt, 'conditioning_mode', 'clip')
    res_transformer = ResidualTransformer(
        code_dim=vq_opt.code_dim, cond_mode='text',
        latent_dim=res_opt.latent_dim, ff_size=res_opt.ff_size,
        num_layers=res_opt.n_layers, num_heads=res_opt.n_heads,
        dropout=res_opt.dropout, clip_dim=512,
        shared_codebook=vq_opt.shared_codebook,
        cond_drop_prob=res_opt.cond_drop_prob,
        share_weight=res_opt.share_weight, clip_version=clip_version,
        conditioning_mode=conditioning_mode,
        num_id_samples=getattr(opt, 'num_id_samples', 50),
        t5_model_name=getattr(opt, 't5_model_name', 't5-base'),
        opt=res_opt,
    )
    is_camera = any(n in res_opt.dataset_name.lower() for n in ("cam", "estate", "realestate"))
    if is_camera:
        ckpt_files = ['net_best_acc.tar', 'net_best_loss.tar', 'latest.tar']
        loaded = False
        for cf in ckpt_files:
            path = pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model', cf)
            if os.path.exists(path):
                ckpt = torch.load(path, map_location=opt.device)
                loaded = True
                print(f'Loading Residual Transformer {res_opt.name} from {cf}')
                break
        if not loaded:
            raise FileNotFoundError(f"No Res-Transformer checkpoint found for {res_opt.name}")
    else:
        ckpt = torch.load(pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name,
                                'model', 'net_best_fid.tar'), map_location=opt.device)
    missing, unexpected = res_transformer.load_state_dict(ckpt['res_transformer'], strict=False)
    assert len(unexpected) == 0
    assert all(k.startswith('clip_model.') or k.startswith('cond_provider.') for k in missing)
    print(f'Loading Residual Transformer {res_opt.name} from epoch {ckpt["ep"]}!')
    return res_transformer


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def apply_gram_schmidt_to_rot6d(data: np.ndarray, format_type: CameraDataFormat) -> np.ndarray:
    """Apply Gram-Schmidt orthonormalization to the 6D rotation columns.

    For FULL_12_ROTMAT format, rot6d lives in columns [6:12].
    For other formats, just return as-is (no raw rot6d columns).
    """
    if format_type != CameraDataFormat.FULL_12_ROTMAT:
        return data
    data = data.copy()
    rot6d = data[:, 6:12]
    R = sixd_to_matrix(rot6d)          # (..., 3, 3)
    clean_6d = matrix_to_sixd(R)       # (..., 6) — already ortho-normalised
    data[:, 6:12] = clean_6d
    return data


def smooth_trajectory(data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply 1-D Gaussian filter along the time axis to each feature."""
    smoothed = np.empty_like(data)
    for d in range(data.shape[1]):
        smoothed[:, d] = gaussian_filter1d(data[:, d], sigma=sigma)
    return smoothed


def compute_l2_position_error(pred: np.ndarray, ref: np.ndarray) -> float:
    """Mean L2 position error (first 3 dims) between pred and ref."""
    min_len = min(len(pred), len(ref))
    diff = pred[:min_len, :3] - ref[:min_len, :3]
    return float(np.mean(np.linalg.norm(diff, axis=1)))


def _orientation_vector(orientations_row, format_type):
    """Compute a forward direction vector from one frame's orientation data."""
    if format_type == CameraDataFormat.FULL_12_ROTMAT:
        rot6d = orientations_row[:6]
        fwd = forward_from_sixd(rot6d)
        return fwd
    elif format_type == CameraDataFormat.FULL_12_EULER:
        pitch, yaw, roll = orientations_row[:3]
        fwd = np.array([
            -np.sin(yaw) * np.cos(pitch),
            np.sin(pitch),
            -np.cos(yaw) * np.cos(pitch),
        ])
        return fwd / (np.linalg.norm(fwd) + 1e-8)
    elif format_type == CameraDataFormat.POSITION_ORIENTATION_6:
        pitch, yaw, roll = orientations_row[:3]
        fwd = np.array([
            -np.sin(yaw) * np.cos(pitch),
            np.sin(pitch),
            -np.cos(yaw) * np.cos(pitch),
        ])
        return fwd / (np.linalg.norm(fwd) + 1e-8)
    elif format_type == CameraDataFormat.LEGACY_5:
        pitch, yaw = orientations_row[:2]
        fwd = np.array([
            -np.sin(yaw) * np.cos(pitch),
            np.sin(pitch),
            -np.cos(yaw) * np.cos(pitch),
        ])
        return fwd / (np.linalg.norm(fwd) + 1e-8)
    else:
        return np.array([0, 0, -1.0])


# ──────────────────────────────────────────────────────────────────────────────
# 2×2 Panel Animated Visualization
# ──────────────────────────────────────────────────────────────────────────────

def render_ablation_comparison(
    paths: dict,           # {'A': (N,D), 'B': (N,D), 'C': (N,D), 'D': (N,D)}
    save_path: str,
    text_prompt: str = "",
    format_type: CameraDataFormat = None,
    fps: int = 30,
    stride: int = 1,
    figsize: tuple = (16, 14),
    arrow_scale_factor: float = 0.06,
    show_trail: bool = True,
    trail_length: int = 30,
):
    """Render a 2×2 animated comparison of four trajectory paths.

    Layout:
        top-left:  Path A (GT Base Only)       top-right: Path B (Full GT Recon)
        bot-left:  Path C (Pred Base Only)      bot-right: Path D (Full Model Pred)
    """
    labels = {
        'A': 'Path A: GT Base-Only (L0)',
        'B': 'Path B: Full GT Recon (All Levels)',
        'C': 'Path C: Pred Base-Only (MTransformer)',
        'D': 'Path D: Full Model Prediction',
    }
    panel_positions = {'A': 1, 'B': 2, 'C': 3, 'D': 4}

    # Parse all paths through UnifiedCameraData → positions + orientations
    parsed = {}
    for key, raw in paths.items():
        ud = UnifiedCameraData(raw, format_type=format_type)
        raw_pos = ud.positions.numpy()
        orientations = ud.orientations.numpy()
        mpl_pos = to_mpl(raw_pos)
        parsed[key] = {
            'positions': mpl_pos,
            'raw_positions': raw_pos,
            'orientations': orientations,
            'format_type': ud.format_type,
        }

    # Compute shared axis limits across all four paths
    all_positions = np.concatenate([p['positions'] for p in parsed.values()], axis=0)
    pos_min = all_positions.min(axis=0)
    pos_max = all_positions.max(axis=0)
    extent = np.max(pos_max - pos_min)
    padding = max(extent * 0.12, 0.05)
    xlim = (pos_min[0] - padding, pos_max[0] + padding)
    ylim = (pos_min[1] - padding, pos_max[1] + padding)
    zlim = (pos_min[2] - padding, pos_max[2] + padding)

    # Arrow scale
    arrow_len = max(arrow_scale_factor * extent, 0.01)
    arrow_len = min(arrow_len, 0.25)

    # Pre-compute orientation vectors for all paths
    for key in parsed:
        vecs = []
        p = parsed[key]
        for i in range(len(p['positions'])):
            fwd_gl = _orientation_vector(p['orientations'][i], p['format_type'])
            fwd_mpl = to_mpl(fwd_gl.reshape(1, 3)).flatten()
            vecs.append(fwd_mpl)
        p['fwd_vectors'] = np.array(vecs)

    # Compute L2 errors relative to Path B
    ref_positions = parsed['B']['raw_positions']
    errors = {}
    for key in ('A', 'C'):
        errors[key] = compute_l2_position_error(
            np.column_stack([parsed[key]['raw_positions']]),  # just positions from raw
            np.column_stack([ref_positions]),
        )
    # Recompute properly (the above was wrong — just use raw positions directly)
    for key in ('A', 'C'):
        pred_pos = parsed[key]['raw_positions']
        ref_pos = parsed['B']['raw_positions']
        min_len = min(len(pred_pos), len(ref_pos))
        diff = pred_pos[:min_len] - ref_pos[:min_len]
        errors[key] = float(np.mean(np.linalg.norm(diff, axis=1)))

    # Determine max frames across all paths
    max_frames = max(len(p['positions']) for p in parsed.values())

    # Build figure
    fig = plt.figure(figsize=figsize)

    # Wrap prompt
    wrap_width = 90
    wrapped_prompt = '\n'.join(textwrap.wrap(text_prompt, width=wrap_width)) if text_prompt else ''
    fig.suptitle(wrapped_prompt, fontsize=11, y=0.98, va='top')

    axes = {}
    artists = {}
    for key in ('A', 'B', 'C', 'D'):
        ax = fig.add_subplot(2, 2, panel_positions[key], projection='3d')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_xlabel('X (Right)', fontsize=8)
        ax.set_ylabel('Depth (Forward)', fontsize=8)
        ax.set_zlabel('Y (Up)', fontsize=8)
        ax.view_init(elev=25, azim=-60)

        title_text = labels[key]
        if key in errors:
            title_text += f'\nL2 pos error vs B: {errors[key]:.4f}'
        ax.set_title(title_text, fontsize=9, pad=8)

        pos = parsed[key]['positions']

        # Full trajectory (light)
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], '-', color='cornflowerblue',
                linewidth=1.2, alpha=0.4, label='Full Path')
        # Start / End markers
        ax.scatter(*pos[0], c='green', s=80, marker='^', label='Start', zorder=5)
        ax.scatter(*pos[-1], c='red', s=80, marker='v', label='End', zorder=5)

        # Animated artists
        trail_line, = ax.plot([], [], [], '-', color='orange', linewidth=2.5, alpha=0.8)
        current_pt = ax.scatter([], [], [], c='red', s=120, zorder=10)

        ax.legend(fontsize=6, loc='upper right')
        axes[key] = ax
        artists[key] = {'trail': trail_line, 'point': current_pt, 'arrow': None}

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    def animate(frame_idx):
        changed = []
        for key in ('A', 'B', 'C', 'D'):
            pos = parsed[key]['positions']
            fwd = parsed[key]['fwd_vectors']
            ax = axes[key]
            n = len(pos)
            # Clamp frame index
            fi = min(frame_idx, n - 1)

            # Update trail
            t_start = max(0, fi - trail_length) if show_trail else fi
            trail_pos = pos[t_start:fi + 1]
            artists[key]['trail'].set_data(trail_pos[:, 0], trail_pos[:, 1])
            artists[key]['trail'].set_3d_properties(trail_pos[:, 2])

            # Update current point
            artists[key]['point']._offsets3d = ([pos[fi, 0]], [pos[fi, 1]], [pos[fi, 2]])

            # Remove previous arrow
            if artists[key]['arrow'] is not None:
                try:
                    artists[key]['arrow'].remove()
                except Exception:
                    pass

            # Draw orientation arrow
            artists[key]['arrow'] = ax.quiver(
                pos[fi, 0], pos[fi, 1], pos[fi, 2],
                fwd[fi, 0] * arrow_len, fwd[fi, 1] * arrow_len, fwd[fi, 2] * arrow_len,
                color='red', arrow_length_ratio=0.3, linewidth=1.5,
            )

            changed.extend([artists[key]['trail'], artists[key]['point']])
        return changed

    # Build frame list with stride
    frame_indices = list(range(0, max_frames, stride))
    if frame_indices[-1] != max_frames - 1:
        frame_indices.append(max_frames - 1)

    interval = 1000 / fps
    anim = FuncAnimation(fig, animate, frames=frame_indices,
                         interval=interval, blit=False, repeat=True)

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    if save_path.endswith('.gif'):
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
    elif save_path.endswith('.mp4'):
        writer = FFMpegWriter(fps=fps, codec='libx264', bitrate=2000)
        anim.save(save_path, writer=writer)
    else:
        save_path += '.mp4'
        writer = FFMpegWriter(fps=fps, codec='libx264', bitrate=2000)
        anim.save(save_path, writer=writer)

    plt.close(fig)
    print(f"Ablation comparison animation saved to {save_path}")


def render_ablation_static(
    paths: dict,
    save_path: str,
    text_prompt: str = "",
    format_type: CameraDataFormat = None,
    arrow_scale_factor: float = 0.06,
    figsize: tuple = (16, 14),
):
    """Render a static 2×2 comparison plot (PNG)."""
    labels = {
        'A': 'Path A: GT Base-Only (L0)',
        'B': 'Path B: Full GT Recon (All Levels)',
        'C': 'Path C: Pred Base-Only (MTransformer)',
        'D': 'Path D: Full Model Prediction',
    }
    panel_positions = {'A': 1, 'B': 2, 'C': 3, 'D': 4}

    parsed = {}
    for key, raw in paths.items():
        ud = UnifiedCameraData(raw, format_type=format_type)
        raw_pos = ud.positions.numpy()
        orientations = ud.orientations.numpy()
        mpl_pos = to_mpl(raw_pos)
        parsed[key] = {
            'positions': mpl_pos,
            'raw_positions': raw_pos,
            'orientations': orientations,
            'format_type': ud.format_type,
        }

    # Pre-compute orientation vectors
    for key in parsed:
        vecs = []
        p = parsed[key]
        for i in range(len(p['positions'])):
            fwd_gl = _orientation_vector(p['orientations'][i], p['format_type'])
            fwd_mpl = to_mpl(fwd_gl.reshape(1, 3)).flatten()
            vecs.append(fwd_mpl)
        p['fwd_vectors'] = np.array(vecs)

    # Shared limits
    all_positions = np.concatenate([p['positions'] for p in parsed.values()], axis=0)
    pos_min = all_positions.min(axis=0)
    pos_max = all_positions.max(axis=0)
    extent = np.max(pos_max - pos_min)
    padding = max(extent * 0.12, 0.05)
    xlim = (pos_min[0] - padding, pos_max[0] + padding)
    ylim = (pos_min[1] - padding, pos_max[1] + padding)
    zlim = (pos_min[2] - padding, pos_max[2] + padding)

    arrow_len = max(arrow_scale_factor * extent, 0.01)
    arrow_len = min(arrow_len, 0.25)

    # L2 errors
    ref_pos = parsed['B']['raw_positions']
    errors = {}
    for key in ('A', 'C'):
        pred_pos = parsed[key]['raw_positions']
        min_len = min(len(pred_pos), len(ref_pos))
        diff = pred_pos[:min_len] - ref_pos[:min_len]
        errors[key] = float(np.mean(np.linalg.norm(diff, axis=1)))

    fig = plt.figure(figsize=figsize)
    wrapped_prompt = '\n'.join(textwrap.wrap(text_prompt, width=90)) if text_prompt else ''
    fig.suptitle(wrapped_prompt, fontsize=11, y=0.98, va='top')

    for key in ('A', 'B', 'C', 'D'):
        ax = fig.add_subplot(2, 2, panel_positions[key], projection='3d')
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
        ax.set_xlabel('X (Right)', fontsize=8)
        ax.set_ylabel('Depth (Forward)', fontsize=8)
        ax.set_zlabel('Y (Up)', fontsize=8)
        ax.view_init(elev=25, azim=-60)

        title_text = labels[key]
        if key in errors:
            title_text += f'\nL2 pos error vs B: {errors[key]:.4f}'
        ax.set_title(title_text, fontsize=9, pad=8)

        pos = parsed[key]['positions']
        fwd = parsed[key]['fwd_vectors']

        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b-', linewidth=2, alpha=0.7, label='Path')
        ax.scatter(*pos[0], c='green', s=100, marker='^', label='Start')
        ax.scatter(*pos[-1], c='red', s=100, marker='v', label='End')

        # Orientation arrows at sampled frames
        step = max(1, len(pos) // 10)
        for i in range(0, len(pos), step):
            ax.quiver(pos[i, 0], pos[i, 1], pos[i, 2],
                      fwd[i, 0] * arrow_len, fwd[i, 1] * arrow_len, fwd[i, 2] * arrow_len,
                      color='red', arrow_length_ratio=0.3, linewidth=1.0)

        ax.legend(fontsize=6, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Ablation comparison static plot saved to {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI & Main
# ──────────────────────────────────────────────────────────────────────────────

def add_ablation_args(parser):
    """Add ablation-specific CLI arguments to an existing EvalT2MOptions parser."""
    parser.add_argument('--smooth', action='store_true',
                        help='Apply 1D Gaussian smoothing to all paths before rendering')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='Sigma for Gaussian smoothing (default: 1.0)')
    parser.add_argument('--stride', type=int, default=1,
                        help='Render every Nth frame for animation (stride=2 halves render time)')
    parser.add_argument('--anim_fps', type=int, default=30,
                        help='FPS for the output animation')
    parser.add_argument('--output_format', type=str, default='mp4', choices=['mp4', 'gif'],
                        help='Output animation format')
    parser.add_argument('--no_static', action='store_true',
                        help='Skip static PNG output')
    parser.add_argument('--no_animation', action='store_true',
                        help='Skip animated MP4/GIF output')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to draw from the dataset split (default: 10)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Which dataset split to draw GT samples from (default: test)')
    return parser


if __name__ == '__main__':
    # ── Parse arguments ──
    base_parser = EvalT2MOptions()
    add_ablation_args(base_parser.parser)
    opt = base_parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else f"cuda:{opt.gpu_id}")
    torch.autograd.set_detect_anomaly(True)

    # Format detection
    viz_format_type = detect_format_from_dataset_name(opt.dataset_name)
    print(f"Dataset: {opt.dataset_name}")
    print(f"Visualization format: {viz_format_type}")

    # ── Directories ──
    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    result_dir = pjoin('./generation', opt.ext)
    os.makedirs(result_dir, exist_ok=True)

    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)

    # ── Load RVQ ──
    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    print(f'VQ model dim_pose = {vq_opt.dim_pose}')
    vq_model, vq_opt = load_vq_model(vq_opt)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    # ── Load R-Transformer ──
    res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, 'opt.txt')
    res_opt = get_opt(res_opt_path, device=opt.device)
    res_model = load_res_model(res_opt, vq_opt, opt)
    assert res_opt.vq_name == model_opt.vq_name

    # ── Load M-Transformer ──
    t2m_transformer = load_trans_model(model_opt, opt, 'latest.tar')

    t2m_transformer.eval()
    vq_model.eval()
    res_model.eval()
    res_model.to(opt.device)
    t2m_transformer.to(opt.device)
    vq_model.to(opt.device)

    # ── Normalisation stats ──
    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'std.npy'))

    def inv_transform(data):
        return data * std + mean

    # ── Resolve data_root ──
    _data_root_defaults = {
        't2m': './dataset/HumanML3D/',
        'kit': './dataset/KIT-ML/',
        'cam': './dataset/CameraTraj/',
        'realestate10k_6': './dataset/RealEstate10K_6feat/',
        'realestate10k_12': './dataset/RealEstate10K_12feat/',
        'realestate10k_quat': './dataset/RealEstate10K_quat/',
        'realestate10k_rotmat': './dataset/RealEstate10K_rotmat/',
    }
    if opt.data_root:
        data_root = opt.data_root
    elif hasattr(model_opt, 'data_root') and model_opt.data_root:
        data_root = model_opt.data_root
    else:
        data_root = _data_root_defaults.get(opt.dataset_name, f'./dataset/{opt.dataset_name}/')
    print(f"Data root: {data_root}")

    # ── Load samples directly from dataset split ──
    # This ensures every sample has GT data for Path A & B
    conditioning_mode = getattr(opt, 'conditioning_mode', 'clip')
    motion_dir = pjoin(data_root, 'new_joint_vecs')
    texts_dir = pjoin(data_root, 'texts')

    split_file = pjoin(data_root, f'{opt.split}.txt')
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with open(split_file, 'r') as f:
        split_sample_names = [line.strip() for line in f if line.strip()]

    # Limit to --num_samples
    num_samples = min(opt.num_samples, len(split_sample_names))
    split_sample_names = split_sample_names[:num_samples]

    # Load GT motion + text prompt for each sample
    gt_sample_names = []
    gt_motions = []
    captions = []
    for sample_name in split_sample_names:
        gt_path = pjoin(motion_dir, f"{sample_name}.npy")
        text_path = pjoin(texts_dir, f"{sample_name}.txt")
        if not os.path.exists(gt_path):
            print(f"  Skipping {sample_name}: GT motion not found at {gt_path}")
            continue
        gt_motion = np.load(gt_path)
        # Read text prompt
        if os.path.exists(text_path):
            with open(text_path, 'r') as tf:
                raw_text = tf.readline().strip()
                caption = raw_text.split('#')[0].strip()
        else:
            caption = f"Sample {sample_name}"
        gt_sample_names.append(sample_name)
        gt_motions.append(gt_motion)
        captions.append(caption)

    if len(captions) == 0:
        raise RuntimeError(f"No valid samples found in {split_file}. "
                           f"Check that {motion_dir} and {texts_dir} contain matching files.")

    print(f"\nLoaded {len(captions)} samples from {opt.split} split")
    print(f"Processing {len(captions)} samples × {opt.repeat_times} repeats\n")

    # Build token lengths from GT motion lengths or default
    if opt.motion_length > 0:
        token_lens = torch.LongTensor([opt.motion_length // 4] * len(captions)).to(opt.device)
    else:
        # Use actual GT lengths (rounded to unit_length)
        raw_lengths = [len(m) for m in gt_motions]
        # Round each length down to nearest multiple of 4 (unit_length)
        rounded_lengths = [(l // 4) for l in raw_lengths]
        token_lens = torch.LongTensor(rounded_lengths).to(opt.device)
    m_length = token_lens * 4

    # Set generation conditions based on conditioning mode
    if conditioning_mode == 'id_embedding':
        # For id_embedding, map sample names to train-split indices
        train_split_file = pjoin(data_root, 'train.txt')
        if os.path.exists(train_split_file):
            with open(train_split_file, 'r') as f:
                train_names = [l.strip() for l in f if l.strip()]
            name_to_id = {n: i for i, n in enumerate(train_names)}
        else:
            name_to_id = {}
        sample_id_list = [name_to_id.get(n, 0) for n in gt_sample_names]
        gen_conds = torch.LongTensor(sample_id_list).to(opt.device)
    else:
        gen_conds = captions

    # ── Main generation loop ──
    for r in range(opt.repeat_times):
        print(f"{'='*70}")
        print(f"  Repeat {r}")
        print(f"{'='*70}")

        with torch.no_grad():
            # ─────────────────────────────────────────────────────────
            # Path C: MTransformer predicted base tokens only (Level 0)
            # ─────────────────────────────────────────────────────────
            mids_base = t2m_transformer.generate(
                gen_conds, token_lens,
                timesteps=opt.time_steps,
                cond_scale=opt.cond_scale,
                temperature=opt.temperature,
                topk_filter_thres=opt.topkr,
                gsample=opt.gumbel_sample,
            )
            # mids_base shape: (B, N) — just the base tokens (level 0)
            # Decode through RVQ with only level 0
            mids_base_only = mids_base.unsqueeze(-1)  # (B, N, 1)
            pred_base_only = vq_model.forward_decoder(mids_base_only)
            pred_base_only = pred_base_only.detach().cpu().numpy()
            path_C_all = inv_transform(pred_base_only)

            # ─────────────────────────────────────────────────────────
            # Path D: Full model prediction (MTransformer + RTransformer)
            # ─────────────────────────────────────────────────────────
            mids_full = res_model.generate(mids_base, gen_conds, token_lens,
                                           temperature=1, cond_scale=5)
            pred_full = vq_model.forward_decoder(mids_full)
            pred_full = pred_full.detach().cpu().numpy()
            path_D_all = inv_transform(pred_full)

        # Process each sample
        for k, caption in enumerate(captions):
            length = m_length[k].item()
            sample_name = gt_sample_names[k]
            gt_motion = gt_motions[k]
            print(f"\n---->Sample {k} [{sample_name}]: \"{caption}\" (length={length}, GT frames={len(gt_motion)})")

            # Encode GT through RVQ to get code indices
            gt_tensor = torch.tensor(gt_motion, dtype=torch.float32).unsqueeze(0)  # (1, T, D)
            # Normalize GT data (stored unnormalized — raw features)
            gt_normalized = (gt_tensor.numpy() - mean) / std
            gt_tensor = torch.tensor(gt_normalized, dtype=torch.float32).to(opt.device)

            with torch.no_grad():
                gt_code_idx, gt_all_codes = vq_model.encode(gt_tensor)
                # gt_code_idx: (1, T', Q), where T' = T // stride

                # ── Path A: GT Base-Only (Level 0 only) ──
                gt_base_only_idx = gt_code_idx[:, :, 0:1]   # (1, T', 1)
                path_A_recon = vq_model.forward_decoder(gt_base_only_idx)
                path_A_recon = path_A_recon.detach().cpu().numpy()
                path_A = inv_transform(path_A_recon)[0]

                # ── Path B: Full GT Reconstruction (all levels) ──
                path_B_recon = vq_model.forward_decoder(gt_code_idx)
                path_B_recon = path_B_recon.detach().cpu().numpy()
                path_B = inv_transform(path_B_recon)[0]

            path_C = path_C_all[k][:length]
            path_D = path_D_all[k][:length]

            # Trim all paths to the same length for fair comparison
            min_len = min(len(path_A), len(path_B), len(path_C), len(path_D))
            path_A = path_A[:min_len]
            path_B = path_B[:min_len]
            path_C = path_C[:min_len]
            path_D = path_D[:min_len]

            # Apply Gram-Schmidt orthonormalization to 6D rotation components
            path_A = apply_gram_schmidt_to_rot6d(path_A, viz_format_type)
            path_B = apply_gram_schmidt_to_rot6d(path_B, viz_format_type)
            path_C = apply_gram_schmidt_to_rot6d(path_C, viz_format_type)
            path_D = apply_gram_schmidt_to_rot6d(path_D, viz_format_type)

            # Optional Gaussian smoothing
            if opt.smooth:
                sigma = opt.sigma
                print(f"  Applying Gaussian smoothing (sigma={sigma})")
                path_A = smooth_trajectory(path_A, sigma=sigma)
                path_B = smooth_trajectory(path_B, sigma=sigma)
                path_C = smooth_trajectory(path_C, sigma=sigma)
                path_D = smooth_trajectory(path_D, sigma=sigma)
                # Re-apply Gram-Schmidt after smoothing to keep rotations valid
                path_A = apply_gram_schmidt_to_rot6d(path_A, viz_format_type)
                path_B = apply_gram_schmidt_to_rot6d(path_B, viz_format_type)
                path_C = apply_gram_schmidt_to_rot6d(path_C, viz_format_type)
                path_D = apply_gram_schmidt_to_rot6d(path_D, viz_format_type)

            paths_dict = {'A': path_A, 'B': path_B, 'C': path_C, 'D': path_D}

            # ── Output directory ──
            sample_dir = pjoin(result_dir, str(k))
            os.makedirs(sample_dir, exist_ok=True)

            # ── Save raw .npy for each path ──
            for pkey, pdata in paths_dict.items():
                np.save(pjoin(sample_dir, f"sample{k}_repeat{r}_path{pkey}.npy"), pdata)

            # ── Render static PNG ──
            if not opt.no_static:
                static_path = pjoin(sample_dir,
                                    f"sample{k}_repeat{r}_len{min_len}_ablation.png")
                render_ablation_static(
                    paths_dict, static_path, text_prompt=caption,
                    format_type=viz_format_type,
                )

            # ── Render animation ──
            if not opt.no_animation:
                ext = getattr(opt, 'output_format', 'mp4')
                anim_path = pjoin(sample_dir,
                                  f"sample{k}_repeat{r}_len{min_len}_ablation.{ext}")
                render_ablation_comparison(
                    paths_dict, anim_path, text_prompt=caption,
                    format_type=viz_format_type,
                    fps=opt.anim_fps,
                    stride=opt.stride,
                )

            # ── Print L2 errors ──
            ref_pos = paths_dict['B'][:, :3]
            for pkey in ('A', 'C', 'D'):
                pred_pos = paths_dict[pkey][:, :3]
                l2 = float(np.mean(np.linalg.norm(pred_pos - ref_pos, axis=1)))
                print(f"  L2 pos error (Path {pkey} vs B): {l2:.6f}")

            print(f"  Outputs saved to {sample_dir}")

    print(f"\n{'='*70}")
    print(f"Ablation comparison complete. Results in {result_dir}")
    print(f"{'='*70}")
