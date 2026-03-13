"""
CLaTr-based Camera Evaluation Functions.

Drop-in replacements for the functions in utils/camera_eval.py,
using the pre-trained CLaTr evaluator instead of the legacy
GloVe+BiGRU evaluator.

These functions compute:
  - FID (from trajectory encoder latents)
  - R-Precision (Top-1/2/3)
  - Matching Score (avg cosine similarity)
  - Diversity
  - Camera-specific metrics (position error, orientation error, smoothness)
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from os.path import join as pjoin
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – needed for 3-D projection

from utils.camera_process import calculate_camera_metrics
from models.evaluator.clatr_models import (
    CLaTrEvalWrapper,
    compute_fid,
    compute_r_precision,
    compute_matching_score,
    compute_diversity,
)


# ---------------------------------------------------------------------------
#  Trajectory comparison plotting
# ---------------------------------------------------------------------------

def _positions_from_trajectory(traj: np.ndarray) -> np.ndarray:
    """Extract (T, 3) positions from a trajectory of arbitrary dim."""
    return traj[:, :3]


def _to_mpl(v: np.ndarray) -> np.ndarray:
    """OpenGL [x, y, z] -> Matplotlib Z-up [x, -z, y]."""
    return np.stack([v[..., 0], -v[..., 2], v[..., 1]], axis=-1)


def plot_trajectory_comparison(
    gt_traj: np.ndarray,
    pred_traj: np.ndarray,
    caption: str,
    save_path: str,
    epoch: int = 0,
):
    """Save a multi-panel PNG comparing GT and predicted camera trajectories.

    Panels:
      1. 3-D overlay of GT (blue) vs Pred (red) with start/end markers
      2. Top-down view (X vs Depth)
      3. Per-frame position error
      4. Per-axis position over time

    Parameters
    ----------
    gt_traj   : (T, D) ground-truth trajectory (may be Z-normalised)
    pred_traj : (T, D) predicted trajectory
    caption   : text prompt
    save_path : output .png path
    epoch     : current training epoch (shown in title)
    """
    gt_pos = _to_mpl(_positions_from_trajectory(gt_traj))
    pr_pos = _to_mpl(_positions_from_trajectory(pred_traj))

    fig = plt.figure(figsize=(16, 12))

    # ---- 1. 3-D overlay ----
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(*gt_pos.T, 'b-', lw=2, label='GT')
    ax1.plot(*pr_pos.T, 'r--', lw=2, label='Pred')
    ax1.scatter(*gt_pos[0], c='green', s=80, marker='^', label='Start')
    ax1.scatter(*gt_pos[-1], c='orange', s=80, marker='v', label='End')
    ax1.set_xlabel('X');  ax1.set_ylabel('Depth');  ax1.set_zlabel('Y')
    ax1.set_title('3-D Trajectory')
    ax1.legend(fontsize=8)

    # ---- 2. Top-down ----
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(gt_pos[:, 0], gt_pos[:, 1], 'b-', lw=2, label='GT')
    ax2.plot(pr_pos[:, 0], pr_pos[:, 1], 'r--', lw=2, label='Pred')
    ax2.scatter(gt_pos[0, 0], gt_pos[0, 1], c='green', s=60, marker='^')
    ax2.scatter(gt_pos[-1, 0], gt_pos[-1, 1], c='orange', s=60, marker='v')
    ax2.set_xlabel('X');  ax2.set_ylabel('Depth')
    ax2.set_title('Top-Down (X vs Depth)')
    ax2.legend(fontsize=8);  ax2.grid(True);  ax2.set_aspect('equal', 'datalim')

    # ---- 3. Position error ----
    min_len = min(len(gt_pos), len(pr_pos))
    pos_err = np.linalg.norm(gt_pos[:min_len] - pr_pos[:min_len], axis=1)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(pos_err, 'g-', lw=2)
    ax3.set_xlabel('Frame');  ax3.set_ylabel('L2 Position Error')
    ax3.set_title(f'Position Error  (mean={pos_err.mean():.4f})')
    ax3.grid(True)

    # ---- 4. Per-axis position ----
    ax4 = fig.add_subplot(2, 2, 4)
    frames = np.arange(len(gt_pos))
    for dim, (name, ls) in enumerate(zip(['X', 'Depth', 'Y'], ['-', '--', ':'])):
        ax4.plot(frames, gt_pos[:, dim], ls, color='blue', lw=1.5,
                 label=f'GT {name}')
    frames_p = np.arange(len(pr_pos))
    for dim, (name, ls) in enumerate(zip(['X', 'Depth', 'Y'], ['-', '--', ':'])):
        ax4.plot(frames_p, pr_pos[:, dim], ls, color='red', lw=1.5,
                 label=f'Pred {name}')
    ax4.set_xlabel('Frame');  ax4.set_ylabel('Position')
    ax4.set_title('Per-Axis Position')
    ax4.legend(fontsize=7, ncol=2);  ax4.grid(True)

    # ---- Suptitle ----
    wrapped = '\n'.join(textwrap.wrap(caption, width=90))
    fig.suptitle(f'Epoch {epoch}  |  {wrapped}', fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_trajectory_comparison_animation(
    gt_traj: np.ndarray,
    pred_traj: np.ndarray,
    caption: str,
    save_path: str,
    epoch: int = 0,
    fps: int = 30,
    trail_length: int = 30,
    stride: int = 1,
):
    """Save a side-by-side MP4 animation comparing GT and predicted trajectories.

    Left panel  : Ground Truth (blue path, green trail)
    Right panel : Predicted    (red path, orange trail)
    Both panels share the same axis limits for fair visual comparison.
    The caption is displayed as the figure suptitle.

    Parameters
    ----------
    gt_traj   : (T, D) ground-truth trajectory
    pred_traj : (T, D) predicted trajectory
    caption   : text prompt
    save_path : output .mp4 path
    epoch     : current training epoch
    fps       : frames per second
    trail_length : number of recent frames highlighted
    stride    : render every N-th frame (speeds up rendering)
    """
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    gt_pos = _to_mpl(_positions_from_trajectory(gt_traj))
    pr_pos = _to_mpl(_positions_from_trajectory(pred_traj))

    # Shared axis limits
    all_pos = np.concatenate([gt_pos, pr_pos], axis=0)
    mins = all_pos.min(axis=0)
    maxs = all_pos.max(axis=0)
    padding = max(np.ptp(all_pos, axis=0).max() * 0.12, 1e-3)

    fig = plt.figure(figsize=(18, 8))
    ax_gt = fig.add_subplot(1, 2, 1, projection='3d')
    ax_pr = fig.add_subplot(1, 2, 2, projection='3d')

    for ax, label in [(ax_gt, 'Ground Truth'), (ax_pr, 'Predicted')]:
        ax.set_xlim(mins[0] - padding, maxs[0] + padding)
        ax.set_ylim(mins[1] - padding, maxs[1] + padding)
        ax.set_zlim(mins[2] - padding, maxs[2] + padding)
        ax.set_xlabel('X');  ax.set_ylabel('Depth');  ax.set_zlabel('Y')
        ax.set_title(label, fontsize=13)

    wrapped = '\n'.join(textwrap.wrap(caption, width=100))
    fig.suptitle(f'Epoch {epoch}  |  {wrapped}', fontsize=10, y=0.98)

    # Static elements
    for ax, pos, clr in [(ax_gt, gt_pos, 'blue'), (ax_pr, pr_pos, 'red')]:
        ax.scatter(*pos[0], c='green', s=120, marker='^', zorder=5, label='Start')
        ax.scatter(*pos[-1], c='orange', s=120, marker='v', zorder=5, label='End')
        ax.legend(fontsize=7, loc='upper left')

    # Artists to update each frame
    gt_full, = ax_gt.plot([], [], [], 'b-', lw=1.2, alpha=0.35)
    gt_trail, = ax_gt.plot([], [], [], 'b-', lw=3, alpha=0.85)
    gt_dot = ax_gt.scatter([], [], [], c='blue', s=160, zorder=5)

    pr_full, = ax_pr.plot([], [], [], 'r-', lw=1.2, alpha=0.35)
    pr_trail, = ax_pr.plot([], [], [], 'r-', lw=3, alpha=0.85)
    pr_dot = ax_pr.scatter([], [], [], c='red', s=160, zorder=5)

    max_frames = max(len(gt_pos), len(pr_pos))

    def _update(frame):
        # GT
        f_gt = min(frame, len(gt_pos) - 1)
        gt_full.set_data_3d(gt_pos[:f_gt + 1, 0], gt_pos[:f_gt + 1, 1], gt_pos[:f_gt + 1, 2])
        ts = max(0, f_gt - trail_length)
        seg = gt_pos[ts:f_gt + 1]
        gt_trail.set_data_3d(seg[:, 0], seg[:, 1], seg[:, 2])
        gt_dot._offsets3d = ([gt_pos[f_gt, 0]], [gt_pos[f_gt, 1]], [gt_pos[f_gt, 2]])

        # Pred
        f_pr = min(frame, len(pr_pos) - 1)
        pr_full.set_data_3d(pr_pos[:f_pr + 1, 0], pr_pos[:f_pr + 1, 1], pr_pos[:f_pr + 1, 2])
        ts = max(0, f_pr - trail_length)
        seg = pr_pos[ts:f_pr + 1]
        pr_trail.set_data_3d(seg[:, 0], seg[:, 1], seg[:, 2])
        pr_dot._offsets3d = ([pr_pos[f_pr, 0]], [pr_pos[f_pr, 1]], [pr_pos[f_pr, 2]])

        return gt_full, gt_trail, gt_dot, pr_full, pr_trail, pr_dot

    frame_indices = list(range(0, max_frames, stride))
    if frame_indices[-1] != max_frames - 1:
        frame_indices.append(max_frames - 1)

    anim = FuncAnimation(fig, _update, frames=frame_indices,
                         interval=1000 / fps, blit=False, repeat=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='CamTraj'), bitrate=1800)
        anim.save(save_path, writer=writer, dpi=100)
    except Exception as e:
        # Fallback: save as GIF if ffmpeg is unavailable
        gif_path = save_path.rsplit('.', 1)[0] + '.gif'
        from matplotlib.animation import PillowWriter
        anim.save(gif_path, writer=PillowWriter(fps=fps), dpi=80)
        print(f"[Warning] FFMpeg unavailable, saved GIF instead: {gif_path}")
    plt.close(fig)


#  VQ-VAE Evaluation (replaces evaluation_camera_vqvae for camera datasets)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluation_camera_vqvae_clatr(
    out_dir, val_loader, net, logger, ep,
    best_recon, best_smoothness,
    best_position_error, best_orientation_error,
    eval_wrapper,            # CLaTrEvalWrapper instance
    best_fid=float("inf"), best_div=float("inf"),
    best_top1=0, best_top2=0, best_top3=0,
    best_matching=-float("inf"),
    save=True, draw=True,
):
    """
    Evaluate VQ-VAE on camera trajectories using the CLaTr evaluator.

    Drop-in replacement for ``evaluation_camera_vqvae``; keeps the same
    return signature so all call-sites remain unchanged.

    Returns
    -------
    (best_fid, best_div, best_top1, best_top2, best_top3, best_matching,
     best_recon, best_smoothness, best_position_error, best_orientation_error, logger)
    """
    net.eval()

    # ── Accumulators ──
    all_gt_emb = []
    all_pred_emb = []
    all_text_emb = []

    total_recon_loss = 0.0
    total_position_error = 0.0
    total_orientation_error = 0.0
    total_smoothness = 0.0
    total_velocity_error = 0.0
    nb_sample = 0
    plot_count = 0
    max_plots = 5  # comparison plots per epoch

    for batch in val_loader:
        # Detect batch format
        if isinstance(batch, torch.Tensor):
            motion = batch.cuda()
            bs, seq = motion.shape[0], motion.shape[1]
            m_length = torch.tensor([seq] * bs)
            captions = None
        else:
            word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch[:7]
            motion = motion.cuda()
            bs = motion.shape[0]
            captions = list(caption) if not isinstance(caption, list) else caption

        # VQ forward
        pred_motion, loss_commit, perplexity = net(motion)

        # ── CLaTr embeddings ──
        if captions is not None and isinstance(eval_wrapper, CLaTrEvalWrapper):
            m_lens_tensor = m_length if isinstance(m_length, torch.Tensor) else torch.tensor(m_length)
            m_lens_tensor = m_lens_tensor.cuda()
            gt_emb = eval_wrapper.get_trajectory_embeddings(motion, m_lens_tensor)
            pred_emb = eval_wrapper.get_trajectory_embeddings(pred_motion, m_lens_tensor)
            text_emb = eval_wrapper.get_text_embeddings(captions)
            all_gt_emb.append(gt_emb)
            all_pred_emb.append(pred_emb)
            all_text_emb.append(text_emb)

        # ── Camera-specific metrics ──
        pred_np = pred_motion.detach().cpu().numpy()
        gt_np = motion.detach().cpu().numpy()
        m_len_list = m_length.tolist() if isinstance(m_length, torch.Tensor) else list(m_length)

        for i in range(bs):
            ml = int(m_len_list[i])
            metrics = calculate_camera_metrics(
                pred_np[i : i + 1, :ml], gt_np[i : i + 1, :ml]
            )
            total_position_error += metrics["mean_position_error"]
            total_orientation_error += metrics["mean_orientation_error"]
            total_smoothness += metrics["pred_smoothness"]
            total_velocity_error += metrics["velocity_error"]

        # ── Comparison plots ──
        if draw and captions is not None and plot_count < max_plots:
            plot_dir = pjoin(out_dir, "eval_plots", f"E{ep:04d}_vq")
            gt_np_plot = motion.detach().cpu().numpy()
            pr_np_plot = pred_motion.detach().cpu().numpy()
            m_len_list_plot = m_length.tolist() if isinstance(m_length, torch.Tensor) else list(m_length)
            for i in range(min(bs, max_plots - plot_count)):
                ml = int(m_len_list_plot[i])
                cap_str = captions[i] if i < len(captions) else "N/A"
                plot_trajectory_comparison(
                    gt_np_plot[i, :ml], pr_np_plot[i, :ml], cap_str,
                    pjoin(plot_dir, f"cmp_{plot_count:02d}.png"), epoch=ep,
                )
                try:
                    plot_trajectory_comparison_animation(
                        gt_np_plot[i, :ml], pr_np_plot[i, :ml], cap_str,
                        pjoin(plot_dir, f"cmp_{plot_count:02d}.mp4"),
                        epoch=ep, fps=20, stride=2,
                    )
                except Exception as e:
                    print(f"[Warning] Animation failed for VQ sample {plot_count}: {e}")
                plot_count += 1

        recon_loss = F.l1_loss(pred_motion, motion)
        total_recon_loss += recon_loss.item() * bs
        nb_sample += bs

    # ── Aggregate ──
    avg_recon = total_recon_loss / max(nb_sample, 1)
    avg_pos_err = total_position_error / max(nb_sample, 1)
    avg_ori_err = total_orientation_error / max(nb_sample, 1)
    avg_smooth = total_smoothness / max(nb_sample, 1)
    avg_vel_err = total_velocity_error / max(nb_sample, 1)

    if all_gt_emb:
        gt_emb_all = np.concatenate(all_gt_emb, axis=0)
        pred_emb_all = np.concatenate(all_pred_emb, axis=0)
        text_emb_all = np.concatenate(all_text_emb, axis=0)

        fid = compute_fid(gt_emb_all, pred_emb_all)

        r_prec_runs = []
        div_runs = []
        gt_div_runs = []
        for run_i in range(3):
            r_prec_runs.append(compute_r_precision(
                text_emb_all, pred_emb_all, top_k=3, seed=run_i))
            div_runs.append(compute_diversity(pred_emb_all, seed=run_i))
            gt_div_runs.append(compute_diversity(gt_emb_all, seed=run_i))

        r_prec = np.mean(r_prec_runs, axis=0)
        match_score = compute_matching_score(text_emb_all, pred_emb_all)
        diversity = float(np.mean(div_runs))
        gt_diversity = float(np.mean(gt_div_runs))
    else:
        fid = float("inf")
        r_prec = np.zeros(3)
        match_score = 0.0
        diversity = 0.0
        gt_diversity = 0.0

    # ── Print ──
    msg = (
        f"--> CLaTr Eva. Ep {ep}: FID={fid:.4f}, Div={diversity:.4f}, "
        f"R-Prec=({r_prec[0]:.4f}, {r_prec[1]:.4f}, {r_prec[2]:.4f}), "
        f"Match={match_score:.4f}, Recon={avg_recon:.4f}, "
        f"PosErr={avg_pos_err:.4f}, OriErr={avg_ori_err:.4f}, "
        f"Smooth={avg_smooth:.4f}, VelErr={avg_vel_err:.4f}"
    )
    print(msg)

    # ── Log ──
    if draw:
        logger.add_scalar("./Test/CLaTr_FID", fid, ep)
        logger.add_scalar("./Test/CLaTr_Diversity", diversity, ep)
        logger.add_scalar("./Test/CLaTr_top1", r_prec[0], ep)
        logger.add_scalar("./Test/CLaTr_top2", r_prec[1], ep)
        logger.add_scalar("./Test/CLaTr_top3", r_prec[2], ep)
        logger.add_scalar("./Test/CLaTr_MatchingScore", match_score, ep)
        logger.add_scalar("./Test/Reconstruction_Loss", avg_recon, ep)
        logger.add_scalar("./Test/Position_Error", avg_pos_err, ep)
        logger.add_scalar("./Test/Orientation_Error", avg_ori_err, ep)
        logger.add_scalar("./Test/Smoothness", avg_smooth, ep)
        logger.add_scalar("./Test/Velocity_Error", avg_vel_err, ep)

    # ── Save best ──
    if fid < best_fid:
        print(f"--> --> CLaTr FID Improved from {best_fid:.5f} to {fid:.5f} !!!")
        best_fid = fid
        if save:
            torch.save(
                {"vq_model": net.state_dict(), "ep": ep},
                os.path.join(out_dir, "net_best_fid.tar"),
            )

    if abs(gt_diversity - diversity) < abs(gt_diversity - best_div):
        print(f"--> --> Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!")
        best_div = diversity

    if r_prec[0] > best_top1:
        print(f"--> --> Top1 Improved from {best_top1:.5f} to {r_prec[0]:.5f} !!!")
        best_top1 = r_prec[0]
    if r_prec[1] > best_top2:
        best_top2 = r_prec[1]
    if r_prec[2] > best_top3:
        best_top3 = r_prec[2]

    if match_score > best_matching:
        print(f"--> --> Matching Score Improved from {best_matching:.5f} to {match_score:.5f} !!!")
        best_matching = match_score
        if save:
            torch.save(
                {"vq_model": net.state_dict(), "ep": ep},
                os.path.join(out_dir, "net_best_matching.tar"),
            )

    if avg_recon < best_recon:
        print(f"--> --> Recon Improved from {best_recon:.5f} to {avg_recon:.5f} !!!")
        best_recon = avg_recon
        if save:
            torch.save(
                {"vq_model": net.state_dict(), "ep": ep},
                os.path.join(out_dir, "net_best_recon.tar"),
            )

    if avg_pos_err < best_position_error:
        print(f"--> --> Position Error Improved from {best_position_error:.5f} to {avg_pos_err:.5f} !!!")
        best_position_error = avg_pos_err
        if save:
            torch.save(
                {"vq_model": net.state_dict(), "ep": ep},
                os.path.join(out_dir, "net_best_position.tar"),
            )

    if avg_ori_err < best_orientation_error:
        print(f"--> --> Orientation Error Improved from {best_orientation_error:.5f} to {avg_ori_err:.5f} !!!")
        best_orientation_error = avg_ori_err
        if save:
            torch.save(
                {"vq_model": net.state_dict(), "ep": ep},
                os.path.join(out_dir, "net_best_orientation.tar"),
            )

    if avg_smooth < best_smoothness:
        print(f"--> --> Smoothness Improved from {best_smoothness:.5f} to {avg_smooth:.5f} !!!")
        best_smoothness = avg_smooth
        if save:
            torch.save(
                {"vq_model": net.state_dict(), "ep": ep},
                os.path.join(out_dir, "net_best_smoothness.tar"),
            )

    net.train()
    return (
        best_fid, best_div, best_top1, best_top2, best_top3, best_matching,
        best_recon, best_smoothness, best_position_error, best_orientation_error,
        logger,
    )


# ---------------------------------------------------------------------------
#  MaskTransformer Evaluation (replaces evaluation_mask_transformer for camera)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluation_mask_transformer_clatr(
    out_dir, val_loader, trans, vq_model, logger, ep,
    best_fid, best_div, best_top1, best_top2, best_top3, best_matching,
    eval_wrapper,        # CLaTrEvalWrapper
    plot_func=None, save_ckpt=False, save_anim=False,
    cond_scale=3, temperature=1, topkr=0.9,
    num_repeat: int = 3,
):
    """
    Evaluate mask transformer on camera trajectories using CLaTr metrics.

    Returns
    -------
    (best_fid, best_div, best_top1, best_top2, best_top3, best_matching, logger)
    """
    trans.eval()
    vq_model.eval()

    all_gt_emb = []
    all_gen_emb = []
    all_text_emb = []
    nb_sample = 0
    plot_count = 0
    max_plots = 5

    for batch in val_loader:
        # Detect batch format
        if len(batch) == 3:
            # id_embedding mode: (sample_idx, motion, m_length) — skip text metrics
            _, motion, m_length = batch
            captions = None
        else:
            word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch[:7]
            captions = list(caption) if not isinstance(caption, list) else caption

        motion = motion.cuda()
        bs = motion.shape[0]
        m_lens_tensor = m_length if isinstance(m_length, torch.Tensor) else torch.tensor(m_length)
        m_lens_tensor = m_lens_tensor.cuda()

        # Generate trajectories
        if captions is not None:
            # Text-conditioned generation
            clip_text = captions
            token_lens = torch.ceil(m_lens_tensor / 4).long()

            mids = trans.generate(
                clip_text, token_lens, timesteps=18,
                cond_scale=cond_scale, temperature=temperature,
                topk_filter_thres=topkr,
            )
            # MaskTransformer returns 2D (batch, seq_len); forward_decoder needs 3D (batch, seq_len, nq)
            if mids.dim() == 2:
                mids = mids.unsqueeze(-1)
            # Decode back to trajectory space
            gen_motion = vq_model.forward_decoder(mids)

            # Trim to original length
            max_gen_len = motion.shape[1]
            if gen_motion.shape[1] < max_gen_len:
                pad = torch.zeros(
                    bs, max_gen_len - gen_motion.shape[1], gen_motion.shape[2],
                    device=gen_motion.device,
                )
                gen_motion = torch.cat([gen_motion, pad], dim=1)
            elif gen_motion.shape[1] > max_gen_len:
                gen_motion = gen_motion[:, :max_gen_len]

            # CLaTr embeddings
            if isinstance(eval_wrapper, CLaTrEvalWrapper):
                gt_emb = eval_wrapper.get_trajectory_embeddings(motion, m_lens_tensor)
                gen_emb = eval_wrapper.get_trajectory_embeddings(gen_motion, m_lens_tensor)
                text_emb = eval_wrapper.get_text_embeddings(captions)
                all_gt_emb.append(gt_emb)
                all_gen_emb.append(gen_emb)
                all_text_emb.append(text_emb)

            # ── Comparison plots ──
            if plot_count < max_plots:
                plot_dir = pjoin(out_dir, "eval_plots", f"E{ep:04d}_mtrans")
                gt_np = motion.detach().cpu().numpy()
                gen_np = gen_motion.detach().cpu().numpy()
                m_len_list = m_lens_tensor.cpu().tolist()
                for i in range(min(bs, max_plots - plot_count)):
                    ml = int(m_len_list[i])
                    cap_str = captions[i] if i < len(captions) else "N/A"
                    plot_trajectory_comparison(
                        gt_np[i, :ml], gen_np[i, :ml], cap_str,
                        pjoin(plot_dir, f"cmp_{plot_count:02d}.png"), epoch=ep,
                    )
                    try:
                        plot_trajectory_comparison_animation(
                            gt_np[i, :ml], gen_np[i, :ml], cap_str,
                            pjoin(plot_dir, f"cmp_{plot_count:02d}.mp4"),
                            epoch=ep, fps=20, stride=2,
                        )
                    except Exception as e:
                        print(f"[Warning] Animation failed for MTrans sample {plot_count}: {e}")
                    plot_count += 1

        nb_sample += bs

        # Save animations
        if save_anim and captions is not None and nb_sample <= 20:
            save_dir = pjoin(out_dir, "animations", f"E{ep:04d}")
            os.makedirs(save_dir, exist_ok=True)
            if plot_func is not None:
                data = torch.cat([motion[:4], gen_motion[:4]], dim=0).detach().cpu().numpy()
                plot_func(data, save_dir)

    # ── Compute metrics (averaged over num_repeat runs for stability) ──
    if all_gt_emb:
        gt_emb_all = np.concatenate(all_gt_emb, axis=0)
        gen_emb_all = np.concatenate(all_gen_emb, axis=0)
        text_emb_all = np.concatenate(all_text_emb, axis=0)

        # FID is deterministic (no sampling), compute once
        fid = compute_fid(gt_emb_all, gen_emb_all)

        # R-Precision and Diversity use random sampling — average over
        # multiple seeds for statistical stability.
        r_prec_runs = []
        div_runs = []
        gt_div_runs = []
        for run_i in range(num_repeat):
            r_prec_runs.append(compute_r_precision(
                text_emb_all, gen_emb_all, top_k=3, seed=run_i))
            div_runs.append(compute_diversity(gen_emb_all, seed=run_i))
            gt_div_runs.append(compute_diversity(gt_emb_all, seed=run_i))

        r_prec = np.mean(r_prec_runs, axis=0)
        match_score = compute_matching_score(text_emb_all, gen_emb_all)
        diversity = float(np.mean(div_runs))
        gt_diversity = float(np.mean(gt_div_runs))

    else:
        fid = float("inf")
        r_prec = np.zeros(3)
        match_score = 0.0
        diversity = 0.0
        gt_diversity = 0.0

    msg = (
        f"--> CLaTr MaskTrans Eva. Ep {ep}: FID={fid:.4f}, Div={diversity:.4f}, "
        f"R-Prec=({r_prec[0]:.4f}, {r_prec[1]:.4f}, {r_prec[2]:.4f}), "
        f"Match={match_score:.4f}"
    )
    print(msg)

    logger.add_scalar("./Test/CLaTr_FID", fid, ep)
    logger.add_scalar("./Test/CLaTr_Diversity", diversity, ep)
    logger.add_scalar("./Test/CLaTr_top1", r_prec[0], ep)
    logger.add_scalar("./Test/CLaTr_top2", r_prec[1], ep)
    logger.add_scalar("./Test/CLaTr_top3", r_prec[2], ep)
    logger.add_scalar("./Test/CLaTr_MatchingScore", match_score, ep)

    # ── Track best ──
    if fid < best_fid:
        print(f"--> --> CLaTr FID Improved: {best_fid:.5f} → {fid:.5f}")
        best_fid = fid
        if save_ckpt:
            torch.save(
                {"t2m_transformer": trans.state_dict(), "ep": ep},
                os.path.join(out_dir, "net_best_fid.tar"),
            )

    if abs(gt_diversity - diversity) < abs(gt_diversity - best_div):
        best_div = diversity

    if r_prec[0] > best_top1:
        print(f"--> --> Top1 Improved: {best_top1:.5f} → {r_prec[0]:.5f}")
        best_top1 = r_prec[0]
    if r_prec[1] > best_top2:
        best_top2 = r_prec[1]
    if r_prec[2] > best_top3:
        best_top3 = r_prec[2]

    if match_score > best_matching:
        print(f"--> --> Match Improved: {best_matching:.5f} → {match_score:.5f}")
        best_matching = match_score
        if save_ckpt:
            torch.save(
                {"t2m_transformer": trans.state_dict(), "ep": ep},
                os.path.join(out_dir, "net_best_matching.tar"),
            )

    trans.train()
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, logger


# ---------------------------------------------------------------------------
#  ResidualTransformer Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluation_res_transformer_clatr(
    out_dir, val_loader, trans, vq_model, logger, ep,
    best_fid, best_div, best_top1, best_top2, best_top3, best_matching,
    eval_wrapper,
    plot_func=None, save_ckpt=False, save_anim=False,
    cond_scale=2, temperature=1,
    num_repeat: int = 3,
):
    """
    Evaluate residual transformer using CLaTr metrics.

    Returns
    -------
    (best_fid, best_div, best_top1, best_top2, best_top3, best_matching, logger)
    """
    trans.eval()
    vq_model.eval()

    all_gt_emb = []
    all_gen_emb = []
    all_text_emb = []
    nb_sample = 0
    plot_count = 0
    max_plots = 5

    for batch in val_loader:
        if len(batch) == 3:
            _, motion, m_length = batch
            captions = None
        else:
            word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch[:7]
            captions = list(caption) if not isinstance(caption, list) else caption

        motion = motion.cuda()
        bs = motion.shape[0]
        m_lens_tensor = m_length if isinstance(m_length, torch.Tensor) else torch.tensor(m_length)
        m_lens_tensor = m_lens_tensor.cuda()

        if captions is not None:
            clip_text = captions
            token_lens = torch.ceil(m_lens_tensor / 4).long()

            # Encode GT → get base codes
            code_idx, all_codes = vq_model.encode(motion)

            if ep == 0:
                # At epoch 0, use base layer directly (no residual generation)
                pred_ids = code_idx[..., 0:1]
            else:
                pred_ids = trans.generate(
                    code_idx[..., 0], clip_text, token_lens,
                    cond_scale=cond_scale, temperature=temperature,
                )

            gen_motion = vq_model.forward_decoder(pred_ids)

            max_gen_len = motion.shape[1]
            if gen_motion.shape[1] < max_gen_len:
                pad = torch.zeros(
                    bs, max_gen_len - gen_motion.shape[1], gen_motion.shape[2],
                    device=gen_motion.device,
                )
                gen_motion = torch.cat([gen_motion, pad], dim=1)
            elif gen_motion.shape[1] > max_gen_len:
                gen_motion = gen_motion[:, :max_gen_len]

            if isinstance(eval_wrapper, CLaTrEvalWrapper):
                gt_emb = eval_wrapper.get_trajectory_embeddings(motion, m_lens_tensor)
                gen_emb = eval_wrapper.get_trajectory_embeddings(gen_motion, m_lens_tensor)
                text_emb = eval_wrapper.get_text_embeddings(captions)
                all_gt_emb.append(gt_emb)
                all_gen_emb.append(gen_emb)
                all_text_emb.append(text_emb)

            # ── Comparison plots ──
            if plot_count < max_plots:
                plot_dir = pjoin(out_dir, "eval_plots", f"E{ep:04d}_rtrans")
                gt_np = motion.detach().cpu().numpy()
                gen_np = gen_motion.detach().cpu().numpy()
                m_len_list = m_lens_tensor.cpu().tolist()
                for i in range(min(bs, max_plots - plot_count)):
                    ml = int(m_len_list[i])
                    cap_str = captions[i] if i < len(captions) else "N/A"
                    plot_trajectory_comparison(
                        gt_np[i, :ml], gen_np[i, :ml], cap_str,
                        pjoin(plot_dir, f"cmp_{plot_count:02d}.png"), epoch=ep,
                    )
                    try:
                        plot_trajectory_comparison_animation(
                            gt_np[i, :ml], gen_np[i, :ml], cap_str,
                            pjoin(plot_dir, f"cmp_{plot_count:02d}.mp4"),
                            epoch=ep, fps=20, stride=2,
                        )
                    except Exception as e:
                        print(f"[Warning] Animation failed for RTrans sample {plot_count}: {e}")
                    plot_count += 1

        nb_sample += bs

    # ── Compute metrics (averaged over num_repeat seeds for stability) ──
    if all_gt_emb:
        gt_emb_all = np.concatenate(all_gt_emb, axis=0)
        gen_emb_all = np.concatenate(all_gen_emb, axis=0)
        text_emb_all = np.concatenate(all_text_emb, axis=0)

        fid = compute_fid(gt_emb_all, gen_emb_all)

        r_prec_runs = []
        div_runs = []
        gt_div_runs = []
        for run_i in range(num_repeat):
            r_prec_runs.append(compute_r_precision(
                text_emb_all, gen_emb_all, top_k=3, seed=run_i))
            div_runs.append(compute_diversity(gen_emb_all, seed=run_i))
            gt_div_runs.append(compute_diversity(gt_emb_all, seed=run_i))

        r_prec = np.mean(r_prec_runs, axis=0)
        match_score = compute_matching_score(text_emb_all, gen_emb_all)
        diversity = float(np.mean(div_runs))
        gt_diversity = float(np.mean(gt_div_runs))
    else:
        fid = float("inf")
        r_prec = np.zeros(3)
        match_score = 0.0
        diversity = 0.0
        gt_diversity = 0.0

    msg = (
        f"--> CLaTr ResTrans Eva. Ep {ep}: FID={fid:.4f}, Div={diversity:.4f}, "
        f"R-Prec=({r_prec[0]:.4f}, {r_prec[1]:.4f}, {r_prec[2]:.4f}), "
        f"Match={match_score:.4f}"
    )
    print(msg)

    logger.add_scalar("./Test/CLaTr_FID", fid, ep)
    logger.add_scalar("./Test/CLaTr_Diversity", diversity, ep)
    logger.add_scalar("./Test/CLaTr_top1", r_prec[0], ep)
    logger.add_scalar("./Test/CLaTr_top2", r_prec[1], ep)
    logger.add_scalar("./Test/CLaTr_top3", r_prec[2], ep)
    logger.add_scalar("./Test/CLaTr_MatchingScore", match_score, ep)

    if fid < best_fid:
        print(f"--> --> CLaTr FID Improved: {best_fid:.5f} → {fid:.5f}")
        best_fid = fid
        if save_ckpt:
            torch.save(
                {"res_transformer": trans.state_dict(), "ep": ep},
                os.path.join(out_dir, "net_best_fid.tar"),
            )

    if abs(gt_diversity - diversity) < abs(gt_diversity - best_div):
        best_div = diversity

    if r_prec[0] > best_top1:
        best_top1 = r_prec[0]
    if r_prec[1] > best_top2:
        best_top2 = r_prec[1]
    if r_prec[2] > best_top3:
        best_top3 = r_prec[2]

    if match_score > best_matching:
        print(f"--> --> Match Improved: {best_matching:.5f} → {match_score:.5f}")
        best_matching = match_score
        if save_ckpt:
            torch.save(
                {"res_transformer": trans.state_dict(), "ep": ep},
                os.path.join(out_dir, "net_best_matching.tar"),
            )

    trans.train()
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, logger
