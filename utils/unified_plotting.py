"""
Unified plotting factory for transformer training scripts.

Returns a single plot function that handles both camera and human-motion
datasets, auto-detecting the appropriate visualisation.
"""

from os.path import join as pjoin

import torch

from .unified_data_format import CameraDataFormat, detect_format_from_dataset_name


def create_plotting_function_for_transformer(dataset_name: str):
    """Return a plot function suitable for transformer training callbacks.

    For camera datasets the function generates animated MP4s via
    ``gen_camera.plot_camera_trajectory_animation``.
    For human-motion datasets it falls back to the standard skeleton renderer.

    Args:
        dataset_name: Dataset identifier (e.g. ``realestate10k_rotmat``).
    """

    is_camera = any(
        tag in dataset_name.lower() for tag in ("cam", "estate", "realestate")
    )
    fmt = detect_format_from_dataset_name(dataset_name) if is_camera else None

    def plot_function(data, save_dir, captions, m_lengths, **kwargs):
        fps = kwargs.get("fps", 30)
        radius = kwargs.get("radius", 4)

        if is_camera:
            from gen_camera import plot_camera_trajectory_animation

            for i, (caption, seq) in enumerate(zip(captions, data)):
                seq = seq[: m_lengths[i]]
                save_path = pjoin(save_dir, "%02d.mp4" % i)
                try:
                    plot_camera_trajectory_animation(
                        data=seq,
                        save_path=save_path,
                        title=caption,
                        fps=fps,
                        show_trail=True,
                        trail_length=30,
                        format_type=fmt,
                    )
                except Exception as e:
                    print(f"Warning: camera plot failed for sample {i}: {e}")
        else:
            from utils.motion_process import recover_from_ric
            from utils.plot_script import plot_3d_motion

            joints_num = kwargs.get("joints_num", 22)
            kinematic_chain = kwargs.get("kinematic_chain", None)
            for i, (caption, seq) in enumerate(zip(captions, data)):
                seq = seq[: m_lengths[i]]
                joint = recover_from_ric(
                    torch.from_numpy(seq).float(), joints_num
                ).numpy()
                save_path = pjoin(save_dir, "%02d.mp4" % i)
                plot_3d_motion(
                    save_path, kinematic_chain, joint,
                    title=caption, fps=fps, radius=radius,
                )

    return plot_function
