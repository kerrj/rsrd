"""
Script for running the tracker.
"""

import datetime
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import torch
torch.set_float32_matmul_precision('high')
import tqdm
import warp as wp

import viser
import tyro
from loguru import logger

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.utils import writer
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.viewer.viewer import Viewer

from toad.optimization.rigid_group_optimizer import (
    RigidGroupOptimizer,
    RigidGroupOptimizerConfig,
)
from toad.cameras import CameraIntr, IPhoneIntr, get_ns_camera_at_origin
from toad.articulated_gaussians import ViserRSRD


@dataclass
class RSRDTrackerConfig:
    """Tracking config to run RSRD."""
    dig_config_path: Path
    video_path: Path
    base_output_folder: Path

    camera_type: CameraIntr = IPhoneIntr()
    optimizer_config: RigidGroupOptimizerConfig = RigidGroupOptimizerConfig()

    detect_hands: bool = False
    save_viser: bool = False
    terminate_on_done: bool = False


def main(exp: RSRDTrackerConfig):
    """Track objects in video using RSRD."""
    exp.base_output_folder.mkdir(parents=True, exist_ok=True)

    # Load video.
    assert exp.video_path.exists(), f"Video path {exp.video_path} does not exist."
    video = cv2.VideoCapture(str(exp.video_path.absolute()))

    # Load DIG model, create viewer.
    train_config, pipeline, _, _ = eval_setup(exp.dig_config_path)
    viewer_lock = Lock()
    viewer = Viewer(
        ViewerConfig(default_composite_depth=False, num_rays_per_chunk=-1),
        exp.dig_config_path.parent,
        pipeline.datamanager.get_datapath(),
        pipeline,
        train_lock=viewer_lock,
    )
    # Need to set up the writer to track number of rays, otherwise the viewer will not calculate the resolution correctly.
    writer.setup_local_writer(train_config.logging, max_iter=train_config.max_num_iterations)

    # TODO(cmk) add `reset_colors` to GARField main branch.
    try:
        pipeline.load_state()
        pipeline.reset_colors()
    except FileNotFoundError:
        print("No state found, starting from scratch")

    # Initialize tracker.
    optimizer = RigidGroupOptimizer(
        exp.optimizer_config,
        pipeline,
        render_lock=viewer_lock,
    )
    logger.info("Initialized tracker.")

    # Load (and generate if necessary) keyframes.
    keyframe_path = exp.base_output_folder / "keyframes.pt"
    if not keyframe_path.exists():
        generate_keyframes(optimizer, video, exp.camera_type, keyframe_path)
    optimizer.load_deltas(keyframe_path)

    server = viser.ViserServer()
    viser_rsrd = ViserRSRD(server, optimizer)

    timesteps = len(optimizer.obj_delta)
    track_slider = server.gui.add_slider("timestep", 0, timesteps - 1, 1, 0)

    while True:
        tstep = track_slider.value
        obj_delta = optimizer.obj_delta[tstep]
        part_deltas = optimizer.part_deltas[tstep]

        viser_rsrd.update_cfg(obj_delta, part_deltas)
        time.sleep(0.05)
        track_slider.value = (tstep + 1) % timesteps


# But this should _really_ be in the rigid optimizer.
def generate_keyframes(
    optimizer: RigidGroupOptimizer,
    motion_clip: cv2.VideoCapture,
    camera_type: CameraIntr,
    keyframe_path: Path,
):
    """Get part poses for each frame in the video, ad save the keyframes to a file. """
    camera = get_ns_camera_at_origin(camera_type)
    num_frames = int(motion_clip.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = motion_clip.get(cv2.CAP_PROP_FPS)

    rgb = get_vid_frame(motion_clip, 0)
    frame = optimizer.create_observation_from_rgb_and_camera(rgb, camera)

    # Initialize.
    renders = optimizer.initialize_obj_pose(frame, render=True, niter=100, n_seeds=5)

    # Save the frames.
    import moviepy.editor as mpy
    out_clip = mpy.ImageSequenceClip(renders, fps=30)
    out_clip.write_videofile(str("test_camopt.mp4"))

    # Add each frame, optimize them separately.
    for frame_id in tqdm.trange(0, num_frames):
        timestamp = frame_id / fps
        rgb = get_vid_frame(motion_clip, timestamp)
        frame = optimizer.create_observation_from_rgb_and_camera(rgb, camera)
        optimizer.add_observation(frame)
        optimizer.fit([-1], 50)

    # Smooth all frames, together.
    # optimizer.optimize_frames(all_frames=True, niter=50)
    logger.warning("Skipping all-frame optimization.")

    # save part trajectories
    optimizer.save_deltas(keyframe_path)


def get_vid_frame(cap: cv2.VideoCapture, timestamp: float) -> np.ndarray:
    """Get frame from video at timestamp (in seconds)."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = min(int(timestamp * fps), int(cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = cap.read()
    if not success:
        raise ValueError(f"Failed to read frame at {timestamp} s.")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


if __name__ == "__main__":
    # Must be called before any other warp API call.
    wp.init()
    tyro.cli(main)
