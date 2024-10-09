"""
Script for running the tracker.
"""

import trimesh
from typing import cast
import time
from pathlib import Path
from threading import Lock
import moviepy.editor as mpy

import cv2
import numpy as np
import torch
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
from toad.optimization.atap_loss import ATAPConfig
from toad.extras.cam_helpers import (
    CameraIntr,
    IPhoneIntr,
    get_ns_camera_at_origin,
    get_vid_frame,
)
from toad.extras.viser_rsrd import ViserRSRD

torch.set_float32_matmul_precision("high")


def main(
    is_obj_jointed: bool,
    dig_config_path: Path,
    video_path: Path,
    output_dir: Path,
    camera_type: CameraIntr = IPhoneIntr(),
    save_hand: bool = True,
):
    """Track objects in video using RSRD."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load video.
    assert video_path.exists(), f"Video path {video_path} does not exist."
    video = cv2.VideoCapture(str(video_path.absolute()))

    # Load DIG model, create viewer.
    train_config, pipeline, _, _ = eval_setup(dig_config_path)
    viewer_lock = Lock()
    Viewer(
        ViewerConfig(default_composite_depth=False, num_rays_per_chunk=-1),
        dig_config_path.parent,
        pipeline.datamanager.get_datapath(),
        pipeline,
        train_lock=viewer_lock,
    )
    # Need to set up the writer to track number of rays, otherwise the viewer will not calculate the resolution correctly.
    writer.setup_local_writer(
        train_config.logging, max_iter=train_config.max_num_iterations
    )

    # TODO(cmk) add `reset_colors` to GARField main branch.
    try:
        pipeline.load_state()
        pipeline.reset_colors()
    except FileNotFoundError:
        print("No state found, starting from scratch")

    # Initialize tracker.
    wp.init()  # Must be called before any other warp API call.
    optimizer_config = RigidGroupOptimizerConfig(
        atap_config=ATAPConfig(
            loss_alpha=(1.0 if is_obj_jointed else 0.1),
        )
    )
    optimizer = RigidGroupOptimizer(
        optimizer_config,
        pipeline,
        render_lock=viewer_lock,
    )
    logger.info("Initialized tracker.")

    # Generate + load keyframes.
    camopt_render_path = output_dir / "camopt_render.mp4"
    track_data_path = output_dir / "keyframes.txt"
    if not track_data_path.exists():
        track_and_save_motion(
            optimizer,
            video,
            camera_type,
            camopt_render_path,
            track_data_path,
            save_hand,
        )

    optimizer.load_tracks(track_data_path)
    hands = optimizer.hands_info
    assert hands is not None

    server = viser.ViserServer()
    viser_rsrd = ViserRSRD(server, optimizer, base_frame_name="/object")

    timesteps = len(optimizer.obj_delta)
    track_slider = server.gui.add_slider("timestep", 0, timesteps - 1, 1, 0)
    play_checkbox = server.gui.add_checkbox("play", True)
    hands_handle = None

    while True:
        if play_checkbox.value:
            track_slider.value = (track_slider.value + 1) % timesteps

        tstep = track_slider.value
        obj_delta = optimizer.obj_delta[tstep]
        part_deltas = optimizer.part_deltas[tstep]

        viser_rsrd.update_cfg(obj_delta, part_deltas)

        if hands.get(tstep, None) is not None:
            left_hand, right_hand = hands[tstep]
            hand_meshes = []
            if left_hand is not None:
                for idx in range(left_hand["verts"].shape[0]):
                    hand_meshes.append(
                        trimesh.Trimesh(
                            vertices=left_hand["verts"][idx],
                            faces=left_hand["faces"].astype(np.int32),
                        )
                    )

            if right_hand is not None:
                for idx in range(right_hand["verts"].shape[0]):
                    hand_meshes.append(
                        trimesh.Trimesh(
                            vertices=right_hand["verts"][idx],
                            faces=right_hand["faces"].astype(np.int32),
                        )
                    )

            hands_handle = server.scene.add_mesh_trimesh(
                "/hands", sum(hand_meshes, trimesh.Trimesh())
            )
        elif hands_handle is not None:
            hands_handle.visible = False

        time.sleep(0.05)


# But this should _really_ be in the rigid optimizer.
def track_and_save_motion(
    optimizer: RigidGroupOptimizer,
    motion_clip: cv2.VideoCapture,
    camera_type: CameraIntr,
    camopt_render_path: Path,
    track_data_path: Path,
    save_hand: bool = False,
):
    """Get part poses for each frame in the video, ad save the keyframes to a file."""
    camera = get_ns_camera_at_origin(camera_type)
    num_frames = int(motion_clip.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = motion_clip.get(cv2.CAP_PROP_FPS)

    rgb = get_vid_frame(motion_clip, 0)
    obs = optimizer.create_observation_from_rgb_and_camera(rgb, camera)

    # Initialize.
    renders = optimizer.initialize_obj_pose(obs, render=True, niter=100, n_seeds=5)

    # Save the frames.
    out_clip = mpy.ImageSequenceClip(renders, fps=30)
    out_clip.write_videofile(str(camopt_render_path))

    # Add each frame, optimize them separately.
    for frame_id in tqdm.trange(0, num_frames):
        timestamp = frame_id / fps
        rgb = get_vid_frame(motion_clip, timestamp)
        obs = optimizer.create_observation_from_rgb_and_camera(rgb, camera)
        optimizer.add_observation(obs)
        optimizer.fit([-1], 50)

    # Get 3D hands.
    hands_dict = {}
    if save_hand:
        for frame_id in tqdm.trange(0, num_frames, desc="Detecting hands"):
            with torch.no_grad():
                optimizer.apply_keyframe(frame_id)
                curr_obs = optimizer.sequence[frame_id]
                outputs = cast(
                    dict[str, torch.Tensor],
                    optimizer.dig_model.get_outputs(curr_obs.frame.camera),
                )
                object_mask = outputs["accumulation"] > optimizer.config.mask_threshold

                # Hands in camera frame.
                left_hand, right_hand = curr_obs.frame.get_hand_3d(
                    object_mask, outputs["depth"], optimizer.dataset_scale
                )
                hands_dict[frame_id] = (left_hand, right_hand)

    # Smooth all frames, together.
    logger.warning("Skipping all-frame smoothing optimization.")

    # Save part trajectories.
    optimizer.save_tracks(track_data_path, hands_dict)


if __name__ == "__main__":
    tyro.cli(main)
