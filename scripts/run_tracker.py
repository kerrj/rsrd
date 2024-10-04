"""
Script for running the tracker.
"""

import pickle
import trimesh
from typing import cast
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import torch
import tqdm
import warp as wp

import viser
import viser.transforms as vtf
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
from toad.extras.cameras import (
    CameraIntr,
    IPhoneIntr,
    get_ns_camera_at_origin,
    get_vid_frame,
)
from toad.extras.viser_rsrd import ViserRSRD

torch.set_float32_matmul_precision("high")


@dataclass
class RSRDTrackerConfig:
    """Tracking config to run RSRD."""

    dig_config_path: Path
    video_path: Path
    output_dir: Path

    camera_type: CameraIntr = IPhoneIntr()
    optimizer_config: RigidGroupOptimizerConfig = RigidGroupOptimizerConfig()


def main(exp: RSRDTrackerConfig):
    """Track objects in video using RSRD."""
    exp.output_dir.mkdir(parents=True, exist_ok=True)

    # Load video.
    assert exp.video_path.exists(), f"Video path {exp.video_path} does not exist."
    video = cv2.VideoCapture(str(exp.video_path.absolute()))

    # Load DIG model, create viewer.
    train_config, pipeline, _, _ = eval_setup(exp.dig_config_path)
    viewer_lock = Lock()
    Viewer(
        ViewerConfig(default_composite_depth=False, num_rays_per_chunk=-1),
        exp.dig_config_path.parent,
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
    optimizer = RigidGroupOptimizer(
        exp.optimizer_config,
        pipeline,
        render_lock=viewer_lock,
    )
    logger.info("Initialized tracker.")

    # Generate + load keyframes.
    keyframe_path = exp.output_dir / "keyframes.pt"
    hand_path = exp.output_dir / "hands.pkl"
    if not keyframe_path.exists():
        generate_keyframes(optimizer, video, exp.camera_type, keyframe_path, hand_path)

    optimizer.load_deltas(keyframe_path)
    with hand_path.open("rb") as f:
        hands = cast(dict[int, list[trimesh.Trimesh]], pickle.load(f))

    server = viser.ViserServer()
    viser_rsrd = ViserRSRD(server, optimizer, base_frame_name="/object")

    timesteps = len(optimizer.obj_delta)
    track_slider = server.gui.add_slider("timestep", 0, timesteps - 1, 1, 0)
    play_checkbox = server.gui.add_checkbox("play", False)
    hands_handle = None

    while True:
        if play_checkbox.value:
            track_slider.value = (track_slider.value + 1) % timesteps

        tstep = track_slider.value
        obj_delta = optimizer.obj_delta[tstep]
        part_deltas = optimizer.part_deltas[tstep]

        viser_rsrd.update_cfg(obj_delta, part_deltas)

        # TODO(cmk) Remvoe magic np.pi/4 here -- related to `get_ns_camera_at_origin`.
        server.scene.add_frame("/cam", wxyz=vtf.SO3.from_x_radians(np.pi / 4).wxyz)
        if hands.get(tstep, None) is not None:
            hands_handle = server.scene.add_mesh_trimesh(
                "/cam/hands", sum(hands[tstep], trimesh.Trimesh())
            )
        elif hands_handle is not None:
            hands_handle.visible = False

        time.sleep(0.05)


# But this should _really_ be in the rigid optimizer.
def generate_keyframes(
    optimizer: RigidGroupOptimizer,
    motion_clip: cv2.VideoCapture,
    camera_type: CameraIntr,
    keyframe_path: Path,
    hand_path: Path,
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
    import moviepy.editor as mpy

    out_clip = mpy.ImageSequenceClip(renders, fps=30)
    out_clip.write_videofile(str("test_camopt.mp4"))

    # Add each frame, optimize them separately.
    for frame_id in tqdm.trange(0, num_frames):
        timestamp = frame_id / fps
        rgb = get_vid_frame(motion_clip, timestamp)
        obs = optimizer.create_observation_from_rgb_and_camera(rgb, camera)
        optimizer.add_observation(obs)
        optimizer.fit([-1], 50)

    # Get 3D hands.
    hands_dict = {}
    for frame_id in tqdm.trange(0, num_frames, desc="Detecting hands"):
        with torch.no_grad():
            optimizer.apply_keyframe(frame_id)
            curr_obs = optimizer.sequence[frame_id]
            outputs = cast(
                dict[str, torch.Tensor],
                optimizer.dig_model.get_outputs(curr_obs.frame.camera),
            )
            rendered_scaled_depth = outputs["depth"] / optimizer.dataset_scale
            object_mask = outputs["accumulation"] > optimizer.config.mask_threshold

            # Hands in camera frame.
            hands_3d = curr_obs.frame.get_hand_3d(object_mask, rendered_scaled_depth)

            for hand in hands_3d:
                hand.vertices *= optimizer.dataset_scale  # Re-apply nerfstudio scaling.
                hand.apply_transform(
                    vtf.SE3.from_rotation(vtf.SO3.from_x_radians(np.pi)).as_matrix()
                )  # OpenCV -> OpenGL (ns).

            if len(hands_3d) > 0:
                hands_dict[frame_id] = hands_3d

    # Smooth all frames, together.
    logger.warning("Skipping all-frame smoothing optimization.")

    # Save part trajectories.
    optimizer.save_deltas(keyframe_path)

    # Save the hands.
    with hand_path.open("wb") as f:
        pickle.dump(hands_dict, f)


if __name__ == "__main__":
    # Must be called before any other warp API call.
    wp.init()
    tyro.cli(main)
