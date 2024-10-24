"""
Run object-centric robot motion planning. (robot do! haha).
"""

from threading import Lock
from typing import Literal
import jax

import jaxlie
import time
from pathlib import Path
import moviepy.editor as mpy

import cv2
import numpy as np
import jax.numpy as jnp
import torch
import warp as wp

import viser
import viser.extras
import viser.transforms as vtf
import tyro
from loguru import logger
import yourdfpy

from nerfstudio.utils.eval_utils import eval_setup

from rsrd.extras.cam_helpers import get_ns_camera_at_origin
from rsrd.extras.zed import Zed
from rsrd.motion.motion_optimizer import (
    RigidGroupOptimizer,
    RigidGroupOptimizerConfig,
)
from rsrd.motion.atap_loss import ATAPConfig
from rsrd.extras.viser_rsrd import ViserRSRD
from rsrd.robot.rsrd_motion_planner import PartMotionPlanner
import rsrd.transforms as tf
from autolab_core import RigidTransform
from jaxmp.extras.urdf_loader import load_urdf

# # set jax device to cpu
# jax.config.update("jax_platform_name", "cpu")

torch.set_float32_matmul_precision("high")


def main(
    hand_mode: Literal["single", "bimanual"],
    dig_config_path: Path,
    track_data_path: Path,
    zed_video_path: Path,
    save_hand: bool = True,
):
    # Load DIG model, create viewer.
    _, pipeline, _, _ = eval_setup(dig_config_path)
    try:
        pipeline.load_state()
        pipeline.reset_colors()
    except FileNotFoundError:
        print("No state found, starting from scratch")

    # Initialize tracker.
    wp.init()  # Must be called before any other warp API call.
    is_obj_jointed = False  # Unused anyway, for registration.
    optimizer_config = RigidGroupOptimizerConfig(
        atap_config=ATAPConfig(
            loss_alpha=(1.0 if is_obj_jointed else 0.1),
        ),
        altitude_down=0.0,
    )
    optimizer = RigidGroupOptimizer(
        optimizer_config,
        pipeline,
    )
    logger.info("Initialized tracker.")

    # Load keyframes.
    optimizer.load_tracks(track_data_path)
    hands = optimizer.hands_info
    assert hands is not None

    # Optimize, based on zed video.
    zed = Zed(str(zed_video_path))
    left, _, depth = zed.get_frame(depth=True)
    assert left is not None and depth is not None
    left = (left.cpu().numpy() * 255).astype(np.uint8)

    # Load zed position.
    T_cam_world = vtf.SE3.from_matrix(
        RigidTransform.load(
            Path(__file__).parent / "../data/zed/zed_to_world.tf"
        ).matrix
    )

    # Load URDF.
    urdf_path = Path(__file__).parent / "../data/yumi_description/urdf/yumi.urdf"
    def filename_handler(fname: str) -> str:
        base_path = urdf_path.parent
        return yourdfpy.filename_handler_magic(fname, dir=base_path)
    urdf = yourdfpy.URDF.load(urdf_path, filename_handler=filename_handler)

    # Optimize object pose.
    camera = get_ns_camera_at_origin(K=zed.get_K(), width=zed.width, height=zed.height)
    first_obs = optimizer.create_observation_from_rgb_and_camera(
        left, camera, metric_depth=depth.cpu().numpy()
    )
    renders = optimizer.initialize_obj_pose(first_obs, render=True, use_depth=True)
    _renders = []
    for r in renders:
        _left = cv2.resize(left, (r.shape[1], r.shape[0]))
        _renders.append(r * 0.8 + _left * 0.2)
    out_clip = mpy.ImageSequenceClip(_renders, fps=30)
    out_clip.write_videofile("foo.mp4")

    T_obj_cam = optimizer.T_objreg_world
    # Convert opengl -> opencv
    T_obj_cam = (
        tf.SE3.from_rotation(tf.SO3.from_x_radians(torch.Tensor([torch.pi]).cuda()))
        @ T_obj_cam
    )
    # Put to world scale.
    T_obj_cam = tf.SE3.from_rotation_and_translation(
        rotation=T_obj_cam.rotation(),
        translation=T_obj_cam.translation() / optimizer.dataset_scale,
    )

    # Optimize robot trajectory.
    T_obj_world = jaxlie.SE3(jnp.array(T_cam_world.wxyz_xyz)) @ jaxlie.SE3(
        jnp.array(T_obj_cam.wxyz_xyz.detach().cpu())
    )
    part_motion_planner = PartMotionPlanner(optimizer, urdf)
    
    # Visualize.
    server = viser.ViserServer()
    server.scene.add_frame(
        "/camera",
        position=T_cam_world.translation(),
        wxyz=T_cam_world.rotation().wxyz,
        show_axes=False,
    )
    server.scene.add_mesh_trimesh("/camera/mesh", zed.zed_mesh)

    server.scene.add_frame(
        "/camera/object",
        position=T_obj_cam.translation().squeeze().cpu().numpy(),
        wxyz=T_obj_cam.rotation().wxyz.squeeze().cpu().numpy(),
        show_axes=False,
    )

    viser_rsrd = ViserRSRD(
        server,
        optimizer,
        root_node_name="/camera/object",
        scale=(1 / optimizer.dataset_scale),
        show_hands=False,
    )
    viser_urdf = viser.extras.ViserUrdf(server, urdf, root_node_name="/yumi")
    viser_urdf.update_cfg(np.zeros(16))

    if hand_mode == "bimanual":
        traj_generator = part_motion_planner.plan_bimanual(T_obj_world)
    else:
        traj_generator = part_motion_planner.plan_single(T_obj_world)

    list_traj = next(traj_generator)
    num_traj, timesteps, _ = list_traj.shape

    traj_list_handler = server.gui.add_slider("Idx", 0, num_traj - 1, 1, 0)
    traj_gen_button = server.gui.add_button("Another traj")
    traj_gen_lock = Lock()
    @traj_gen_button.on_click
    def _(_):
        nonlocal list_traj
        traj_gen_button.disabled = True
        with traj_gen_lock:
            list_traj = next(traj_generator)
        traj_gen_button.disabled = False
        traj_list_handler.value = 0
        traj_list_handler.max = list_traj.shape[0] - 1

    track_slider = server.gui.add_slider("timestep", 0, timesteps - 1, 1, 0)
    play_checkbox = server.gui.add_checkbox("play", True)

    part_deltas = optimizer.part_deltas[0]
    viser_rsrd.update_cfg(part_deltas)

    run_traj_handler = server.gui.add_button("Run traj") 
    @run_traj_handler.on_click
    def _(_):
        nonlocal list_traj
        traj_list_handler.disabled = True
        traj_gen_button.disabled = True
        traj_idx = traj_list_handler.value
        approach_traj = part_motion_planner.get_approach_motion(list_traj[traj_idx, 0], 0.1)
        traj = np.concatenate([approach_traj, list_traj[traj_idx]])
        list_traj = traj[np.newaxis]
        traj_list_handler.value = 0
        traj_list_handler.max = 0


    while True:
        if play_checkbox.value:
            track_slider.value = (track_slider.value + 1) % timesteps

        traj = list_traj[traj_list_handler.value]
        tstep = track_slider.value
        part_deltas = optimizer.part_deltas[tstep]
        viser_rsrd.update_cfg(part_deltas)
        viser_rsrd.update_hands(tstep)
        viser_urdf.update_cfg(np.array(traj[tstep]))

        time.sleep(0.05)




if __name__ == "__main__":
    tyro.cli(main)
