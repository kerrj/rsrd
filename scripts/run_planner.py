"""
Run object-centric robot motion planning. (robot do! haha).
"""

# JAX by default pre-allocates 75%, which will cause OOM with nerfstudio model loading.
# This line needs to go above any jax import.
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

import json
from typing import Literal, Optional
from threading import Lock

import jaxlie
import time
from pathlib import Path
import moviepy.editor as mpy

import cv2
import numpy as onp
import jax.numpy as jnp
import torch
import warp as wp

import viser
import viser.extras
import viser.transforms as vtf
import tyro
from loguru import logger

from nerfstudio.utils.eval_utils import eval_setup
from jaxmp.extras import load_urdf
from jaxmp.coll import Convex

from rsrd.extras.cam_helpers import get_ns_camera_at_origin
from rsrd.robot.motion_plan_yumi import YUMI_REST_POSE
from rsrd.extras.zed import Zed
from rsrd.motion.motion_optimizer import (
    RigidGroupOptimizer,
    RigidGroupOptimizerConfig,
)
from rsrd.motion.atap_loss import ATAPConfig
from rsrd.extras.viser_rsrd import ViserRSRD
from rsrd.robot.planner import PartMotionPlanner
import rsrd.transforms as tf
from autolab_core import RigidTransform


def main(
    hand_mode: Literal["single", "bimanual"],
    track_dir: Path,
    zed_video_path: Optional[Path] = None,
):
    optimizer = get_optimizer(track_dir)
    logger.info("Initialized tracker.")

    server = viser.ViserServer()

    # Load URDF.
    urdf = load_urdf(
        robot_urdf_path=Path(__file__).parent
        / "../data/yumi_description/urdf/yumi.urdf"
    )
    viser_urdf = viser.extras.ViserUrdf(server, urdf, root_node_name="/yumi")
    server.scene.add_grid("ground", width=2, height=2, cell_size=0.1)
    planner = PartMotionPlanner(optimizer, urdf)
    viser_urdf.update_cfg(onp.array(YUMI_REST_POSE))

    # Load camera position.
    T_cam_world = vtf.SE3.from_matrix(
        RigidTransform.load(
            Path(__file__).parent / "../data/zed/zed_to_world.tf"
        ).matrix
    )
    server.scene.add_frame(
        "/camera",
        position=T_cam_world.translation(),
        wxyz=T_cam_world.rotation().wxyz,
        show_axes=False,
    )

    # Load object position, if available.
    if zed_video_path is not None:
        T_obj_world, (points, colors) = get_T_obj_world_from_zed(
            optimizer, zed_video_path, T_cam_world, track_dir
        )
        server.scene.add_point_cloud(
            "/camera/points",
            points=points,
            colors=colors,
            point_size=0.005,
        )
        free_object = False  # Fix at initialization zed point.
        
        if hand_mode == "bimanual":
            traj_generator = planner.plan_bimanual(T_obj_world)
        else:
            traj_generator = planner.plan_single(T_obj_world)

    else:
        T_obj_world = jaxlie.SE3.from_translation(jnp.array([0.4, 0.0, 0.0]))
        free_object = True
        traj_generator = None

    obj_frame_handle = server.scene.add_transform_controls(
        "/object",
        position=onp.array(T_obj_world.translation().squeeze()),
        wxyz=onp.array(T_obj_world.rotation().wxyz.squeeze()),
        active_axes=(free_object, free_object, free_object),
        scale=0.2,
    )

    # Load RSRD object.
    viser_rsrd = ViserRSRD(
        server,
        optimizer,
        root_node_name="/object",
        scale=(1 / optimizer.dataset_scale),
        show_hands=False,
    )
    initial_part_deltas = optimizer.part_deltas[0]
    viser_rsrd.update_cfg(initial_part_deltas)

    timesteps = optimizer.part_deltas.shape[0]
    move_obj_handler = server.gui.add_checkbox("Move object", free_object, disabled=(not free_object))
    generate_traj_handler = server.gui.add_button("Generate trajectory", disabled=free_object)
    traj_list_handler = server.gui.add_slider("trajectory", 0, 1, 1, 0, disabled=True)
    track_slider = server.gui.add_slider("timestep", 0, timesteps - 1, 1, 0)
    play_checkbox = server.gui.add_checkbox("play", True)
    traj_gen_lock = Lock()

    list_traj = jnp.array(YUMI_REST_POSE).reshape(1, 1, -1).repeat(timesteps, axis=1)

    # While moving object / fixing object, trajectories need to be regenerated.
    # So it should clear all the cache.
    @move_obj_handler.on_update
    def _(_):
        nonlocal list_traj, traj_generator
        generate_traj_handler.disabled = move_obj_handler.value
        curr_free_axes = move_obj_handler.value
        obj_frame_handle.active_axes = (curr_free_axes, curr_free_axes, curr_free_axes)

        traj_list_handler.value = 0
        traj_list_handler.disabled = True
        if list_traj is not None:
            list_traj = jnp.array(YUMI_REST_POSE).reshape(1, 1, -1).repeat(timesteps, axis=1)

        T_obj_world = jaxlie.SE3(
            jnp.array([*obj_frame_handle.wxyz, *obj_frame_handle.position])
        )
        if hand_mode == "bimanual":
            traj_generator = planner.plan_bimanual(T_obj_world)
        else:
            traj_generator = planner.plan_single(T_obj_world)

    @generate_traj_handler.on_click
    def _(_):
        nonlocal list_traj
        assert traj_generator is not None
        generate_traj_handler.disabled = True
        move_obj_handler.disabled = True
        with traj_gen_lock:
            list_traj = next(traj_generator)
        generate_traj_handler.disabled = False
        traj_list_handler.max = list_traj.shape[0] - 1
        traj_list_handler.value = 0
        traj_list_handler.disabled = False
        move_obj_handler.disabled = False

    while True:
        if play_checkbox.value:
            track_slider.value = (track_slider.value + 1) % timesteps

        traj = list_traj[traj_list_handler.value]
        tstep = track_slider.value

        part_deltas = optimizer.part_deltas[tstep]
        viser_rsrd.update_cfg(part_deltas)
        viser_rsrd.update_hands(tstep)
        viser_urdf.update_cfg(onp.array(traj[tstep]))

        time.sleep(0.05)


def get_optimizer(
    track_dir: Path,
) -> RigidGroupOptimizer:
    # Save the paths to the cache file.
    track_cache_path = track_dir / "cache_info.json"
    assert track_cache_path.exists()
    cache_data = json.loads(track_cache_path.read_text())
    is_obj_jointed = bool(cache_data["is_obj_jointed"])
    dig_config_path = Path(cache_data["dig_config_path"])
    track_data_path = track_dir / "keyframes.txt"

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
    # Load keyframes.
    optimizer.load_tracks(track_data_path)
    hands = optimizer.hands_info
    assert hands is not None

    return optimizer


def get_T_obj_world_from_zed(
    optimizer: RigidGroupOptimizer,
    zed_video_path: Path,
    T_cam_world: vtf.SE3,
    track_dir: Path,
) -> tuple[jaxlie.SE3, tuple[onp.ndarray, onp.ndarray]]:
    """
    Get T_obj_world by registering the object in the scene.
    The ZED video shows the static scene with the object.
    """
    # Optimize, based on zed video.
    zed = Zed(str(zed_video_path))
    left, _, depth = zed.get_frame(depth=True)
    assert left is not None and depth is not None
    points, colors = zed.project_depth(
        left, depth, torch.Tensor(zed.get_K()).cuda(), subsample=8
    )

    # Optimize object pose.
    left_uint8 = (left.cpu().numpy() * 255).astype(onp.uint8)
    camera = get_ns_camera_at_origin(K=zed.get_K(), width=zed.width, height=zed.height)
    first_obs = optimizer.create_observation_from_rgb_and_camera(
        left_uint8, camera, metric_depth=depth.cpu().numpy()
    )
    renders = optimizer.initialize_obj_pose(first_obs, render=True, use_depth=True)
    _renders = []
    for r in renders:
        _left = cv2.resize(left_uint8, (r.shape[1], r.shape[0]))
        _left = cv2.cvtColor(_left, cv2.COLOR_BGR2RGB)
        _renders.append(r * 0.8 + _left * 0.2)
    out_clip = mpy.ImageSequenceClip(_renders, fps=30)
    out_clip.write_videofile(str(track_dir / "zed_registration.mp4"))

    T_obj_cam = optimizer.T_objreg_world
    # Convert opengl -> opencv
    T_obj_cam = (
        tf.SE3.from_rotation(tf.SO3.from_x_radians(torch.Tensor([torch.pi]).cuda()))
        @ T_obj_cam
    )
    # Put to world scale; tracking is performed + saved in nerfstudio coordinates.
    T_obj_cam = tf.SE3.from_rotation_and_translation(
        rotation=T_obj_cam.rotation(),
        translation=T_obj_cam.translation() / optimizer.dataset_scale,
    )

    # Optimize robot trajectory.
    T_obj_world = jaxlie.SE3(jnp.array(T_cam_world.wxyz_xyz)) @ jaxlie.SE3(
        jnp.array(T_obj_cam.wxyz_xyz.detach().cpu())
    )

    return T_obj_world, (points, colors)


if __name__ == "__main__":
    tyro.cli(main)
