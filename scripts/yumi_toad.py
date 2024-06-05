from __future__ import annotations

import torch
import viser
import viser.transforms as vtf
import time
import numpy as np
import trimesh
from typing import Optional, List, Any, Tuple, Callable
import tyro
import moviepy.editor as mpy
from copy import deepcopy

from nerfstudio.utils.eval_utils import eval_setup
from pathlib import Path
from threading import Lock
import warp as wp
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.viewer.viewer import Viewer
from nerfstudio.configs.base_config import ViewerConfig
from garfield.garfield_gaussian_pipeline import GarfieldGaussianPipeline
from nerfstudio.utils import writer
from nerfstudio.models.splatfacto import SH2RGB

from curobo.types.state import JointState
from curobo.geom.types import Mesh
from curobo.rollout.rollout_base import Goal, RolloutBase, RolloutMetrics
from curobo.types.math import Pose

from autolab_core import RigidTransform

from toad.yumi_curobo import YumiCurobo, createTableWorld
from toad.zed import Zed
from toad.optimization.rigid_group_optimizer import RigidGroupOptimizer
from toad.toad_optimizer import ToadOptimizer
from toad.yumi.yumi_robot import YumiRobot, REAL_ROBOT_SPEED, YUMI_REST_POSE_LEFT, YUMI_REST_POSE_RIGHT


def create_zed_and_toad(
    server: viser.ViserServer,
    config_path: Path,
    camera_frame_name: str = "camera",
) -> Tuple[Zed, viser.FrameHandle, Callable[[Path], ToadOptimizer]]:
    # Visualize the camera.
    zed = Zed()
    camera_tf = RigidTransform.load("data/zed_to_world.tf")
    camera_frame = server.scene.add_frame(
        f"{camera_frame_name}",
        position=camera_tf.translation,  # rough alignment.
        wxyz=camera_tf.quaternion,
        show_axes=True,
        axes_length=0.05,
        axes_radius=0.007,
    )
    server.scene.add_mesh_trimesh(
        f"{camera_frame_name}/mesh",
        mesh=zed.zed_mesh,
        scale=0.001,
        position=zed.cam_to_zed.translation,
        wxyz=zed.cam_to_zed.quaternion,
    )
    # server.scene.add_label(
    #     f"{camera_frame_name}/label",
    #     text="Camera",
    # )

    # Create the ToadOptimizer.
    def create_toad_opt(keyframe_path: Path) -> ToadOptimizer:
        toad_opt = ToadOptimizer(
            config_path,
            zed.get_K(),
            zed.width,
        zed.height,
            # zed.width,
            # zed.height,
            init_cam_pose=torch.from_numpy(
                vtf.SE3(
                    wxyz_xyz=np.array([*camera_frame.wxyz, *camera_frame.position])
                ).as_matrix()[None, :3, :]
            ).float(),
        )

        l, _, depth = zed.get_frame(depth=True)  # type: ignore
        toad_opt.set_frame(l,depth)
        with zed.raft_lock:
            toad_opt.init_obj_pose()
        toad_opt.optimizer.load_trajectory(keyframe_path)
        return toad_opt

    return zed, camera_frame, create_toad_opt


def create_objects_in_scene(
    server: viser.ViserServer,
    toad_opt: ToadOptimizer,
    object_frame_name: str = "camera/object",
) -> Tuple[viser.GuiInputHandle, viser.GuiInputHandle, List[viser.FrameHandle]]:
    part_handle = server.gui.add_number(
        "Moving", -1, -1, toad_opt.num_groups - 1, 1, disabled=True
    )
    anchor_handle = server.gui.add_number(
        "Anchor", -1, -1, toad_opt.num_groups - 1, 1, disabled=True
    )
    reset_part_anchor_handle = server.gui.add_button("Reset part and anchor")
    @reset_part_anchor_handle.on_click
    def _(_):
        part_handle.value = -1
        anchor_handle.value = -1
        for i in range(toad_opt.num_groups):
            curry_mesh(i)

    # Make the meshes s.t. that if you double-click, it becomes the anchor.
    def curry_mesh(idx):
        mesh = toad_opt.toad_object.meshes[idx]
        if part_handle.value == idx:
            mesh.visual.vertex_colors = [100, 255, 100, 255]
        elif anchor_handle.value == idx:
            mesh.visual.vertex_colors = [255, 100, 100, 255]

        handle = server.scene.add_mesh_trimesh(
            f"{object_frame_name}/group_{idx}/mesh",
            mesh=mesh,
        )
        @handle.on_click
        def _(_):
            container_handle = server.scene.add_3d_gui_container(
                f"{object_frame_name}/group_{idx}/container",
            )
            with container_handle:
                anchor_button = server.gui.add_button("Anchor")
                @anchor_button.on_click
                def _(_):
                    anchor_handle.value = idx
                    if part_handle.value == idx:
                        part_handle.value = -1
                    curry_mesh(idx)
                move_button = server.gui.add_button("Move")
                @move_button.on_click
                def _(_):
                    part_handle.value = idx
                    if anchor_handle.value == idx:
                        anchor_handle.value = -1
                    curry_mesh(idx)
                close_button = server.gui.add_button("Close")
                @close_button.on_click
                def _(_):
                    container_handle.remove()

        return handle

    frame_list = []
    for i, tf in enumerate(toad_opt.get_parts2cam(keyframe=0)):
        frame_list.append(
            server.scene.add_frame(
                f"{object_frame_name}/group_{i}",
                position=tf.translation(),
                wxyz=tf.rotation().wxyz,
                show_axes=True,
                axes_length=0.04,
                axes_radius=.004
            )
        )
        curry_mesh(i)
        grasps = toad_opt.toad_object.grasps[i].numpy() # [N_grasps, 7]
        grasp_mesh = toad_opt.toad_object.grasp_axis_mesh()
        for j, grasp in enumerate(grasps):
            server.scene.add_mesh_trimesh(
                f"{object_frame_name}/group_{i}/grasp_{j}",
                grasp_mesh,
                position=grasp[:3],
                wxyz=grasp[3:],
            )

    return part_handle, anchor_handle, frame_list


def main(
    # config_path: Path = Path("outputs/nerfgun_poly_far/dig/2024-06-02_234451/config.yml"),
    # keyframe_path: Path = Path("renders/nerfgun_poly_far/keyframes.pt")
    # config_path: Path = Path("outputs/garfield_poly/dig/2024-06-03_183227/config.yml"),
    # keyframe_path: Path = Path("renders/garfield_poly/keyframes.pt")
    config_path: Path = Path("outputs/sunglasses3/dig/2024-06-03_175202/config.yml"),
    keyframe_path: Path = Path("renders/sunglasses3/keyframes.pt")
    # config_path: Path = Path("outputs/scissors/dig/2024-06-03_135548/config.yml"),
    # keyframe_path: Path = Path("renders/scissors/keyframes.pt")
):
    """Quick interactive demo for object traj following.

    Args:
        config_path: Path to the nerfstudio config file.
        keyframe_path: Path to the keyframe file.
    """

    server = viser.ViserServer()

    # Needs to be called before any warp code gets called.
    wp.init()

    robot = YumiRobot(
        target=server,
        minibatch_size=1,
    )

    zed, camera_frame, create_toad_opt = create_zed_and_toad(
        server,
        config_path,
        camera_frame_name="camera",
    )
    toad_opt = create_toad_opt(keyframe_path)
    reinitialize_toad_button = server.gui.add_button("Reinitialize Toad")
    @reinitialize_toad_button.on_click
    def _(_):
        nonlocal toad_opt
        toad_opt = create_toad_opt(keyframe_path)

    assert toad_opt.optimizer.keyframes is not None, "Keyframes not loaded."

    part_handle, anchor_handle, frame_list = create_objects_in_scene(
        server,
        toad_opt,
        object_frame_name="camera/object"
    )
    def update_keyframe(keyframe_idx):
        part2cam = toad_opt.get_parts2cam(keyframe=keyframe_idx)
        for i in range(len(frame_list)):
            frame_list[i].position = part2cam[i].translation()
            frame_list[i].wxyz = part2cam[i].rotation().wxyz

        if toad_opt.optimizer.keyframes is not None:
            toad_opt.optimizer.apply_keyframe(keyframe_handle.value)
            hands = toad_opt.optimizer.hand_frames[keyframe_handle.value]
            for ih,h in enumerate(hands):
                h_world = h.copy()
                h_world.apply_transform(toad_opt.optimizer.get_registered_o2w().cpu().numpy())
                server.scene.add_mesh_trimesh(f"hand{ih}", h_world, scale=1/toad_opt.optimizer.dataset_scale)

    keyframe_handle = server.gui.add_slider("Keyframe", 0, len(toad_opt.optimizer.keyframes) - 1, 1, 0)
    @keyframe_handle.on_update
    def _(_):
        update_keyframe(keyframe_handle.value)

    # Create the trajectories!
    traj, traj_handle, play_handle = None, None, None
    goal_button = server.gui.add_button("Calculate working grasps")
    real_button = server.gui.add_button("Move real robot", disabled=(robot.real is None))
    reset_button = server.gui.add_button("Reset real robot")

    @reset_button.on_click
    def _(_):
        robot.reset_real()
        if robot.real is not None:
            return

        assert robot.real is not None
        robot.real.move_joints_sync(
            l_joints=np.array(list(YUMI_REST_POSE_LEFT.values())[:7]).astype(np.float64),
            r_joints=np.array(list(YUMI_REST_POSE_RIGHT.values())[:7]).astype(np.float64),
        )
        robot.real.left.sync()
        robot.real.right.sync()
    
    # For now, we set the active arm manually...
    plan_dropdown = server.gui.add_dropdown(
        "Active Arm",
        options=("left", "right")
    )

    @goal_button.on_click
    def _(_):
        nonlocal traj, traj_handle, play_handle

        if part_handle.value == -1:
            print("Please select a part to move.")
            return
        goal_button.disabled = True

        cam_vtf = vtf.SE3(wxyz_xyz=np.array([*camera_frame.wxyz, *camera_frame.position]))
        poses_part2cam = toad_opt.get_parts2cam(keyframe=0)
        poses_part2world = [cam_vtf.multiply(pose) for pose in poses_part2cam]

        # Update the world objects.
        toad_obj = toad_opt.toad_object
        mesh_curobo_config = toad_obj.to_world_config(
            poses_wxyz_xyz=[pose.wxyz_xyz for pose in poses_part2world]
        )
        robot.plan.update_world_objects(mesh_curobo_config)

        # Get the grasp path for the selected part.
        moving_grasp = toad_obj.grasps[part_handle.value]
        moving_grasps_gripper = toad_obj.to_gripper_frame(moving_grasp)
        moving_grasp_path_wxyz_xyz = torch.cat([
            torch.Tensor(
                cam_vtf
                .multiply(toad_opt.get_parts2cam(keyframe=keyframe_idx)[part_handle.value])
                .multiply(moving_grasps_gripper)
                .wxyz_xyz
            ).unsqueeze(1)  # [num_grasps, 7] -> [num_grasps, 1, 7]
            for keyframe_idx in range(len(toad_opt.optimizer.keyframes))
        ], dim=1).float().cuda() # [num_grasps, num_keyframes, 7]

        batch_size = moving_grasp_path_wxyz_xyz.shape[0]
        if plan_dropdown.value == "left":
            robot.plan.activate_arm("left", robot.q_right)
        else:
            robot.plan.activate_arm("right", robot.q_left)

        with zed.raft_lock:
            moving_js, moving_success = robot.plan.ik(
                goal_wxyz_xyz=moving_grasp_path_wxyz_xyz[:, 0, :],
                q_init=robot.q_left.expand(batch_size, -1),
            )

        if moving_js is None:
            print("Failed to solve IK (moving).")
            goal_button.disabled = False
            return
        assert isinstance(moving_js, torch.Tensor) and isinstance(moving_success, torch.Tensor)

        if not moving_success.any():
            print("IK solution is invalid (moving).")
            goal_button.disabled = False
            return

        # Plan on the gripper pose.
        moving_grasp = toad_obj.grasps[part_handle.value]
        moving_grasps_gripper = toad_obj.to_gripper_frame(
            moving_grasp,
            robot.tooltip_to_gripper
        )
        moving_grasp_path_wxyz_xyz = torch.cat([
            torch.Tensor(
                cam_vtf
                .multiply(toad_opt.get_parts2cam(keyframe=keyframe_idx)[part_handle.value])
                .multiply(moving_grasps_gripper)
                .wxyz_xyz
            ).unsqueeze(1)  # [num_grasps, 7] -> [num_grasps, 1, 7]
            for keyframe_idx in range(len(toad_opt.optimizer.keyframes))
        ], dim=1).float().cuda() # [num_grasps, num_keyframes, 7]

        with zed.raft_lock:
            # ... should be on cpu, but just in case.
            traj, succ = robot.plan_jax.plan_from_waypoints(
                poses=moving_grasp_path_wxyz_xyz[moving_success],
                arm=plan_dropdown.value,
            )

        assert len(traj.shape) == 3 and traj.shape[-1] == 8
        if not succ.any():
            print("Failed to generate a valid trajectory.")
            goal_button.disabled = False
            return
        goal_button.disabled = False

        # Override the gripper width, for both arms.
        traj[..., 7:] = 0.025
        # traj = traj[succ]

        # For the valid motion trajs, also plan a grasp motion to the object.
        # moving_grasp = toad_obj.grasps[part_handle.value]
        # moving_grasps_gripper = toad_obj.to_gripper_frame(moving_grasp)
        # moving_grasp_path_wxyz_xyz = torch.cat([
        #     torch.Tensor(
        #         cam_vtf
        #         .multiply(toad_opt.get_parts2cam(keyframe=keyframe_idx)[part_handle.value])
        #         .multiply(moving_grasps_gripper)
        #         .wxyz_xyz
        #     ).unsqueeze(1)  # [num_grasps, 7] -> [num_grasps, 1, 7]
        #     for keyframe_idx in range(len(toad_opt.optimizer.keyframes))
        # ], dim=1).float().cuda() # [num_grasps, num_keyframes, 7]
        # with zed.raft_lock:
        #     approach_traj, approach_succ = robot.plan.gen_motion_from_goal(
        #         goal_wxyz_xyz=moving_grasp_path_wxyz_xyz[moving_success, 0, :],
        #         q_init=robot.plan.home_pos.view(1, -1).expand(traj.shape[0], -1),
        #     )
        with zed.raft_lock:
            approach_traj, approach_succ = robot.plan.gen_motion_from_goal_joints(
                goal_js=traj[:, 0, :],  # [num_traj, 8]
                q_init=robot.plan.home_pos.view(1, -1).expand(traj.shape[0], -1),
            )

        if approach_traj is None or approach_succ is None:
            print("Approach traj failed.")
            goal_button.disabled = False
            return
        if not approach_succ.any():
            print("Approach traj invalid.")
            goal_button.disabled = False
            return
        approach_traj[..., 7:] = 0.025

        traj = torch.cat([approach_traj, traj], dim=1)
        succ = succ & approach_succ
        traj = traj[succ]

        traj_handle = server.gui.add_slider("Trajectory Index", 0, len(traj) - 1, 1, 0)
        play_handle = server.gui.add_slider("play", min=0, max=traj.shape[1]-1, step=1, initial_value=0)

        def move_to_traj_position():
            assert traj is not None and traj_handle is not None and play_handle is not None
            assert isinstance(traj, torch.Tensor)
            if plan_dropdown.value == "left":
                robot.q_left = traj[traj_handle.value][play_handle.value].view(-1)
            else:
                robot.q_right = traj[traj_handle.value][play_handle.value].view(-1)
            if play_handle.value >= approach_traj.shape[1]:
                update_keyframe(play_handle.value - approach_traj.shape[1])

        @traj_handle.on_update
        def _(_):
            move_to_traj_position()
        @play_handle.on_update
        def _(_):
            move_to_traj_position()

        move_to_traj_position()

        @real_button.on_click
        def _(_):
            if robot.real is None:
                print("Real robot is not available.")
                return
            if traj is None or traj_handle is None or play_handle is None:
                print("No trajectory available.")
                return
            approach_length = approach_traj.shape[1]
            if plan_dropdown.value == "left":
                robot.real.left.open_gripper()
                robot.real.left.move_joints_traj(
                    joints=traj[traj_handle.value][:approach_length, :7].cpu().numpy().astype(np.float64),
                    speed=REAL_ROBOT_SPEED
                )
                time.sleep(1)
                robot.real.left.sync()
                robot.real.left.close_gripper()
                robot.real.left.move_joints_traj(
                    joints=traj[traj_handle.value][approach_length:, :7].cpu().numpy().astype(np.float64),
                    speed=REAL_ROBOT_SPEED
                )
                time.sleep(1)
                robot.real.left.sync()
                robot.real.left.open_gripper()
            if plan_dropdown.value == "right":
                robot.real.right.open_gripper()
                robot.real.right.move_joints_traj(
                    joints=traj[traj_handle.value][:approach_length, :7].cpu().numpy().astype(np.float64),
                    speed=REAL_ROBOT_SPEED
                )

                time.sleep(1)
                robot.real.right.sync()
                robot.real.right.close_gripper()
                robot.real.right.move_joints_traj(
                    joints=traj[traj_handle.value][approach_length:, :7].cpu().numpy().astype(np.float64),
                    speed=REAL_ROBOT_SPEED
                )
                time.sleep(1)
                robot.real.right.sync()
                robot.real.right.open_gripper()
                robot.real.right.sync()

    while True:
        left, right, depth = zed.get_frame() # internally gets the lock.

        toad_opt.set_frame(left,depth)
        with zed.raft_lock:
            # toad_opt.step_opt(niter=50)
            update_keyframe(keyframe_handle.value)

        K = torch.from_numpy(zed.get_K()).float().cuda()
        assert isinstance(left, torch.Tensor) and isinstance(depth, torch.Tensor)
        points, colors = Zed.project_depth(left, depth, K)

        server.scene.add_point_cloud(
            "camera/points",
            points=points,
            colors=colors,
            point_size=0.001,
        )

if __name__ == "__main__":
    tyro.cli(main)