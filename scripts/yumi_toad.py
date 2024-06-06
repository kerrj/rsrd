from __future__ import annotations

import torch
import viser
import viser.transforms as vtf
import time
import numpy as np
import trimesh
from typing import Optional, List, Any, Tuple, Callable, Literal
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


def gen_approach_and_motion(
    robot: YumiRobot,
    arm: Literal["left", "right"],
    gripper_path_wxyz_xyz: torch.Tensor,  # gripper poses
    tt_path_wxyz_xyz: torch.Tensor,  # tooltip poses
    raft_lock: Lock,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    # gripper and tt_ should be [num_grasps, tstep, 7].
    # They should be for the same poses! This is just because waypoint gen is happpening @ gripper.
    batch_size = gripper_path_wxyz_xyz.shape[0]
    tstep = gripper_path_wxyz_xyz.shape[1]
    assert gripper_path_wxyz_xyz.shape == tt_path_wxyz_xyz.shape == (batch_size, tstep, 7)

    robot.plan.activate_arm(arm, (
        (robot.q_right_home if arm == "left" else robot.q_left_home)
    ))

    with raft_lock:
        waypoint_js, waypoint_js_success = robot.plan.ik(
            goal_wxyz_xyz=tt_path_wxyz_xyz[:, 0, :],
            q_init=robot.plan.home_pos.expand(batch_size, -1),
        )

    if waypoint_js is None:
        print("Failed to solve IK (moving).")
        return None, None
    assert isinstance(waypoint_js, torch.Tensor) and isinstance(waypoint_js_success, torch.Tensor)
    if not waypoint_js_success.any():
        print("IK solution is invalid (moving).")
        return None, None
    waypoint_js[..., 7:] = 0.025

    with raft_lock:
        # ... should be on cpu, but just in case.
        waypoint_traj, waypoint_succ = robot.plan_jax.plan_from_waypoints(
            poses=gripper_path_wxyz_xyz[waypoint_js_success],
            arm=arm,
        )
    if waypoint_traj is None:
        print("Failed to generate a valid trajectory.")
        return None, None
    if not waypoint_succ.any():
        print("No valid waypoint traj.")
        return None, None
    waypoint_traj[..., 7:] = 0.025
    waypoint_traj = waypoint_traj[waypoint_succ]

    # I should... probably do collcheck w the waypoints.

    with raft_lock:
        approach_traj, approach_succ = robot.plan.gen_motion_from_goal_joints(
            goal_js=waypoint_traj[:, 0, :],  # [num_traj, 8]
            q_init=robot.plan.home_pos.view(1, -1).expand(waypoint_traj.shape[0], -1),
        )

    if approach_traj is None or approach_succ is None:
        print("Approach traj failed.")
        return None, None
    if not approach_succ.any():
        print("Approach traj invalid.")
        return None, None

    approach_traj, waypoint_traj = approach_traj[approach_succ], waypoint_traj[approach_succ]

    return (
        approach_traj,
        waypoint_traj
    )


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
    
    # phone_mesh = trimesh.creation.
    # server.scene.add_mesh_trimesh(
    #     f"{camera_frame_name}/phone",
    # )
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
                axes_length=0.03,
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
    # config_path: Path = Path("outputs/sunglasses3/dig/2024-06-03_175202/config.yml"),
    # keyframe_path: Path = Path("renders/sunglasses3/keyframes.pt")
    config_path: Path = Path("outputs/scissors/dig/2024-06-03_135548/config.yml"),
    keyframe_path: Path = Path("renders/scissors/keyframes.pt")
    # config_path: Path = Path("outputs/wooden_drawer/dig/2024-06-03_160055/config.yml"),
    # keyframe_path: Path = Path("renders/wooden_drawer/keyframes.pt")
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
    update_keyframe(0)

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
    # plan_dropdown = server.gui.add_dropdown(
    #     "Active Arm",
    #     options=("left", "right")
    # )

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
        moving_grasps_gripper = toad_obj.to_gripper_frame(moving_grasp, robot.tooltip_to_gripper)
        moving_grasp_path_wxyz_xyz_gripper = torch.cat([
            torch.Tensor(
                cam_vtf
                .multiply(toad_opt.get_parts2cam(keyframe=keyframe_idx)[part_handle.value])
                .multiply(moving_grasps_gripper)
                .wxyz_xyz
            ).unsqueeze(1)  # [num_grasps, 7] -> [num_grasps, 1, 7]
            for keyframe_idx in range(len(toad_opt.optimizer.keyframes))
        ], dim=1).float().cuda() # [num_grasps, num_keyframes, 7]

        moving_grasp = toad_obj.grasps[part_handle.value]
        moving_grasps_gripper = toad_obj.to_gripper_frame(moving_grasp)
        moving_grasp_path_wxyz_xyz_tt = torch.cat([
            torch.Tensor(
                cam_vtf
                .multiply(toad_opt.get_parts2cam(keyframe=keyframe_idx)[part_handle.value])
                .multiply(moving_grasps_gripper)
                .wxyz_xyz
            ).unsqueeze(1)  # [num_grasps, 7] -> [num_grasps, 1, 7]
            for keyframe_idx in range(len(toad_opt.optimizer.keyframes))
        ], dim=1).float().cuda() # [num_grasps, num_keyframes, 7]

        if anchor_handle.value != -1:
            anchor_grasp = toad_obj.grasps[anchor_handle.value]
            anchor_grasps_gripper = toad_obj.to_gripper_frame(anchor_grasp, robot.tooltip_to_gripper)
            anchor_grasp_path_wxyz_xyz_gripper = torch.cat([
                torch.Tensor(
                    cam_vtf
                    .multiply(toad_opt.get_parts2cam(keyframe=keyframe_idx)[anchor_handle.value])
                    .multiply(anchor_grasps_gripper)
                    .wxyz_xyz
                ).unsqueeze(1)  # [num_grasps, 7] -> [num_grasps, 1, 7]
                for keyframe_idx in range(len(toad_opt.optimizer.keyframes))
            ], dim=1).float().cuda() # [num_grasps, num_keyframes, 7]

            anchor_grasp = toad_obj.grasps[anchor_handle.value]
            anchor_grasps_gripper = toad_obj.to_gripper_frame(anchor_grasp)
            anchor_grasp_path_wxyz_xyz_tt = torch.cat([
                torch.Tensor(
                    cam_vtf
                    .multiply(toad_opt.get_parts2cam(keyframe=keyframe_idx)[anchor_handle.value])
                    .multiply(anchor_grasps_gripper)
                    .wxyz_xyz
                ).unsqueeze(1)  # [num_grasps, 7] -> [num_grasps, 1, 7]
                for keyframe_idx in range(len(toad_opt.optimizer.keyframes))
            ], dim=1).float().cuda() # [num_grasps, num_keyframes, 7]

        l_moving_approach, l_moving_waypoint = gen_approach_and_motion(
            robot=robot,
            arm="left",
            gripper_path_wxyz_xyz=moving_grasp_path_wxyz_xyz_gripper,
            tt_path_wxyz_xyz=moving_grasp_path_wxyz_xyz_tt,
            raft_lock=zed.raft_lock,
        )

        if anchor_handle.value != -1:
            l_anchor_approach, l_anchor_waypoint = gen_approach_and_motion(
                robot=robot,
                arm="left",
                gripper_path_wxyz_xyz=anchor_grasp_path_wxyz_xyz_gripper,
                tt_path_wxyz_xyz=anchor_grasp_path_wxyz_xyz_tt,
                raft_lock=zed.raft_lock,
            )

        r_moving_approach, r_moving_waypoint = gen_approach_and_motion(
            robot=robot,
            arm="right",
            gripper_path_wxyz_xyz=moving_grasp_path_wxyz_xyz_gripper,
            tt_path_wxyz_xyz=moving_grasp_path_wxyz_xyz_tt,
            raft_lock=zed.raft_lock,
        )

        if anchor_handle.value != -1:
            r_anchor_approach, r_anchor_waypoint = gen_approach_and_motion(
                robot=robot,
                arm="right",
                gripper_path_wxyz_xyz=anchor_grasp_path_wxyz_xyz_gripper,
                tt_path_wxyz_xyz=anchor_grasp_path_wxyz_xyz_tt,
                raft_lock=zed.raft_lock,
            )

        robot.plan.update_world_objects()  # clear the objects.

        if anchor_handle.value != -1:
            working_trajs = []
            if (l_moving_approach is not None and r_anchor_approach is not None):
                assert r_anchor_waypoint is not None and l_moving_waypoint is not None
                # need to do some mxn matching here.
                for moving_idx in range(l_moving_approach.shape[0]):
                    for anchor_idx in range(r_anchor_approach.shape[0]):
                        approach_traj = robot.concat_joints(r_anchor_approach[anchor_idx], l_moving_approach[moving_idx])
                        waypoint_traj = robot.concat_joints(r_anchor_waypoint[anchor_idx], l_moving_waypoint[moving_idx])
                        traj = torch.cat([approach_traj, waypoint_traj], dim=0)
                        assert len(traj.shape) == 2 and traj.shape[-1] == 16
                        d_world, d_self = robot.plan.in_collision_full(traj)
                        if (d_self <= 0).all():
                            working_trajs.append(traj)
            elif (r_moving_approach is not None and l_anchor_approach is not None):
                assert r_moving_waypoint is not None and l_anchor_waypoint is not None
                for moving_idx in range(r_moving_approach.shape[0]):
                    for anchor_idx in range(l_anchor_approach.shape[0]):
                        approach_traj = robot.concat_joints(r_moving_approach[moving_idx], l_anchor_approach[anchor_idx])
                        waypoint_traj = robot.concat_joints(r_moving_waypoint[moving_idx], l_anchor_waypoint[anchor_idx])
                        traj = torch.cat([approach_traj, waypoint_traj], dim=0)
                        assert len(traj.shape) == 2 and traj.shape[-1] == 16
                        d_world, d_self = robot.plan.in_collision_full(traj)
                        if (d_self <= 0).all():
                            working_trajs.append(traj)
            else:
                print("No bimanual setup worked.")
                goal_button.disabled = False
                return

            traj = torch.stack(working_trajs, dim=0)
        else:
            # Single arm.
            working_trajs = []
            if l_moving_approach is not None:
                _traj_l = torch.cat([l_moving_approach, l_moving_waypoint], dim=1).cuda()  # [num_traj, tstep, 8]
                _traj_r = robot.q_right_home.expand(_traj_l.shape).cuda()
                traj = robot.concat_joints(_traj_r, _traj_l)
                working_trajs.append(traj)
            if r_moving_approach is not None:
                _traj_r = torch.cat([r_moving_approach, r_moving_waypoint], dim=1).cuda()
                _traj_l = robot.q_left_home.expand(_traj_r.shape).cuda()
                traj = robot.concat_joints(_traj_l, _traj_r)
                working_trajs.append(traj)

            if len(working_trajs) == 0:
                print("No valid single arm setup.")
                goal_button.disabled = False
                return
            elif len(working_trajs) == 1:
                traj = working_trajs[0]
            else:
                traj = torch.cat(working_trajs, dim=0)
        
        goal_button.disabled = False
        assert traj is not None
        assert len(traj.shape) == 3 and traj.shape[-1] == 16

        traj_handle = server.gui.add_slider("Trajectory Index", 0, traj.shape[0] - 1, 1, 0)
        play_handle = server.gui.add_slider("play", min=0, max=traj.shape[1]-1, step=1, initial_value=0)
        approach_len = robot.plan._motion_gen.interpolation_steps

        def move_to_traj_position():
            assert traj is not None and traj_handle is not None and play_handle is not None
            # robot.q_left = traj[traj_handle.value][play_handle.value].view(-1)
            robot.q_all = traj[traj_handle.value][play_handle.value].view(-1)
            if play_handle.value >= approach_len:
                update_keyframe(play_handle.value - approach_len)

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
            assert traj is not None
            assert traj_handle is not None

            robot.real.left.open_gripper()
            robot.real.right.open_gripper()

            robot.real.move_joints_sync(
                l_joints = robot.get_left(traj[traj_handle.value])[:approach_len, :7].cpu().numpy().astype(np.float64),
                r_joints = robot.get_right(traj[traj_handle.value])[:approach_len, :7].cpu().numpy().astype(np.float64),
                speed=REAL_ROBOT_SPEED
            )
            time.sleep(1)
            robot.real.left.sync()
            robot.real.right.sync()

            robot.real.left.close_gripper()
            robot.real.right.close_gripper()

            robot.real.move_joints_sync(
                l_joints = robot.get_left(traj[traj_handle.value])[approach_len:, :7].cpu().numpy().astype(np.float64),
                r_joints = robot.get_right(traj[traj_handle.value])[approach_len:, :7].cpu().numpy().astype(np.float64),
                speed=REAL_ROBOT_SPEED
            )
            time.sleep(1)
            robot.real.left.sync()
            robot.real.right.sync()

            robot.real.left.open_gripper()
            robot.real.right.open_gripper()

            # robot.real.move_joints_sync(
            #     l_joints = robot.get_left(traj[traj_handle.value])[:approach_len, :7].cpu().numpy().astype(np.float64),
            #     r_joints = robot.get_right(traj[traj_handle.value])[:approach_len, :7][::-1].cpu().numpy().astype(np.float64),
            #     speed=REAL_ROBOT_SPEED
            # )
            # time.sleep(1)
            # robot.real.left.sync()
            # robot.real.right.sync()

    # @goal_button.on_click
    # def _(_):
    #     nonlocal traj, traj_handle, play_handle

    #     if part_handle.value == -1:
    #         print("Please select a part to move.")
    #         return
    #     goal_button.disabled = True

    #     cam_vtf = vtf.SE3(wxyz_xyz=np.array([*camera_frame.wxyz, *camera_frame.position]))
    #     poses_part2cam = toad_opt.get_parts2cam(keyframe=0)
    #     poses_part2world = [cam_vtf.multiply(pose) for pose in poses_part2cam]

    #     # Update the world objects.
    #     toad_obj = toad_opt.toad_object
    #     mesh_curobo_config = toad_obj.to_world_config(
    #         poses_wxyz_xyz=[pose.wxyz_xyz for pose in poses_part2world]
    #     )
    #     robot.plan.update_world_objects(mesh_curobo_config)

    #     # Get the grasp path for the selected part.
    #     moving_grasp = toad_obj.grasps[part_handle.value]
    #     moving_grasps_gripper = toad_obj.to_gripper_frame(moving_grasp)
    #     moving_grasp_path_wxyz_xyz = torch.cat([
    #         torch.Tensor(
    #             cam_vtf
    #             .multiply(toad_opt.get_parts2cam(keyframe=keyframe_idx)[part_handle.value])
    #             .multiply(moving_grasps_gripper)
    #             .wxyz_xyz
    #         ).unsqueeze(1)  # [num_grasps, 7] -> [num_grasps, 1, 7]
    #         for keyframe_idx in range(len(toad_opt.optimizer.keyframes))
    #     ], dim=1).float().cuda() # [num_grasps, num_keyframes, 7]

    #     batch_size = moving_grasp_path_wxyz_xyz.shape[0]
    #     if plan_dropdown.value == "left":
    #         robot.plan.activate_arm("left", robot.q_right)
    #     else:
    #         robot.plan.activate_arm("right", robot.q_left)

    #     with zed.raft_lock:
    #         moving_js, moving_success = robot.plan.ik(
    #             goal_wxyz_xyz=moving_grasp_path_wxyz_xyz[:, 0, :],
    #             q_init=robot.plan.home_pos.expand(batch_size, -1),
    #         )

    #     if moving_js is None:
    #         print("Failed to solve IK (moving).")
    #         goal_button.disabled = False
    #         return
    #     assert isinstance(moving_js, torch.Tensor) and isinstance(moving_success, torch.Tensor)

    #     if not moving_success.any():
    #         print("IK solution is invalid (moving).")
    #         goal_button.disabled = False
    #         return

    #     # Plan on the gripper pose.
    #     moving_grasp = toad_obj.grasps[part_handle.value]
    #     moving_grasps_gripper = toad_obj.to_gripper_frame(
    #         moving_grasp,
    #         robot.tooltip_to_gripper
    #     )
    #     moving_grasp_path_wxyz_xyz = torch.cat([
    #         torch.Tensor(
    #             cam_vtf
    #             .multiply(toad_opt.get_parts2cam(keyframe=keyframe_idx)[part_handle.value])
    #             .multiply(moving_grasps_gripper)
    #             .wxyz_xyz
    #         ).unsqueeze(1)  # [num_grasps, 7] -> [num_grasps, 1, 7]
    #         for keyframe_idx in range(len(toad_opt.optimizer.keyframes))
    #     ], dim=1).float().cuda() # [num_grasps, num_keyframes, 7]

    #     with zed.raft_lock:
    #         # ... should be on cpu, but just in case.
    #         waypoint_traj, waypoint_succ = robot.plan_jax.plan_from_waypoints(
    #             poses=moving_grasp_path_wxyz_xyz[moving_success],
    #             arm=plan_dropdown.value,
    #         )

    #     assert len(waypoint_traj.shape) == 3 and waypoint_traj.shape[-1] == 8
    #     if not waypoint_succ.any():
    #         print("Failed to generate a valid trajectory.")
    #         goal_button.disabled = False
    #         return

    #     # Override the gripper width, for both arms.
    #     waypoint_traj[..., 7:] = 0.025

    #     # For the valid motion trajs, also plan a grasp motion to the object.
    #     # Plan in joint space, to avoid discontinuities w/ the motion trajectory.
    #     with zed.raft_lock:
    #         approach_traj, approach_succ = robot.plan.gen_motion_from_goal_joints(
    #             goal_js=waypoint_traj[:, 0, :],  # [num_traj, 8]
    #             q_init=robot.plan.home_pos.view(1, -1).expand(waypoint_traj.shape[0], -1),
    #         )

    #     if approach_traj is None or approach_succ is None:
    #         print("Approach traj failed.")
    #         goal_button.disabled = False
    #         return
    #     if not approach_succ.any():
    #         print("Approach traj invalid.")
    #         goal_button.disabled = False
    #         return
    #     approach_traj[..., 7:] = 0.025

    #     succ = approach_succ & waypoint_succ
    #     waypoint_traj = waypoint_traj[succ]
    #     approach_traj = approach_traj[succ]

    #     # OK, so given the trajectory, we plan the anchor motion, if it exists.
    #     if anchor_handle.value != -1:
    #         anchor_grasp = toad_obj.grasps[anchor_handle.value]
    #         anchor_grasps_gripper = toad_obj.to_gripper_frame(anchor_grasp)
    #         anchor_grasp_wxyz_xyz = torch.Tensor(
    #             cam_vtf
    #             .multiply(toad_opt.get_parts2cam(keyframe=0)[anchor_handle.value])
    #             .multiply(anchor_grasps_gripper)
    #             .wxyz_xyz
    #         )  # [num_grasps, 7]

    #         # Activate the anchor arm.
    #         batch_size = anchor_grasp_wxyz_xyz.shape[0]
    #         if plan_dropdown.value == "left":
    #             robot.plan.activate_arm("right", robot.q_right)
    #         else:
    #             robot.plan.activate_arm("left", robot.q_left)

    #         with zed.raft_lock:
    #             anchor_traj, anchor_success = robot.plan.gen_motion_from_goal(
    #                 goal_wxyz_xyz=anchor_grasp_wxyz_xyz,
    #                 q_init=robot.plan.home_pos.expand(batch_size, -1),
    #             )

    #         if anchor_success is None or (not anchor_success.any()):
    #             print("Failed to solve motion (anchor).")
    #             goal_button.disabled = False
    #             return
    #         assert isinstance(anchor_traj, torch.Tensor)
    #         anchor_traj = anchor_traj[anchor_success]
    #         anchor_traj[..., 7:] = 0.025

    #         # Collision checking with the moving arm.
    #         # Find the "most compatible" anchor with the moving arm.
    #         # If there are N motion trajs, and M anchor trajs, we have N*M pairs.
    #         motion_and_anchor_success = torch.zeros((waypoint_traj.shape[0],)).bool()
    #         anchor_for_motion_traj = torch.zeros((waypoint_traj.shape[0], anchor_traj.shape[1], anchor_traj.shape[2])).long()
    #         for motion_idx in range(waypoint_traj.shape[0]):
    #             # Fix the motion arm, in starting pose.
    #             if plan_dropdown.value == "left":
    #                 robot.plan.activate_arm("right", waypoint_traj[motion_idx, 0, :7])
    #             else:
    #                 robot.plan.activate_arm("left", waypoint_traj[motion_idx, 0, :7])

    #             valid_anchor_list = []
    #             for anchor_idx in range(anchor_traj.shape[0]):
    #                 d_world, d_self = robot.plan.in_collision(anchor_traj[anchor_idx, -1, :7])
    #                 if d_self <= 0:  # world collision should've already been filtered by motiongen.
    #                     valid_anchor_list.append(anchor_idx)
    #                 else:
    #                     print(f"{motion_idx} {anchor_idx} collision: {d_world} {d_self}")
                
    #             if len(valid_anchor_list) == 0:
    #                 motion_and_anchor_success[motion_idx] = False
    #                 continue

    #             # ... let's just use the first one for now.
    #             motion_and_anchor_success[motion_idx] = True
    #             anchor_for_motion_traj[motion_idx] = anchor_traj[valid_anchor_list[0]]
            
    #         if not motion_and_anchor_success.any():
    #             print("No valid anchor for any motion traj.")
    #             goal_button.disabled = False
    #             return
    #         anchor_traj = anchor_for_motion_traj.to(robot.plan.device)
    #         waypoint_traj = waypoint_traj[motion_and_anchor_success].to(robot.plan.device)
    #         approach_traj = approach_traj[motion_and_anchor_success].to(robot.plan.device)
            
    #     else:
    #         anchor_traj = None

    #     goal_button.disabled = False
    #     traj_handle = server.gui.add_slider("Trajectory Index", 0, len(waypoint_traj) - 1, 1, 0)
        
    #     play_traj_length = (
    #         (0 if anchor_traj is None else anchor_traj.shape[1]) +
    #         approach_traj.shape[1] +
    #         waypoint_traj.shape[1]
    #     )
    #     play_handle = server.gui.add_slider("play", min=0, max=play_traj_length-1, step=1, initial_value=0)

    #     def move_to_traj_position():
    #         assert waypoint_traj is not None and traj_handle is not None and play_handle is not None
    #         # assert isinstance(traj, torch.Tensor)
    #         if plan_dropdown.value == "left":
    #             if anchor_traj is not None:
    #                 if play_handle.value < anchor_traj.shape[1]:
    #                     robot.q_right = anchor_traj[traj_handle.value][play_handle.value].view(-1)
    #                     robot.q_left = torch.Tensor(list(YUMI_REST_POSE_LEFT.values()))
    #                 elif play_handle.value < anchor_traj.shape[1] + approach_traj.shape[1]:
    #                     robot.q_left = approach_traj[traj_handle.value][play_handle.value - anchor_traj.shape[1]].view(-1)
    #                     robot.q_right = anchor_traj[traj_handle.value][-1].view(-1)
    #                 else:
    #                     robot.q_left = waypoint_traj[traj_handle.value][play_handle.value - (anchor_traj.shape[1] + approach_traj.shape[1])].view(-1)
    #                     robot.q_right = anchor_traj[traj_handle.value][-1].view(-1)
    #             else:
    #                 if play_handle.value < approach_traj.shape[1]:
    #                     robot.q_left = approach_traj[traj_handle.value][play_handle.value].view(-1)
    #                     robot.q_right = torch.Tensor(list(YUMI_REST_POSE_RIGHT.values()))
    #                 else:
    #                     robot.q_left = waypoint_traj[traj_handle.value][play_handle.value - approach_traj.shape[1]].view(-1)
    #                     robot.q_right = torch.Tensor(list(YUMI_REST_POSE_RIGHT.values()))
    #         else:  # plan_dropdown.value == "right"
    #             if anchor_traj is not None:
    #                 if play_handle.value < anchor_traj.shape[1]:
    #                     robot.q_left = anchor_traj[traj_handle.value][play_handle.value].view(-1)
    #                     robot.q_right = torch.Tensor(list(YUMI_REST_POSE_RIGHT.values()))
    #                 elif play_handle.value < anchor_traj.shape[1] + approach_traj.shape[1]:
    #                     robot.q_right = approach_traj[traj_handle.value][play_handle.value - anchor_traj.shape[1]].view(-1)
    #                     robot.q_left = anchor_traj[traj_handle.value][-1].view(-1)
    #                 else:
    #                     robot.q_right = waypoint_traj[traj_handle.value][play_handle.value - (anchor_traj.shape[1] + approach_traj.shape[1])].view(-1)
    #                     robot.q_left = anchor_traj[traj_handle.value][-1].view(-1)
    #             else:
    #                 if play_handle.value < approach_traj.shape[1]:
    #                     robot.q_right = approach_traj[traj_handle.value][play_handle.value].view(-1)
    #                     robot.q_left = torch.Tensor(list(YUMI_REST_POSE_LEFT.values()))
    #                 else:
    #                     robot.q_right = waypoint_traj[traj_handle.value][play_handle.value - approach_traj.shape[1]].view(-1)
    #                     robot.q_left = torch.Tensor(list(YUMI_REST_POSE_LEFT.values()))

    #     @traj_handle.on_update
    #     def _(_):
    #         move_to_traj_position()
    #     @play_handle.on_update
    #     def _(_):
    #         move_to_traj_position()

    #     move_to_traj_position()

    #     @real_button.on_click
    #     def _(_):
    #         if robot.real is None:
    #             print("Real robot is not available.")
    #             return
    #         if waypoint_traj is None or traj_handle is None or play_handle is None:
    #             print("No trajectory available.")
    #             return
    #         if plan_dropdown.value == "left":
    #             if anchor_traj is not None:
    #                 robot.real.right.open_gripper()
    #                 robot.real.right.move_joints_traj(
    #                     joints=anchor_traj[traj_handle.value][:, :7].cpu().numpy().astype(np.float64),
    #                     speed=REAL_ROBOT_SPEED
    #                 )
    #                 time.sleep(1)
    #                 robot.real.right.sync()
    #                 robot.real.right.close_gripper()

    #             robot.real.left.open_gripper()
    #             robot.real.left.move_joints_traj(
    #                 joints=approach_traj[traj_handle.value][:, :7].cpu().numpy().astype(np.float64),
    #                 speed=REAL_ROBOT_SPEED
    #             )
    #             time.sleep(1)
    #             robot.real.left.sync()
    #             robot.real.left.close_gripper()
    #             robot.real.left.move_joints_traj(
    #                 joints=waypoint_traj[traj_handle.value][:, :7].cpu().numpy().astype(np.float64),
    #                 speed=REAL_ROBOT_SPEED
    #             )
    #             time.sleep(1)
    #             robot.real.left.sync()
    #             robot.real.left.open_gripper()

    #         if plan_dropdown.value == "right":
    #             if anchor_traj is not None:
    #                 robot.real.left.open_gripper()
    #                 robot.real.left.move_joints_traj(
    #                     joints=anchor_traj[traj_handle.value][:, :7].cpu().numpy().astype(np.float64),
    #                     speed=REAL_ROBOT_SPEED
    #                 )
    #                 time.sleep(1)
    #                 robot.real.left.sync()
    #                 robot.real.left.close_gripper()

    #             robot.real.right.open_gripper()
    #             robot.real.right.move_joints_traj(
    #                 joints=approach_traj[traj_handle.value][:, :7].cpu().numpy().astype(np.float64),
    #                 speed=REAL_ROBOT_SPEED
    #             )

    #             time.sleep(1)
    #             robot.real.right.sync()
    #             robot.real.right.close_gripper()
    #             robot.real.right.move_joints_traj(
    #                 joints=waypoint_traj[traj_handle.value][:, :7].cpu().numpy().astype(np.float64),
    #                 speed=REAL_ROBOT_SPEED
    #             )
    #             time.sleep(1)
    #             robot.real.right.sync()
    #             robot.real.right.open_gripper()
    #             robot.real.right.sync()

    while True:
        left, right, depth = zed.get_frame() # internally gets the lock.

        toad_opt.set_frame(left,depth)
        # with zed.raft_lock:
        #     # toad_opt.step_opt(niter=50)
        #     update_keyframe(keyframe_handle.value)

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