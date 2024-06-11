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
from toad.toad_object import GraspableToadObject

from yumi_toad import create_objects_in_scene, create_zed_and_toad

def get_grasp_paths(
    toad_opt: ToadOptimizer,
    part_handle_value: int,
    anchor_handle_value: int,
    robot: YumiRobot,
    cam_vtf: vtf.SE3,
):
    # Get the grasp path for the selected part.
    toad_obj = toad_opt.toad_object
    moving_grasp = toad_obj.grasps[part_handle_value]
    moving_grasps_gripper = toad_obj.to_gripper_frame(moving_grasp, robot.tooltip_to_gripper)
    moving_grasp_path_wxyz_xyz_gripper = torch.cat([
        torch.Tensor(
            cam_vtf
            .multiply(toad_opt.get_parts2cam(keyframe=keyframe_idx)[part_handle_value])
            .multiply(moving_grasps_gripper)
            .wxyz_xyz
        ).unsqueeze(1)  # [num_grasps, 7] -> [num_grasps, 1, 7]
        for keyframe_idx in range(len(toad_opt.optimizer.keyframes))
    ], dim=1).float().cuda() # [num_grasps, num_keyframes, 7]

    moving_grasp = toad_obj.grasps[part_handle_value]
    moving_grasps_gripper = toad_obj.to_gripper_frame(moving_grasp)
    moving_grasp_path_wxyz_xyz_tt = torch.cat([
        torch.Tensor(
            cam_vtf
            .multiply(toad_opt.get_parts2cam(keyframe=keyframe_idx)[part_handle_value])
            .multiply(moving_grasps_gripper)
            .wxyz_xyz
        ).unsqueeze(1)  # [num_grasps, 7] -> [num_grasps, 1, 7]
        for keyframe_idx in range(len(toad_opt.optimizer.keyframes))
    ], dim=1).float().cuda() # [num_grasps, num_keyframes, 7]

    # if anchor_handle_value != -1:
    anchor_grasp = toad_obj.grasps[anchor_handle_value]
    anchor_grasps_gripper = toad_obj.to_gripper_frame(anchor_grasp, robot.tooltip_to_gripper)
    anchor_grasp_path_wxyz_xyz_gripper = torch.cat([
        torch.Tensor(
            cam_vtf
            .multiply(toad_opt.get_parts2cam(keyframe=keyframe_idx)[anchor_handle_value])
            .multiply(anchor_grasps_gripper)
            .wxyz_xyz
        ).unsqueeze(1)  # [num_grasps, 7] -> [num_grasps, 1, 7]
        for keyframe_idx in range(len(toad_opt.optimizer.keyframes))
    ], dim=1).float().cuda() # [num_grasps, num_keyframes, 7]

    anchor_grasp = toad_obj.grasps[anchor_handle_value]
    anchor_grasps_gripper = toad_obj.to_gripper_frame(anchor_grasp)
    anchor_grasp_path_wxyz_xyz_tt = torch.cat([
        torch.Tensor(
            cam_vtf
            .multiply(toad_opt.get_parts2cam(keyframe=keyframe_idx)[anchor_handle_value])
            .multiply(anchor_grasps_gripper)
            .wxyz_xyz
        ).unsqueeze(1)  # [num_grasps, 7] -> [num_grasps, 1, 7]
        for keyframe_idx in range(len(toad_opt.optimizer.keyframes))
    ], dim=1).float().cuda() # [num_grasps, num_keyframes, 7]

    return (
        moving_grasp_path_wxyz_xyz_gripper,
        moving_grasp_path_wxyz_xyz_tt,
        anchor_grasp_path_wxyz_xyz_gripper,
        anchor_grasp_path_wxyz_xyz_tt,
    )


def main(
    config_path: Path = Path("outputs/nerfgun_poly_far/dig/2024-06-02_234451/config.yml"),
    keyframe_path: Path = Path("renders/nerfgun_poly_far/keyframes.pt")
    # config_path: Path = Path("outputs/garfield_poly/dig/2024-06-03_183227/config.yml"),
    # keyframe_path: Path = Path("renders/garfield_poly/keyframes.pt")
    # config_path: Path = Path("outputs/sunglasses3/dig/2024-06-03_175202/config.yml"),
    # keyframe_path: Path = Path("renders/sunglasses3/keyframes.pt")
    # config_path: Path = Path("outputs/scissors/dig/2024-06-03_135548/config.yml"),
    # keyframe_path: Path = Path("renders/scissors/keyframes.pt")
    # config_path: Path = Path("outputs/wooden_drawer/dig/2024-06-03_160055/config.yml"),
    # keyframe_path: Path = Path("renders/wooden_drawer/keyframes.pt")
    # config_path: Path = Path("outputs/painter_t/dig/2024-06-05_103134/config.yml"),
    # keyframe_path: Path = Path("renders/painter_t/keyframes.pt")
    # config_path: Path = Path("outputs/cal_bear_naked/dig/2024-06-04_215034/config.yml"),
    # keyframe_path: Path = Path("renders/cal_bear_naked/keyframes.pt")
    # config_path: Path = Path("outputs/big_painter_t/dig/2024-06-05_122620/config.yml"),
    # keyframe_path: Path = Path("renders/big_painter_t/keyframes.pt")
    # config_path: Path = Path("outputs/buddha_empty/dig/2024-06-02_224243/config.yml"),
    # keyframe_path: Path = Path("renders/buddha_empty/keyframes.pt")
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
    real_button = server.gui.add_button("Move real robot")
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

        (
            moving_grasp_path_wxyz_xyz_gripper,
            moving_grasp_path_wxyz_xyz_tt,
            anchor_grasp_path_wxyz_xyz_gripper,
            anchor_grasp_path_wxyz_xyz_tt,
        ) = get_grasp_paths(
            toad_opt=toad_opt,
            part_handle_value=part_handle.value,
            anchor_handle_value=anchor_handle.value,
            robot=robot,
            cam_vtf=cam_vtf,
        )

        approach_offsets = vtf.SE3.from_translation(np.array([[0, 0, d] for d in np.linspace(-0.08, 0.0, 10)]))
        moving_grasp_path_wxyz_xyz_gripper_approach = np.zeros((moving_grasp_path_wxyz_xyz_gripper.shape[0], 10, 7))
        anchor_grasp_path_wxyz_xyz_gripper_approach = np.zeros((anchor_grasp_path_wxyz_xyz_gripper.shape[0], 10, 7))
        for i in range(moving_grasp_path_wxyz_xyz_gripper.shape[0]):
            moving_grasp_path_wxyz_xyz_gripper_approach[i] = vtf.SE3(moving_grasp_path_wxyz_xyz_gripper[i, 0].cpu().numpy()).multiply(approach_offsets).wxyz_xyz
            anchor_grasp_path_wxyz_xyz_gripper_approach[i] = vtf.SE3(anchor_grasp_path_wxyz_xyz_gripper[i, 0].cpu().numpy()).multiply(approach_offsets).wxyz_xyz
        anchor_grasp_path_wxyz_xyz_gripper_approach = torch.Tensor(anchor_grasp_path_wxyz_xyz_gripper_approach).float().cuda()
        moving_grasp_path_wxyz_xyz_gripper_approach = torch.Tensor(moving_grasp_path_wxyz_xyz_gripper_approach).float().cuda()


        with zed.raft_lock:
            robot.plan.activate_arm("left", locked_arm_q=robot.q_right_home)
            l_moving_ik, l_moving_ik_succ = robot.plan.ik(
                goal_wxyz_xyz=moving_grasp_path_wxyz_xyz_tt[:, 0],
            )
            l_anchor_ik, l_anchor_ik_succ = robot.plan.ik(
                goal_wxyz_xyz=anchor_grasp_path_wxyz_xyz_tt[:, 0],
            )
            robot.plan.activate_arm("right", locked_arm_q=robot.q_left_home)
            r_moving_ik, r_moving_ik_succ = robot.plan.ik(
                goal_wxyz_xyz=moving_grasp_path_wxyz_xyz_tt[:, 0],
            )
            r_anchor_ik, r_anchor_ik_succ= robot.plan.ik(
                goal_wxyz_xyz=anchor_grasp_path_wxyz_xyz_tt[:, 0],
            )

        # ... should cache this?
        with zed.raft_lock:
            found_traj = False
            if (l_moving_ik is not None and r_anchor_ik is not None and l_moving_ik_succ is not None and r_anchor_ik_succ is not None):
                for l_idx in range(l_moving_ik.shape[0]):
                    if not l_moving_ik_succ[l_idx]:
                        continue
                    if found_traj:
                        break
                    for r_idx in range(r_anchor_ik.shape[0]):
                        if not r_anchor_ik_succ[r_idx]:
                            continue
                        if found_traj:
                            break
                        d_world, d_self = robot.plan.in_collision_full(
                            robot.concat_joints(r_anchor_ik[r_idx], l_moving_ik[l_idx])
                        )
                        print("self-collision")
                        if d_self <= 0:
                            # check if waypoint works.

                            l_waypoint_traj, l_waypoint_succ = robot.plan_jax.plan_from_waypoints(
                                poses=torch.cat([moving_grasp_path_wxyz_xyz_gripper_approach[l_idx], moving_grasp_path_wxyz_xyz_gripper[l_idx]]).unsqueeze(0),
                                arm="left",
                            )
                            r_waypoint_traj, r_waypoint_succ = robot.plan_jax.plan_from_waypoints(
                                poses=torch.cat([anchor_grasp_path_wxyz_xyz_gripper_approach[l_idx], anchor_grasp_path_wxyz_xyz_gripper[l_idx]]).unsqueeze(0),
                                arm="right",
                            )
                            
                            # check that the approach part is collision-free.
                            _approach_traj = robot.concat_joints(r_waypoint_traj.squeeze(), l_waypoint_traj.squeeze())[:moving_grasp_path_wxyz_xyz_gripper_approach.shape[1]]
                            d_world, d_self = robot.plan.in_collision_full(_approach_traj)
                            if (d_world > 0).any():
                                continue

                            if not (l_waypoint_succ & r_waypoint_succ).any():
                                continue

                            # check if the waypoints cause the arms to collide with one another.
                            waypoint_traj = robot.concat_joints(r_waypoint_traj.squeeze(), l_waypoint_traj.squeeze())
                            d_world, d_self = robot.plan.in_collision_full(waypoint_traj)

                            if (d_self > 0).any():
                                continue

                            robot.plan.activate_arm("left", locked_arm_q=robot.q_right_home)
                            l_approach_traj, l_approach_succ = robot.plan.gen_motion_from_goal_joints(
                                goal_js=l_waypoint_traj[0, 0, :].unsqueeze(0),  # [num_traj, 8]
                                q_init=robot.q_left_home.view(-1, 8),
                            )
                            robot.plan.activate_arm("right", locked_arm_q=robot.q_left_home)
                            r_approach_traj, r_approach_succ = robot.plan.gen_motion_from_goal_joints(
                                goal_js=r_waypoint_traj[0, 0, :].unsqueeze(0),  # [num_traj, 8]
                                q_init=robot.q_right_home.view(-1, 8),
                            )

                            if (l_approach_succ is None or r_approach_succ is None):
                                continue
                            if not (l_approach_succ & r_approach_succ).any():
                                continue

                            approach_traj = robot.concat_joints(r_approach_traj.squeeze(), l_approach_traj.squeeze())
                            d_world, d_self = robot.plan.in_collision_full(approach_traj)

                            if (d_self > 0).any():
                                continue

                            if (
                                l_approach_succ is not None and l_approach_traj is not None and 
                                r_approach_succ is not None and r_approach_traj is not None and
                                l_approach_succ.all() and r_approach_succ.all()
                            ):
                                # ... it has succeeded.
                                traj = torch.cat([approach_traj, waypoint_traj], dim=0)
                                print("Found working traj")
                                # yield traj
                                found_traj = True

            if (l_anchor_ik is not None and r_moving_ik is not None and l_anchor_ik_succ is not None and r_moving_ik_succ is not None):
                for l_idx in range(l_anchor_ik.shape[0]):
                    if not l_anchor_ik_succ[l_idx]:
                        continue
                    if found_traj:
                        break
                    for r_idx in range(r_moving_ik.shape[0]):
                        if not r_moving_ik_succ[r_idx]:
                            continue
                        if found_traj:
                            break
                        d_world, d_self = robot.plan.in_collision_full(
                            robot.concat_joints(r_moving_ik[r_idx], l_anchor_ik[l_idx])
                        )
                        if d_self <= 0:
                            # check if waypoint works.
                            l_waypoint_traj, l_waypoint_succ = robot.plan_jax.plan_from_waypoints(
                                poses=torch.cat([anchor_grasp_path_wxyz_xyz_gripper_approach[l_idx], anchor_grasp_path_wxyz_xyz_gripper[l_idx]]).unsqueeze(0),
                                arm="left",
                            )
                            r_waypoint_traj, r_waypoint_succ = robot.plan_jax.plan_from_waypoints(
                                poses=torch.cat([moving_grasp_path_wxyz_xyz_gripper_approach[l_idx], moving_grasp_path_wxyz_xyz_gripper[l_idx]]).unsqueeze(0),
                                arm="right",
                            )

                            if not (l_waypoint_succ & r_waypoint_succ).any():
                                continue

                            # check if the waypoints cause the arms to collide with one another.
                            waypoint_traj = robot.concat_joints(r_waypoint_traj.squeeze(), l_waypoint_traj.squeeze())
                            d_world, d_self = robot.plan.in_collision_full(waypoint_traj)

                            if (d_self > 0).any():
                                continue

                            robot.plan.activate_arm("left", locked_arm_q=robot.q_right_home)
                            l_approach_traj, l_approach_succ = robot.plan.gen_motion_from_goal_joints(
                                goal_js=l_waypoint_traj[0, 0, :].unsqueeze(0),  # [num_traj, 8]
                                q_init=robot.q_left_home.view(-1, 8),
                            )
                            robot.plan.activate_arm("right", locked_arm_q=robot.q_left_home)
                            r_approach_traj, r_approach_succ = robot.plan.gen_motion_from_goal_joints(
                                goal_js=r_waypoint_traj[0, 0, :].unsqueeze(0),  # [num_traj, 8]
                                q_init=robot.q_right_home.view(-1, 8),
                            )

                            if (l_approach_succ is None or r_approach_succ is None):
                                continue
                            if not (l_approach_succ & r_approach_succ).any():
                                continue

                            approach_traj = robot.concat_joints(r_approach_traj.squeeze(), l_approach_traj.squeeze())
                            d_world, d_self = robot.plan.in_collision_full(approach_traj)

                            if (d_self > 0).any():
                                continue

                            if (
                                l_approach_succ is not None and l_approach_traj is not None and 
                                r_approach_succ is not None and r_approach_traj is not None and
                                l_approach_succ.all() and r_approach_succ.all()
                            ):
                                # ... it has succeeded.
                                traj = torch.cat([approach_traj, waypoint_traj], dim=0)
                                print("Found working traj")
                                # yield traj
                                found_traj = True

        goal_button.disabled = False
        if found_traj is not True:
            print("No working trajectory found.")
            return
        assert traj is not None
        assert len(traj.shape) == 2 and traj.shape[-1] == 16

        play_handle = server.gui.add_slider("play", min=0, max=traj.shape[0]-1, step=1, initial_value=0)
        approach_len = robot.plan._motion_gen.interpolation_steps

        def move_to_traj_position():
            assert traj is not None and play_handle is not None
            # robot.q_left = traj[traj_handle.value][play_handle.value].view(-1)
            robot.q_all = traj[play_handle.value].view(-1)
            if play_handle.value >= approach_len:
                update_keyframe(play_handle.value - approach_len)

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

            robot.real.left.open_gripper()
            robot.real.right.open_gripper()

            robot.real.move_joints_sync(
                l_joints = robot.get_left(traj)[:approach_len, :7].cpu().numpy().astype(np.float64),
                r_joints = robot.get_right(traj)[:approach_len, :7].cpu().numpy().astype(np.float64),
                speed=REAL_ROBOT_SPEED
            )
            time.sleep(1)
            robot.real.left.sync()
            robot.real.right.sync()

            robot.real.left.close_gripper()
            robot.real.right.close_gripper()

            robot.real.move_joints_sync(
                l_joints = robot.get_left(traj)[approach_len:, :7].cpu().numpy().astype(np.float64),
                r_joints = robot.get_right(traj)[approach_len:, :7].cpu().numpy().astype(np.float64),
                speed=REAL_ROBOT_SPEED
            )
            time.sleep(1)
            robot.real.left.sync()
            robot.real.right.sync()

            robot.real.left.open_gripper()
            robot.real.right.open_gripper()
            
            robot.real.move_joints_sync(
                l_joints = robot.get_left(traj)[:approach_len, :7].flip(0).cpu().numpy().astype(np.float64),
                r_joints = robot.get_right(traj)[:approach_len, :7].flip(0).cpu().numpy().astype(np.float64),
                speed=REAL_ROBOT_SPEED
            )
            time.sleep(1)
            robot.real.left.sync()
            robot.real.right.sync()

    while True:
        left, right, depth = zed.get_frame() # internally gets the lock.
        toad_opt.set_frame(left,depth)

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