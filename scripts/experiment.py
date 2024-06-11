import time
from pathlib import Path
from datetime import datetime
from typing import Literal, Tuple, Optional
from toad.zed import Zed
from autolab_core import RigidTransform
import viser
import numpy as np
import viser.transforms as vtf
import torch
from toad.yumi.yumi_robot import YumiRobot, REAL_ROBOT_SPEED, YUMI_REST_POSE_LEFT, YUMI_REST_POSE_RIGHT
from toad.toad_optimizer import ToadOptimizer
from yumi_toad import create_objects_in_scene
from yumi_toad_2 import get_grasp_paths
import json
import tyro
import concurrent.futures
# from multiprocessing.pool import ThreadPool, Pool
from threading import Thread

# Save results to:
# exps/{experiment_name}/{timestep}/...
# - init_zed.svo (a brief capture of the initial state, for 1 second).
# - traj.svo (taken during physical robot execution).
# - log.txt (success/failure. time taken for path generation. init success.)

EXPERIMENTS = {
    "nerfgun" : (
        "outputs/nerfgun_poly_far/dig/2024-06-02_234451/config.yml",
        "renders/nerfgun_poly_far/keyframes.pt",
        "bimanual"
        ),
    "eyeglasses": (
        "outputs/sunglasses3/dig/2024-06-03_175202/config.yml",
        "renders/sunglasses3/keyframes.pt",
        "bimanual"
        ),
    "scissors": (
        "outputs/scissors/dig/2024-06-03_135548/config.yml",
        "renders/scissors/keyframes.pt",
        "bimanual"
        ),
    "bear": (
        "outputs/cal_bear_naked/dig/2024-06-04_215034/config.yml",
        "renders/cal_bear_naked/keyframes.pt",
        "bimanual"
        ),
    "red_box": (
        "outputs/buddha_empty/dig/2024-06-02_224243/config.yml",
        "renders/buddha_empty/keyframes.pt",
        "single"
    ),
    "painter": (
        "outputs/big_painter_t/dig/2024-06-05_122620/config.yml",
        "renders/big_painter_t/keyframes.pt",
        "bimanual"
    ),
    # "wooden_drawer": (
    #     "outputs/wooden_drawer/dig/2024-06-03_160055/config.yml",
    #     "renders/wooden_drawer/keyframes.pt"
    #     ),
}


def generate_path_single(
    toad_opt: ToadOptimizer,
    cam_vtf: vtf.SE3,
    robot: YumiRobot,
    part_idx: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
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
        _,
        _,
    ) = get_grasp_paths(
        toad_opt=toad_opt,
        part_handle_value=part_idx,
        anchor_handle_value=-1,
        robot=robot,
        cam_vtf=cam_vtf,
    )

    # Calculate the approach path.
    approach_offsets = vtf.SE3.from_translation(np.array([[0, 0, d] for d in np.linspace(-0.08, 0.0, 10)]))
    moving_grasp_path_wxyz_xyz_gripper_approach = np.zeros((moving_grasp_path_wxyz_xyz_gripper.shape[0], 10, 7))
    for i in range(moving_grasp_path_wxyz_xyz_gripper.shape[0]):
        moving_grasp_path_wxyz_xyz_gripper_approach[i] = vtf.SE3(moving_grasp_path_wxyz_xyz_gripper[i, 0].cpu().numpy()).multiply(approach_offsets).wxyz_xyz
    moving_grasp_path_wxyz_xyz_gripper_approach = torch.Tensor(moving_grasp_path_wxyz_xyz_gripper_approach).float().cuda()

    robot.plan.activate_arm("left", locked_arm_q=robot.q_right_home)
    l_moving_ik, l_moving_ik_succ = robot.plan.ik(
        goal_wxyz_xyz=moving_grasp_path_wxyz_xyz_tt[:, 0],
    )
    robot.plan.activate_arm("right", locked_arm_q=robot.q_left_home)
    r_moving_ik, r_moving_ik_succ = robot.plan.ik(
        goal_wxyz_xyz=moving_grasp_path_wxyz_xyz_tt[:, 0],
    )

    traj, approach_traj, waypoint_traj = None, None, None
    found_traj = False
    d_world_thresh = 0.005
    if l_moving_ik is not None and l_moving_ik_succ is not None:
        for l_idx in range(l_moving_ik.shape[0]):
            if not l_moving_ik_succ[l_idx]:
                continue
            if found_traj:
                break

            q_right_rest = robot.q_right_home.view(l_moving_ik[l_idx].shape).cuda()
            d_world, d_self = robot.plan.in_collision_full(
                robot.concat_joints(q_right_rest, l_moving_ik[l_idx])
            )
            if d_self <= 0 and d_world <= d_world_thresh:
                # check if waypoint works.
                l_waypoint_traj, l_waypoint_succ = robot.plan_jax.plan_from_waypoints(
                    poses=torch.cat(
                        [
                            moving_grasp_path_wxyz_xyz_gripper_approach[l_idx],
                            moving_grasp_path_wxyz_xyz_gripper[l_idx],
                        ]
                    ).unsqueeze(0),
                    arm="left",
                )

                # check that the approach part is collision-free.
                q_right_rest = robot.q_right_home.view(-1, 8).expand(l_waypoint_traj.shape[1], -1).cuda()
                _approach_traj = robot.concat_joints(
                    q_right_rest, l_waypoint_traj.squeeze()
                )[: moving_grasp_path_wxyz_xyz_gripper_approach.shape[1]]
                d_world, d_self = robot.plan.in_collision_full(_approach_traj)
                if (d_world > d_world_thresh).any():
                    continue

                if not l_waypoint_succ.any():
                    continue

                # check if the waypoints cause the arms to collide with one another.
                waypoint_traj = robot.concat_joints(q_right_rest, l_waypoint_traj.squeeze())
                d_world, d_self = robot.plan.in_collision_full(waypoint_traj)

                if (d_self > 0).any():
                    continue

                robot.plan.activate_arm("left", locked_arm_q=robot.q_right_home)
                l_approach_traj, l_approach_succ = robot.plan.gen_motion_from_goal_joints(
                    goal_js=l_waypoint_traj[0, 0, :].unsqueeze(0),  # [num_traj, 8]
                    q_init=robot.q_left_home.view(-1, 8),
                )

                if l_approach_succ is None or l_approach_traj is None:
                    continue
                if not l_approach_succ.any():
                    continue

                q_right_rest = robot.q_right_home.view(-1, 8).expand(l_approach_traj.shape[1], -1).cuda()
                approach_traj = robot.concat_joints(q_right_rest, l_approach_traj.squeeze())
                d_world, d_self = robot.plan.in_collision_full(approach_traj)

                if (d_self > 0).any():
                    continue

                if (
                    l_approach_succ is not None and l_approach_traj is not None and l_approach_succ.all()
                ):
                    # ... it has succeeded.
                    traj = torch.cat([approach_traj, waypoint_traj], dim=0)
                    print("Found working traj")
                    # yield traj
                    found_traj = True
    if r_moving_ik is not None and r_moving_ik_succ is not None:
        for r_idx in range(r_moving_ik.shape[0]):
            if not r_moving_ik_succ[r_idx]:
                continue
            if found_traj:
                break

            q_left_rest = robot.q_left_home.view(-1, 8).cuda()
            d_world, d_self = robot.plan.in_collision_full(
                robot.concat_joints(r_moving_ik[r_idx], q_left_rest)
            )
            if d_self <= 0 and d_world <= d_world_thresh:
                # check if waypoint works.
                r_waypoint_traj, r_waypoint_succ = robot.plan_jax.plan_from_waypoints(
                    poses=torch.cat(
                        [
                            moving_grasp_path_wxyz_xyz_gripper_approach[r_idx],
                            moving_grasp_path_wxyz_xyz_gripper[r_idx],
                        ]
                    ).unsqueeze(0),
                    arm="right",
                )

                # check that the approach part is collision-free.
                q_left_rest = robot.q_left_home.view(-1, 8).expand(r_waypoint_traj.shape[1], -1).cuda()
                _approach_traj = robot.concat_joints(
                    q_left_rest, r_waypoint_traj.squeeze()
                )[: moving_grasp_path_wxyz_xyz_gripper_approach.shape[1]]
                d_world, d_self = robot.plan.in_collision_full(_approach_traj)
                if (d_world > d_world_thresh).any():
                    continue

                if not r_waypoint_succ.any():
                    continue

                # check if the waypoints cause the arms to collide with one another.
                waypoint_traj = robot.concat_joints(r_waypoint_traj.squeeze(), q_left_rest).cuda()
                d_world, d_self = robot.plan.in_collision_full(waypoint_traj)

                if (d_self > 0).any():
                    continue

                robot.plan.activate_arm("right", locked_arm_q=robot.q_left_home)
                r_approach_traj, r_approach_succ = robot.plan.gen_motion_from_goal_joints(
                    goal_js=r_waypoint_traj[0, 0, :].unsqueeze(0),  # [num_traj, 8]
                    q_init=robot.q_right_home.view(-1, 8),
                )

                if r_approach_succ is None:
                    continue
                if not r_approach_succ.any():
                    continue

                q_left_rest = robot.q_left_home.view(-1, 8).expand(r_approach_traj.shape[1], -1).cuda()
                approach_traj = robot.concat_joints( r_approach_traj.squeeze(), q_left_rest)
                d_world, d_self = robot.plan.in_collision_full(approach_traj)

                if (d_self > 0).any():
                    continue

                if (
                    r_approach_succ is not None and r_approach_traj is not None and r_approach_succ.all()
                ):
                    # ... it has succeeded.
                    traj = torch.cat([approach_traj, waypoint_traj], dim=0)
                    print("Found working traj")
                    # yield traj
                    found_traj = True

    if found_traj is not True or traj is None:
        print("No working trajectory found.")
        return None

    assert traj is not None and approach_traj is not None and waypoint_traj is not None
    assert len(traj.shape) == 2 and traj.shape[-1] == 16

    assert len(approach_traj.shape) == 2 and approach_traj.shape[-1] == 16
    assert len(waypoint_traj.shape) == 2 and waypoint_traj.shape[-1] == 16

    # Need to swap around the approach, because that's the first part of the trajectory.
    approach_traj, waypoint_traj = torch.cat([approach_traj, waypoint_traj[:10]], dim=0), waypoint_traj[10:]

    assert len(approach_traj.shape) == 2 and approach_traj.shape[-1] == 16
    assert len(waypoint_traj.shape) == 2 and waypoint_traj.shape[-1] == 16

    return (approach_traj, waypoint_traj)


def generate_path_bimanual(
    toad_opt: ToadOptimizer,
    cam_vtf: vtf.SE3,
    robot: YumiRobot,
    part_idx: int,
    anchor_idx: int,
    lift_traj: bool = False,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
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
        part_handle_value=part_idx,
        anchor_handle_value=anchor_idx,
        robot=robot,
        cam_vtf=cam_vtf,
    )

    # Calculate the approach path.
    approach_offsets = vtf.SE3.from_translation(
        np.array(
            [[0, 0, d] for d in np.linspace(-0.05, 0.0, 10)]
        )
    )
    moving_grasp_path_wxyz_xyz_gripper_approach = np.zeros((moving_grasp_path_wxyz_xyz_gripper.shape[0], 10, 7))
    anchor_grasp_path_wxyz_xyz_gripper_approach = np.zeros((anchor_grasp_path_wxyz_xyz_gripper.shape[0], 10, 7))
    for i in range(moving_grasp_path_wxyz_xyz_gripper.shape[0]):
        moving_grasp_path_wxyz_xyz_gripper_approach[i] = vtf.SE3(moving_grasp_path_wxyz_xyz_gripper[i, 0].cpu().numpy()).multiply(approach_offsets).wxyz_xyz
        anchor_grasp_path_wxyz_xyz_gripper_approach[i] = vtf.SE3(anchor_grasp_path_wxyz_xyz_gripper[i, 0].cpu().numpy()).multiply(approach_offsets).wxyz_xyz
    anchor_grasp_path_wxyz_xyz_gripper_approach = torch.Tensor(anchor_grasp_path_wxyz_xyz_gripper_approach).float().cuda()
    moving_grasp_path_wxyz_xyz_gripper_approach = torch.Tensor(moving_grasp_path_wxyz_xyz_gripper_approach).float().cuda()

    if lift_traj:
        lift_amount = 0.08
        moving_grasp_lift = moving_grasp_path_wxyz_xyz_gripper[:, :1].expand(-1, 10, -1).contiguous()
        moving_grasp_lift[..., 6] = moving_grasp_lift[..., 6] + torch.linspace(0, lift_amount, 10).expand(moving_grasp_lift.shape[0], -1).to(moving_grasp_lift.device)
        moving_grasp_path_wxyz_xyz_gripper[..., 6] = moving_grasp_path_wxyz_xyz_gripper[..., 6] + lift_amount
        moving_grasp_path_wxyz_xyz_gripper = torch.cat([
            moving_grasp_lift,
            moving_grasp_path_wxyz_xyz_gripper,
        ], dim=1)

        moving_grasp_lift = moving_grasp_path_wxyz_xyz_tt[:, :1].expand(-1, 10, -1).contiguous()
        moving_grasp_lift[..., 6] = moving_grasp_lift[..., 6] + torch.linspace(0, lift_amount, 10).expand(moving_grasp_lift.shape[0], -1).to(moving_grasp_lift.device)
        moving_grasp_path_wxyz_xyz_tt[..., 6] = moving_grasp_path_wxyz_xyz_tt[..., 6] + lift_amount
        moving_grasp_path_wxyz_xyz_tt = torch.cat([
            moving_grasp_lift,
            moving_grasp_path_wxyz_xyz_tt,
        ], dim=1)

        anchor_grasp_lift = anchor_grasp_path_wxyz_xyz_gripper[:, :1].expand(-1, 10, -1).contiguous()
        anchor_grasp_lift[..., 6] = anchor_grasp_lift[..., 6] + torch.linspace(0, lift_amount, 10).expand(moving_grasp_lift.shape[0], -1).to(moving_grasp_lift.device)
        anchor_grasp_path_wxyz_xyz_gripper[..., 6] = anchor_grasp_path_wxyz_xyz_gripper[..., 6] + lift_amount
        anchor_grasp_path_wxyz_xyz_gripper = torch.cat([
            anchor_grasp_lift,
            anchor_grasp_path_wxyz_xyz_gripper,
        ], dim=1)

        anchor_grasp_lift = anchor_grasp_path_wxyz_xyz_tt[:, :1].expand(-1, 10, -1).contiguous()
        anchor_grasp_lift[..., 6] = anchor_grasp_lift[..., 6] + torch.linspace(0, lift_amount, 10).expand(moving_grasp_lift.shape[0], -1).to(moving_grasp_lift.device)
        anchor_grasp_path_wxyz_xyz_tt[..., 6] = anchor_grasp_path_wxyz_xyz_tt[..., 6] + lift_amount
        anchor_grasp_path_wxyz_xyz_tt = torch.cat([
            anchor_grasp_lift,
            anchor_grasp_path_wxyz_xyz_tt,
        ], dim=1)

    robot.plan.update_world_objects()

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

    l_moving_ik[..., -1] = 0.025
    l_anchor_ik[..., -1] = 0.025
    r_moving_ik[..., -1] = 0.025
    r_anchor_ik[..., -1] = 0.025

    # robot.plan.update_world_objects(mesh_curobo_config)

    # ... should cache this?
    traj, approach_traj, waypoint_traj = None, None, None
    found_traj = False
    d_world_thresh = 0.005
    # d_world_thresh = 0.02
    l_waypoint_cache, r_waypoint_cache = {}, {}
    if (l_moving_ik is not None and r_anchor_ik is not None and l_moving_ik_succ is not None and r_anchor_ik_succ is not None):
        print(l_moving_ik[l_moving_ik_succ].shape, r_anchor_ik[r_anchor_ik_succ].shape)
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
                print(d_world, d_self)
                if d_self <= 0 and d_world <= d_world_thresh:
                    # check if waypoint works.
                    print("Checking waypoints...")
                    if l_idx in l_waypoint_cache:
                        l_waypoint_traj, l_waypoint_succ = l_waypoint_cache[l_idx]
                        l_waypoint_traj, l_waypoint_succ = l_waypoint_traj.cuda(), l_waypoint_succ.cuda()
                    else:
                        l_waypoint_traj, l_waypoint_succ = robot.plan_jax.plan_from_waypoints(
                            poses=torch.cat([moving_grasp_path_wxyz_xyz_gripper_approach[l_idx], moving_grasp_path_wxyz_xyz_gripper[l_idx]]).unsqueeze(0),
                            arm="left",
                        )
                        l_waypoint_cache[l_idx] = (l_waypoint_traj.cpu(), l_waypoint_succ.cpu())

                    if r_idx in r_waypoint_cache:
                        r_waypoint_traj, r_waypoint_succ = r_waypoint_cache[r_idx]
                        r_waypoint_succ, r_waypoint_traj = r_waypoint_succ.cuda(), r_waypoint_traj.cuda()
                    else:
                        r_waypoint_traj, r_waypoint_succ = robot.plan_jax.plan_from_waypoints(
                            poses=torch.cat([anchor_grasp_path_wxyz_xyz_gripper_approach[r_idx], anchor_grasp_path_wxyz_xyz_gripper[r_idx]]).unsqueeze(0),
                            arm="right",
                        )
                        r_waypoint_cache[r_idx] = (r_waypoint_traj.cpu(), r_waypoint_succ.cpu())

                    l_waypoint_traj[..., -1] = 0.025 # gripper fully open.
                    r_waypoint_traj[..., -1] = 0.025 # gripper fully open.
                    # check that the approach part is collision-free.
                    _approach_traj = robot.concat_joints(r_waypoint_traj.squeeze(), l_waypoint_traj.squeeze())[:moving_grasp_path_wxyz_xyz_gripper_approach.shape[1]]
                    d_world, d_self = robot.plan.in_collision_full(_approach_traj)
                    if (d_world > d_world_thresh).any():
                        print("World coll fail...")
                        continue

                    if not (l_waypoint_succ & r_waypoint_succ).any():
                        print("Waypoint fail...")
                        continue

                    # check if the waypoints cause the arms to collide with one another.
                    waypoint_traj = robot.concat_joints(r_waypoint_traj.squeeze(), l_waypoint_traj.squeeze())
                    d_world, d_self = robot.plan.in_collision_full(waypoint_traj)

                    if (d_self > 0).any():
                        print("self coll waypoint fail...")
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
                        print("Approach fail...")
                        continue
                    if not (l_approach_succ & r_approach_succ).any():
                        print("Approach fail...")
                        continue

                    approach_traj = robot.concat_joints(r_approach_traj.squeeze(), l_approach_traj.squeeze())
                    d_world, d_self = robot.plan.in_collision_full(approach_traj)

                    if (d_self > 0).any():
                        continue
                    if (d_world > 0).any():
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

    l_waypoint_cache, r_waypoint_cache = {}, {}
    if (l_anchor_ik is not None and r_moving_ik is not None and l_anchor_ik_succ is not None and r_moving_ik_succ is not None):
        print(l_anchor_ik[l_anchor_ik_succ].shape, r_moving_ik[r_moving_ik_succ].shape)
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
                if d_self <= 0 and d_world <= d_world_thresh:
                    # check if waypoint works.
                    print("Checking waypoints...")
                    l_waypoint_traj, l_waypoint_succ = robot.plan_jax.plan_from_waypoints(
                        poses=torch.cat([anchor_grasp_path_wxyz_xyz_gripper_approach[l_idx], anchor_grasp_path_wxyz_xyz_gripper[l_idx]]).unsqueeze(0),
                        arm="left",
                    )
                    r_waypoint_traj, r_waypoint_succ = robot.plan_jax.plan_from_waypoints(
                        poses=torch.cat([moving_grasp_path_wxyz_xyz_gripper_approach[r_idx], moving_grasp_path_wxyz_xyz_gripper[r_idx]]).unsqueeze(0),
                        arm="right",
                    )

                    if not (l_waypoint_succ & r_waypoint_succ).any():
                        print("Waypoint fail...")
                        continue

                    l_waypoint_traj[..., -1] = 0.025 # gripper fully open.
                    r_waypoint_traj[..., -1] = 0.025 # gripper fully open.

                    # check that the approach part is collision-free.
                    _approach_traj = robot.concat_joints(r_waypoint_traj.squeeze(), l_waypoint_traj.squeeze())[:moving_grasp_path_wxyz_xyz_gripper_approach.shape[1]]
                    d_world, d_self = robot.plan.in_collision_full(_approach_traj)
                    if (d_world > d_world_thresh).any():
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
                    if (d_world > 0).any():
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

    if found_traj is not True or traj is None:
        print("No working trajectory found.")
        return None

    assert traj is not None and approach_traj is not None and waypoint_traj is not None
    assert len(traj.shape) == 2 and traj.shape[-1] == 16

    assert len(approach_traj.shape) == 2 and approach_traj.shape[-1] == 16
    assert len(waypoint_traj.shape) == 2 and waypoint_traj.shape[-1] == 16

    # Need to swap around the approach, because that's the first part of the trajectory.
    # if lift_traj:
    #     approach_traj, waypoint_traj = torch.cat([approach_traj, waypoint_traj[:20]], dim=0), waypoint_traj[20:]
    # else:
    approach_traj, waypoint_traj = torch.cat([approach_traj, waypoint_traj[:10]], dim=0), waypoint_traj[10:]

    assert len(approach_traj.shape) == 2 and approach_traj.shape[-1] == 16
    assert len(waypoint_traj.shape) == 2 and waypoint_traj.shape[-1] == 16

    return (approach_traj, waypoint_traj)


def run_real_robot(
    robot: YumiRobot,
    approach_traj: torch.Tensor,
    waypoint_traj: torch.Tensor,
): 
    print("moving robot...")
    robot.real.left.open_gripper(); robot.real.right.open_gripper()
    robot.real.move_joints_sync(
        l_joints = robot.get_left(approach_traj)[:, :7].cpu().numpy().astype(np.float64),
        r_joints = robot.get_right(approach_traj)[:, :7].cpu().numpy().astype(np.float64),
        speed=REAL_ROBOT_SPEED
    ); time.sleep(1)
    robot.real.left.sync(); robot.real.right.sync()

    robot.real.left.close_gripper(); robot.real.right.close_gripper()
    robot.real.move_joints_sync(
        l_joints = robot.get_left(waypoint_traj)[:, :7].cpu().numpy().astype(np.float64),
        r_joints = robot.get_right(waypoint_traj)[:, :7].cpu().numpy().astype(np.float64),
        speed=REAL_ROBOT_SPEED
    ); time.sleep(1)
    robot.real.left.sync(); robot.real.right.sync()

    robot.real.left.move_gripper(0.01); robot.real.right.move_gripper(0.01)

    robot.real.move_joints_sync(
        l_joints = robot.get_left(approach_traj)[:, :7].flip(0).cpu().numpy().astype(np.float64),
        r_joints = robot.get_right(approach_traj)[:, :7].flip(0).cpu().numpy().astype(np.float64),
        speed=REAL_ROBOT_SPEED
    ); time.sleep(1)
    robot.real.left.sync(); robot.real.right.sync()


def async_record_zed(
    zed: Zed,
    done: list[bool],
):
    assert len(done) == 1
    assert done[0] is False
    while not done[0]:
        zed.get_frame(depth=False)
        time.sleep(1 / 10.0)
    print("Exited async_record_zed!! Nice!!")


def start_trial(
    exp_name: str,
    toad_opt: ToadOptimizer,
    zed: Zed,
    cam_vtf: vtf.SE3,
    robot: YumiRobot,
    server: viser.ViserServer,
) -> None:
    # Stuff that's different for all trials, remain here (e.g., re-initializing rigidoptimizer.)
    curr_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_dir = Path("exps") / exp_name / curr_time

    # Things we want to log, qualitatively:
    # - whether initial state looks good (GT matches estimate).
    # - whether a valid trajectory was generated.
    # - whether the trajectory was executed successfully.
    # - part idx.
    # Things we want to log, quantitatively:
    # - objreg2objinit (can be used to recover obj2cam).
    # - trajectory planned (if generated).

    init_gui = server.gui.add_dropdown("Init state", ("YES", "NO", ""), initial_value="", disabled=True)
    valid_traj_gui = server.gui.add_dropdown("Valid traj", ("YES", "NO", ""), initial_value="", disabled=True)
    succ_traj_gui = server.gui.add_dropdown("Successful traj", ("YES", "NO", ""), initial_value="", disabled=True)
    exec_traj_button = server.gui.add_button("Execute traj", disabled=True)

    planned_traj = None
    part_idx, anchor_idx = None, None

    def cleanup():
        nonlocal init_gui, valid_traj_gui, succ_traj_gui, exec_traj_button
        # Record the data we wanted earlier, and write it as a json file.
        data = {
            "init_state": init_gui.value,  # I.e., whether the initial pose est is aligned with real.
            "valid_traj": valid_traj_gui.value,  # I.e., whether the trajectory was valid.
            "succ_traj": succ_traj_gui.value,  # I.e., whether the trajectory successfully manipulated the obj.
            "objreg2objinit": toad_opt.optimizer.objreg2objinit.cpu().numpy().tolist(),
            "planned_traj": planned_traj.cpu().numpy().tolist() if planned_traj is not None else [],
            "part_idx": part_idx if part_idx is not None else "",
            "anchor_idx": anchor_idx if anchor_idx is not None else "",
        }
        log_file = log_dir / "log.json"
        log_file.write_text(json.dumps(data, indent=4))

        # Clean up the gui elements.
        init_gui.remove()
        valid_traj_gui.remove()
        succ_traj_gui.remove()
        exec_traj_button.remove()

    # Create the directory.
    log_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the object pose -- first, resetting the RigidOptimizer.
    toad_opt.reset_optimizer()
    toad_opt.initialized = False

    l, _, depth = zed.get_frame(depth=True)  # type: ignore
    toad_opt.set_frame(l,depth)
    print("Running camopt...", end="", flush=True)
    with zed.raft_lock:
        toad_opt.init_obj_pose()
    print("done.")
    toad_opt.optimizer.load_trajectory(EXPERIMENTS[exp_name][1])
    frame_list = create_objects_in_scene(
        server, toad_opt, object_frame_name="camera/object", create_gui=False
    )

    # Add pointcloud to the scene, to check mesh alignment.
    left, right, depth = zed.get_frame()  # internally gets the lock.
    K = torch.from_numpy(zed.get_K()).float().cuda()
    assert isinstance(left, torch.Tensor) and isinstance(depth, torch.Tensor)
    points, colors = Zed.project_depth(left, depth, K)
    server.scene.add_point_cloud(
        "camera/points",
        points=points,
        colors=colors,
        point_size=0.001,
    )

    # Record zed for one second, for the initial state.
    zed.start_record(str(log_dir / "init_zed.svo"))
    done = [False]
    thread = Thread(target=async_record_zed, args=(zed, done))
    thread.start()
    time.sleep(1)
    done[0] = True
    thread.join()
    zed.stop_record()

    # Check if the initial state looks good.
    print("Waiting for init gui log...")
    init_gui.disabled = False
    while init_gui.value == "":
        time.sleep(1 / 10.0)
    if init_gui.value == "NO":
        cleanup()
        return
    init_gui.disabled = True

    # Generate the motions based on the keyframes.
    # Get the part and anchor indices.
    traj = None
    part_idx, anchor_idx = None, None
    with zed.raft_lock:
        if EXPERIMENTS[exp_name][2] == "single":
            hand_cands = toad_opt.optimizer.compute_single_hand_assignment()
            for curr_part_idx in hand_cands:
                print("Trying part idx:", curr_part_idx)
                traj = generate_path_single(
                    toad_opt, cam_vtf, robot, curr_part_idx,
                )
                if traj is not None:
                    part_idx = curr_part_idx
                    break
        else:
            hand_cands = toad_opt.optimizer.compute_two_hand_assignment()
            for curr_part_idx, curr_anchor_idx in hand_cands:
                print("Trying part idx:", curr_part_idx, "anchor idx:", curr_anchor_idx)
                traj = generate_path_bimanual(
                    toad_opt, cam_vtf, robot, curr_part_idx, curr_anchor_idx,
                )
                if traj is not None:
                    part_idx, anchor_idx = curr_part_idx, curr_anchor_idx
                    break
    if traj is None:
        valid_traj_gui.value = "NO"
        cleanup()
        return

    else:
        valid_traj_gui.disabled = False
        approach_traj, waypoint_traj = traj

        if part_idx is not None:
            _mesh = toad_opt.toad_object.meshes[part_idx]
            _mesh.visual.vertex_colors = [100, 255, 100, 255]  # type: ignore
            server.scene.add_mesh_trimesh(
                "camera/object/group_{idx}/mesh",
                _mesh,
            )
        elif anchor_idx is not None:
            _mesh = toad_opt.toad_object.meshes[anchor_idx]
            _mesh.visual.vertex_colors = [255, 100, 100, 255]  # type: ignore
            server.scene.add_mesh_trimesh(
                "camera/object/group_{idx}/mesh",
                _mesh,
            )

        # Visualize trajectory.
        traj_slider = server.gui.add_slider("Traj", 0, (approach_traj.shape[0] + waypoint_traj.shape[0]) - 1, 1, 0)
        @traj_slider.on_update
        def _(_):
            if traj_slider.value < approach_traj.shape[0]:
                robot.q_all = approach_traj[traj_slider.value].view(-1)
                # Update keyframes.
                part2cam = toad_opt.get_parts2cam(keyframe=0)
                for i in range(len(frame_list)):
                    frame_list[i].position = part2cam[i].translation()
                    frame_list[i].wxyz = part2cam[i].rotation().wxyz
            else:
                keyframe_idx = traj_slider.value - approach_traj.shape[0]
                robot.q_all = waypoint_traj[keyframe_idx].view(-1)
                # Update keyframes.
                part2cam = toad_opt.get_parts2cam(keyframe=keyframe_idx)
                for i in range(len(frame_list)):
                    frame_list[i].position = part2cam[i].translation()
                    frame_list[i].wxyz = part2cam[i].rotation().wxyz

        # Wait for validation to run the trajectory.
        run_traj = False

        print("Waiting for whether traj is valid...")
        @exec_traj_button.on_click
        def _(_):
            nonlocal run_traj
            run_traj = True
        while not run_traj:
            traj_slider.value = (traj_slider.value + 1) % (approach_traj.shape[0] + waypoint_traj.shape[0])
            if valid_traj_gui.value == "NO":
                traj_slider.remove()
                cleanup()
                return
            elif valid_traj_gui.value == "YES":
                print("Waiting for button press to run the trajectory...")
                valid_traj_gui.disabled = True
                exec_traj_button.disabled = False
            time.sleep(1 / 10.0)
        traj_slider.remove()

        planned_traj = torch.cat([approach_traj, waypoint_traj], dim=0)

        # Play the trajectory, in real. Will wait until real robot is available.
        print("Waiting for real robot...")
        while robot.real is None:
            time.sleep(1)

        print("starting recording, running physical.")

        cleanup()
        # run_real_robot(robot, approach_traj, waypoint_traj)
        # with Pool(processes=1) as pool:
        #     pool.apply_async(run_real_robot, (robot, approach_traj, waypoint_traj))
        #     time.sleep(30)

        # zed.start_record(str(log_dir / "traj.svo"))
        # with Pool(processes=1) as pool:
        #     pool.apply_async(async_record_zed, (zed,))
        #     time.sleep(1.0)
        # zed.stop_record()
        # with Pool(processes=2) as pool:
        #     pool.apply_async(run_real_robot, (robot, approach_traj, waypoint_traj))
        #     pool.apply_async(async_record_zed, (zed,))
        #     time.sleep(30)

        zed.start_record(str(log_dir / "traj.svo"))
        done = [False]
        thread = Thread(target=async_record_zed, args=(zed, done))
        thread.start()
        run_real_robot(robot, approach_traj, waypoint_traj)
        done[0] = True
        thread.join()
        zed.stop_record()

        # print("stopped traj recording.")

        # succ_traj_gui.disabled = False
        # while succ_traj_gui.value == "":
        #     time.sleep(1 / 10.0)

    # Cleanup gui + log states before returning.


def initialize_zed(server: viser.ViserServer, camera_frame_name: str = "camera") -> Tuple[Zed, viser.FrameHandle]:
    zed = Zed()
    camera_tf = RigidTransform.load("data/zed_to_world.tf")
    camera_frame = server.scene.add_frame(
        f"{camera_frame_name}",
        position=camera_tf.translation,  # rough alignment.
        wxyz=camera_tf.quaternion,
        show_axes=False,
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
    return zed, camera_frame


def main(exp_name: Literal["nerfgun", "eyeglasses", "scissors", "bear", "red_box", "painter"]):
    server = viser.ViserServer()

    # Instantiate the stuff that needs to be re-used across sessions:
    # - Zed camera, and camera frame.
    zed, cam_frame = initialize_zed(server, "camera")
    cam_vtf = vtf.SE3(wxyz_xyz=np.array([*cam_frame.wxyz, *cam_frame.position]))

    # - YuMi robot (with ability to reinitialize)!
    robot = YumiRobot(
        target=server,
        minibatch_size=1,
    )
    assert robot.real is not None, "This is for physical exps -- need robot."
    reset_robot_button = server.gui.add_button("Reset+home robot")
    @reset_robot_button.on_click
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

    # - Toad object (once it's generated, keep it.)
    toad_opt = ToadOptimizer(
        Path(EXPERIMENTS[exp_name][0]),
        zed.get_K(),
        zed.width,
        zed.height,
        init_cam_pose=torch.from_numpy(
            cam_vtf.as_matrix()[None, :3, :]
        ).float(),
    )

    # - Start the trial.
    # start_trial_button = server.gui.add_button("Start trial")
    # @start_trial_button.on_click
    # def _(_):
    #     start_trial_button.disabled = True
    start_trial(exp_name, toad_opt, zed, cam_vtf, robot, server)
        # start_trial_button.disabled = False

    while True:
        time.sleep(1)

if __name__ == "__main__":
    tyro.cli(main)
