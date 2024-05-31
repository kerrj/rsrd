"""
Quick interactive demo for yumi IK, with curobo.
"""

import torch
import viser
import viser.transforms as vtf
import time
import numpy as np
import trimesh
from typing import Optional, List, Any
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

from yumirws.yumi import YuMi


def main(
    # config_path: Path = Path("outputs/buddha_balls_poly/dig/2024-05-23_184345/config.yml"),
    # config_path: Path = Path("outputs/mallet/dig/2024-05-27_180206/config.yml"),
    config_path: Path = Path("outputs/nerfgun/dig/2024-05-30_210410/config.yml"),
    keyframe_path: Path = Path("renders/nerfgun/keyframes.pt")
):
    """Quick interactive demo for object traj following.

    Args:
        config_path: Path to the nerfstudio config file.
        keyframe_path: Path to the keyframe file.
    """

    server = viser.ViserServer()

    # YuMi robot code must be placed before any curobo code!
    with server.add_gui_folder("Robot control"):
        gripper_open_button = server.add_gui_button("Open physical gripper")
        gripper_close_button = server.add_gui_button("Close physical gripper")
        robot_handler = server.add_gui_button("Move robot")
        restart_button = server.add_gui_button("Restart")

        robot = YuMi()
        @gripper_open_button.on_click
        def _(_):
            robot.left.open_gripper()
            robot.right.open_gripper()
        @gripper_close_button.on_click
        def _(_):
            robot.left.close_gripper()
            robot.right.close_gripper()
        @robot_handler.on_click
        def _(_):
            robot.move_joints_sync(
                l_joints=urdf.get_left_joints().view(1, 7).cpu().numpy().astype(np.float64),
                r_joints=urdf.get_right_joints().view(1, 7).cpu().numpy().astype(np.float64),
                speed=(0.1, np.pi),
            )
            robot.left.sync()
            robot.right.sync()
        @restart_button.on_click
        def _(_):
            nonlocal robot
            del robot
            robot = YuMi()
        

    # Needs to be called before any warp pose gets called.
    wp.init()

    urdf = YumiCurobo(
        server,
        ik_solver_batch_size=240,
        motion_gen_batch_size=240,
    )

    zed = Zed()

    # Visualize the camera.
    camera_tf = RigidTransform.load("data/zed_to_world.tf")
    camera_frame = server.add_frame(
        "camera",
        position=camera_tf.translation,  # rough alignment.
        wxyz=camera_tf.quaternion,
        show_axes=True,
        axes_length=0.1,
        axes_radius=0.005,
    )
    server.add_mesh_trimesh(
        "camera/mesh",
        mesh=zed.zed_mesh,
        scale=0.001,
        position=zed.cam_to_zed.translation,
        wxyz=zed.cam_to_zed.quaternion,
    )    

    l, _, depth = zed.get_frame(depth=True)  # type: ignore
    toad_opt = ToadOptimizer(
        config_path,
        zed.get_K(),
        l.shape[1],
        l.shape[0],
        # zed.width,
        # zed.height,
        init_cam_pose=torch.from_numpy(
            vtf.SE3(
                wxyz_xyz=np.array([*camera_frame.wxyz, *camera_frame.position])
            ).as_matrix()[None, :3, :]
        ).float(),
    )

    # Initialize the object pose.
    l, _, depth = zed.get_frame(depth=True)  # type: ignore
    toad_opt.set_frame(l,depth)
    with zed.raft_lock:
        toad_opt.init_obj_pose()

    # Visualize the keyframes.
    keyframe_ind = server.add_gui_slider("Keyframe Index",0.0,1.0,.01,0.0,disabled=True)
    toad_opt.optimizer.load_trajectory(keyframe_path)
    kf_disable_checkbox = server.add_gui_checkbox("Disable keyframes", False)
    @kf_disable_checkbox.on_update
    def _(_):
        if kf_disable_checkbox.value:
            keyframe_ind.disabled = True
        else:
            keyframe_ind.disabled = False

    part_handle = server.add_gui_number("Part", 0, 0, toad_opt.num_groups-1, 1, disabled=True)
    # Set up the robot.
    def curry_mesh(idx, tf):
        server.add_frame(
            f"camera/object/group_{idx}",
            position=tf.translation(),
            wxyz=tf.rotation().wxyz,
            show_axes=True,
            axes_length=0.02,
            axes_radius=.002
        )
        mesh = toad_opt.toad_object.meshes[idx]
        handle = server.add_mesh_trimesh(
            f"camera/object/group_{idx}/mesh",
            mesh=mesh,
        )
        grasps = toad_opt.toad_object.grasps[idx] # [N_grasps, 7]
        grasp_mesh = toad_opt.toad_object.grasp_axis_mesh()
        for j, grasp in enumerate(grasps):
            server.add_mesh_trimesh(
                f"camera/object/group_{idx}/grasp_{j}",
                grasp_mesh,
                position=grasp[:3],
                wxyz=grasp[3:],
            )
        @handle.on_click
        def _(_):
            part_handle.value = idx
        return handle
    for i, tf in enumerate(toad_opt.get_parts2cam()):
        curry_mesh(i, tf)

    traj, traj_handle, play_handle = None, None, None
    button_handle = server.add_gui_button("Calculate working grasps")

    @button_handle.on_click
    def _(_):
        nonlocal traj, traj_handle, play_handle
        button_handle.disabled = True

        # Update the object's pose, for collisionbody.
        world_config = createTableWorld()
        assert world_config.mesh is not None  # should not be none after post_init.
        poses_part2cam = toad_opt.get_parts2cam(keyframe=0)
        poses_part2world = [vtf.SE3(wxyz_xyz=np.array([*camera_frame.wxyz, *camera_frame.position])).multiply(pose) for pose in poses_part2cam]
        poses_wxyz_xyz = [pose.wxyz_xyz for pose in poses_part2world]
        mesh_list = toad_opt.toad_object.to_world_config(poses_wxyz_xyz=poses_wxyz_xyz)
        world_config.mesh.extend(mesh_list)  # type: ignore
        urdf.update_world(world_config)

        grasps = toad_opt.toad_object.grasps[part_handle.value]  # [N_grasps, 7]
        grasps_gripper = toad_opt.toad_object.to_gripper_frame(grasps, urdf._tooltip_to_gripper)
        grasp_cand_list = poses_part2world[part_handle.value].multiply(grasps_gripper)

        # right now, at least, let's assume one hand.
        goal_l_wxyz_xyz = torch.cat([
            torch.Tensor(grasp_cand_list.wxyz_xyz),
            torch.Tensor([[0, 1, 0, 0, 0.4, 0.2, 0.5]]).expand(grasp_cand_list.wxyz_xyz.shape[0], 7)
        ])
        goal_r_wxyz_xyz = torch.cat([
            torch.Tensor([[0, 1, 0, 0, 0.4, -0.2, 0.5]]).expand(grasp_cand_list.wxyz_xyz.shape[0], 7),
            torch.Tensor(grasp_cand_list.wxyz_xyz),
        ])         

        start_state = JointState.from_position(urdf.home_pos)
        with zed.raft_lock:
            approach_traj, approach_success = urdf.motiongen(
                goal_l_wxyz_xyz=goal_l_wxyz_xyz,
                goal_r_wxyz_xyz=goal_r_wxyz_xyz,
                start_state=start_state,
                get_pos_only=True
            )
        assert isinstance(approach_traj, torch.Tensor) and isinstance(approach_success, torch.Tensor)
        if not approach_success.any():
            button_handle.disabled = False
            return

        if toad_opt.optimizer.keyframes is not None:
            prev_js = approach_traj[:, -1, :]  # [batch, dof]
            js_list, js_success_list = [], []
            
            # IK-based waypoint following.
            with zed.raft_lock:
                last_keyframe = 0
                for i in range(0, len(toad_opt.optimizer.keyframes)):
                    poses_part2cam = toad_opt.get_parts2cam(keyframe=i)
                    poses_part2world = [vtf.SE3(wxyz_xyz=np.array([*camera_frame.wxyz, *camera_frame.position])).multiply(pose) for pose in poses_part2cam]
                    poses_wxyz_xyz = [pose.wxyz_xyz for pose in poses_part2world]

                    world_config = createTableWorld()
                    assert world_config.mesh is not None  # should not be none after post_init.
                    mesh_list = toad_opt.toad_object.to_world_config(poses_wxyz_xyz=poses_wxyz_xyz)
                    world_config.mesh.extend(mesh_list)  # type: ignore
                    urdf.update_world(world_config)

                    poses_part2cam = toad_opt.get_parts2cam(keyframe=i)
                    grasp_cand_list = poses_part2world[part_handle.value].multiply(grasps_gripper)
                    goal_l_wxyz_xyz = torch.cat([
                        torch.Tensor(grasp_cand_list.wxyz_xyz),
                        torch.Tensor([[0, 1, 0, 0, 0.4, 0.2, 0.5]]).expand(grasp_cand_list.wxyz_xyz.shape[0], 7)
                    ])
                    goal_r_wxyz_xyz = torch.cat([
                        torch.Tensor([[0, 1, 0, 0, 0.4, -0.2, 0.5]]).expand(grasp_cand_list.wxyz_xyz.shape[0], 7),
                        torch.Tensor(grasp_cand_list.wxyz_xyz),
                    ])
                    start = time.time()
                    curr_js, curr_js_success = urdf.ik(
                        goal_l_wxyz_xyz=goal_l_wxyz_xyz,
                        goal_r_wxyz_xyz=goal_r_wxyz_xyz,
                        initial_js=prev_js[..., :14].squeeze(),  # needs [batch, dof], not [batch, 1, dof].
                        get_pos_only=True
                    )
                    assert isinstance(curr_js, torch.Tensor) and isinstance(curr_js_success, torch.Tensor)
                    d_world, d_self = urdf.in_collision(curr_js[..., :14].contiguous())
                    curr_js_success = curr_js_success # & (d_world <= 0.005)
                    print(f"Time taken for IK: {time.time() - start:.2f} s")
                    assert isinstance(curr_js, torch.Tensor) and isinstance(curr_js_success, torch.Tensor)
                    assert len(curr_js.shape) == 3 and curr_js.shape[1] == 1, f"Should be [batch, 1, dof], but got shape: {curr_js.shape}"

                    # Consider the case where collision causes *all* trajectories to fail.
                    # Then, this trajectory shouldn't be considered.
                    if not curr_js_success.any():
                        break

                    # if ((prev_js[:, :14] - curr_js[:, 0, :14]).abs() > 2*np.pi).any():
                    #     mask = (prev_js[:, :14] - curr_js[:, 0, :14]) > 2*np.pi
                    #     curr_js[:, 0, :14][mask] += 2*np.pi
                    #     mask = (prev_js[..., :14] - curr_js[..., 0, :14]) < -2*np.pi
                    #     curr_js[:, 0, :14][mask] -= 2*np.pi
                    js_list.append(curr_js[..., :14])
                    js_success_list.append(curr_js_success)
                    prev_js = curr_js
                    last_keyframe = i
                
                # # from the list of successful waypoints, we can optimize the trajectory.
                # poses_part2cam = toad_opt.get_parts2cam(keyframe=last_keyframe)
                # poses_part2world = [vtf.SE3(wxyz_xyz=np.array([*camera_frame.wxyz, *camera_frame.position])).multiply(pose) for pose in poses_part2cam]
                # grasp_cand_list = poses_part2world[part_handle.value].multiply(grasps_gripper)
                # goal_l_wxyz_xyz = torch.cat([
                #     torch.Tensor(grasp_cand_list.wxyz_xyz),
                #     torch.Tensor([[0, 1, 0, 0, 0.4, 0.2, 0.5]]).expand(grasp_cand_list.wxyz_xyz.shape[0], 7)
                # ])
                # goal_r_wxyz_xyz = torch.cat([
                #     torch.Tensor([[0, 1, 0, 0, 0.4, -0.2, 0.5]]).expand(grasp_cand_list.wxyz_xyz.shape[0], 7),
                #     torch.Tensor(grasp_cand_list.wxyz_xyz),
                # ])
                # goal_l = Pose(goal_l_wxyz_xyz[:, 4:].contiguous().cuda(), goal_l_wxyz_xyz[:, :4].contiguous().cuda())
                # goal_r = Pose(goal_r_wxyz_xyz[:, 4:].contiguous().cuda(), goal_r_wxyz_xyz[:, :4].contiguous().cuda())
                # # import pdb; pdb.set_trace()
                # start_state = JointState.from_position(approach_traj[:, -1, :14].cuda())
                # goal_state = JointState.from_position(js_list[-1].squeeze().cuda())

                # world_config = createTableWorld()
                # urdf.update_world(world_config)

                # # this OOM's the memory...
                # foo = urdf._motion_gen.js_trajopt_solver.solve_batch(
                #     Goal(
                #         # goal_state=goal_state,
                #         # batch=goal_l_wxyz_xyz.shape[0],
                #         # current_state=start_state,
                #         # links_goal_pose={
                #         #     "gripper_l_base": goal_l,
                #         #     "gripper_r_base": goal_r
                #         # },
                #         # goal_state=goal_state,
                #         goal_pose = goal_l,
                #         batch=goal_l_wxyz_xyz.shape[0],
                #         current_state=start_state,
                #         links_goal_pose={
                #             "gripper_l_base": goal_l,
                #             "gripper_r_base": goal_r
                #         },
                #     ),
                #     seed_traj=JointState(torch.cat(js_list[:28], dim=1).cuda()),
                #     num_seeds=1,
                #     newton_iters=2,
                # )
                # waypoint_traj = foo.interpolated_solution.position
                # waypoint_success = foo.success.unsqueeze(-1).expand(-1, waypoint_traj.shape[1])

            # # TODO Retract trajectory.
            # with zed.raft_lock:
            #     poses_part2cam = toad_opt.get_parts2cam(keyframe=last_keyframe)
            #     poses_part2world = [vtf.SE3(wxyz_xyz=np.array([*camera_frame.wxyz, *camera_frame.position])).multiply(pose) for pose in poses_part2cam]
            #     grasp_cand_list = poses_part2world[part_handle.value].multiply(grasps_gripper)
            #     goal_l_wxyz_xyz = torch.Tensor([[0, 1, 0, 0, 0.4, 0.2, 0.5]]).expand(grasp_cand_list.wxyz_xyz.shape[0]*2, 7)
            #     goal_r_wxyz_xyz = torch.Tensor([[0, 1, 0, 0, 0.4, -0.2, 0.5]]).expand(grasp_cand_list.wxyz_xyz.shape[0]*2, 7)
            #     start = time.time()
            #     retract_traj, retract_success = urdf.motiongen(
            #         goal_l_wxyz_xyz=goal_l_wxyz_xyz,
            #         goal_r_wxyz_xyz=goal_r_wxyz_xyz,
            #         start_state=prev_js[..., :14].squeeze(),
            #         get_pos_only=True
            #     )
            #     assert isinstance(retract_traj, torch.Tensor) and isinstance(retract_success, torch.Tensor)
            #     print(f"Time taken for retract: {time.time() - start:.2f} s")

            traj = torch.cat([approach_traj, *js_list], dim=1)
            success = torch.cat([approach_success, *js_success_list], dim=1).all(dim=-1)  # i.e., all waypoints are successful.
            # traj = torch.cat([approach_traj, waypoint_traj], dim=1)
            # success = torch.cat([approach_success, waypoint_success], dim=1).all(dim=-1)  # i.e., all waypoints are successful.
            traj = traj[success]

        else:
            traj = approach_traj[approach_success]

        button_handle.disabled = False

        # Reset visualization!
        if traj_handle is not None:
            traj_handle.remove()
        if play_handle is not None:
            play_handle.remove()

        assert traj is not None

        traj_handle = server.add_gui_slider("trajectory", 0, len(traj)-1, 1, 0)
        play_handle = server.add_gui_slider("play", 0, traj.shape[1]-1, 1, 0)
        move_traj_handle = server.add_gui_button("Play traj")

        @traj_handle.on_update
        def _(_):
            assert traj is not None
            urdf.joint_pos = traj[int(traj_handle.value), int(play_handle.value)]

        @play_handle.on_update
        def _(_):
            assert traj is not None
            urdf.joint_pos = traj[int(traj_handle.value), int(play_handle.value)]

        @move_traj_handle.on_click
        def _(_):
            with zed.raft_lock:
                assert traj is not None
                robot.move_joints_sync(
                    l_joints=traj[int(traj_handle.value)][:30, :7].cpu().numpy().astype(np.float64),
                    r_joints=traj[int(traj_handle.value)][:30, 7:].cpu().numpy().astype(np.float64),
                    speed=(0.1, np.pi),
                    zone="z1",
                )
                robot.left.sync()
                robot.right.sync()
                robot.left.close_gripper()
                robot.right.close_gripper()
                time.sleep(1)
                foo = traj[int(traj_handle.value)][30:].cpu().numpy().astype(np.float64)
                for _ in range(5):
                    blur_foo = 0.5*foo + 0.25*np.roll(foo, 1, axis=0) + 0.25*np.roll(foo, -1, axis=0)
                    blur_foo[0] = foo[0]
                    blur_foo[-1] = foo[-1]
                    foo = blur_foo

                robot.move_joints_sync(
                    l_joints=foo[:, :7],
                    r_joints=foo[:, 7:],
                    speed=(0.05, np.pi/5),
                    zone="z10",
                )
                robot.left.sync()
                robot.right.sync()
                # robot.move_joints_sync(
                #     l_joints=traj[int(traj_handle.value)][30:, :7].cpu().numpy().astype(np.float64),
                #     r_joints=traj[int(traj_handle.value)][30:, 7:].cpu().numpy().astype(np.float64),
                #     speed=(0.1, np.pi),
                #     zone="z10",
                # )


    while True:
        left, right, depth = zed.get_frame()
        if toad_opt.initialized:
            with zed.raft_lock:
                toad_opt.set_frame(left,depth)
                toad_opt.step_opt(niter=50)
                keyframe = None if keyframe_ind.disabled else int(keyframe_ind.value*(len(toad_opt.optimizer.keyframes)-1))
                tf_list = toad_opt.get_parts2cam(keyframe=keyframe)
                for idx, tf in enumerate(tf_list):
                    server.add_frame(
                        f"camera/object/group_{idx}",
                        position=tf.translation(),
                        wxyz=tf.rotation().wxyz,
                        show_axes=True,
                        axes_length=0.02,
                        axes_radius=.002
                    )
                if keyframe is not None:
                    toad_opt.optimizer.apply_keyframe(keyframe)
                    hands = toad_opt.optimizer.hand_frames[keyframe]
                    for ih,h in enumerate(hands):
                        h_world = h.copy()
                        h_world.apply_transform(toad_opt.optimizer.get_registered_o2w().cpu().numpy())
                        server.add_mesh_trimesh(f"hand{ih}", h_world, scale=1/toad_opt.optimizer.dataset_scale)
            
        # Visualize pointcloud.
        K = torch.from_numpy(zed.get_K()).float().cuda()
        assert isinstance(left, torch.Tensor) and isinstance(depth, torch.Tensor)
        points, colors = Zed.project_depth(left, depth, K)
        server.add_point_cloud(
            "camera/points",
            points=points,
            colors=colors,
            point_size=0.001,
        )


if __name__ == "__main__":
    tyro.cli(main)
