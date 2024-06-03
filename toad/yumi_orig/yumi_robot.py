from __future__ import annotations

import time
from pathlib import Path
from typing import List, Literal, Dict, Any, Union, Optional, Tuple

import torch
import numpy as np
import viser
import viser.transforms as vtf
from viser.extras import ViserUrdf
import trimesh.creation
import tyro

# Yumi
from yumirws.yumi import YuMi

from toad.yumi.yumi_planner import YumiPlanner
from toad.toad_object import GraspableToadObject
from toad.yumi.yumi_jax_planner import YumiJaxPlanner


# Physical height of the table.
TABLE_HEIGHT=-0.006

# Real robot's speed, settings, etc.
REAL_ROBOT_SPEED=(0.1, np.pi/4)


class YumiRobot:
    vis: ViserUrdf
    """The URDF object for the visualizer."""
    plan: YumiPlanner
    """The motion planner for the robot, using curobo."""
    plan_jax: YumiJaxPlanner
    """The motion planner for the robot, using jax."""
    real: Optional[YuMi] = None
    """The physical robot object, if available."""

    _curr_cfg: torch.Tensor
    """Current joint configuration. (16,), including gripper width."""
    main_ee: Literal["gripper_l_base", "gripper_r_base"]
    """The main end-effector of the robot -- this matters! Trajopt tends to work better w the main ee."""

    def __init__(
        self,
        target: Union[viser.ViserServer, viser.ClientHandle],
        batch_size: int = 1,
        main_ee: Literal["gripper_l_base", "gripper_r_base"] = "gripper_l_base",
    ):
        _base_dir = Path(__file__).parent.parent.parent

        # Initialize the visualizer.
        self.vis = ViserUrdf(
            target, _base_dir / Path("data/yumi_description/urdf/yumi.urdf")
        )
        target.scene.add_grid(
            name="grid",
            width=0.6,
            height=0.8,
            position=(0.5, 0, TABLE_HEIGHT),
            section_size=0.05,
        )
        self.vis._joint_frames[0].remove()
        self.main_ee = main_ee

        # Initialize the planner.
        self.plan = YumiPlanner(
            batch_size=batch_size,
            table_height=TABLE_HEIGHT,
            main_ee=main_ee,
        )

        # Initialize the planner using jax.
        # Note that this loads urdf from load_descriptions... not the same urdf. 
        self.plan_jax = YumiJaxPlanner(main_ee=main_ee)

        # Initialize the real robot.
        try:
            self.real = YuMi()
        except:
            print("Failed to initialize the real robot, continuing w/o it.")

        # Get the tooltip-to-gripper offset. TODO remove hardcoding...
        self.tooltip_to_gripper = vtf.SE3.from_translation(np.array([0.0, 0.0, 0.128]))
        self.tooltip_to_approach = vtf.SE3.from_translation(np.array([0.0, 0.0, 0.148]))

        # Set the current joint position to the home position.
        self.joint_pos = self.plan.home_pos

    @property
    def joint_pos(self) -> torch.Tensor:
        """Gets the joint position of the robot. (16,), including gripper width."""
        return self._curr_cfg

    @joint_pos.setter
    def joint_pos(self, joint_pos: torch.Tensor):
        """Sets the joint position of the robot, and updates the visualizer.
        joint_pos may be of shape (14,) or (16,). If (16,), the last two elements are the gripper width.
        If (14,), the gripper width is set to 0.025 (max gripper width)."""
        if len(joint_pos.shape) != 1:
            joint_pos = joint_pos.squeeze()

        assert joint_pos.shape == (14,) or joint_pos.shape == (16,)

        # Update the joint positions in the visualizer.
        if joint_pos.shape == (16,):
            gripper_width = joint_pos[-2:]
        else:
            gripper_width = torch.Tensor([0.025, 0.025]).to(joint_pos.device)
        _joint_pos = torch.concat(
            (
                (joint_pos[7:14], joint_pos[:7], gripper_width) if self.main_ee == "gripper_l_base"
                else (joint_pos[:7], joint_pos[7:14], gripper_width)  # right
            )
        ).detach().cpu().numpy()
        self.vis.update_cfg(_joint_pos)

        self._curr_cfg = joint_pos


def main(
    mode: Literal["ik", "motiongen", "ik-chain", "goal", "waypoint"] = "ik",
):
    server = viser.ViserServer()
    robot = YumiRobot(server, batch_size=(240 if mode == "goal" else 1))

    drag_l_handle = server.scene.add_transform_controls(
        name="drag_l_handle",
        scale=0.1,
        position=(0.4, 0.2, 0.5),
        wxyz=(0, 1, 0, 0)
    )
    drag_r_handle = server.scene.add_transform_controls(
        name="drag_r_handle",
        scale=0.1,
        position=(0.4, -0.2, 0.5),
        wxyz=(0, 1, 0, 0)
    )

    spheres_list = []
    vis_spheres_checkbox = server.add_gui_checkbox("Visualize spheres", False)
    @vis_spheres_checkbox.on_update
    def _(_):
        nonlocal spheres_list
        if len(spheres_list) > 0:
            for sphere in spheres_list:
                sphere.remove()
            spheres_list = []
        if vis_spheres_checkbox.value:
            spheres = robot.plan.get_robot_as_spheres(robot.joint_pos[:14])
            for idx, sphere in enumerate(spheres):
                spheres_list.append(server.add_mesh_trimesh(f"spheres/sphere_{idx}", sphere))

    def update_joints():
        js, success = robot.plan.ik(
            torch.Tensor([*drag_l_handle.wxyz, *drag_l_handle.position]).view(1, 7),
            torch.Tensor([*drag_r_handle.wxyz, *drag_r_handle.position]).view(1, 7),
            initial_js=robot.joint_pos[:14],
        )
        if js is None:
            print("Failed to solve IK.")
            return
        assert isinstance(js, torch.Tensor) and isinstance(success, torch.Tensor)
        assert js.shape == (1, 14) and success.shape == (1,)
        # if success.all():
        robot.joint_pos = js[0]

    poses = robot.plan.fk(robot.plan.home_pos[:14])
    drag_l_handle.position = poses.link_poses['gripper_l_base'].position.flatten().cpu().numpy()
    drag_l_handle.wxyz = poses.link_poses['gripper_l_base'].quaternion.flatten().cpu().numpy()
    drag_r_handle.position = poses.link_poses['gripper_r_base'].position.flatten().cpu().numpy()
    drag_r_handle.wxyz = poses.link_poses['gripper_r_base'].quaternion.flatten().cpu().numpy()

    update_joints()

    if mode == "ik":
        @drag_r_handle.on_update
        def _(_):
            update_joints()
            d_world, d_self = robot.plan.in_collision(robot.joint_pos[:14])
            is_collide_world.value = (d_world > 0).any().item()
            is_collide_self.value = (d_self > 0).any().item()
        @drag_l_handle.on_update
        def _(_):
            update_joints()
            d_world, d_self = robot.plan.in_collision(robot.joint_pos[:14])
            is_collide_world.value = (d_world > 0).any().item()
            is_collide_self.value = (d_self > 0).any().item()

        is_collide_world = server.add_gui_checkbox("Collide (world)", False, disabled=True)
        is_collide_self = server.add_gui_checkbox("Collide (self)", False, disabled=True)

    elif mode == "motiongen" or mode == "ik-chain" or mode == "waypoint":
        waypoint_button = server.add_gui_button("Add waypoint")
        drag_button = server.add_gui_button("Match drag")
        drag_slider = server.add_gui_slider("Time", 0, 1, 0.01, 0, disabled=True)

        traj = None
        waypoint_queue = [[], []]  # left, right.
        @waypoint_button.on_click
        def _(_):
            waypoint_queue[0].append(torch.Tensor([*drag_l_handle.wxyz, *drag_l_handle.position]).view(1, 7))
            waypoint_queue[1].append(torch.Tensor([*drag_r_handle.wxyz, *drag_r_handle.position]).view(1, 7))

        @drag_button.on_click
        def _(_):
            nonlocal traj
            drag_slider.disabled = True
            drag_button.disabled = True

            if len(waypoint_queue[0]) == 0:
                waypoint_queue[0].append(torch.Tensor([*drag_l_handle.wxyz, *drag_l_handle.position]).view(1, 7))
                waypoint_queue[1].append(torch.Tensor([*drag_r_handle.wxyz, *drag_r_handle.position]).view(1, 7))

            start_pose = robot.joint_pos

            start = time.time()
            traj_pieces = []
            prev_start_state = start_pose

            if mode == "motiongen":
                for i in range(len(waypoint_queue[0])):
                    js, success = robot.plan.gen_motion_from_goal(
                        waypoint_queue[0][i],
                        waypoint_queue[1][i],
                        initial_js=prev_start_state,
                    )
                    if js is None:
                        drag_slider.value = 0
                        drag_button.disabled = False
                        drag_slider.disabled = True
                        print("Failed to generate motion.")
                        break

                    assert isinstance(js, torch.Tensor) and isinstance(success, torch.Tensor)
                    assert len(js.shape) == 3 and js.shape[-1] == 14 and js.shape[0] == 1
                    assert len(success.shape) == 1 and success.shape[0] == js.shape[0]
                    if not success.all():
                        break
                    js = js[0]  # [1, time, 14] -> [time, 14]
                    prev_start_state = js[-1:]
                    traj_pieces.append(js)

                if len(traj_pieces) == 0:
                    drag_button.disabled = False
                    drag_slider.disabled = True
                    return
                traj = torch.concat(traj_pieces, dim=0)

            elif mode == "ik-chain":
                path_l_wxyz_xyz = torch.cat([_.unsqueeze(1) for _ in waypoint_queue[0]], dim=1)
                path_r_wxyz_xyz = torch.cat([_.unsqueeze(1) for _ in waypoint_queue[1]], dim=1)
                initial_js = prev_start_state[:14].unsqueeze(0)
                assert path_l_wxyz_xyz.shape == (1, len(waypoint_queue[0]), 7)
                assert initial_js.shape == (1, 14)
                js, success = robot.plan.gen_motion_from_ik_chain(
                    path_l_wxyz_xyz,
                    path_r_wxyz_xyz,
                    initial_js=initial_js,
                )
                if js is None:
                    drag_button.disabled = False
                    drag_slider.disabled = True
                    return
                assert isinstance(js, torch.Tensor) and isinstance(success, torch.Tensor)
                assert js.shape == (1, len(waypoint_queue[0]), 14) and success.shape == (1, len(waypoint_queue[0]),)
                traj = torch.cat([
                    initial_js,
                    js[0]  # (len(waypoint_queue[0]), 14
                ])  # --> (len(waypoint_queue[0])+1, 14)

            elif mode == "waypoint":
                path_l_wxyz_xyz = torch.cat([_.unsqueeze(1) for _ in waypoint_queue[0]], dim=1)
                path_r_wxyz_xyz = torch.cat([_.unsqueeze(1) for _ in waypoint_queue[1]], dim=1)
                assert path_l_wxyz_xyz.shape == (1, len(waypoint_queue[0]), 7)

                path_l_wxyz_xyz = path_l_wxyz_xyz.squeeze(0)
                path_r_wxyz_xyz = path_r_wxyz_xyz.squeeze(0)
                assert path_l_wxyz_xyz.shape == (len(waypoint_queue[0]), 7)
                assert path_r_wxyz_xyz.shape == (len(waypoint_queue[1]), 7)

                traj = robot.plan_jax.plan_from_waypoints(
                    path_l_wxyz_xyz,
                    path_r_wxyz_xyz,
                )

            else:
                raise ValueError(f"Unknown mode: {mode}")

            print("MotionGen took", time.time() - start, "seconds")
            waypoint_queue[0] = []
            waypoint_queue[1] = []

            assert len(traj.shape) == 2 and traj.shape[-1] == 14
            robot.joint_pos = traj[0]
            drag_slider.value = 0
            drag_button.disabled = False
            drag_slider.disabled = False

        @drag_slider.on_update
        def _(_):
            assert traj is not None
            idx = int(drag_slider.value * (len(traj)-1))
            robot.joint_pos = traj[idx]

    elif mode == "goal":
        drag_l_handle.remove()
        drag_r_handle.remove()

        cube = trimesh.creation.box((0.03, 0.03, 0.03))
        cube_toad = GraspableToadObject.from_mesh(cube)
        cube_tf = server.add_transform_controls(
            name="obj",
            position=(0.4, 0, 0.5),
            wxyz=(1, 0, 0, 0),
            scale=0.08,
        )
        cube_mesh = server.add_mesh_trimesh("obj/mesh", cube)
        grasp_mesh = cube_toad.grasp_axis_mesh()
        for i, grasp in enumerate(cube_toad.grasps[0]):
            server.add_mesh_trimesh(
                f"obj/grasp_{i}",
                grasp_mesh,
                position=grasp[:3].numpy(),
                wxyz=grasp[3:].numpy(),
            )

        plan_dropdown = server.add_gui_dropdown("Plan", ["either", "left", "right"], "either")
        goal_button = server.add_gui_button("Move to goal")

        traj, traj_handle, play_handle = None, None, None
        @goal_button.on_click
        def _(_):
            nonlocal traj, traj_handle, play_handle
            goal_button.disabled = True

            obj_pose = vtf.SE3(np.array([*cube_tf.wxyz, *cube_tf.position]))
            print("obj pose", obj_pose)
            mesh_list = cube_toad.to_world_config(
                poses_wxyz_xyz=[obj_pose.wxyz_xyz]
            )
            robot.plan.update_world_objects(mesh_list)

            grasps = cube_toad.grasps[0]  # [N_grasps, 7]
            grasps_gripper = cube_toad.to_gripper_frame(
                grasps, robot.tooltip_to_gripper
            )
            grasp_cand_list = obj_pose.multiply(grasps_gripper)
            goal_wxyz_xyz = torch.Tensor(grasp_cand_list.wxyz_xyz)

            if plan_dropdown.value == "either":
                traj, success = robot.plan.gen_motion_from_goal_either(
                    goal_wxyz_xyz=goal_wxyz_xyz,
                    initial_js=robot.plan.home_pos[:14].expand(goal_wxyz_xyz.shape[0], 14).contiguous(),
                )
            elif plan_dropdown.value == "left":
                traj, success = robot.plan.gen_motion_from_goal(
                    goal_l_wxyz_xyz=goal_wxyz_xyz,
                    goal_r_wxyz_xyz=torch.Tensor([0, 1, 0, 0, 0.4, -0.2, 0.5]).expand(goal_wxyz_xyz.shape[0], 7),
                    initial_js=robot.plan.home_pos[:14].expand(goal_wxyz_xyz.shape[0], 14),
                )
            else:
                traj, success = robot.plan.gen_motion_from_goal(
                    goal_l_wxyz_xyz=torch.Tensor([0, 1, 0, 0, 0.4, 0.2, 0.5]).expand(goal_wxyz_xyz.shape[0], 7),
                    goal_r_wxyz_xyz=goal_wxyz_xyz,
                    initial_js=robot.plan.home_pos[:14].expand(goal_wxyz_xyz.shape[0], 14),
                )

            if traj is None:
                print("Failed to generate motion.")
                goal_button.disabled = False
                return

            assert isinstance(traj, torch.Tensor) and isinstance(success, torch.Tensor)

            if not success.any():
                print("No successful trajectory found.")
                goal_button.disabled = False
                return

            traj = traj[success]
            assert len(traj.shape) == 3, "Should be [batch, time, dof]."

            if traj_handle is not None:
                traj_handle.remove()
            traj_handle = server.add_gui_slider("Trajectory Index", 0, len(traj) - 1, 1, 0)

            if play_handle is not None:
                play_handle.remove()
            play_handle = server.add_gui_slider("play", min=0, max=traj.shape[1]-1, step=1, initial_value=0)

            goal_button.disabled = False

            def move_to_traj_position():
                assert traj is not None and traj_handle is not None and play_handle is not None
                assert isinstance(traj, torch.Tensor)
                robot.joint_pos = traj[traj_handle.value][play_handle.value].view(1, 14)

            @traj_handle.on_update
            def _(_):
                move_to_traj_position()
            @play_handle.on_update
            def _(_):
                move_to_traj_position()

    while True:
        time.sleep(1)

if __name__ == "__main__":
    tyro.cli(main)
