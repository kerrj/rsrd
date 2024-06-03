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

from toad.yumi.yumi_arm_planner import YumiArmPlanner, YUMI_REST_POSE_LEFT, YUMI_REST_POSE_RIGHT
from toad.toad_object import GraspableToadObject


# Physical height of the table.
TABLE_HEIGHT=-0.006

# Real robot's speed, settings, etc.
REAL_ROBOT_SPEED=(0.1, np.pi/4)


class YumiRobot:
    vis: ViserUrdf
    """The URDF object for the visualizer."""
    plan: YumiArmPlanner
    """The motion planner for the robot, using curobo."""
    real: Optional[YuMi] = None
    """The physical robot object, if available."""

    _q_left: torch.Tensor
    """The current joint configuration of the left arm, (8,) including gripper."""
    _q_right: torch.Tensor
    """The current joint configuration of the right arm, (8,) including gripper."""

    def __init__(
        self,
        target: Union[viser.ViserServer, viser.ClientHandle],
        minibatch_size: int = 1,
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

        # Initialize the planner.
        self.plan = YumiArmPlanner(
            minibatch_size=minibatch_size,
            table_height=TABLE_HEIGHT,
        )

        # Initialize the real robot.
        try:
            self.real = YuMi()
        except:
            print("Failed to initialize the real robot, continuing w/o it.")

        # Get the tooltip-to-gripper offset. TODO remove hardcoding...
        self.tooltip_to_gripper = vtf.SE3.from_translation(np.array([0.0, 0.0, 0.128]))

        # Set the current joint position to the home position.
        self._q_left = torch.Tensor(list(YUMI_REST_POSE_LEFT.values())).to(self.plan.device)
        self._q_right = torch.Tensor(list(YUMI_REST_POSE_RIGHT.values())).to(self.plan.device)
        self.vis.update_cfg(self.concat_joints(self._q_right, self._q_left).cpu().numpy())

    @property
    def q_left(self) -> torch.Tensor:
        return self._q_left

    @q_left.setter
    def q_left(self, q: torch.Tensor):
        assert q.shape == (8,)
        assert q.device == self.plan.device
        self._q_left = q
        self.vis.update_cfg(self.concat_joints(self._q_right, q).cpu().numpy())

    @property
    def q_right(self) -> torch.Tensor:
        return self._q_right
    
    @q_right.setter
    def q_right(self, q: torch.Tensor):
        assert q.shape == (8,)
        assert q.device == self.plan.device
        self._q_right = q
        self.vis.update_cfg(self.concat_joints(q, self._q_left).cpu().numpy())

    @staticmethod
    def concat_joints(q_right: torch.Tensor, q_left: torch.Tensor) -> torch.Tensor:
        """Concatenate the left and right joint configurations."""
        assert q_right.shape == q_left.shape == (8,)
        q_right_arm, q_right_gripper = q_right[:7], q_right[7:]
        q_left_arm, q_left_gripper = q_left[:7], q_left[7:]
        return torch.cat((q_right_arm, q_left_arm, q_right_gripper, q_left_gripper))


def main(
    mode: Literal["ik", "goal", "waypoint"] = "ik",
):
    server = viser.ViserServer()
    robot = YumiRobot(server, minibatch_size=(240 if mode == "goal" else 1))

    is_collide_world = server.gui.add_checkbox("Collide (world)", False, disabled=True)
    is_collide_self = server.gui.add_checkbox("Collide (self)", False, disabled=True)

    if mode == "ik":
        active_arm_handle = server.gui.add_dropdown(
            "Active Arm",
            options=("left", "right")
        )
        @active_arm_handle.on_update
        def _(_):
            if active_arm_handle.value == "left":
                robot.plan.activate_arm("left", robot.q_right)
            else:
                robot.plan.activate_arm("right", robot.q_left)

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
            wxyz=(0, 1, 0, 0),
        )

        def update_joints():
            if robot.plan.active_arm == "left":
                js, success = robot.plan.ik(
                    torch.Tensor([*drag_l_handle.wxyz, *drag_l_handle.position]).view(1, 7),
                    q_init=robot.q_left.view(1, -1),
                )
            else:
                js, success = robot.plan.ik(
                    torch.Tensor([*drag_r_handle.wxyz, *drag_r_handle.position]).view(1, 7),
                    q_init=robot.q_right.view(1, -1),
                )
            if js is None:
                print("Failed to solve IK.")
                return
            assert isinstance(js, torch.Tensor) and isinstance(success, torch.Tensor)
            assert js.shape == (1, 8) and success.shape == (1,)

            if not success.any():
                print("IK solution is invalid.")

            if robot.plan.active_arm == "left":
                robot.q_left = js[0]
            else:
                robot.q_right = js[0]

        @drag_r_handle.on_update
        def _(_):
            update_joints()
            if robot.plan.active_arm == "left":
                d_world, d_self = robot.plan.in_collision(robot.q_left)
            else:
                d_world, d_self = robot.plan.in_collision(robot.q_right)
            is_collide_world.value = (d_world > 0).any().item()
            is_collide_self.value = (d_self > 0).any().item()

        @drag_l_handle.on_update
        def _(_):
            update_joints()
            if active_arm_handle.value == "left":
                d_world, d_self = robot.plan.in_collision(robot.q_left)
            else:
                d_world, d_self = robot.plan.in_collision(robot.q_right)
            is_collide_world.value = (d_world > 0).any().item()
            is_collide_self.value = (d_self > 0).any().item()

        active_arm_handle.value = "left"
        robot.plan.activate_arm("left", robot.q_right)
        update_joints()

        robot.plan.activate_arm("right", robot.q_left)
        active_arm_handle.value = "right"
        update_joints()

    elif mode == "goal":
        cube = trimesh.creation.box((0.03, 0.03, 0.03))
        cube_toad = GraspableToadObject.from_mesh(cube)
        cube_tf = server.scene.add_transform_controls(
            name="obj",
            position=(0.4, 0, 0.5),
            wxyz=(1, 0, 0, 0),
            scale=0.08,
        )
        cube_mesh = server.scene.add_mesh_trimesh("obj/mesh", cube)
        grasp_mesh = cube_toad.grasp_axis_mesh()
        for i, grasp in enumerate(cube_toad.grasps[0]):
            server.scene.add_mesh_trimesh(
                f"obj/grasp_{i}",
                grasp_mesh,
                position=grasp[:3].numpy(),
                wxyz=grasp[3:].numpy(),
            )

        plan_dropdown = server.gui.add_dropdown("Plan", ("left", "right"), "left")
        goal_button = server.gui.add_button("Move to goal")

        traj, traj_handle, play_handle = None, None, None
        @goal_button.on_click
        def _(_):
            nonlocal traj, traj_handle, play_handle
            goal_button.disabled = True
            if traj_handle is not None:
                traj_handle.remove()
            if play_handle is not None:
                play_handle.remove()

            obj_pose = vtf.SE3(np.array([*cube_tf.wxyz, *cube_tf.position]))
            print("obj pose", obj_pose)
            mesh_list = cube_toad.to_world_config(
                poses_wxyz_xyz=[obj_pose.wxyz_xyz]
            )
            robot.plan.update_world_objects(mesh_list)

            grasps = cube_toad.grasps[0]  # [N_grasps, 7]
            grasps_gripper = cube_toad.to_gripper_frame(grasps)

            grasp_cand_list = obj_pose.multiply(grasps_gripper)
            goal_wxyz_xyz = torch.Tensor(grasp_cand_list.wxyz_xyz)

            robot.plan.activate_arm(
                plan_dropdown.value,
                robot.q_right if plan_dropdown.value == "left" else robot.q_left,
            )
            traj, success = robot.plan.gen_motion_from_goal(
                goal_wxyz_xyz=goal_wxyz_xyz,
                q_init=robot.plan.home_pos.expand(goal_wxyz_xyz.shape[0], -1),
            )

            if traj is None:
                print("Failed to generate a trajectory.")
                goal_button.disabled = False
                return
            assert isinstance(traj, torch.Tensor) and isinstance(success, torch.Tensor)
            if not success.any():
                print("Failed to generate a valid trajectory.")
                goal_button.disabled = False
                return
            
            goal_button.disabled = False
            traj = traj[success]
            assert len(traj.shape) == 3 and traj.shape[-1] == 8

            traj_handle = server.gui.add_slider("Trajectory Index", 0, len(traj) - 1, 1, 0)
            play_handle = server.gui.add_slider("play", min=0, max=traj.shape[1]-1, step=1, initial_value=0)

            def move_to_traj_position():
                assert traj is not None and traj_handle is not None and play_handle is not None
                assert isinstance(traj, torch.Tensor)
                if plan_dropdown.value == "left":
                    robot.q_left = traj[traj_handle.value][play_handle.value].view(-1)
                else:
                    robot.q_right = traj[traj_handle.value][play_handle.value].view(-1)

            @traj_handle.on_update
            def _(_):
                move_to_traj_position()
            @play_handle.on_update
            def _(_):
                move_to_traj_position()

    while True:
        time.sleep(10)

if __name__ == "__main__":
    tyro.cli(main)
