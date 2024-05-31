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


# Physical height of the table.
TABLE_HEIGHT=-0.006

# Real robot's speed, settings, etc.
REAL_ROBOT_SPEED=(0.1, np.pi/4)


class YumiRobot:
    vis: ViserUrdf
    """The URDF object for the visualizer."""
    plan: YumiPlanner
    """The motion planner for the robot, using curobo."""
    real: Optional[YuMi] = None
    """The physical robot object, if available."""

    _curr_cfg: torch.Tensor
    """Current joint configuration. (16,), including gripper width."""

    def __init__(
        self,
        target: Union[viser.ViserServer, viser.ClientHandle],
        batch_size: int = 1,
    ):
        _base_dir = Path(__file__).parent.parent.parent

        # Initialize the visualizer.
        self.vis = ViserUrdf(
            target, _base_dir / Path("data/yumi_description/urdf/yumi.urdf")
        )
        target.add_grid(
            name="grid",
            width=0.6,
            height=0.8,
            position=(0.5, 0, TABLE_HEIGHT),
            section_size=0.05,
        )
        self.vis._joint_frames[0].remove()

        # Initialize the planner.
        self.plan = YumiPlanner(
            batch_size=batch_size,
            table_height=TABLE_HEIGHT,
        )

        # Initialize the real robot.
        self.real = YuMi()

        # Get the tooltip-to-gripper offset. TODO remove hardcoding...
        self.tooltip_to_gripper = vtf.SE3.from_translation(np.array([0.0, 0.0, 0.128]))

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
            (joint_pos[7:14], joint_pos[:7], gripper_width)
        ).detach().cpu().numpy()
        self.vis.update_cfg(_joint_pos)

        self._curr_cfg = joint_pos


def main(
    mode: Literal["ik", "motiongen", "ik-chain"] = "ik"
):
    server = viser.ViserServer()

    robot = YumiRobot(server)

    drag_l_handle = server.add_transform_controls(
        name="drag_l_handle",
        scale=0.1,
        position=(0.4, 0.2, 0.5),
        wxyz=(0, 1, 0, 0)
    )
    drag_r_handle = server.add_transform_controls(
        name="drag_r_handle",
        scale=0.1,
        position=(0.4, -0.2, 0.5),
        wxyz=(0, 1, 0, 0)
    )

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
        if success.all():
            robot.joint_pos = js[0]

    update_joints()

    if mode == "ik":
        @drag_r_handle.on_update
        def _(_):
            update_joints()
        @drag_l_handle.on_update
        def _(_):
            update_joints()

    elif mode == "motiongen" or mode == "ik-chain":
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

    while True:
        time.sleep(1)

if __name__ == "__main__":
    tyro.cli(main)