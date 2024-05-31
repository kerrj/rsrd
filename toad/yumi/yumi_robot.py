from __future__ import annotations

import time
from pathlib import Path
from typing import List, Literal, Dict, Any, Union, Optional, Tuple

import torch
import numpy as np
import viser
import viser.transforms as vtf
from viser.extras import ViserUrdf
import trimesh
import trimesh.creation

# Yumi
from yumirws.yumi import YuMi

from toad.yumi.yumi_planner import YumiPlanner

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
            position=(0.5, 0, 0.00),
            section_size=0.05,
        )
        self.vis._joint_frames[0].remove()

        # Initialize the planner.
        self.plan = YumiPlanner(
            batch_size=batch_size,
        )

        # Initialize the real robot.
        self.real = YuMi()

        # Get the tooltip-to-gripper offset. TODO remove hardcoding...
        self.tooltip_to_gripper = vtf.SE3.from_translation(np.array([0.0, 0.0, 0.128]))

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


if __name__ == "__main__":
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

    js, success = robot.plan.ik(
        torch.Tensor([*drag_l_handle.wxyz, *drag_l_handle.position]).view(1, 7),
        torch.Tensor([*drag_r_handle.wxyz, *drag_r_handle.position]).view(1, 7),
        get_pos_only=True,
    )
    assert isinstance(js, torch.Tensor) and isinstance(success, torch.Tensor)
    assert js.shape == (1, 1, 16) and success.shape == (1, 1)
    if success.all():
        robot.joint_pos = js[0, 0]

    waypoint_button = server.add_gui_button("Add waypoint")
    drag_button = server.add_gui_button("Match drag")
    drag_slider = server.add_gui_slider("Time", 0, 1, 0.01, 0, disabled=True)

    traj = None
    waypoint_queue = [[], []]
    @waypoint_button.on_click
    def _(_):
        waypoint_queue[0].append(torch.Tensor([*drag_l_handle.wxyz, *drag_l_handle.position]).view(1, 7))
        waypoint_queue[1].append(torch.Tensor([*drag_r_handle.wxyz, *drag_r_handle.position]).view(1, 7))

    @drag_button.on_click
    def _(_):
        global traj
        drag_slider.disabled = True
        drag_button.disabled = True

        if len(waypoint_queue[0]) == 0:
            waypoint_queue[0].append(torch.Tensor([*drag_l_handle.wxyz, *drag_l_handle.position]).view(1, 7))
            waypoint_queue[1].append(torch.Tensor([*drag_r_handle.wxyz, *drag_r_handle.position]).view(1, 7))

        start_pose = robot.joint_pos

        start = time.time()
        traj_pieces = []
        prev_start_state = start_pose
        for i in range(len(waypoint_queue[0])):
            js, success = robot.plan.motiongen(
                waypoint_queue[0][i],
                waypoint_queue[1][i],
                start_state=prev_start_state,
                get_pos_only=True,
            )
            assert isinstance(js, torch.Tensor) and isinstance(success, torch.Tensor)
            assert len(js.shape) == 2 and js.shape[-1] == 14 and success.shape == (1,)
            if not success.all():
                drag_slider.value = 0
                drag_button.disabled = False
                drag_slider.disabled = False
                print("Failed to generate motion.")
                break
            prev_start_state = js[-1:]
            traj_pieces.append(js)

        traj = torch.concat(traj_pieces, dim=0)
        print("MotionGen took", time.time() - start, "seconds")
        waypoint_queue[0] = []
        waypoint_queue[1] = []

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