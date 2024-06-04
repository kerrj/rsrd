from __future__ import annotations

import time
from pathlib import Path
from typing import List, Literal, Dict, Any, Union, Optional, Tuple
from functools import partial

import torch
import numpy as np
import viser
import viser.transforms as vtf
from viser.extras import ViserUrdf
import trimesh.creation
import tyro
import yourdfpy
from copy import deepcopy

# Yumi
from yumirws.yumi import YuMi

from toad.yumi.yumi_jax_planner import YumiJaxPlanner
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
    """The motion planner for the robot, using curobo. One arm at a time."""
    plan_jax: YumiJaxPlanner
    """Brent's jax-based motion planner, used for waypoint following."""
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

        # Initialize the real robot.
        # YuMi robot code must be placed before any curobo code!
        try:
            self.real = YuMi()
        except:
            print("Failed to initialize the real robot, continuing w/o it.")

        # Initialize the visualizer.
        urdf_path = _base_dir / Path("data/yumi_description/urdf/yumi.urdf")
        self.vis = ViserUrdf(
            target, urdf_path
        )
        # Initialize the planner.
        self.plan = YumiArmPlanner(
            minibatch_size=minibatch_size,
            table_height=TABLE_HEIGHT,
        )
        self.plan_jax = YumiJaxPlanner(urdf_path)  # ... hacky wrapper around it..

        target.scene.add_grid(
            name="grid",
            width=0.6,
            height=0.8,
            position=(0.5, 0, TABLE_HEIGHT),
            section_size=0.05,
        )

        # Get the tooltip-to-gripper offset. TODO remove hardcoding...
        self.tooltip_to_gripper = vtf.SE3.from_translation(np.array([0.0, 0.0, 0.13]))
        self.home_robot()

    def home_robot(self):
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

    @property
    def q_all(self) -> torch.Tensor:
        return self.concat_joints(self.q_right, self.q_left)
    
    @q_all.setter
    def q_all(self, q: torch.Tensor):
        assert q.shape == (16,)
        assert q.device == self.plan.device
        self.q_right = torch.Tensor([*q[:7], q[14]]).to(self.plan.device)
        self.q_left = torch.Tensor([*q[7:14], q[15]]).to(self.plan.device)

    @staticmethod
    def get_left(q_all: torch.Tensor) -> torch.Tensor:
        assert q_all.shape[-1] == 16
        return torch.cat([q_all[..., 7:14], q_all[..., 15:]], dim=-1)

    @staticmethod
    def get_right(q_all: torch.Tensor) -> torch.Tensor:
        assert q_all.shape[-1] == 16
        return torch.cat([q_all[..., :7], q_all[..., 14:15]], dim=-1)

    @staticmethod
    def concat_joints(q_right: torch.Tensor, q_left: torch.Tensor) -> torch.Tensor:
        """Concatenate the left and right joint configurations."""
        assert q_right.shape == q_left.shape == (8,), "Invalid joint shapes -- expected (8,), but got {} and {}.".format(q_right.shape, q_left.shape)
        q_right_arm, q_right_gripper = q_right[:7], q_right[7:]
        q_left_arm, q_left_gripper = q_left[:7], q_left[7:]
        return torch.cat((q_right_arm, q_left_arm, q_right_gripper, q_left_gripper))

    def reset_real(self):
        """Reset the real robot, if available, and re-instanciate it."""
        if self.real is not None:
            del self.real
        self.real = None

        try:
            self.real = YuMi()
        except:
            print("Failed to initialize the real robot, continuing w/o it.")


def main(
    mode: Literal["off", "ik", "goal", "waypoint"] = "off",
    toad_object_path: Optional[Path] = None,
):
    server = viser.ViserServer()
    robot = YumiRobot(server, minibatch_size=(240 if mode == "goal" else 1))

    if mode == "off":
        vis_spheres = server.gui.add_checkbox("Visualize Spheres", False)

        robot.plan.activate_arm("left", robot.q_right)
        sphere_frame = server.scene.add_frame("sphere", visible=False)
        spheres = robot.plan.get_robot_as_spheres(robot.q_left)
        for i, sphere in enumerate(spheres):
            server.scene.add_mesh_trimesh(f"sphere/sphere_{i}", sphere)

        @vis_spheres.on_update
        def _(_):
            sphere_frame.visible = vis_spheres.value

    elif mode == "ik":
        is_collide_world = server.gui.add_checkbox("Collide (world)", False, disabled=True)
        is_collide_self = server.gui.add_checkbox("Collide (self)", False, disabled=True)

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
            is_collide_world.value = (d_world > 0).any().item()  # type: ignore
            is_collide_self.value = (d_self > 0).any().item()  # type: ignore

        @drag_l_handle.on_update
        def _(_):
            update_joints()
            if active_arm_handle.value == "left":
                d_world, d_self = robot.plan.in_collision(robot.q_left)
            else:
                d_world, d_self = robot.plan.in_collision(robot.q_right)
            is_collide_world.value = (d_world > 0).any().item()  # type: ignore
            is_collide_self.value = (d_self > 0).any().item()  # type: ignore

        active_arm_handle.value = "left"
        robot.plan.activate_arm("left", robot.q_right)
        update_joints()

        robot.plan.activate_arm("right", robot.q_left)
        active_arm_handle.value = "right"
        update_joints()

        real_button = server.gui.add_button("Move real robot", disabled=(robot.real is None))
        @real_button.on_click
        def _(_):
            assert robot.real is not None
            real_button.disabled = True
            robot.real.move_joints_sync(
                l_joints=robot.q_left[:7].cpu().numpy().astype(np.float64),
                r_joints=robot.q_right[:7].cpu().numpy().astype(np.float64),
                speed=REAL_ROBOT_SPEED
            )
            robot.real.left.sync()
            robot.real.right.sync()
            real_button.disabled = False

    elif mode == "goal":
        cube = trimesh.creation.box((0.03, 0.03, 0.03))
        cube_toad = GraspableToadObject.from_mesh([cube])
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

    elif mode == "waypoint":
        # assert toad_object_path is not None, "Please provide a toad object path for waypoint mode."
        if toad_object_path is None:
            tsteps = 30
            moving_cube = trimesh.creation.box((0.02, 0.02, 0.02))
            moving_translation = np.concatenate([
                np.linspace(0, 0.1, tsteps).reshape(-1, 1),
                np.linspace(0, 0, tsteps).reshape(-1, 1),
                np.linspace(0, 0, tsteps).reshape(-1, 1),
            ], axis=-1)
            assert moving_translation.shape == (tsteps, 3), "Invalid translation shape, expected (tsteps, 3) and got {}.".format(moving_translation.shape)
            moving_keyframes = torch.from_numpy(vtf.SE3.from_translation(moving_translation).wxyz_xyz)
            moving_keyframes = moving_keyframes.unsqueeze(1)

            anchor_cube = trimesh.creation.box((0.02, 0.02, 0.02))
            anchor_translation = np.concatenate([
                np.linspace(-0.05, -0.05, tsteps).reshape(-1, 1),
                np.linspace(0, 0, tsteps).reshape(-1, 1),
                np.linspace(0, 0, tsteps).reshape(-1, 1),
            ], axis=-1)
            assert anchor_translation.shape == (tsteps, 3), "Invalid translation shape, expected (tsteps, 3) and got {}.".format(anchor_translation.shape)
            anchor_keyframes = torch.from_numpy(vtf.SE3.from_translation(anchor_translation).wxyz_xyz)
            anchor_keyframes = anchor_keyframes.unsqueeze(1)

            toad_obj = GraspableToadObject.from_mesh([moving_cube, anchor_cube])
            mesh_list = toad_obj.meshes
            keyframes = torch.cat([moving_keyframes, anchor_keyframes], dim=1)

        else:
            toad_obj = GraspableToadObject.load(toad_object_path)
            mesh_list = toad_obj.meshes
            keyframes = toad_obj.keyframes

        assert isinstance(keyframes, torch.Tensor) and len(keyframes.shape) == 3

        toad_tf = server.scene.add_transform_controls(
            name="obj",
            position=(0.4, 0, 0.5),
            wxyz=(1, 0, 0, 0),
            scale=0.1,
        )

        # Add the object to the world -- update the transforms based on the part2obj traj.
        list_frame_handle = []
        for i, mesh in enumerate(mesh_list):
            list_frame_handle.append(
                server.scene.add_frame(
                    f"obj/{i}",
                    position=keyframes[0, i, 4:].cpu().numpy(),
                    wxyz=keyframes[0, i, :4].cpu().numpy(),
                    axes_length=0.02,
                    axes_radius=0.005,
                )
            )
        keyframe_handle = server.gui.add_slider("Keyframe", 0, len(keyframes) - 1, 1, 0)
        def update_keyframe(keyframe_idx):
            for i in range(len(mesh_list)):
                list_frame_handle[i].position = keyframes[keyframe_idx, i, 4:].cpu().numpy()
                list_frame_handle[i].wxyz = keyframes[keyframe_idx, i, :4].cpu().numpy()

        @keyframe_handle.on_update
        def _(_):
            update_keyframe(keyframe_handle.value)

        # Set the anchor/part for the object.
        part_handle = server.gui.add_number("Part", -1, len(mesh_list) - 1, 1, -1, disabled=True)
        anchor_handle = server.gui.add_number("Anchor", -1, len(mesh_list) - 1, 1, -1, disabled=True)

        if toad_object_path is None:
            part_handle.value = 0
            anchor_handle.value = 1

        reset_part_anchor_handle = server.gui.add_button("Reset part and anchor")
        @reset_part_anchor_handle.on_click
        def _(_):
            part_handle.value = -1
            anchor_handle.value = -1
            for i in range(len(mesh_list)):
                curry_mesh(i)

        def curry_mesh(idx):
            mesh = toad_obj.meshes[idx]
            if part_handle.value == idx:
                mesh.visual.vertex_colors = [100, 255, 100, 255]
            elif anchor_handle.value == idx:
                mesh.visual.vertex_colors = [255, 100, 100, 255]

            handle = server.scene.add_mesh_trimesh(
                f"obj/{idx}/mesh",
                mesh=mesh,
            )
            @handle.on_click
            def _(_):
                container_handle = server.scene.add_3d_gui_container(
                    f"obj/{idx}/container",
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

        for i in range(len(mesh_list)):
            curry_mesh(i)

        grasp_mesh = toad_obj.grasp_axis_mesh()
        for obj_idx in range(len(mesh_list)):
            for grasp_idx, grasp in enumerate(toad_obj.grasps[obj_idx]):
                server.scene.add_mesh_trimesh(
                    f"obj/{obj_idx}/grasp_{grasp_idx}",
                    grasp_mesh,
                    position=grasp[:3].numpy(),
                    wxyz=grasp[3:].numpy(),
                )

        plan_dropdown = server.gui.add_dropdown("Plan", ("left", "right"), "left")
        @plan_dropdown.on_update
        def _(_):
            nonlocal traj_handle, play_handle
            robot.home_robot()

        goal_button = server.gui.add_button("Move to goal")
        real_button = server.gui.add_button("Move real robot")

        traj, traj_handle, play_handle = None, None, None
        @goal_button.on_click
        def _(_):
            nonlocal traj, traj_handle, play_handle
            goal_button.disabled = True
            try:
                if traj_handle is not None:
                    traj_handle.remove()
                if play_handle is not None:
                    play_handle.remove()
            except:
                pass

            if len(mesh_list) > 1 and (part_handle.value == -1):
                print("Please select a part and an anchor.")
                goal_button.disabled = False
                return

            obj_tf_vtf = vtf.SE3(np.array([*toad_tf.wxyz, *toad_tf.position]))

            mesh_curobo_config = toad_obj.to_world_config(
                poses_wxyz_xyz=[(obj_tf_vtf.multiply(vtf.SE3(wxyz_xyz))).wxyz_xyz for wxyz_xyz in keyframes[0].cpu().numpy()]
            )
            robot.plan.update_world_objects(mesh_curobo_config)

            # Get the grasp path for the selected part.
            moving_grasp = toad_obj.grasps[part_handle.value]
            moving_grasps_gripper = toad_obj.to_gripper_frame(moving_grasp)
            moving_grasp_path_wxyz_xyz = torch.cat([
                torch.Tensor(
                    obj_tf_vtf.multiply(
                        vtf.SE3(keyframes[keyframe_idx, part_handle.value, :].flatten().cpu().numpy())
                    )
                    .multiply(moving_grasps_gripper)
                    .wxyz_xyz
                ).unsqueeze(1)  # [num_grasps, 7] -> [num_grasps, 1, 7]
                for keyframe_idx in range(len(keyframes))
            ], dim=1).float().cuda() # [num_grasps, num_keyframes, 7]

            batch_size = moving_grasp_path_wxyz_xyz.shape[0]
            if plan_dropdown.value == "left":
                robot.plan.activate_arm("left", robot.q_right)
            else:
                robot.plan.activate_arm("right", robot.q_left)
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
            moving_grasps_gripper = toad_obj.to_gripper_frame(moving_grasp, robot.tooltip_to_gripper)
            moving_grasp_path_wxyz_xyz = torch.cat([
                torch.Tensor(
                    obj_tf_vtf.multiply(
                        vtf.SE3(keyframes[keyframe_idx, part_handle.value, :].flatten().cpu().numpy())
                    )
                    .multiply(moving_grasps_gripper)
                    .wxyz_xyz
                ).unsqueeze(1)  # [num_grasps, 7] -> [num_grasps, 1, 7]
                for keyframe_idx in range(len(keyframes))
            ], dim=1).float().cuda() # [num_grasps, num_keyframes, 7]

            # if plan_dropdown.value == "left":
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
                # robot.q_all = traj[traj_handle.value][play_handle.value].view(-1)
                update_keyframe(play_handle.value)

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
                if plan_dropdown.value == "left":
                    robot.real.left.move_joints_traj(
                        joints=traj[traj_handle.value][..., :7].cpu().numpy().astype(np.float64),
                        speed=REAL_ROBOT_SPEED
                    )
                    robot.real.left.sync()
                if plan_dropdown.value == "right":
                    robot.real.right.move_joints_traj(
                        joints=traj[traj_handle.value][..., :7].cpu().numpy().astype(np.float64),
                        speed=REAL_ROBOT_SPEED
                    )
                    robot.real.right.sync()


    while True:
        time.sleep(10)

if __name__ == "__main__":
    tyro.cli(main)
