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
from dataclasses import dataclass
from threading import RLock
from copy import deepcopy
import tqdm

# cuRobo
from curobo.types.math import Pose
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig, CudaRobotModelState, CudaRobotGeneratorConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.util_file import join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig, IKResult
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig, MotionGenResult, PoseCostMetric, MotionGenStatus
from curobo.geom.types import WorldConfig
from curobo.geom.sdf.world import CollisionCheckerType, WorldCollision, WorldCollisionConfig
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.geom.types import Sphere, Cuboid, Mesh
from curobo.wrap.reacher.trajopt import TrajOptResult, TrajOptSolver, TrajOptSolverConfig

# Generally, really, really need to be careful of race condition here.

YUMI_REST_POSE_LEFT = {
    "yumi_joint_1_l": -1.24839656,
    "yumi_joint_2_l": -1.09802876,
    "yumi_joint_7_l": 1.06634394,
    "yumi_joint_3_l": 0.31386161 - 0.2,
    "yumi_joint_4_l": 1.90125141,
    "yumi_joint_5_l": 1.3205139,
    "yumi_joint_6_l": 2.43563939,
    "gripper_l_joint": 0.025,
}
YUMI_REST_POSE_RIGHT = {
    "yumi_joint_1_r": 1.21442839,
    "yumi_joint_2_r": -1.03205606,
    "yumi_joint_7_r": -1.10072738,
    "yumi_joint_3_r": 0.2987352 - 0.2,
    "yumi_joint_4_r": -1.85257716,
    "yumi_joint_5_r": 1.25363652,
    "yumi_joint_6_r": -2.42181893,
    "gripper_r_joint": 0.025,
}
YUMI_REST_POSE = {
    "yumi_joint_1_r": 1.21442839,
    "yumi_joint_2_r": -1.03205606,
    "yumi_joint_7_r": -1.10072738,
    "yumi_joint_3_r": 0.2987352 - 0.2,
    "yumi_joint_4_r": -1.85257716,
    "yumi_joint_5_r": 1.25363652,
    "yumi_joint_6_r": -2.42181893,
    "yumi_joint_1_l": -1.24839656,
    "yumi_joint_2_l": -1.09802876,
    "yumi_joint_7_l": 1.06634394,
    "yumi_joint_3_l": 0.31386161 - 0.2,
    "yumi_joint_4_l": 1.90125141,
    "yumi_joint_5_l": 1.3205139,
    "yumi_joint_6_l": 2.43563939,
    "gripper_r_joint": 0.025,
    "gripper_l_joint": 0.025,
}


class YumiArmPlanner:
    """Motion planner for *one* of the yumi arms, using cuRobo for collision checking and motion generation.
    Should have 8 degrees of freedom (7, then 1 for the gripper.)
    
    Can swap to the other arm, but can only plan for one arm at a time."""

    _base_dir: Path
    """All paths are provided relative to the root of the repository."""
    _tensor_args: TensorDeviceType
    """Convenience object to store tensor type and device, for curobo."""
    _robot_world: RobotWorld
    """The robot world object, which contains the robot model and the world model (collision, etc)."""
    _robot_world_full: RobotWorld
    """(14 / 16 dof) robot world obj, for checking both arms for collision."""
    _ik_solver:IKSolver
    """The inverse kinematics solver object."""
    _motion_gen: MotionGen
    """The motion generation object."""
    _minibatch_size: int
    """The minibatch size for the planner."""
    table_height: float
    """The height of the table, in world coordinates."""
    _active_arm: Literal["left", "right"] = "left"
    """The arm that is currently active."""
    _freeze_gripper: bool = True
    """Whether to freeze the gripper joint when planning (i.e., remove it from DoF.)"""

    def __init__(
        self,
        minibatch_size: int,
        table_height: float,
    ):
        # check if cuda is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available -- curobo requires CUDA.")
        self._base_dir = Path(__file__).parent.parent.parent
        self._tensor_args = TensorDeviceType()

        # Create a base table for world collision.
        self._minibatch_size = minibatch_size
        self.table_height = table_height
        self._setup(
            world_config=self.createTableWorld(table_height),
        )
        self._active_arm_lock = RLock()

    def _setup(
        self,
        world_config: WorldConfig,
    ):
        robot_cfg = self.create_cfg("left", freeze_gripper=self._freeze_gripper)
        _robot_world_cfg = RobotWorldConfig.load_from_config(
            robot_config=robot_cfg,
            world_model=world_config,
            tensor_args=self._tensor_args,
            collision_activation_distance=0.01,
        )
        self._robot_world = RobotWorld(_robot_world_cfg)
        assert isinstance(self._robot_world.world_model, WorldCollision)

        _robot_cfg_full = self.create_cfg_full(freeze_gripper=self._freeze_gripper)
        _robot_world_cfg_full = RobotWorldConfig.load_from_config(
            robot_config=_robot_cfg_full,
            world_model=world_config,
            tensor_args=self._tensor_args,
            collision_activation_distance=0.01,
        )
        self._robot_world_full = RobotWorld(_robot_world_cfg_full)

        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            None,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=10,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=self._tensor_args,
            collision_activation_distance=0.01,
            world_coll_checker=self._robot_world.world_model,
            use_cuda_graph=True,
        )
        self._ik_solver = IKSolver(ik_config)

        # Set up motion planning! This considers collisions.
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            None,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_ik_seeds=10,
            num_trajopt_seeds=4,
            interpolation_dt=0.25,
            trajopt_dt=0.25,
            collision_activation_distance=0.005,  # keeping this small is important, because gripper balls interfere w/ opt.
            interpolation_steps=100,
            world_coll_checker=self._robot_world.world_model,
            self_collision_check=True,
            self_collision_opt=True,
            use_cuda_graph=True,
        )
        self._motion_gen = MotionGen(motion_gen_config)
        # Warmup seems to get messed up with the arm-swapping here.
        # self._motion_gen.warmup(
        #     batch=self._minibatch_size,
        # )

        self.update_world_objects([])  # for ground, and camera.

    def create_cfg(
        self,
        active_arm: Literal["left", "right"],
        locked_arm_q: Optional[torch.Tensor] = None,
        freeze_gripper: bool = True,
    ) -> RobotConfig:
        """Load the robot model, while also locking the joints of the one of the arms. Locks the other arm at `locked_arm_q`."""
        # Create the robot model.
        cfg_file = load_yaml(join_path(self._base_dir, f"data/yumi.yml"))
        cfg_file["robot_cfg"]["kinematics"]["external_robot_configs_path"] = self._base_dir
        cfg_file["robot_cfg"]["kinematics"]["external_asset_path"] = self._base_dir

        # lock the joints of the other side, as well as the gripper joint!
        # If locking joint pos is provided, use that. Otherwise, lock at the rest pose.
        active_poses = YUMI_REST_POSE_LEFT if active_arm == "left" else YUMI_REST_POSE_RIGHT
        locking_poses = YUMI_REST_POSE_RIGHT if active_arm == "left" else YUMI_REST_POSE_LEFT
        active_poses, locking_poses = deepcopy(active_poses), deepcopy(locking_poses)
        if locked_arm_q is not None:
            assert locked_arm_q.shape == (8,)
            cfg_file["robot_cfg"]["kinematics"]["lock_joints"] = {
                k: v for k, v in zip(locking_poses.keys(), locked_arm_q.cpu().numpy())
            }
        else:
            cfg_file["robot_cfg"]["kinematics"]["lock_joints"] = locking_poses

        cfg_file["robot_cfg"]["kinematics"]["ee_link"] = (
            # "gripper_l_base" if active_arm == "left" else "gripper_r_base"
            "left_dummy_point" if active_arm == "left" else "right_dummy_point"
        )

        if freeze_gripper:
            cfg_file["robot_cfg"]["kinematics"]["cspace"]["joint_names"] = list(active_poses.keys())[:-1]
            cfg_file["robot_cfg"]["kinematics"]["cspace"]["retract_config"] = list(active_poses.values())[:-1]
            cfg_file["robot_cfg"]["kinematics"]["cspace"]["null_space_weight"] = [1]*7
            cfg_file["robot_cfg"]["kinematics"]["cspace"]["cspace_distance_weight"] = [1]*7

            # Add gripper to the lock joints.
            cfg_file["robot_cfg"]["kinematics"]["lock_joints"][
                list(active_poses.keys())[-1]
            ] = active_poses[list(active_poses.keys())[-1]]
        else:
            cfg_file["robot_cfg"]["kinematics"]["cspace"]["joint_names"] = list(active_poses.keys())
            cfg_file["robot_cfg"]["kinematics"]["cspace"]["retract_config"] = list(active_poses.values())
            cfg_file["robot_cfg"]["kinematics"]["cspace"]["null_space_weight"] = [1]*8
            cfg_file["robot_cfg"]["kinematics"]["cspace"]["cspace_distance_weight"] = [1]*8

        robot_cfg = RobotConfig.from_dict(cfg_file, self._tensor_args)
        return robot_cfg

    def create_cfg_full(
        self,
        freeze_gripper: bool = True,
    ):
        # Create the robot model.
        cfg_file = load_yaml(join_path(self._base_dir, f"data/yumi.yml"))
        cfg_file["robot_cfg"]["kinematics"]["external_robot_configs_path"] = self._base_dir
        cfg_file["robot_cfg"]["kinematics"]["external_asset_path"] = self._base_dir

        # lock the joints of the other side, as well as the gripper joint!
        # If locking joint pos is provided, use that. Otherwise, lock at the rest pose.
        cfg_file["robot_cfg"]["kinematics"]["ee_link"] = "right_dummy_point"  # ordering affects joints.

        poses = deepcopy(YUMI_REST_POSE)

        if freeze_gripper:
            cfg_file["robot_cfg"]["kinematics"]["cspace"]["joint_names"] = list(poses.keys())[:-2]
            cfg_file["robot_cfg"]["kinematics"]["cspace"]["retract_config"] = list(poses.values())[:-2]
            cfg_file["robot_cfg"]["kinematics"]["cspace"]["null_space_weight"] = [1]*14
            cfg_file["robot_cfg"]["kinematics"]["cspace"]["cspace_distance_weight"] = [1]*14

            # Add gripper to the lock joints.
            cfg_file["robot_cfg"]["kinematics"]["lock_joints"][
                list(poses.keys())[-2]
            ] = poses[list(poses.keys())[-2]]
            cfg_file["robot_cfg"]["kinematics"]["lock_joints"][
                list(poses.keys())[-1]
            ] = poses[list(poses.keys())[-1]]
        else:
            cfg_file["robot_cfg"]["kinematics"]["cspace"]["joint_names"] = list(poses.keys())
            cfg_file["robot_cfg"]["kinematics"]["cspace"]["retract_config"] = list(poses.values())
            cfg_file["robot_cfg"]["kinematics"]["cspace"]["null_space_weight"] = [1]*16
            cfg_file["robot_cfg"]["kinematics"]["cspace"]["cspace_distance_weight"] = [1]*16

        robot_cfg = RobotConfig.from_dict(cfg_file, self._tensor_args)
        return robot_cfg

    @property
    def active_arm(self) -> Literal["left", "right"]:
        with self._active_arm_lock:
            return self._active_arm

    @property
    def gripper_frozen(self) -> bool:
        with self._active_arm_lock:
            return self._freeze_gripper

    def activate_arm(self, arm: Literal["left", "right"], locked_arm_q: Optional[torch.Tensor] = None, freeze_gripper: bool = True):
        """Set the movable arm, and update the robot world."""
        if arm == self._active_arm:
            return
        with self._active_arm_lock:
            self._active_arm = arm
            robot_cfg = self.create_cfg(arm, locked_arm_q=locked_arm_q, freeze_gripper=freeze_gripper)
            # Running the update function is important.
            # Also, note that curobo originally has a bug in `copy_` for inplace self.lock_jointstate update.
            self._robot_world.kinematics.update_kinematics_config(robot_cfg.kinematics.kinematics_config)
            self._motion_gen.kinematics.update_kinematics_config(robot_cfg.kinematics.kinematics_config)
            self._ik_solver.kinematics.update_kinematics_config(robot_cfg.kinematics.kinematics_config)

    @property
    def device(self) -> torch.device:
        return self._tensor_args.device

    @property
    def home_pos(self) -> torch.Tensor:
        """The home position of the robot, from yumi's retract config."""
        _home_pos = self._robot_world.kinematics.retract_config
        assert isinstance(_home_pos, torch.Tensor)
        if self._freeze_gripper:
            _home_pos = torch.cat([_home_pos, torch.Tensor([0.025]).to(self.device)], dim=-1)
        assert _home_pos.shape == (8,)
        return _home_pos

    def fk(self, q: torch.Tensor) -> CudaRobotModelState:
        """Get the forward kinematics of the robot, given the joint positions."""
        if len(q.shape) == 1:
            q = q.unsqueeze(0)
        assert len(q.shape) == 2
        q = q.to(self.device)
        with self._active_arm_lock:
            state = self._robot_world.kinematics.get_state(q)
        return state

    def ik(self, goal_wxyz_xyz: torch.Tensor, q_init: Optional[torch.Tensor] = None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get the inverse kinematics of the robot, given the goal position. This considers collision!
        Args:
            goal_wxyz_xyz: The goal position, in world coordinates (n, wxyz_xyz)
            q_init: The initial joint position to seed the IK from. If None, uses the retract position. (n, 8)

        Returns:
            q: The joint position that achieves the goal. (n, 8)
            succ: Whether the IK was successful. (n,)
        """
        batch_size = goal_wxyz_xyz.shape[0]
        # assert batch_size % self._minibatch_size == 0, f"Batch size ({batch_size}) must be divisible by {self._minibatch_size}."
        assert goal_wxyz_xyz.shape == (batch_size, 7)
        assert q_init is None or q_init.shape == (batch_size, 8), f"q_init must be None or (batch_size, 8), got {q_init.shape}."

        with self._active_arm_lock:
            # if q_init is None:
            #     q_init = self.home_pos.view(1, -1).repeat(batch_size, 1)

            if q_init is not None:
                q_init = q_init.to(self.device)
                assert q_init.shape == (batch_size, 8)

                if self._freeze_gripper:
                    q_init = q_init[:, :-1].contiguous()  # remove gripper joint from dof.
                
                q_init = q_init.unsqueeze(0)  # [1, batch, dof] for seed.

            assert goal_wxyz_xyz.shape == (batch_size, 7)
            goal_wxyz_xyz = goal_wxyz_xyz.to(self.device)
            goal_pose = Pose(goal_wxyz_xyz[:, 4:], goal_wxyz_xyz[:, :4])

            # Loop through IK.
            result_list = []
            # for i in tqdm.trange(0, batch_size, self._minibatch_size, desc="IK"):
            result = self._ik_solver.solve_batch(
                goal_pose=goal_pose,
                seed_config=q_init
            )
            result_list.append(result)

            # js_solution is [batch, time, dof] tensor!
            q = torch.cat([
                result.solution for result in result_list  # type: ignore
            ], dim=0)  # [batch, time, dof]
            succ = torch.cat([
                result.success for result in result_list  # type: ignore
            ], dim=0)

            q = q[:, 0, :]  # [batch, dof]
            succ = succ[:, 0]  # [batch]

            if self._freeze_gripper:
                q = torch.cat([q, torch.Tensor([0.025]).expand(batch_size, 1).to(self.device)], dim=-1)

        assert q.shape == (batch_size, 8) and succ.shape == torch.Size([batch_size])
        return (q, succ)

    def gen_motion_from_goal(
        self,
        goal_wxyz_xyz: torch.Tensor,
        q_init: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get the inverse kinematics of the robot, given the goal position.
        Args:
            goal_wxyz_xyz: The goal position, in world coordinates (n, wxyz_xyz)
            q_init: The initial joint position to seed the IK from. If None, uses the retract position. (n, 8)

        Returns:
            q: The trajectories that achieves the goal. (n, time, 8)
            succ: Whether the IK was successful. (n,)
        """
        batch_size = goal_wxyz_xyz.shape[0]
        assert batch_size % self._minibatch_size == 0, f"Batch size ({batch_size}) must be divisible by {self._minibatch_size}."
        assert goal_wxyz_xyz.shape == (batch_size, 7)
        assert q_init is None or q_init.shape == (batch_size, 8), f"q_init must be None or (batch_size {batch_size}, 8), got {q_init.shape}."

        with self._active_arm_lock:
            if q_init is None:
                q_init = self.home_pos.view(1, -1).repeat(batch_size, 1)

            goal_wxyz_xyz = goal_wxyz_xyz.to(self.device)
            q_init = q_init.to(self.device)
            assert goal_wxyz_xyz.shape == (batch_size, 7) and q_init.shape == (batch_size, 8)

            goal_pose = Pose(goal_wxyz_xyz[:, 4:], goal_wxyz_xyz[:, :4])
            if self._freeze_gripper:
                q_init = q_init[:, :-1].contiguous()  # remove gripper joint from dof.
            start_state = JointState.from_position(q_init)

            result_list: List[MotionGenResult] = []
            for i in tqdm.trange(0, batch_size, self._minibatch_size, desc="Motion"):
                result = self._motion_gen.plan_batch(
                    start_state=start_state[i : i+self._minibatch_size],  # type: ignore
                    goal_pose=goal_pose[i : i+self._minibatch_size],
                    plan_config=MotionGenPlanConfig(
                        max_attempts=1,
                        parallel_finetune=True,
                        # enable_graph=True,
                    ),
                )
                result_list.append(result)

            tsteps = self._motion_gen.interpolation_steps
            for result in result_list:
                # If no solution is found, we set an invalid plan + set success ot false...
                # if result.interpolated_plan is None:
                #     result.interpolated_plan = JointState.from_position(torch.zeros(
                #         (self._minibatch_size, tsteps, q_init.shape[-1]),  # match dof.
                #         device=self.device,
                #     ))
                #     result.success = torch.zeros(batch_size, device=self.device).bool()
                assert self._minibatch_size == 1, "This block assumes minibatch size of 1..."
                if result.status == MotionGenStatus.IK_FAIL or result.status == MotionGenStatus.TRAJOPT_FAIL:
                    result.interpolated_plan = JointState.from_position(torch.zeros(
                        (tsteps, q_init.shape[-1]),  # match dof.
                        device=self.device,
                    ))
                    result.success = torch.zeros(1, device=self.device).bool()
                # elif result.interpolated_plan is None or len(result.interpolated_plan.position.shape) == 3:
                #     result.interpolated_plan = JointState.from_position(torch.zeros(
                #         (tsteps, q_init.shape[-1]),  # match dof.
                #         device=self.device,
                #     ))
                #     result.success = torch.zeros(1, device=self.device).bool()

            # result.interpolated_plan.position with minibatch_size 1 [time, dof] --> [batch, time, dof].
            q = torch.cat([
                result.interpolated_plan.position.unsqueeze(0) for result in result_list  # type: ignore
            ], dim=0)  # [batch, time, dof]
            succ = torch.cat([
                result.success for result in result_list  # type: ignore
            ], dim=0)  # [batch,]

            if self._freeze_gripper:
                q = torch.cat([q, torch.Tensor([0.025]).expand(batch_size, tsteps, 1).to(self.device)], dim=-1)

        assert q.shape == (batch_size, tsteps, 8) and succ.shape == torch.Size([batch_size])
        return (q, succ)

    def gen_motion_from_goal_joints(
        self,
        goal_js: torch.Tensor,
        q_init: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get the inverse kinematics of the robot, given the goal position.
        Args:
            goal_wxyz_xyz: The goal position, in world coordinates (n, wxyz_xyz)
            q_init: The initial joint position to seed the IK from. If None, uses the retract position. (n, 8)

        Returns:
            q: The trajectories that achieves the goal. (n, time, 8)
            succ: Whether the IK was successful. (n,)
        """
        batch_size = goal_js.shape[0]
        assert batch_size % self._minibatch_size == 0, f"Batch size ({batch_size}) must be divisible by {self._minibatch_size}."
        assert goal_js.shape == (batch_size, 8)
        assert q_init is None or q_init.shape == (batch_size, 8), f"q_init must be None or (batch_size {batch_size}, 8), got {q_init.shape}."

        with self._active_arm_lock:
            if q_init is None:
                q_init = self.home_pos.view(1, -1).repeat(batch_size, 1)

            goal_js = goal_js.to(self.device)
            q_init = q_init.to(self.device)
            assert goal_js.shape == (batch_size, 8) and q_init.shape == (batch_size, 8)

            if self._freeze_gripper:
                q_init = q_init[:, :-1].contiguous()  # remove gripper joint from dof.
                goal_js = goal_js[:, :-1].contiguous()  # remove gripper joint from dof.
            start_state = JointState.from_position(q_init)
            goal_state = JointState.from_position(goal_js)

            result_list: List[MotionGenResult] = []
            assert self._minibatch_size == 1, "This block assumes minibatch size of 1..."
            for i in tqdm.trange(0, batch_size, self._minibatch_size, desc="Motion"):
                result = self._motion_gen.plan_single_js(
                    start_state=start_state[i : i+self._minibatch_size],  # type: ignore
                    goal_state=goal_state[i : i+self._minibatch_size],
                    plan_config=MotionGenPlanConfig(
                        max_attempts=1,
                        parallel_finetune=True,
                        # enable_graph=True,
                    ),
                )
                result_list.append(result)

            tsteps = self._motion_gen.interpolation_steps
            for result in result_list:
                # If no solution is found, we set an invalid plan + set success ot false...
                # if result.interpolated_plan is None:
                #     result.interpolated_plan = JointState.from_position(torch.zeros(
                #         (self._minibatch_size, tsteps, q_init.shape[-1]),  # match dof.
                #         device=self.device,
                #     ))
                #     result.success = torch.zeros(batch_size, device=self.device).bool()
                assert self._minibatch_size == 1, "This block assumes minibatch size of 1..."
                if result.status == MotionGenStatus.IK_FAIL or result.status == MotionGenStatus.TRAJOPT_FAIL or result.interpolated_plan is None:
                    result.interpolated_plan = JointState.from_position(torch.zeros(
                        (tsteps, q_init.shape[-1]),  # match dof.
                        device=self.device,
                    ))
                    result.success = torch.zeros(1, device=self.device).bool()
                # elif result.interpolated_plan is None or len(result.interpolated_plan.position.shape) == 3:
                #     result.interpolated_plan = JointState.from_position(torch.zeros(
                #         (tsteps, q_init.shape[-1]),  # match dof.
                #         device=self.device,
                #     ))
                #     result.success = torch.zeros(1, device=self.device).bool()
                elif result.interpolated_plan.position.shape[0] != tsteps:
                    position = result.interpolated_plan.position
                    assert isinstance(position, torch.Tensor)
                    position = torch.cat([
                        position,
                        position[-1].unsqueeze(0).repeat(tsteps - position.shape[0], 1),
                    ])
                    result.interpolated_plan = JointState.from_position(position)

            # result.interpolated_plan.position with minibatch_size 1 [time, dof] --> [batch, time, dof].
            q = torch.cat([
                result.interpolated_plan.position.unsqueeze(0) for result in result_list  # type: ignore
            ], dim=0)  # [batch, time, dof]
            succ = torch.cat([
                result.success for result in result_list  # type: ignore
            ], dim=0)  # [batch,]

            if self._freeze_gripper:
                q = torch.cat([q, torch.Tensor([0.025]).expand(batch_size, tsteps, 1).to(self.device)], dim=-1)

        assert q.shape == (batch_size, tsteps, 8) and succ.shape == torch.Size([batch_size])
        return (q, succ)

    @staticmethod
    def createTableWorld(table_height):
        """Create a simple world with a table, with the top surface at z={table_height}."""
        cfg_dict = {
            "cuboid": {
                "table": {
                    "dims": [1.0, 1.0, 0.2],  # x, y, z
                    "pose": [0.0, 0.0, -0.1+table_height, 1, 0, 0, 0.0],  # x, y, z, qw, qx, qy, qz
                },
                "camera": {
                    "dims": [0.001*65, 0.001*200, 0.001*250],  # x, y, z
                    "pose": [0.075, 0.0, 0.12, 1, 0, 0, 0.0],  # x, y, z, qw, qx, qy, qz
                    # Per slack:
                    # X: 75mm
                    # Y: 0mm
                    # Z: 120mm
                    # X size:65mm
                    # Y size: 200mm
                    # Z size: 250 mm
                }
            },
        }
        return WorldConfig.from_dict(cfg_dict)

    def update_world(self, world_config: WorldConfig):
        """Update world collision model."""
        # Need to clear the world cache, as per in: https://github.com/NVlabs/curobo/issues/263.
        self._robot_world.clear_world_cache()
        self._robot_world.update_world(world_config)
        self._robot_world_full.clear_world_cache()
        self._robot_world_full.update_world(world_config)

    def update_world_objects(self, objects: Optional[Union[List[Sphere], List[Mesh]]] = None):
        """Update the world objects -- reinstantiates the world table, too."""
        # Need to clear the world cache, as per in: https://github.com/NVlabs/curobo/issues/263.
        world_config = self.createTableWorld(self.table_height)
        assert isinstance(world_config.mesh, list) and isinstance(world_config.sphere, list)
        if objects is not None:
            for obj in objects:
                if isinstance(obj, Sphere):
                    world_config.sphere.append(obj)
                elif isinstance(obj, Mesh):
                    world_config.mesh.append(obj)
                else:
                    raise ValueError(f"Unsupported object type., {type(objects[0])}")
        self.update_world(world_config)

    def in_collision(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns d_world, d_self.
        get_world_coll_dist accepts b, horizon, dof.
        If q is 1D, it is treated as a single trajectory, and reshaped to 1x1xdof.
        If q is 2D, it is treated as a batch of trajectories, and reshaped to bx1xdof.
        """
        if len(q.shape) == 1:
            q = q.unsqueeze(0).unsqueeze(0)
        elif len(q.shape) == 2:
            q = q.unsqueeze(1)
        assert len(q.shape) == 3
        d_world, d_self = self._robot_world.get_world_self_collision_distance_from_joint_trajectory(q)

        return (d_world.squeeze(), d_self.squeeze())

    def in_collision_full(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(q.shape) == 1:
            q = q.unsqueeze(0).unsqueeze(0)
        elif len(q.shape) == 2:
            q = q.unsqueeze(1)
        assert len(q.shape) == 3
        d_world, d_self = self._robot_world_full.get_world_self_collision_distance_from_joint_trajectory(q)

        return (d_world.squeeze(), d_self.squeeze())

    def get_robot_as_spheres(
        self,
        joint_pos: torch.Tensor,
    ) -> List[trimesh.Trimesh]:
        """Returns the robot as a list of spheres, given the joint positions."""
        assert joint_pos.shape == (8,)
        spheres: List[Sphere] = self._robot_world.kinematics.get_robot_as_spheres(joint_pos)[0]

        return [
            trimesh.creation.uv_sphere(
                radius=sphere.radius,
                transform=trimesh.transformations.translation_matrix(sphere.position)
                )
            for sphere in spheres
        ]
