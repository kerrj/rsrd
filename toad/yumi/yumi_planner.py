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

# cuRobo
from curobo.types.math import Pose
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig, CudaRobotModelState, CudaRobotGeneratorConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.util_file import join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig, IKResult
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig, MotionGenResult, PoseCostMetric
from curobo.geom.types import WorldConfig
from curobo.geom.sdf.world import CollisionCheckerType, WorldCollision, WorldCollisionConfig
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.geom.types import Sphere, Cuboid
from curobo.wrap.reacher.trajopt import TrajOptResult, TrajOptSolver, TrajOptSolverConfig


def createTableWorld():
    """Create a simple world with a table, with the top surface at z=0."""
    cfg_dict = {
        "cuboid": {
            "table": {
                "dims": [1.0, 1.0, 0.2],  # x, y, z
                "pose": [0.0, 0.0, 0.00-0.1, 1, 0, 0, 0.0],  # x, y, z, qw, qx, qy, qz
            },
        },
    }
    return WorldConfig.from_dict(cfg_dict)


class YumiPlanner:
    """Motion planner for the yumi, using cuRobo for collision checking and motion generation."""
    _base_dir: Path
    """All paths are provided relative to the root of the repository."""
    _tensor_args: TensorDeviceType
    """Convenience object to store tensor type and device, for curobo."""
    _robot_world: RobotWorld
    """The robot world object, which contains the robot model and the world model (collision, etc)."""
    _ik_solver: IKSolver
    """The IK solver object."""
    _motion_gen: MotionGen
    """The motion generation object."""
    _batch_size: int

    def __init__(
        self,
        batch_size: int,
    ):
        # check if cuda is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available -- curobo requires CUDA.")
        self._base_dir = Path(__file__).parent.parent.parent
        self._tensor_args = TensorDeviceType()

        # Create a base table for world collision.
        self._batch_size = batch_size
        self._setup(
            world_config=createTableWorld(),
        )

    def _setup(
        self,
        world_config: WorldConfig,
    ):
        # Create the robot model.
        cfg_file = load_yaml(join_path(self._base_dir, f"data/yumi.yml"))
        cfg_file["robot_cfg"]["kinematics"]["external_robot_configs_path"] = self._base_dir
        cfg_file["robot_cfg"]["kinematics"]["external_asset_path"] = self._base_dir
        robot_cfg = RobotConfig.from_dict(cfg_file, self._tensor_args)

        _robot_world_cfg = RobotWorldConfig.load_from_config(
            robot_config=robot_cfg,
            world_model=world_config,
            tensor_args=self._tensor_args,
            collision_activation_distance=0.01,
        )
        self._robot_world = RobotWorld(_robot_world_cfg)
        assert isinstance(self._robot_world.world_model, WorldCollision)

        # Set up IK. Doesn't consider collisions.
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
            use_cuda_graph=True,
        )
        self._ik_solver = IKSolver(ik_config)
        self._ik_solver_batch_size = self._batch_size 

        # Set up trajopt.
        trajopt_config = TrajOptSolverConfig.load_from_robot_config(
            robot_cfg,
            world_coll_checker=self._robot_world.world_model,
        )
        self._trajopt_solver = TrajOptSolver(trajopt_config)

    @property
    def device(self) -> torch.device:
        return self._tensor_args.device

    @property
    def home_pos(self) -> torch.Tensor:
        """The home position of the robot, from yumi's retract config."""
        _home_pos = self._robot_world.kinematics.retract_config
        assert type(_home_pos) == torch.Tensor and _home_pos.shape == (14,)
        return _home_pos

    def update_world(self, world_config: WorldConfig):
        """Update world collision model."""
        # Need to clear the world cache, as per in: https://github.com/NVlabs/curobo/issues/263.
        self._robot_world.clear_world_cache()
        self._robot_world.update_world(world_config)

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

    def fk(self, q: torch.Tensor) -> CudaRobotModelState:
        if len(q.shape) == 1:
            q = q.unsqueeze(0)
        assert len(q.shape) == 2
        q = q.to(self.device)
        state = self._robot_world.kinematics.get_state(q)
        return state

    def ik(
        self,
        goal_l_wxyz_xyz: torch.Tensor,
        goal_r_wxyz_xyz: torch.Tensor,
        initial_js: Optional[torch.Tensor] = None,
        get_pos_only: bool = False,
    ) -> Union[List[IKResult], Tuple[torch.Tensor, torch.Tensor]]:
        """Solve IK for both arms simultaneously."""
        assert len(goal_l_wxyz_xyz.shape) == 2 and len(goal_r_wxyz_xyz.shape) == 2
        assert goal_l_wxyz_xyz.shape[-1] == 7 and goal_r_wxyz_xyz.shape[-1] == 7
        assert goal_l_wxyz_xyz.shape == goal_r_wxyz_xyz.shape
        assert goal_l_wxyz_xyz.shape[0] % self._ik_solver_batch_size == 0
        assert goal_r_wxyz_xyz.shape[0] % self._ik_solver_batch_size == 0

        goal_l_wxyz_xyz = goal_l_wxyz_xyz.to(self.device).float()
        goal_r_wxyz_xyz = goal_r_wxyz_xyz.to(self.device).float()

        goal_l = Pose(goal_l_wxyz_xyz[:, 4:], goal_l_wxyz_xyz[:, :4])
        goal_r = Pose(goal_r_wxyz_xyz[:, 4:], goal_r_wxyz_xyz[:, :4])

        if initial_js is not None:
            initial_js = initial_js.to(self.device).float()
        
            if len(initial_js.shape) == 1:
                initial_js = initial_js.unsqueeze(0).expand(goal_l_wxyz_xyz.shape[0], -1)
            assert len(initial_js.shape) == 2, f"Should be of size [batch, dof], instead got: {initial_js.shape}"
            assert initial_js.shape[0] == goal_l_wxyz_xyz.shape[0]
            initial_js = initial_js.unsqueeze(1) # n, batch, dof.

        result_list: List[IKResult] = []
        for i in range(0, goal_l_wxyz_xyz.shape[0], self._ik_solver_batch_size):
            result = self._ik_solver.solve_batch(
                goal_l[i:i+self._ik_solver_batch_size],
                link_poses={
                    "gripper_l_base": goal_l[i:i+self._ik_solver_batch_size],
                    "gripper_r_base": goal_r[i:i+self._ik_solver_batch_size]
                },
                seed_config=None if initial_js is None else initial_js[i:i+self._ik_solver_batch_size]  # None, or [n, batch, dof].
            )
            result_list.append(result)

        if get_pos_only:
            positions = torch.cat([
                result.js_solution.position for result in result_list  # type: ignore
            ], dim=0)  # [batch, time, dof]
            success = torch.cat([
                result.success for result in result_list  # type: ignore
            ], dim=0)

            # Pad success shape, to be [batch, 1].
            if len(success.shape) == 1:
                success = success.unsqueeze(-1)
            return positions, success

        return result_list

    def motiongen(
        self,
        goal_l_wxyz_xyz: torch.Tensor,
        goal_r_wxyz_xyz: torch.Tensor,
        start_state: Optional[Union[JointState, torch.Tensor]] = None,
        start_l_wxyz_xyz: Optional[torch.Tensor] = None,
        start_r_wxyz_xyz: Optional[torch.Tensor] = None,
        get_pos_only: bool = False,
    ) -> Union[List[MotionGenResult], Tuple[torch.Tensor, torch.Tensor]]:
        # Takes around ~0.1 seconds
        assert len(goal_l_wxyz_xyz.shape) == 2 and len(goal_r_wxyz_xyz.shape) == 2
        assert goal_l_wxyz_xyz.shape[0] % self._motion_gen_batch_size == 0
        assert goal_r_wxyz_xyz.shape[0] % self._motion_gen_batch_size == 0

        if start_state is None:
            if (start_l_wxyz_xyz is not None and start_r_wxyz_xyz is not None):
                start_l_wxyz_xyz = start_l_wxyz_xyz.to(self.device).float()
                start_r_wxyz_xyz = start_r_wxyz_xyz.to(self.device).float()
                ik_result = self.ik(start_l_wxyz_xyz, start_r_wxyz_xyz)
                joint_state_list = [ik_result[i].js_solution for i in range(len(ik_result))]
                start_state = JointState.from_position(torch.cat([js.position for js in joint_state_list], dim=0))  # type: ignore
            else:
                raise ValueError("Either start_state or start_l_wxyz_xyz and start_r_wxyz_xyz must be provided.")

        # Make sure that start_state is a batched tensor (bxdof), without gripper.
        if isinstance(start_state, JointState):
            start_joints = start_state.position
        elif isinstance(start_state, torch.Tensor):
            start_joints = start_state
        assert isinstance(start_joints, torch.Tensor)
        if len(start_joints.shape) == 1:
            start_joints = start_joints.expand(goal_l_wxyz_xyz.shape[0], -1)
        assert len(start_joints.shape) == 2 and start_joints.shape[0] == goal_l_wxyz_xyz.shape[0]
        if start_joints.shape[-1] == 16:
            start_joints = start_joints[:, :14]
        start_state = JointState.from_position(start_joints.cuda())

        # Set the goal poses.
        goal_l_wxyz_xyz = goal_l_wxyz_xyz.to(self.device).float()
        goal_r_wxyz_xyz = goal_r_wxyz_xyz.to(self.device).float()

        goal_l_pose = Pose(goal_l_wxyz_xyz[:, 4:], goal_l_wxyz_xyz[:, :4])
        goal_r_pose = Pose(goal_r_wxyz_xyz[:, 4:], goal_r_wxyz_xyz[:, :4])

        # get the current joint locations
        result_list: List[MotionGenResult] = []
        for i in range(0, goal_l_wxyz_xyz.shape[0], self._batch_size):
            result = self._motion_gen.plan_batch(
                start_state[i : i + self._batch_size],
                goal_l_pose[i : i + self._batch_size],
                MotionGenPlanConfig(
                    max_attempts=60,
                    parallel_finetune=True,
                    # enable_finetune_trajopt=False,
                    enable_graph=True,
                    need_graph_success=True,
                    ik_fail_return=5,
                ),
                link_poses={
                    "gripper_l_base": goal_l_pose[i : i + self._batch_size],
                    "gripper_r_base": goal_r_pose[i : i + self._batch_size],
                },  # type: ignore
            )
            result_list.append(result)

        if get_pos_only:
            positions = torch.cat([
                result.interpolated_plan.position for result in result_list  # type: ignore
            ], dim=0)  # [batch, time, dof]
            success = torch.cat([
                result.success for result in result_list  # type: ignore
            ], dim=0)

            # # Pad success shape, to be [batch, time].
            # if len(success.shape) == 1:
            #     success = success.unsqueeze(-1).expand(-1, positions.shape[1])
            # assert positions.shape[0] == success.shape[0]
            # assert positions.shape[1] == success.shape[1]

            return positions, success

        return result_list