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
from curobo.geom.types import Sphere, Cuboid, Mesh
from curobo.wrap.reacher.trajopt import TrajOptResult, TrajOptSolver, TrajOptSolverConfig

# TODO decide if 14, or 16,. -- gripper...?
# TODO add size checks at some point.


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
    """The batch size for the planner."""
    table_height: float
    """The height of the table, in world coordinates."""

    def __init__(
        self,
        batch_size: int,
        table_height: float,
    ):
        # check if cuda is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available -- curobo requires CUDA.")
        self._base_dir = Path(__file__).parent.parent.parent
        self._tensor_args = TensorDeviceType()

        # Create a base table for world collision.
        self._batch_size = batch_size
        self.table_height = table_height
        self._setup(
            world_config=self.createTableWorld(table_height),
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

        # Set up motion planning! This considers collisions.
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            None,
            rotation_threshold=0.05,
            position_threshold=0.01,
            num_ik_seeds=10,
            num_trajopt_seeds=4,
            interpolation_dt=0.25,
            trajopt_dt=0.25,
            collision_activation_distance=0.01,
            interpolation_steps=30,
            # world_coll_checker=self._robot_world.world_model,
            use_cuda_graph=True,
        )
        self._motion_gen = MotionGen(motion_gen_config)
        self._motion_gen.warmup(
            batch=self._batch_size,
            warmup_js_trajopt=False,
        )

    @property
    def device(self) -> torch.device:
        return self._tensor_args.device

    @property
    def home_pos(self) -> torch.Tensor:
        """The home position of the robot, from yumi's retract config."""
        _home_pos = self._robot_world.kinematics.retract_config
        assert type(_home_pos) == torch.Tensor and _home_pos.shape == (14,)
        return _home_pos

    @staticmethod
    def createTableWorld(table_height):
        """Create a simple world with a table, with the top surface at z={table_height}."""
        cfg_dict = {
            "cuboid": {
                "table": {
                    "dims": [1.0, 1.0, 0.2],  # x, y, z
                    "pose": [0.0, 0.0, 0.00-table_height, 1, 0, 0, 0.0],  # x, y, z, qw, qx, qy, qz
                },
            },
        }
        return WorldConfig.from_dict(cfg_dict)

    def update_world(self, world_config: WorldConfig):
        """Update world collision model."""
        # Need to clear the world cache, as per in: https://github.com/NVlabs/curobo/issues/263.
        self._robot_world.clear_world_cache()
        self._robot_world.update_world(world_config)

    def update_world_objects(self, objects: Union[List[Sphere], List[Mesh]]):
        """Update the world objects -- reinstantiates the world table, too."""
        # Need to clear the world cache, as per in: https://github.com/NVlabs/curobo/issues/263.
        world_config = self.createTableWorld(self.table_height)
        assert isinstance(world_config.mesh, list) and isinstance(world_config.sphere, list)
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

    def fk(self, q: torch.Tensor) -> CudaRobotModelState:
        """Get the forward kinematics of the robot, given the joint positions."""
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
        get_result: bool = False,
    ) -> Union[List[IKResult], Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """Solve IK for both arms simultaneously.
        goal_*_wxyz_xyz: [n, 7] tensor, where the first 4 elements are the quaternion, and the last 3 are the position.
        initial_js is a [n, dof] or [dof] tensor, where dof is 14.

        returns [n, dof] (position) and [n,] (success).

        If get_result is True, returns the generated IKResults.
        If no solutions are found using `pos_only`, will return None.
        """
        assert len(goal_l_wxyz_xyz.shape) == 2 and len(goal_r_wxyz_xyz.shape) == 2
        assert goal_l_wxyz_xyz.shape[-1] == 7 and goal_r_wxyz_xyz.shape[-1] == 7
        assert goal_l_wxyz_xyz.shape == goal_r_wxyz_xyz.shape
        assert goal_l_wxyz_xyz.shape[0] % self._ik_solver_batch_size == 0
        assert goal_r_wxyz_xyz.shape[0] % self._ik_solver_batch_size == 0

        batch_size = goal_l_wxyz_xyz.shape[0]

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
            assert initial_js.shape[-1] == 14

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

        if get_result:
            return result_list

        result_list = list(filter(lambda x: x.success.any(), result_list))
        if len(result_list) == 0:
            return None, None

        # js_solution is [batch, time, dof] tensor!
        positions = torch.cat([
            result.js_solution.position for result in result_list  # type: ignore
        ], dim=0)  # [batch, time, dof]
        success = torch.cat([
            result.success for result in result_list  # type: ignore
        ], dim=0)

        assert positions.shape[1] == 1  # Only one time step.
        positions = positions.squeeze(1)
        success = success.squeeze(1)
        positions = positions[..., :14]  # Remove the gripper joints.

        assert positions.shape == (batch_size, 14) and success.shape == (batch_size,)

        return positions, success


    def gen_motion_from_goal(
        self,
        goal_l_wxyz_xyz: torch.Tensor,
        goal_r_wxyz_xyz: torch.Tensor,
        initial_js: Optional[torch.Tensor] = None,
        start_l_wxyz_xyz: Optional[torch.Tensor] = None,
        start_r_wxyz_xyz: Optional[torch.Tensor] = None,
        get_result: bool = False,
    ) -> Union[List[MotionGenResult], Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """
        Generate collision-free motion plans for both arms simultaneously.
        goal_*_wxyz_xyz: [batch, 7] tensor, where the first 4 elements are the quaternion, and the last 3 are the position.

        returns [batch, time, dof] (position) and [batch,] (success).

        If get_result is True, returns the MotionGenResult list.
        If no solutions are found using `pos_only`, will return None.
        """
        # Takes around ~0.1 seconds
        assert len(goal_l_wxyz_xyz.shape) == 2 and len(goal_r_wxyz_xyz.shape) == 2
        assert goal_l_wxyz_xyz.shape[0] % self._batch_size == 0
        assert goal_r_wxyz_xyz.shape[0] % self._batch_size == 0
        batch_size = goal_l_wxyz_xyz.shape[0]

        if initial_js is None:
            if (start_l_wxyz_xyz is not None and start_r_wxyz_xyz is not None):
                start_l_wxyz_xyz = start_l_wxyz_xyz.to(self.device).float()
                start_r_wxyz_xyz = start_r_wxyz_xyz.to(self.device).float()
                js, _ = self.ik(
                    start_l_wxyz_xyz, start_r_wxyz_xyz
                )
                assert js is not None, "Failed to generate initial joint states."
                assert isinstance(js, torch.Tensor)
                initial_js = js
            else:
                raise ValueError("Either start_state or start_l_wxyz_xyz and start_r_wxyz_xyz must be provided.")

        # Make sure that start_state is a batched tensor (bxdof), without gripper.
        assert isinstance(initial_js, torch.Tensor)
        if len(initial_js.shape) == 1:
            initial_js = initial_js.expand(goal_l_wxyz_xyz.shape[0], -1)
        assert len(initial_js.shape) == 2 and initial_js.shape[0] == goal_l_wxyz_xyz.shape[0]
        if initial_js.shape[-1] == 16:
            initial_js = initial_js[:, :14]
        start_state = JointState.from_position(initial_js.cuda())

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

        if get_result:
            return result_list

        result_list = list(filter(lambda x: (x.interpolated_plan is not None), result_list))
        if len(result_list) == 0:
            return None, None

        positions = torch.cat([
            result.interpolated_plan.position for result in result_list  # type: ignore
        ], dim=0)  # [batch, time, dof]
        success = torch.cat([
            result.success for result in result_list  # type: ignore
        ], dim=0)

        if len(positions.shape) == 2:
            assert batch_size == 1, "Number of trajs is not 1, but positions is TxDOF."
            positions = positions.unsqueeze(0)

        assert positions.shape == (batch_size, self._motion_gen.interpolation_steps, 14)
        assert success.shape == (batch_size,)

        return positions, success

    def gen_motion_from_ik_chain(
        self,
        path_l_wxyz_xyz: torch.Tensor,
        path_r_wxyz_xyz: torch.Tensor,
        initial_js: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Generate a motion plan from a given path. If IK fails at any point, the motion generation stops.
        path_*_wxyz_xyz: [batch, time, 7] tensor, where the first 4 elements are the quaternion, and the last 3 are the position.
        initial_js: [batch, dof] tensor, where dof is 14.
        If no solutions are found, will return None.

        returns [batch, time, dof] (position) and [batch, time] (success).
        success might not be true throughout the full trajectory! is success for every timestep in every batch.
        """
        assert path_l_wxyz_xyz.shape == path_r_wxyz_xyz.shape
        assert len(path_l_wxyz_xyz.shape) == 3 and path_l_wxyz_xyz.shape[-1] == 7
        assert path_l_wxyz_xyz.shape[0] % self._batch_size == 0
        assert (
            initial_js.shape[0] == path_l_wxyz_xyz.shape[0]
            and initial_js.shape[-1] == 14
            and len(initial_js.shape) == 2
        )
        batch_size = path_l_wxyz_xyz.shape[0]

        # TODO We also want to smoothen this out!
        # TODO this ... doesn't consider collision.

        js_list: List[torch.Tensor] = []
        js_success_list: List[torch.Tensor] = []
        for idx in range(path_l_wxyz_xyz.shape[1]):
            goal_l_wxyz_xyz = path_l_wxyz_xyz[:, idx].view(-1, 7)
            goal_r_wxyz_xyz = path_r_wxyz_xyz[:, idx].view(-1, 7)

            assert goal_l_wxyz_xyz.shape == (batch_size, 7)
            assert goal_r_wxyz_xyz.shape == (batch_size, 7)
            assert initial_js.shape == (batch_size, 14)

            # Get the current joint locations.
            curr_js, curr_success = self.ik(
                goal_l_wxyz_xyz=goal_l_wxyz_xyz,
                goal_r_wxyz_xyz=goal_r_wxyz_xyz,
                initial_js=initial_js,
            )
            assert isinstance(curr_js, torch.Tensor) and isinstance(curr_success, torch.Tensor)
            assert curr_js.shape == (batch_size, 14) and curr_success.shape == (batch_size,)

            initial_js = curr_js
            js_list.append(initial_js.unsqueeze(1))  # [batch, 1, dof]
            js_success_list.append(curr_success.unsqueeze(dim=1))  # [batch, 1]

        # Stack along time dimension -- [batch, time, dof].
        return torch.cat(js_list, dim=1), torch.cat(js_success_list, dim=1)
