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
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig, MotionGenResult
from curobo.geom.types import WorldConfig
from curobo.geom.sdf.world import CollisionCheckerType, WorldCollision, WorldCollisionConfig
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.geom.types import Sphere, Cuboid

def createTableWorld():
    """Create a simple world with a table, at z=0."""
    cfg_dict = {
        "cuboid": {
            "table": {
                "dims": [1.0, 1.0, 0.2],  # x, y, z
                "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0.0],  # x, y, z, qw, qx, qy, qz
            },
        },
    }
    return WorldConfig.from_dict(cfg_dict)

class YumiCurobo:
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
    _viser_urdf: ViserUrdf
    """The URDF object for the visualizer."""
    _tooltip_to_gripper: vtf.SE3
    """The offset from the tooltip to the gripper."""
    _curr_cfg: torch.Tensor
    """The current joint configuration of the robot."""
    _home_pos: torch.Tensor
    """The home position of the robot."""

    def __init__(
        self,
        target: Union[viser.ViserServer, viser.ClientHandle],
        world_config: Optional[WorldConfig] = None,
        ik_solver_batch_size: int = 1,
        motion_gen_batch_size: int = 1,
    ):
        # check if cuda is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available -- curobo requires CUDA.")
        self._base_dir = Path(__file__).parent.parent
        self._tensor_args = TensorDeviceType()

        if world_config is None:
            world_config = createTableWorld()

        self._setup(target, world_config, ik_solver_batch_size, motion_gen_batch_size)

        # Add ground plane visualization, and remove camera_link.
        target.add_grid(
            name="grid",
            width=1,
            height=1,
            position=(0.5, 0, 0),
            section_size=0.05,
        )
        self._viser_urdf._joint_frames[0].remove()

    def _setup(
        self,
        target: Union[viser.ViserServer, viser.ClientHandle],
        world_config: WorldConfig,
        ik_solver_batch_size: int,
        motion_gen_batch_size: int,
    ):
        # Create the robot model.
        cfg_file = load_yaml(join_path(self._base_dir, f"data/yumi.yml"))
        urdf_path = cfg_file["robot_cfg"]["kinematics"]["urdf_path"]
        cfg_file["robot_cfg"]["kinematics"]["external_robot_configs_path"] = self._base_dir
        cfg_file["robot_cfg"]["kinematics"]["external_asset_path"] = self._base_dir
        robot_cfg = RobotConfig.from_dict(cfg_file, self._tensor_args)

        _robot_world_cfg = RobotWorldConfig.load_from_config(
            robot_config=robot_cfg,
            world_model=world_config,
            tensor_args=self._tensor_args,
            collision_activation_distance=0.0,
        )
        self._robot_world = RobotWorld(_robot_world_cfg)
        assert isinstance(self._robot_world.world_model, WorldCollision)

        # Set up IK.
        # NOTE(cmk): I contemplated putting the collision checker into the IK
        # for collision-free IK, but this makes the code much slower (0.02 s -> ~1s.)
        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            None,
            rotation_threshold=0.05,
            position_threshold=0.002,
            num_seeds=10,
            self_collision_check=True,
            self_collision_opt=False,
            # world_coll_checker=self._robot_world.world_model,
            tensor_args=self._tensor_args,
            use_cuda_graph=True,
        )
        self._ik_solver = IKSolver(ik_config)
        self._ik_solver_batch_size = ik_solver_batch_size

        # Set up motion generation.
        # Note that this ignores collision for now, for speed.
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            None,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_ik_seeds=10,
            num_graph_seeds=4,
            num_trajopt_seeds=4,
            interpolation_dt=0.25,
            trajopt_dt=0.25,
            collision_activation_distance=0.0,
            interpolation_steps=100,
            world_coll_checker=self._robot_world.world_model,
            use_cuda_graph=True,
        )
        self._motion_gen = MotionGen(motion_gen_config)
        self._motion_gen_batch_size = motion_gen_batch_size
        self._motion_gen.warmup(
            batch=motion_gen_batch_size,
            warmup_js_trajopt=False,
            parallel_finetune=False,
        )

        self._viser_urdf = ViserUrdf(target, Path(urdf_path))

        # Get the tooltip-to-gripper offset. TODO remove hardcoding...
        self._tooltip_to_gripper = vtf.SE3.from_translation(np.array([0.0, 0.0, 0.128]))

        # Initialize the robot to the retracted position.
        self.joint_pos = self.home_pos

    @property
    def device(self) -> torch.device:
        return self._tensor_args.device

    @property
    def home_pos(self) -> torch.Tensor:
        _home_pos = self._robot_world.kinematics.retract_config
        assert type(_home_pos) == torch.Tensor and _home_pos.shape == (14,)
        return _home_pos

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._curr_cfg

    @joint_pos.setter
    def joint_pos(self, joint_pos: torch.Tensor):
        if len(joint_pos.shape) != 1:
            joint_pos = joint_pos.squeeze()

        assert joint_pos.shape == (14,) or joint_pos.shape == (16,)

        # Update the joint positions in the visualizer.
        if joint_pos.shape == (16,):
            gripper_width = joint_pos[-2:]
        else:
            gripper_width = torch.Tensor([0.02, 0.02]).to(joint_pos.device)
        _joint_pos = torch.concat(
            (joint_pos[7:14], joint_pos[:7], gripper_width)
        ).detach().cpu().numpy()
        self._viser_urdf.update_cfg(_joint_pos)

        self._curr_cfg = joint_pos

    def get_left_joints(self) -> torch.Tensor:
        return self.joint_pos[:7]

    def get_right_joints(self) -> torch.Tensor:
        return self.joint_pos[7:14]

    def update_world(self, world_config: WorldConfig):
        # Need to clear the world cache, as per in: https://github.com/NVlabs/curobo/issues/263.
        self._robot_world.clear_world_cache()
        self._robot_world.update_world(world_config)

    def get_robot_as_spheres(
        self,
        joint_pos: torch.Tensor,
    ) -> List[trimesh.Trimesh]:
        assert joint_pos.shape == (14,) or joint_pos.shape == (16,)
        spheres: List[Sphere] = self._robot_world.kinematics.get_robot_as_spheres(joint_pos)[0]

        return [
            trimesh.creation.uv_sphere(
                radius=sphere.radius,
                transform=trimesh.transformations.translation_matrix(sphere.position)
                )
            for sphere in spheres
        ]

    def in_collision(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns d_world, d_self."""
        # get_world_coll_dist accepts b, h, dof.
        q = urdf.joint_pos[:-2].unsqueeze(0).unsqueeze(0)
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
    ) -> List[IKResult]:
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

        result_list = []
        for i in range(0, goal_l_wxyz_xyz.shape[0], self._ik_solver_batch_size):
            result = self._ik_solver.solve_batch(
                goal_l[i:i+self._ik_solver_batch_size],
                link_poses={
                    "gripper_l_base": goal_l[i:i+self._ik_solver_batch_size],
                    "gripper_r_base": goal_r[i:i+self._ik_solver_batch_size]
                },
                seed_config=initial_js.expand(1, 1, -1),
            )
            result_list.append(result)

        return result_list

    def get_js_from_ik(
        self,
        ik_result_list: List[IKResult],
        filter_success: bool = True,
        filter_collision: bool = True,
    ) -> Optional[torch.Tensor]:
        ik_result_list = list(filter(lambda x: x.success.any(), ik_result_list))
        if len(ik_result_list) == 0:
            return None

        traj_all = []
        for ik_results in ik_result_list:
            traj = ik_results.js_solution[ik_results.success].position
            assert isinstance(traj, torch.Tensor) and len(traj.shape) == 2
            if traj.shape[0] == 0:
                continue
            d_world, d_self = (
                urdf._robot_world.get_world_self_collision_distance_from_joint_trajectory(
                    traj.unsqueeze(1)
                )
            )
            traj = traj[(d_world.squeeze() <= 0) & (d_self.squeeze() <= 0)]
            if len(traj) > 0:
                traj_all.append(traj.squeeze(1))

        if len(traj_all) == 0:
            print("No collision-free IK solution found.")
            return None

        return torch.cat(traj_all)

    def motiongen(
        self,
        goal_l_wxyz_xyz: torch.Tensor,
        goal_r_wxyz_xyz: torch.Tensor,
        start_state: Optional[JointState] = None,
        start_l_wxyz_xyz: Optional[torch.Tensor] = None,
        start_r_wxyz_xyz: Optional[torch.Tensor] = None,
    ) -> List[MotionGenResult]:
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
        start_joints = start_state.position
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
        result_list = []
        print(start_state.shape, goal_l_pose.shape, goal_r_pose.shape)
        for i in range(0, goal_l_wxyz_xyz.shape[0], self._motion_gen_batch_size):
            result = self._motion_gen.plan_batch(
                start_state[i:i+self._motion_gen_batch_size],
                goal_l_pose[i:i+self._motion_gen_batch_size],
                MotionGenPlanConfig(max_attempts=10, parallel_finetune=True),
                link_poses={
                    "gripper_l_base": goal_l_pose[i:i+self._motion_gen_batch_size],
                    "gripper_r_base": goal_r_pose[i:i+self._motion_gen_batch_size]
                },  # type: ignore
            )
            result_list.append(result)

        return result_list


if __name__ == "__main__":
    server = viser.ViserServer()

    urdf = YumiCurobo(
        server,
        world_config=createTableWorld()
    )
    server.add_grid(
        name="grid",
        width=1,
        height=1,
        position=(0.5, 0, 0),
        section_size=0.05,
    )

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

    joints_from_ik = urdf.ik(
        torch.Tensor([*drag_l_handle.wxyz, *drag_l_handle.position]).view(1, 7),
        torch.Tensor([*drag_r_handle.wxyz, *drag_r_handle.position]).view(1, 7),
    ).js_solution.position
    assert isinstance(joints_from_ik, torch.Tensor)
    urdf.joint_pos = joints_from_ik
    print(urdf.in_collision(urdf.joint_pos))

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

        start_pose = JointState.from_position(urdf.joint_pos)

        start = time.time()
        traj_pieces = []
        prev_start_state = start_pose
        for i in range(len(waypoint_queue[0])):
            motion_gen_result = urdf.motiongen(
                waypoint_queue[0][i],
                waypoint_queue[1][i],
                start_state=prev_start_state,
            )
            prev_start_state = motion_gen_result.get_interpolated_plan()[-1:]
            traj_piece = motion_gen_result.get_interpolated_plan().position
            assert isinstance(traj_piece, torch.Tensor)
            traj_pieces.append(traj_piece)

        traj = torch.concat(traj_pieces)
        print("MotionGen took", time.time() - start, "seconds")
        waypoint_queue[0] = []
        waypoint_queue[1] = []

        urdf.joint_pos = traj[0]
        drag_slider.value = 0
        drag_button.disabled = False
        drag_slider.disabled = False

    @drag_slider.on_update
    def _(_):
        assert traj is not None
        idx = int(drag_slider.value * (len(traj)-1))
        urdf.joint_pos = traj[idx]

    # pos_l, quat_l = drag_l_handle.position, drag_l_handle.wxyz
    # pos_r, quat_r = drag_r_handle.position, drag_r_handle.wxyz
    # urdf.motiongen(
    #     torch.tensor(pos_l), torch.tensor(quat_l),
    #     torch.tensor(pos_l) + torch.tensor([0, 0.1, 0]), torch.tensor(quat_l),
    #     torch.tensor(pos_r), torch.tensor(quat_r),
    #     torch.tensor(pos_r) + torch.tensor([0, 0.1, 0]), torch.tensor(quat_r),
    # )

    # def update_joints():
    #     start = time.time()
    #     pos_l, quat_l = drag_l_handle.position, drag_l_handle.wxyz
    #     pos_r, quat_r = drag_r_handle.position, drag_r_handle.wxyz
    #     ik_result = urdf.ik(
    #         torch.tensor(pos_l), torch.tensor(quat_l),
    #         torch.tensor(pos_r), torch.tensor(quat_r),
    #     ) # .position.cpu().numpy()  # type: ignore
    #     urdf.joint_pos = ik_result.js_solution.position.squeeze().cpu().numpy()
    #     print("ik took", time.time() - start, "seconds")
    # @drag_r_handle.on_update
    # def _(_):
    #     update_joints()
    # @drag_l_handle.on_update
    # def _(_):
    #     update_joints()
    # update_joints()

    # drag_button = server.add_gui_button("Match drag")
    # @drag_button.on_click
    # def _(_):
    #     global traj
    #     update_joints()
    import trimesh
    def get_bounding_spheres(mesh: trimesh.Trimesh):
        # get the bounding spheres of the mesh
        # this is a simple way to get a set of spheres that
        # cover the entire mesh volume
        spheres = trimesh.nsphere.minimum_nsphere(mesh.vertices, 1)
        return spheres

    populate_balls = server.add_gui_button("Populate balls")
    @populate_balls.on_click
    def _(_):
        # get the links in the urdf one by one
        for link_name, mesh in urdf._viser_urdf._urdf.scene.geometry.items():
            assert isinstance(mesh, trimesh.Trimesh)
            T_parent_child = urdf._viser_urdf._urdf.get_transform(
                link_name, urdf._viser_urdf._urdf.scene.graph.transforms.parents[link_name]
            )

            # Scale + transform the mesh. (these will mutate it!)
            #
            # It's important that we use apply_transform() instead of unpacking
            # the rotation/translation terms, since the scene graph transform
            # can also contain scale and reflection terms.
            mesh = mesh.copy()
            voxs = mesh.voxelized(0.01)
            server.add_mesh_trimesh(f'dummy/{link_name}', mesh)
            break
    while True:
        time.sleep(1)
