"""
Part-motion centric motion planner.
"""

from typing import Generator, cast
import jax
import jax.numpy as jnp
import numpy as onp
import jaxlie
from loguru import logger
import trimesh
import yourdfpy

from rsrd.motion.motion_optimizer import RigidGroupOptimizer
from rsrd.robot.graspable_obj import GraspableObject
from jaxmp.kinematics import JaxKinTree
from jaxmp.coll import RobotColl, collide, Convex
from jaxmp.jaxls.solve_ik import solve_ik
from rsrd.robot.motion_plan_yumi import YUMI_REST_POSE
from rsrd.robot.motion_plan_yumi import motion_plan_yumi

solve_ik_vmap = jax.vmap(
    solve_ik, in_axes=(None, 0, None, None, None, None, None, None)
)
mp_yumi_vmap = jax.vmap(motion_plan_yumi, in_axes=(None, None, 0))


class PartMotionPlanner:
    optimizer: RigidGroupOptimizer
    urdf: yourdfpy.URDF
    kin: JaxKinTree
    robot_coll: RobotColl
    object: GraspableObject

    left_joint_idx: int
    right_joint_idx: int
    finger_indices: jnp.ndarray

    def __init__(
        self,
        optimizer: RigidGroupOptimizer,
        urdf: yourdfpy.URDF,
        motion_subsample_rate: int = 2,
    ):
        self.optimizer = optimizer
        self.urdf = urdf
        self.kin = JaxKinTree.from_urdf(self.urdf)

        # In MJX, all convex-* coll. distances are [interpenetration dist](negative) or [1.0].
        # To counter this, we inflate the meshes a bit -- the convex bodies tend to be an _underestimate_ from the decimation process.
        def coll_handler(meshes: list[trimesh.Trimesh]) -> Convex:
            _meshes = []
            for mesh in meshes:
                mesh.vertices += 0.01 * mesh.vertex_normals
                _meshes.append(mesh)
            return Convex.from_meshes(_meshes)

        # Allow "collisions" within an arm.
        self_coll_ignore = []
        for link_0 in urdf.robot.links:
            for link_1 in urdf.robot.links:
                link_0_name = link_0.name
                link_1_name = link_1.name
                if link_0 == link_1:
                    self_coll_ignore.append((link_0_name, link_1_name))
                if "_r" in link_0_name and "_r" in link_1_name:
                    self_coll_ignore.append((link_0_name, link_1_name))
                if "_l" in link_0_name and "_l" in link_1_name:
                    self_coll_ignore.append((link_0_name, link_1_name))
                if link_0 == "yumi_link_1_r" or link_1 == "yumi_link_1_r":
                    self_coll_ignore.append((link_0_name, link_1_name))
                if link_0 == "yumi_link_1_l" or link_1 == "yumi_link_1_l":
                    self_coll_ignore.append((link_0_name, link_1_name))

                if link_0 == "yumi_link_2_r" or link_1 == "yumi_link_2_r":
                    self_coll_ignore.append((link_0_name, link_1_name))
                if link_0 == "yumi_link_2_l" or link_1 == "yumi_link_2_l":
                    self_coll_ignore.append((link_0_name, link_1_name))
                
                if "point" in link_0_name or "point" in link_1_name:
                    self_coll_ignore.append((link_0_name, link_1_name))

        self.robot_coll = RobotColl.from_urdf(
            self.urdf, self_coll_ignore=self_coll_ignore, coll_handler=coll_handler
        )

        self.object = GraspableObject(self.optimizer)

        self.motion_subsample_rate = motion_subsample_rate

        # These TCP points are defined in `data/yumi_descriptions/yumi.urdf`.
        left_joint_name, right_joint_name = "left_dummy_joint", "right_dummy_joint"
        self.left_joint_idx = self.kin.joint_names.index(left_joint_name)
        self.right_joint_idx = self.kin.joint_names.index(right_joint_name)
        self.finger_indices = jnp.array(
            ["finger" in link_name for link_name in self.robot_coll.coll_link_names]
        )

    def plan_single(
        self, T_obj_world: jaxlie.SE3
    ) -> Generator[onp.ndarray, None, None]:
        len_traj = self.object.optimizer.part_deltas.shape[0]
        timesteps = jnp.arange(len_traj, step=self.motion_subsample_rate)

        for part_idx, joint_idx, grasp_idx in self._get_start_cand_single(
            T_obj_world
        ):
            logger.info(f"Trying part {part_idx}, joint {joint_idx}.")
            Ts_grasp_world = self.object.get_T_grasps_world(
                part_idx, timesteps, T_obj_world
            ).wxyz_xyz[grasp_idx, None, :, :]
            traj, succ = mp_yumi_vmap(
                self.kin,
                jnp.array([joint_idx]),
                Ts_grasp_world,
            )  # N_grasps, timesteps, 16

            succ_traj = traj[succ[:, 0]]
            in_coll = jax.vmap(self.get_self_coll, in_axes=0)(succ_traj[:, 0])
            succ_traj = succ_traj[~in_coll]
            logger.info(f"Found {len(succ_traj)} successful trajectories.")

            if len(succ_traj) > 0:
                for i in range(len(succ_traj)):
                    curr_traj = succ_traj[i]
                    curr_traj = jnp.repeat(
                        curr_traj, self.motion_subsample_rate, axis=0
                    )[: len_traj]
                    yield onp.array(curr_traj)

        raise ValueError("No more valid trajectories found.")

    def plan_bimanual(self, T_obj_world: jaxlie.SE3) -> Generator[onp.ndarray, None, None]:
        len_traj = self.object.optimizer.part_deltas.shape[0]
        timesteps = jnp.arange(len_traj, step=self.motion_subsample_rate)

        for (
            part_idx_0, part_idx_1, joint_idx_0, joint_idx_1, grasp_idx
        ) in self._get_start_cand_bimanual(T_obj_world):
            logger.info(f"Trying part {part_idx_0}, {part_idx_1}, joint {joint_idx_0}, {joint_idx_1}.")
            # Batch the grasps.
            for i in range(0, grasp_idx.shape[0], 500):
                _grasp_idx = grasp_idx[i : min(i + 500, grasp_idx.shape[0])]

                logger.info(f"Attempting trajs from {len(_grasp_idx)} grasps.")
                Ts_grasp_world_0 = self.object.get_T_grasps_world(
                    part_idx_0, timesteps, T_obj_world
                ).wxyz_xyz[_grasp_idx[:, 0], None, :, :]
                Ts_grasp_world_1 = self.object.get_T_grasps_world(
                    part_idx_1, timesteps, T_obj_world
                ).wxyz_xyz[_grasp_idx[:, 1], None, :, :]
                traj, succ = mp_yumi_vmap(
                    self.kin,
                    jnp.array([joint_idx_0, joint_idx_1]),
                    jnp.concatenate([Ts_grasp_world_0, Ts_grasp_world_1], axis=1),
                )
                succ_traj = traj[succ[:, 0]]
                in_coll = jax.vmap(self.get_self_coll, in_axes=0)(succ_traj[:, 0])
                succ_traj = succ_traj[~in_coll]
                logger.info(f"Found {len(succ_traj)} successful trajectories.")

                if len(succ_traj) > 0:
                    for i in range(len(succ_traj)):
                        curr_traj = succ_traj[i]
                        curr_traj = jnp.repeat(
                            curr_traj, self.motion_subsample_rate, axis=0
                        )[: len_traj]
                        yield onp.array(curr_traj)

        raise ValueError("No more valid trajectories found.")

    def _get_start_cand_single(self, T_obj_world: jaxlie.SE3):
        """
        Find a list of collision-free IK solutions to reach the object.
        """
        part_indices = self.object.rank_parts_to_move_single()

        for part_idx in part_indices:
            for joint_idx in [self.left_joint_idx, self.right_joint_idx]:
                succ_grasp_idx = self._get_coll_free_grasp_indices(
                    part_idx, joint_idx, T_obj_world
                )
                yield (part_idx, joint_idx, succ_grasp_idx)

    def _get_start_cand_bimanual(self, T_obj_world: jaxlie.SE3):
        part_indices = self.object.rank_parts_to_move_bimanual()

        grasp_idx_cache = {}  # Cache the grasp indices for each part.
        for part_idx_0, part_idx_1 in part_indices:
            for joint_idx_0, joint_idx_1 in [
                (self.left_joint_idx, self.right_joint_idx),
                (self.right_joint_idx, self.left_joint_idx)
            ]:
                if (part_idx_0, joint_idx_0) not in grasp_idx_cache:
                    grasp_idx_cache[(part_idx_0, joint_idx_0)] = self._get_coll_free_grasp_indices(
                        part_idx_0, joint_idx_0, T_obj_world
                    )
                if (part_idx_1, joint_idx_1) not in grasp_idx_cache:
                    grasp_idx_cache[(part_idx_1, joint_idx_1)] = self._get_coll_free_grasp_indices(
                        part_idx_1, joint_idx_1, T_obj_world
                    )

                # make a MxN grid of all possible combinations of grasp indices
                grasp_idx_0, grasp_idx_1 = jnp.meshgrid(
                    grasp_idx_cache[(part_idx_0, joint_idx_0)],
                    grasp_idx_cache[(part_idx_1, joint_idx_1)],
                    indexing='ij'
                )
                succ_grasp_idx = jnp.stack([grasp_idx_0.ravel(), grasp_idx_1.ravel()], axis=-1)

                yield (
                    part_idx_0,
                    part_idx_1,
                    joint_idx_0,
                    joint_idx_1,
                    succ_grasp_idx,
                )
    
    def _get_coll_free_grasp_indices(self, part_index: int, joint_idx: int, T_obj_world: jaxlie.SE3) -> jax.Array:
        """
        Get collision-free initial (t=0) grasp indices (between the robot and the object).
        """
        T_grasp_world = self.object.get_T_grasps_world(
            part_index, jnp.array([0]), T_obj_world
        )
        _, joints = solve_ik_vmap(
            self.kin,
            T_grasp_world,
            jnp.array([joint_idx]),
            5.0,
            1.0,
            0.01,
            100.0,
            jnp.array(YUMI_REST_POSE),
        )

        # Check that the IK actually goes to the correct place.
        in_position = jnp.isclose(
            jaxlie.SE3(
                self.kin.forward_kinematics(joints)[..., joint_idx, :]
            ).translation(),
            T_grasp_world.translation().squeeze(),
            atol=0.01,
        ).all(axis=-1)

        # Collision check.
        robot_coll = cast(Convex, self.robot_coll.coll).transform(
            jaxlie.SE3(
                self.kin.forward_kinematics(joints)[
                    None, ..., self.robot_coll.link_joint_idx, :
                ]
            ),
            mesh_axis=2,
        )
        part_coll = [
            Convex.from_meshes([self.object.parts[idx].mesh.convex_hull]).transform(
                self.object.get_T_part_world(jnp.array(0), T_obj_world)[idx]
            )
            for idx in range(len(self.object.parts))
        ]
        coll_dist = jnp.full(
            (joints.shape[0], len(self.robot_coll.coll_link_names)),
            float("inf"),
        )
        for coll in part_coll:
            # [*batch_axes, codim]; here through reshape we know it's [obj, links].
            # TODO the collision math is buggy!! Fix this.
            # dist = collide(robot_coll, coll.reshape(1, -1, mesh_axis=1)).dist
            dist = collide(robot_coll, coll.reshape(-1, 1, mesh_axis=0)).dist.squeeze()
            coll_dist = jnp.minimum(coll_dist, dist)

        # Make the collision check more lax for gripper ends.
        in_collision_non_fingers = jnp.any(coll_dist[..., ~self.finger_indices] < -0.000, axis=-1)
        in_collision_fingers = jnp.any(coll_dist[..., self.finger_indices] < -0.000, axis=-1)
        in_collision = in_collision_non_fingers | in_collision_fingers

        mask = ~in_collision & in_position
        succ_joints = joints[mask]
        succ_grasp_idx = jnp.where(mask)[0]

        return succ_grasp_idx
    
    def get_self_coll(self, joints: jnp.ndarray) -> jnp.ndarray:
        """
        Get self-collision distances for the robot.
        """
        robot_coll = cast(Convex, self.robot_coll.coll).transform(
            jaxlie.SE3(
                wxyz_xyz=self.kin.forward_kinematics(joints)[
                    ..., self.robot_coll.link_joint_idx, :
                ]
            ),
            mesh_axis=0,
        )
        num_links = len(self.robot_coll.coll_link_names)
        dist = collide(
            robot_coll.reshape(num_links, 1, mesh_axis=0),
            robot_coll.reshape(1, num_links, mesh_axis=1),
        ).dist
        return jnp.any(dist * self.robot_coll.self_coll_matrix < -0.005)
