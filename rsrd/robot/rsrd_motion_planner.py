"""
Part-motion centric motion planner.
"""

from typing import Generator
import jax
import jax.numpy as jnp
import numpy as onp
import jaxlie
from loguru import logger
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
    ):
        self.optimizer = optimizer
        self.urdf = urdf
        self.kin = JaxKinTree.from_urdf(self.urdf)
        self.robot_coll = RobotColl.from_urdf(self.urdf, coll_handler=Convex.from_meshes)
        self.object = GraspableObject(self.optimizer)

        left_joint_name, right_joint_name = "left_dummy_joint", "right_dummy_joint"
        self.left_joint_idx = self.kin.joint_names.index(left_joint_name)
        self.right_joint_idx = self.kin.joint_names.index(right_joint_name)
        self.finger_indices = jnp.array(
            ["finger" in link_name for link_name in self.robot_coll.coll_link_names]
        )

    def plan_single(
        self, T_obj_world: jaxlie.SE3
    ) -> Generator[onp.ndarray, None, None]:
        timesteps = self.object.optimizer.part_deltas.shape[0]

        for part_idx, joint_idx, grasp_idx, joints in self._get_start_cand_single(
            T_obj_world
        ):
            logger.info(f"Trying part {part_idx}, joint {joint_idx}.")
            Ts_grasp_world = self.object.get_T_grasps_world(
                part_idx, jnp.arange(timesteps), T_obj_world
            ).wxyz_xyz[grasp_idx, None, :, :]
            traj, succ = mp_yumi_vmap(
                self.kin,
                jnp.array([joint_idx]),
                Ts_grasp_world,
            )  # N_grasps, timesteps, 16

            succ_traj = traj[succ[:, 0]]
            if len(succ_traj) > 0:
                for i in range(len(succ_traj)):
                    logger.info("Returning trajectory.")
                    yield onp.array(succ_traj[i])

        raise ValueError("No more valid trajectories found.")

    def plan_bimanual(self, urdf, T_obj_world: jaxlie.SE3) -> Generator[onp.ndarray, None, None]:
        timesteps = self.object.optimizer.part_deltas.shape[0]
        for (
            part_idx_0, part_idx_1, joint_idx_0, joint_idx_1, grasp_idx, joints
        ) in self._get_start_cand_bimanual(urdf, T_obj_world):
            logger.info(f"Trying part {part_idx_0}, {part_idx_1}, joint {joint_idx_0}, {joint_idx_1}.")
            logger.info(f"Attempting trajs from {len(grasp_idx)} grasps.")
            Ts_grasp_world_0 = self.object.get_T_grasps_world(
                part_idx_0, jnp.arange(timesteps), T_obj_world
            ).wxyz_xyz[grasp_idx, None, :, :]
            Ts_grasp_world_1 = self.object.get_T_grasps_world(
                part_idx_1, jnp.arange(timesteps), T_obj_world
            ).wxyz_xyz[grasp_idx, None, :, :]
            traj, succ = mp_yumi_vmap(
                self.kin,
                jnp.array([joint_idx_0, joint_idx_1]),
                jnp.concatenate([Ts_grasp_world_0, Ts_grasp_world_1], axis=1),
            )
            succ_traj = traj[succ[:, 0]]
            if len(succ_traj) > 0:
                for i in range(len(succ_traj)):
                    logger.info("Returning trajectory.")
                    yield onp.array(succ_traj[i])

        raise ValueError("No more valid trajectories found.")

    def _get_start_cand_single(self, T_obj_world: jaxlie.SE3):
        """
        Find a list of collision-free IK solutions to reach the object.
        """
        part_indices = self.object.rank_parts_to_move_single()

        # Get a pointcloud approximation of the object.
        part_coll = [
            Convex.from_meshes(self.object.parts[idx].mesh.convex_hull).transform(
                self.object.get_T_part_world(jnp.array(0), T_obj_world)[idx]
            )
            for idx in range(len(self.object.parts))
        ]

        for part_idx in part_indices:
            T_grasp_world = self.object.get_T_grasps_world(
                part_idx, jnp.array([0]), T_obj_world
            )
            # should be (num_grasps, 1, 7)
            for joint_idx in [self.left_joint_idx, self.right_joint_idx]:
                # IK solve.
                _, joints = solve_ik_vmap(
                    self.kin,
                    T_grasp_world,
                    # jnp.array([joint_idx]),
                    (joint_idx,),
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
                robot_coll = self.robot_coll.coll.transform(
                    jaxlie.SE3(
                        self.kin.forward_kinematics(joints)[
                            ..., self.robot_coll.link_joint_idx, :
                        ]
                    )
                )
                in_collision = jnp.zeros(joints.shape[0], dtype=bool)
                for coll in part_coll:
                    # [*batch_axes, codim]; here through reshape we know it's [obj, links].
                    dist = collide(robot_coll, coll.reshape(-1, 1)).dist

                    # Make the collision check more lax for gripper ends.
                    dist = (
                        jnp.any(dist[:, ~self.finger_indices] < -0.01, axis=-1).any(axis=-1) |
                        jnp.any(dist[:, self.finger_indices] < -0.03, axis=-1).any(axis=-1)
                    )

                    in_collision = in_collision | dist

                succ_joints = joints[~in_collision & in_position]
                succ_grasp_idx = jnp.where(~in_collision & in_position)[0]

                yield (part_idx, joint_idx, succ_grasp_idx, succ_joints)

    def _get_start_cand_bimanual(self, urdf, T_obj_world: jaxlie.SE3):
        part_indices = self.object.rank_parts_to_move_bimanual()

        # Get a pointcloud approximation of the object.
        part_coll = [
            Convex.from_meshes([self.object.parts[idx].mesh.convex_hull]).transform(
                self.object.get_T_part_world(jnp.array(0), T_obj_world)[idx]
            )
            for idx in range(len(self.object.parts))
        ]

        for part_idx_0, part_idx_1 in part_indices:
            T_grasp_world_0 = self.object.get_T_grasps_world(
                part_idx_0, jnp.array([0]), T_obj_world
            )
            T_grasp_world_1 = self.object.get_T_grasps_world(
                part_idx_1, jnp.array([0]), T_obj_world
            )
            # should be (num_grasps, 1, 7)
            for joint_idx_0, joint_idx_1 in [
                (self.left_joint_idx, self.right_joint_idx),
                (self.right_joint_idx, self.left_joint_idx),
            ]:
                # IK solve.
                # TODO(cmk) this is pairwise comparison, not N^2.
                # Maybe I can use the other _get_start_cand_single function.
                T_grasps_world = jaxlie.SE3(
                    jnp.stack(
                        [T_grasp_world_0.wxyz_xyz, T_grasp_world_1.wxyz_xyz], axis=1
                    )
                )
                _, joints = solve_ik_vmap(
                    self.kin,
                    T_grasps_world,
                    (joint_idx_0, joint_idx_1),
                    5.0,
                    1.0,
                    0.01,
                    100.0,
                    jnp.array(YUMI_REST_POSE),
                )

                # Check that the IK actually goes to the correct place.
                in_position = jnp.isclose(
                    jaxlie.SE3(
                        self.kin.forward_kinematics(joints)[..., joint_idx_0, :]
                    ).translation(),
                    T_grasp_world_0.translation().squeeze(),
                    atol=0.01,
                ).all(axis=-1)

                in_position = in_position & jnp.isclose(
                    jaxlie.SE3(
                        self.kin.forward_kinematics(joints)[..., joint_idx_1, :]
                    ).translation(),
                    T_grasp_world_1.translation().squeeze(),
                    atol=0.01,
                ).all(axis=-1)

                # Collision check.
                robot_coll = self.robot_coll.coll.transform(
                    jaxlie.SE3(
                        self.kin.forward_kinematics(joints)[
                            ..., self.robot_coll.link_joint_idx, :
                        ]
                    ),
                    mesh_axis=1,
                )
                # in_collision = jnp.zeros(joints.shape[0], dtype=bool)
                coll_dist = jnp.full(
                    (joints.shape[0], len(self.robot_coll.coll_link_names)),
                    float("inf"),
                )
                for coll in part_coll:
                    # [*batch_axes, codim]; here through reshape we know it's [obj, links].
                    dist = collide(robot_coll, coll.reshape(1, -1, mesh_axis=1)).dist

                    # Make the collision check more lax for gripper ends.
                    # dist = (
                    #     jnp.any(dist[:, ~self.finger_indices] < -0.03, axis=-1).any(axis=-1) |
                    #     jnp.any(dist[:, self.finger_indices] < -0.1, axis=-1).any(axis=-1)
                    # )
                    coll_dist = jnp.minimum(coll_dist, dist)

                # import trimesh
                # urdf._target.add_mesh_trimesh(
                #     "foo",
                #     sum([coll.to_trimesh() for coll in part_coll], trimesh.Trimesh()),
                # )
                # vis_idx = 21
                # _robot_coll = self.robot_coll.coll.transform( jaxlie.SE3( self.kin.forward_kinematics(joints[vis_idx])[ ..., self.robot_coll.link_joint_idx, : ]))
                # urdf._target.add_mesh_trimesh("bar", _robot_coll.to_trimesh())
                # breakpoint()
                in_collision = jnp.any(coll_dist[:, ~self.finger_indices] < 0, axis=-1) | jnp.any(coll_dist[:, self.finger_indices] < 0.02, axis=-1)
                mask = ~in_collision & in_position
                succ_joints = joints[mask]
                succ_grasp_idx = jnp.where(mask)[0]

                yield (part_idx_0, part_idx_1, joint_idx_0, joint_idx_1, succ_grasp_idx, succ_joints)