"""
Part-motion centric motion planner.
"""

import jax
from loguru import logger
import numpy as onp
import jax.numpy as jnp
import jaxlie
import trimesh
import yourdfpy

from jaxmp.coll.collision_sdf import dist_signed
from jaxmp.coll.collision_types import SphereColl
from rsrd.motion.motion_optimizer import RigidGroupOptimizer
from rsrd.robot.graspable_obj import AntipodalGrasps, GraspableObject, GraspablePart
from jaxmp.kinematics import JaxKinTree
from jaxmp.coll.collision_types import RobotColl
from jaxmp.jaxls.solve_ik import solve_ik
from rsrd.robot.motion_plan_yumi import motion_plan_yumi, YUMI_REST_POSE
from rsrd.util.common import MANO_KEYPOINTS
import rsrd.transforms as tf


class PartMotionPlanner:
    optimizer: RigidGroupOptimizer
    urdf: yourdfpy.URDF
    kin: JaxKinTree
    coll: RobotColl
    object: GraspableObject

    def __init__(
        self,
        optimizer: RigidGroupOptimizer,
        urdf: yourdfpy.URDF,
    ):
        self.optimizer = optimizer
        self.urdf = urdf
        self.kin = JaxKinTree.from_urdf(self.urdf)
        self.coll = RobotColl.from_urdf(self.urdf)
        self.object = GraspableObject(self.optimizer)

    def ik_single(
        self, T_obj_world: jaxlie.SE3
    ):
        """
        Find a list of collision-free IK solutions to reach the object.
        """
        part_indices = self.object.rank_parts_to_move_single()

        left_joint_name, right_joint_name = "left_dummy_joint", "right_dummy_joint"
        left_joint_idx = self.kin.joint_names.index(left_joint_name)
        right_joint_idx = self.kin.joint_names.index(right_joint_name)

        # Get a pointcloud approximation of the object.
        object_mesh = self.object.get_obj_mesh_in_world(0, T_obj_world)
        points = jnp.array(trimesh.sample.volume_mesh(object_mesh, 1000))
        obj_coll = SphereColl(points, jnp.full((points.shape[0],), 0.005))

        # Find T_grasp_world
        solve_ik_vmap = jax.vmap(solve_ik, in_axes=(None, 0, None, None, None, None, None, None))
        dist_signed_vmap = jax.vmap(dist_signed, in_axes=(0, None))
        for part_idx in range(len(self.object.parts)):
            part_idx = 3
            T_grasp_world = self.object.get_T_grasps_world(
                part_idx, jnp.array([0]), T_obj_world
            )
            # should be (num_grasps, 1, 7)
            T_grasp_world = jaxlie.SE3(T_grasp_world.wxyz_xyz[:48, :, :])
            joints_list = []
            for joint_idx in [left_joint_idx, right_joint_idx]:
                _, joints = solve_ik_vmap(
                    self.kin,
                    T_grasp_world,
                    (joint_idx,),
                    5.0,
                    1.0,
                    0.01,
                    100.0,
                    jnp.array(YUMI_REST_POSE),
                )
                robot_coll = self.coll.transform(
                    jaxlie.SE3(self.kin.forward_kinematics(joints))
                ).as_cylinders()
                breakpoint()
                dist_signed_vmap(robot_coll, obj_coll)
                # check for collision
                joints_list.append(joints)
            breakpoint()
            break
        return joints, T_grasp_world


    # def plan_single(
    #     self, T_obj_world: jaxlie.SE3
    # ) -> tuple[onp.ndarray | None, int, int, onp.ndarray | None]:
    #     part_indices = self.rank_parts_to_move_single()

    #     left_joint_name, right_joint_name = "left_dummy_joint", "right_dummy_joint"
    #     left_joint_idx = self.kin.joint_names.index(left_joint_name)
    #     right_joint_idx = self.kin.joint_names.index(right_joint_name)

    #     succ_traj = None
    #     part_idx, grasp_idx = 0, 0
    #     Ts_grasp_world = None

    #     mp_yumi_vmap = jax.vmap(motion_plan_yumi, in_axes=(None, None, 0, None))
    #     for part_idx in part_indices:
    #         if succ_traj is not None:
    #             break
    #         logger.info(f"Planning for part {part_idx}")
    #         Ts_part_world = self._get_part_pose(part_idx, T_obj_world)

    #         for joint_idx in [left_joint_idx, right_joint_idx]:
    #             if succ_traj is not None:
    #                 break

    #             num_grasps = self.parts[part_idx]._grasps.centers.shape[0]
    #             for grasp_idx in range(num_grasps):
    #                 if succ_traj is not None:
    #                     break

    #                 # Need _some_ filtering here.
    #                 # e.g., using the top 5 grasps that are physically good.
    #                 # also need to put in the collision in.
                    
    #                 # ok.
    #                 # IK, filter by collision (hardcoded for this proj), then generate trajs.
    #                 # Wonder how much the first two can filter out.

    #                 logger.info(f"- trying {grasp_idx}")

    #                 Ts_grasp_world = self._create_grasp_trajs(
    #                     part_idx, grasp_idx, Ts_part_world
    #                 ).wxyz_xyz[:, None, ::5, :]
    #                 traj, succ = mp_yumi_vmap(
    #                     self.kin,
    #                     jnp.array([joint_idx]),
    #                     Ts_grasp_world[0],
    #                     jnp.array(YUMI_REST_POSE),
    #                 )  # N_grasps, timesteps, 16
    #                 if succ.any():
    #                     succ_traj = onp.array(traj[jnp.where(succ)[0][0]])
    #                     Ts_grasp_world = onp.array(
    #                         Ts_grasp_world[jnp.where(succ)[0][0]]
    #                     )
    #                     part_idx, grasp_idx = part_idx, grasp_idx
    #                     break

    #     return succ_traj, part_idx, grasp_idx, Ts_grasp_world

    def rank_parts_to_move_bimanual(self) -> list[tuple[int, int]]:
        raise NotImplementedError

    def plan_bimanual(self, part_idx1: int, part_idx2: int) -> None:
        raise NotImplementedError

    # def _create_grasp_trajs(
    #     self, part_idx: int, grasp_idx: int, Ts_part_world: jaxlie.SE3
    # ) -> jaxlie.SE3:
    #     part = self.parts[part_idx]
    #     T_grasp_part = jaxlie.SE3(
    #         jnp.concatenate(
    #             [
    #                 part._grasps.to_se3().wxyz_xyz,
    #                 part._grasps.to_se3(flip_axis=True).wxyz_xyz,
    #             ],
    #             axis=0,
    #         )
    #     )

    #     T_grasp_grasp = self._get_grasp_augs()

    #     T_grasp_part = jaxlie.SE3(T_grasp_part.wxyz_xyz[grasp_idx]).multiply(
    #         jaxlie.SE3(T_grasp_grasp.wxyz_xyz)
    #     )

    #     num_grasps = T_grasp_part.wxyz_xyz.shape[0]
    #     timesteps = Ts_part_world.wxyz_xyz.shape[0]

    #     Ts_grasp_world = jaxlie.SE3(Ts_part_world.wxyz_xyz[None, :]).multiply(
    #         jaxlie.SE3(T_grasp_part.wxyz_xyz[:, None])
    #     )

    #     assert Ts_grasp_world.wxyz_xyz.shape == (num_grasps, timesteps, 7)
    #     return Ts_grasp_world
