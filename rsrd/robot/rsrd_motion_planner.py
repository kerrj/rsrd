"""
Part-motion centric motion planner.
"""

from loguru import logger
import numpy as onp
import jax.numpy as jnp
import jaxlie
import yourdfpy

from rsrd.motion.motion_optimizer import RigidGroupOptimizer
from rsrd.robot.graspable_obj import AntipodalGrasps, GraspableObj
from rsrd.robot.kinematics import JaxKinTree
from rsrd.robot.motion_plan_yumi import motion_plan_yumi
from rsrd.util.common import MANO_KEYPOINTS
import rsrd.transforms as tf


class PartMotionPlanner:
    optimizer: RigidGroupOptimizer
    urdf: yourdfpy.URDF
    kin: JaxKinTree
    parts: list[GraspableObj]

    def __init__(self, optimizer: RigidGroupOptimizer, urdf: yourdfpy.URDF) -> None:
        self.optimizer = optimizer
        self.urdf = urdf
        self.kin = JaxKinTree.from_urdf(urdf)
        self._create_graspable_parts()

    def _create_graspable_parts(self) -> None:
        self.parts = []
        for mask in self.optimizer.group_masks:
            part_points = self.optimizer.init_means[mask].cpu().numpy()
            part_points -= part_points.mean(axis=0)
            graspable_part = GraspableObj.from_points(part_points)

            # We generate grasps at nerfstudio scale, then scale them to world scale.
            graspable_part.mesh.vertices /= self.optimizer.dataset_scale
            graspable_part.grasps = AntipodalGrasps(
                centers=graspable_part.grasps.centers / self.optimizer.dataset_scale,
                axes=graspable_part.grasps.axes,
            )

            self.parts.append(graspable_part)

    def rank_parts_to_move_single(self) -> list[int]:
        assert self.optimizer.hands_info is not None
        sum_part_dist = onp.array([0.0] * len(self.parts))
        for tstep, (l_hand, r_hand) in self.optimizer.hands_info.items():
            for part_idx in range(len(self.parts)):
                delta = self.optimizer.part_deltas[tstep, part_idx]
                part_means = (
                    self.optimizer.init_means[self.optimizer.group_masks[part_idx]]
                    .detach()
                    .cpu()
                    .numpy()
                )
                part_means = jaxlie.SE3(
                    jnp.array(delta.detach().cpu().numpy())
                ) @ jnp.array(part_means)

                part_dist = onp.inf
                for hand in [l_hand, r_hand]:
                    if hand is not None:
                        for hand_idx in range(hand["keypoints_3d"].shape[0]):
                            pointer = hand["keypoints_3d"][hand_idx, MANO_KEYPOINTS["index"]]
                            part_dist = min(
                                jnp.linalg.norm(pointer - part_means, axis=1).min().item(), part_dist
                            )
                sum_part_dist[part_idx] += part_dist

        # argsort
        part_indices = sum_part_dist.argsort()
        return part_indices.tolist()

    def plan_single(self, T_obj_world: jaxlie.SE3) -> jnp.ndarray:
        part_indices = self.rank_parts_to_move_single()

        left_joint_name, right_joint_name = "left_dummy_joint", "right_dummy_joint"
        left_joint_idx = self.kin.joint_names.index(left_joint_name)
        right_joint_idx = self.kin.joint_names.index(right_joint_name)

        succ_traj = None
        part_idx, grasp_idx = None, None
        for part_idx in part_indices[1:]:
            logger.info(f"Planning for part {part_idx}")
            Ts_part_world = self._create_part_traj(part_idx, T_obj_world)
            for joint_idx in [left_joint_idx, right_joint_idx]:
                trajs = self._create_grasp_trajs(part_idx, Ts_part_world).wxyz_xyz
                for grasp_idx in range(trajs.shape[0]):
                    Ts_grasp_world = trajs[grasp_idx][None, ...]
                    traj = motion_plan_yumi(
                        self.kin, jnp.array([joint_idx]), Ts_grasp_world
                    )
                    Ts_joint_world = self.kin.forward_kinematics(traj)[:, joint_idx]
                    succ = jnp.all(
                        jnp.isclose(
                            jaxlie.SE3(Ts_grasp_world).as_matrix(),
                            jaxlie.SE3(Ts_joint_world).as_matrix(),
                            rtol=1e-2, atol=1e-2
                        )
                    )
                    if succ:
                        succ_traj = traj
                        part_idx, grasp_idx = part_idx, grasp_idx
                        break
                if succ_traj is not None:
                    break
            if succ_traj is not None:
                break
        else:
            raise ValueError("Failed to plan for both arms")

        assert succ_traj is not None
        return succ_traj, part_idx, grasp_idx, Ts_grasp_world

    def rank_parts_to_move_bimanual(self) -> list[tuple[int, int]]:
        raise NotImplementedError

    def plan_bimanual(self, part_idx1: int, part_idx2: int) -> None:
        raise NotImplementedError

    def _create_part_traj(self, part_idx: int, T_obj_world: jaxlie.SE3) -> jaxlie.SE3:
        Ts_part_obj = (
            tf.SE3(self.optimizer.init_p2o[part_idx]) @
            tf.SE3(self.optimizer.part_deltas[:, part_idx])
        ).wxyz_xyz.detach().cpu().numpy()
        Ts_part_obj[..., 4:] = Ts_part_obj[..., 4:] / self.optimizer.dataset_scale
        Ts_part_obj = jaxlie.SE3(Ts_part_obj)
        Ts_part_world = T_obj_world @ Ts_part_obj
        return Ts_part_world

    def _create_grasp_trajs(
        self, part_idx: int, Ts_part_world: jaxlie.SE3
    ) -> jaxlie.SE3:
        part = self.parts[part_idx]
        T_grasp_part = jaxlie.SE3(
            jnp.concatenate(
                [
                    part.grasps.to_se3().wxyz_xyz,
                    part.grasps.to_se3(flip_axis=True).wxyz_xyz,
                ],
                axis=0,
            )
        )

        T_grasp_grasp = self._get_grasp_augs()

        T_grasp_part = jaxlie.SE3(T_grasp_part.wxyz_xyz[:, None]).multiply(
            jaxlie.SE3(T_grasp_grasp.wxyz_xyz[None, :])
        )  # [N_grasps, N_augs, 7]
        T_grasp_part = jaxlie.SE3(
            T_grasp_part.wxyz_xyz.reshape(-1, 7)
        )  # [N_grasps * N_augs, 7]

        num_grasps = T_grasp_part.wxyz_xyz.shape[0]
        timesteps = Ts_part_world.wxyz_xyz.shape[0]

        Ts_grasp_world = jaxlie.SE3(Ts_part_world.wxyz_xyz[None, :]).multiply(
            jaxlie.SE3(T_grasp_part.wxyz_xyz[:, None])
        )

        assert Ts_grasp_world.wxyz_xyz.shape == (num_grasps, timesteps, 7)
        return Ts_grasp_world

    @staticmethod
    def _get_grasp_augs(
        num_rotations: int = 8,
        num_translations: int = 3,
    ) -> jaxlie.SE3:
        rot_augs = jaxlie.SE3.from_rotation(
            rotation=(
                jaxlie.SO3.from_x_radians(jnp.linspace(-jnp.pi, jnp.pi, num_rotations))
            )
        )
        trans_augs = jaxlie.SE3.from_translation(
            translation=jnp.array(
                [
                    [d, 0, 0]
                    for d in (
                        jnp.linspace(-0.005, 0.005, num_translations)
                        if num_translations > 1
                        else jnp.linspace(0, 0, 1)
                    )
                ]
            )
        )
        T_grasp_grasp = jaxlie.SE3(
            jnp.repeat(trans_augs.wxyz_xyz, rot_augs.wxyz_xyz.shape[0], axis=0)
        ).multiply(
            jaxlie.SE3(jnp.tile(rot_augs.wxyz_xyz, (trans_augs.wxyz_xyz.shape[0], 1)))
        )
        T_grasp_grasp = jaxlie.SE3(T_grasp_grasp.wxyz_xyz.reshape(-1, 7))
        return T_grasp_grasp