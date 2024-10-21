"""
Turn points to mesh.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, cast
from jax import Array
import jax
import jaxlie
import numpy as onp
import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Float
import trimesh
import trimesh.bounds
from jaxmp.extras.grasp_antipodal import AntipodalGrasps

from rsrd.motion.motion_optimizer import RigidGroupOptimizer
import rsrd.transforms as tf
from rsrd.util.common import MANO_KEYPOINTS

try:
    import open3d as o3d
except:
    o3d = None


@dataclass
class GraspableObject:
    optimizer: RigidGroupOptimizer
    parts: list[GraspablePart]

    def __init__(
        self,
        optimizer: RigidGroupOptimizer,
    ) -> None:
        self.optimizer = optimizer
        self._create_graspable_parts()

    def _create_graspable_parts(self) -> None:
        self.parts = []
        for mask in self.optimizer.group_masks:
            part_points = self.optimizer.init_means[mask].cpu().numpy()
            part_points -= part_points.mean(axis=0)
            graspable_part = GraspablePart.from_points(part_points)

            # We generate grasps at nerfstudio scale, then scale them to world scale.
            graspable_part.mesh.vertices /= self.optimizer.dataset_scale
            with jdc.copy_and_mutate(graspable_part) as graspable_part:
                graspable_part._grasps = AntipodalGrasps(
                    centers=graspable_part._grasps.centers / self.optimizer.dataset_scale,
                    axes=graspable_part._grasps.axes,
                )

            self.parts.append(graspable_part)

    def get_T_part_world(
        self, timesteps: jnp.ndarray, T_obj_world: jaxlie.SE3
    ) -> list[jaxlie.SE3]:
        """
        Get the part poses in world frame.
        """
        list_Ts_part_world: list[jaxlie.SE3] = []
        for part_idx in range(len(self.parts)):
            Ts_part_obj = (
                (
                    tf.SE3(self.optimizer.init_p2o[part_idx])
                    @ tf.SE3(self.optimizer.part_deltas[:, part_idx])
                )
                .wxyz_xyz.detach()
                .cpu()
                .numpy()
            )
            Ts_part_obj[..., 4:] = Ts_part_obj[..., 4:] / self.optimizer.dataset_scale
            Ts_part_obj = jaxlie.SE3(jnp.array(Ts_part_obj)[timesteps])
            Ts_part_world = T_obj_world @ Ts_part_obj
            list_Ts_part_world.append(Ts_part_world)
        return list_Ts_part_world

    def get_T_grasps_world(
        self, part_idx: int, timesteps: jnp.ndarray, T_obj_world: jaxlie.SE3
    ) -> jaxlie.SE3:
        """
        Get [num_grasps, timesteps, 7].
        """
        T_part_world = self.get_T_part_world(timesteps, T_obj_world)[part_idx]
        T_grasp_part = self.parts[part_idx].grasp

        T_grasp_world = jaxlie.SE3(T_part_world.wxyz_xyz[None, :]).multiply(
            jaxlie.SE3(T_grasp_part.wxyz_xyz[:, None])
        )
        return T_grasp_world

    def get_obj_mesh_in_world(self, timestep: int, T_obj_world: jaxlie.SE3) -> trimesh.Trimesh:
        list_Ts_part_world = self.get_T_part_world(jnp.array([timestep]), T_obj_world)
        sum_mesh = trimesh.Trimesh()
        for part_idx, Ts_part_world in enumerate(list_Ts_part_world):
            part_mesh = self.parts[part_idx].mesh.copy()
            part_mesh.apply_transform(Ts_part_world.as_matrix().squeeze())
            sum_mesh += part_mesh
        return sum_mesh

    def rank_parts_to_move_single(self) -> list[int]:
        mat_left, mat_right = self._get_matrix_for_hand_assignments()
        sum_part_dist = (mat_left + mat_right).sum(axis=0)
        part_indices = sum_part_dist.argsort()
        return part_indices.tolist()
    
    def rank_parts_to_move_bimanual(self) -> list[tuple[int, int]]:
        mat_left, mat_right = self._get_matrix_for_hand_assignments()

        num_groups = mat_left.shape[1]
        assignments = []
        for li in range(num_groups):
            for ri in range(num_groups):
                if li == ri:
                    continue
                dist = mat_left[:, li].sum() + mat_right[:, ri].sum()
                assignments.append((li, ri, dist))
        assignments.sort(key=lambda x: x[2])
        return [(a[0], a[1]) for a in assignments]
    
    def _get_matrix_for_hand_assignments(self) -> tuple[onp.ndarray, onp.ndarray]:
        assert self.optimizer.hands_info is not None
        num_timesteps = len(self.optimizer.hands_info)
        num_parts = len(self.parts)
        sum_part_dist_left = onp.zeros((num_timesteps, num_parts))
        sum_part_dist_right = onp.zeros((num_timesteps, num_parts))
        for idx, (tstep, (l_hand, r_hand)) in enumerate(self.optimizer.hands_info.items()):
            for part_idx in range(len(self.parts)):
                delta = self.optimizer.part_deltas[tstep, part_idx]
                part_means = (
                    self.optimizer.init_means[self.optimizer.group_masks[part_idx]]
                    .detach()
                    .cpu()
                    .numpy()
                )
                part_means -= part_means.mean(axis=0)
                part_means = (
                    jaxlie.SE3(jnp.array(self.optimizer.init_p2o[part_idx].cpu()))
                    @ jaxlie.SE3(jnp.array(delta.detach().cpu().numpy()))
                    @ jnp.array(part_means)
                )

                if l_hand is None:
                    l_hand = {"keypoints_3d": onp.zeros((0,))}
                if r_hand is None:
                    r_hand = {"keypoints_3d": onp.zeros((0,))}

                for hand_idx in range(l_hand["keypoints_3d"].shape[0]):
                    pointer = l_hand["keypoints_3d"][hand_idx, MANO_KEYPOINTS["index"]]
                    d_pointer = jnp.linalg.norm(pointer - part_means, axis=1).min().item()
                    thumb = l_hand["keypoints_3d"][hand_idx, MANO_KEYPOINTS["thumb"]]
                    d_thumb = jnp.linalg.norm(thumb - part_means, axis=1).min().item()
                    sum_part_dist_left[idx, part_idx] += (d_pointer + d_thumb) / 2

                for hand_idx in range(r_hand["keypoints_3d"].shape[0]):
                    pointer = r_hand["keypoints_3d"][hand_idx, MANO_KEYPOINTS["index"]]
                    d_pointer = jnp.linalg.norm(pointer - part_means, axis=1).min().item()
                    thumb = r_hand["keypoints_3d"][hand_idx, MANO_KEYPOINTS["thumb"]]
                    d_thumb = jnp.linalg.norm(thumb - part_means, axis=1).min().item()
                    sum_part_dist_right[idx, part_idx] += (d_pointer + d_thumb) / 2
            
        return sum_part_dist_left, sum_part_dist_right


@jdc.pytree_dataclass
class GraspablePart:
    mesh: jdc.Static[trimesh.Trimesh]
    _grasps: AntipodalGrasps

    @staticmethod
    def from_mesh(mesh: trimesh.Trimesh, max_grasps: int = 10) -> GraspablePart:
        grasps = AntipodalGrasps.from_sample_mesh(mesh, max_samples=100)
        grasps = jax.tree.map(lambda x: x[:max_grasps], grasps)
        return GraspablePart(mesh=mesh, _grasps=grasps)

    @staticmethod
    def from_points(points: onp.ndarray) -> GraspablePart:
        mesh = GraspablePart._points_to_mesh(points)
        return GraspablePart.from_mesh(mesh)

    @staticmethod
    def _points_to_mesh(points: onp.ndarray) -> trimesh.Trimesh:
        """Converts a point cloud to a mesh, using alpha hulls and smoothing."""
        if o3d is None:
            return trimesh.PointCloud(vertices=points).convex_hull

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        pc.estimate_normals()

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pc, 0.04)
        mesh.compute_vertex_normals()

        mesh = trimesh.Trimesh(
            vertices=onp.asarray(mesh.vertices),
            faces=onp.asarray(mesh.triangles),
        )

        # Smooth the mesh.
        _mesh = trimesh.smoothing.filter_mut_dif_laplacian(
            mesh,
        )

        # Simplify the mesh.
        _mesh = _mesh.simplify_quadric_decimation(100)  # This affects speed by a lot!

        # Correct normals are important for grasp sampling!
        try:
            # If mesh is too big... then shrink it down.
            obb_orig, obb_extents = trimesh.bounds.oriented_bounds(_mesh)
            _mesh_obb = _mesh.copy()
            _mesh_obb.vertices = trimesh.transform_points(_mesh.vertices, obb_orig)
            _mesh.vertices = trimesh.transform_points(
                _mesh_obb.vertices, onp.linalg.inv(obb_orig)
            )

            _mesh.fix_normals()
            _mesh.fill_holes()
        except:
            _mesh = trimesh.PointCloud(vertices=points).convex_hull

        return _mesh

    @staticmethod
    def get_grasp_augs(
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

    @property
    def grasp(self):
        """
        Get the grasp poses in part frame, augmented by rotations and translations.
        """
        T_grasp_part = jaxlie.SE3(
            jnp.concatenate(
                [
                    self._grasps.to_se3().wxyz_xyz,
                    self._grasps.to_se3(flip_axis=True).wxyz_xyz,
                ],
                axis=0,
            )
        )
        T_grasp_grasp = self.get_grasp_augs()
        T_grasp_part = jaxlie.SE3(
            jaxlie.SE3(T_grasp_part.wxyz_xyz[:, None]).multiply(
                jaxlie.SE3(T_grasp_grasp.wxyz_xyz[None, :])
            ).wxyz_xyz.reshape(-1, 7)
        )
        return T_grasp_part

