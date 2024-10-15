"""
Turn points to mesh.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, cast
from jax import Array
import jaxlie
import numpy as onp
import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Float
import trimesh
import trimesh.bounds
import open3d as o3d


@dataclass
class GraspableObj:
    mesh: trimesh.Trimesh
    grasps: AntipodalGrasps

    @staticmethod
    def from_mesh(mesh: trimesh.Trimesh) -> GraspableObj:
        return GraspableObj(
            mesh=mesh,
            grasps=AntipodalGrasps.from_sample_mesh(mesh),
        )

    @staticmethod
    def from_points(points: onp.ndarray) -> GraspableObj:
        mesh = GraspableObj._points_to_mesh(points)
        return GraspableObj(
            mesh=mesh,
            grasps=AntipodalGrasps.from_sample_mesh(mesh),
        )

    @staticmethod
    def _points_to_mesh(points: onp.ndarray) -> trimesh.Trimesh:
        """Converts a point cloud to a mesh, using alpha hulls and smoothing."""
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


@jdc.pytree_dataclass
class AntipodalGrasps:
    centers: Float[Array, "*batch 3"]
    axes: Float[Array, "*batch 3"]

    def __len__(self) -> int:
        return self.centers.shape[0]

    @staticmethod
    def from_sample_mesh(
        mesh: trimesh.Trimesh,
        max_samples=1000,
        max_width=float("inf"),
        max_angle_deviation=onp.pi / 4,
    ) -> AntipodalGrasps:
        """
        Sample antipodal grasps from a given mesh, using rejection sampling.
        May return fewer grasps than `max_samples`.
        """
        grasp_centers, grasp_axes = [], []

        sampled_points, sampled_face_indices = cast(
            tuple[onp.ndarray, onp.ndarray],
            trimesh.sample.sample_surface(mesh, max_samples),
        )
        min_dot_product = onp.cos(max_angle_deviation)

        for sample_idx in range(max_samples):
            p1 = sampled_points[sample_idx]
            n1 = mesh.face_normals[sampled_face_indices[sample_idx]]

            # Raycast!
            locations, _, index_tri = mesh.ray.intersects_location(
                p1.reshape(1, 3), -n1.reshape(1, 3), multiple_hits=False
            )

            if len(locations) == 0:
                continue

            p2 = locations[0]
            n2 = mesh.face_normals[index_tri[0]]

            # Check grasp width.
            grasp_width = onp.linalg.norm(p2 - p1)
            if grasp_width > max_width:
                continue

            # Check for antipodal condition.
            grasp_center = (p1 + p2) / 2
            grasp_direction = p1 - p2
            grasp_direction /= onp.linalg.norm(grasp_direction)

            if (
                onp.dot(n1, grasp_direction) > min_dot_product
                and onp.dot(n2, -grasp_direction) > min_dot_product
            ):
                grasp_centers.append(grasp_center)
                grasp_axes.append(grasp_direction)

        return AntipodalGrasps(
            centers=jnp.array(grasp_centers),
            axes=jnp.array(grasp_axes),
        )

    def to_trimesh(
        self,
        axes_radius: float = 0.005,
        axes_height: float = 0.1,
        indices: Optional[tuple[int, ...]] = None,
        along_axis: Literal["x", "y", "z"] = "x",
    ) -> trimesh.Trimesh:
        """
        Convert the grasp to a trimesh object.
        """
        # Create "base" grasp visualization, centered at origin + lined up with x-axis.
        transform = onp.eye(4)
        if along_axis == "x":
            rotation = trimesh.transformations.rotation_matrix(onp.pi / 2, [0, 1, 0])
        elif along_axis == "y":
            rotation = trimesh.transformations.rotation_matrix(onp.pi / 2, [1, 0, 0])
        else:
            rotation = onp.eye(4)

        transform[:3, :3] = rotation[:3, :3]
        mesh = trimesh.creation.cylinder(
            radius=axes_radius, height=axes_height, transform=transform
        )
        mesh.visual.vertex_colors = [150, 150, 255, 255]  # type: ignore[attr-defined]

        meshes = []
        grasp_transforms = self.to_se3(along_axis=along_axis).as_matrix()
        for idx in range(self.centers.shape[0]):
            if indices is not None and idx not in indices:
                continue
            mesh_copy = mesh.copy()
            mesh_copy.apply_transform(grasp_transforms[idx])
            meshes.append(mesh_copy)

        return sum(meshes, trimesh.Trimesh())

    def to_se3(
        self, along_axis: Literal["x", "y", "z"] = "x", flip_axis: bool = False
    ) -> jaxlie.SE3:
        # Create rotmat, first assuming the x-axis is the grasp axis.
        x_axes = self.axes
        if flip_axis:
            x_axes = -x_axes

        delta = x_axes + (
            jnp.sign(x_axes[..., 0] + 1e-6)[..., None]
            * jnp.roll(x_axes, shift=1, axis=-1)
        )
        y_axes = jnp.cross(x_axes, delta)
        y_axes = y_axes / (jnp.linalg.norm(y_axes, axis=-1, keepdims=True) + 1e-6)
        assert jnp.isclose(x_axes, y_axes).all(axis=-1).sum() == 0

        z_axes = jnp.cross(x_axes, y_axes)

        if along_axis == "x":
            rotmat = jnp.stack([x_axes, y_axes, z_axes], axis=-1)
        elif along_axis == "y":
            rotmat = jnp.stack([z_axes, x_axes, y_axes], axis=-1)
        elif along_axis == "z":
            rotmat = jnp.stack([y_axes, z_axes, x_axes], axis=-1)

        assert jnp.isnan(rotmat).sum() == 0

        # Use the axis-angle representation to create the rotation matrix.
        return jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.from_matrix(rotmat), self.centers
        )
