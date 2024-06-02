from __future__ import annotations
from typing import List, Optional, Union
import torch
import numpy as np
import trimesh
import trimesh.repair
import trimesh.creation
import viser.transforms as vtf
import open3d as o3d
import dataclasses
from pathlib import Path
import tyro

from curobo.geom.types import Mesh, Sphere


@dataclasses.dataclass(frozen=True)
class ToadObject:
    """
    Note that all objects/parts/grasps are stored in *metric* scale, not nerfstudio.
    """

    points: torch.Tensor
    """Gaussian centers of the object. Shape: (N, 3)."""
    clusters: torch.Tensor
    """Cluster labels for each point. Shape: (N,)."""
    _meshes: List[trimesh.Trimesh]
    """List of meshes, one for each cluster. Each mesh is centered at the centroid / part frame."""
    scene_scale: float
    """Scale of the scene. Used to convert the object to metric scale. Default: 1.0.
    This is defined as: (point in metric) = (point in scene) / scene_scale."""

    @staticmethod
    def from_ply(ply_file: Path) -> ToadObject:
        pcd_object = trimesh.load(ply_file)
        assert type(pcd_object) == trimesh.PointCloud
        pcd_object.vertices = pcd_object.vertices - np.mean(pcd_object.vertices, axis=0)
        scene_scale = pcd_object.metadata['_ply_raw']['vertex']['data']['scene_scale'][0]

        # It's possible that polycam scale is off by a factor of 10...?
        while (pcd_object.vertices.max(axis=0) - pcd_object.vertices.min(axis=0)).max() / scene_scale > 1:
            scene_scale *= 10

        # Meshify in nerfstudio/provided scene scale, but store in metric scale.
        part_mesh_list = []
        cluster_labels = pcd_object.metadata['_ply_raw']['vertex']['data']['cluster_labels'].astype(np.int32)
        for i in range(cluster_labels.max() + 1):
            mask = cluster_labels == i
            part_vertices = np.array(pcd_object.vertices[mask])
            assert isinstance(part_vertices, np.ndarray)
            part_mesh = ToadObject._points_to_mesh(part_vertices)
            part_mesh.vertices -= part_vertices.mean(axis=0)  # Center the mesh at the centroid.
            part_mesh.vertices /= scene_scale
            part_mesh_list.append(part_mesh)

        return ToadObject(
            points=torch.tensor(pcd_object.vertices) / scene_scale,
            clusters=torch.tensor(cluster_labels),
            scene_scale=scene_scale,
            _meshes=part_mesh_list,
        )
    
    @staticmethod
    def from_points_and_clusters(
        points: np.ndarray,
        clusters: np.ndarray,
        scene_scale: float = 1.0
    ) -> ToadObject:
        assert points.shape[1] == 3
        assert len(points) == len(clusters)

        # Meshify in nerfstudio/provided scene scale, but store in metric scale.
        part_mesh_list = []
        for i in range(clusters.max() + 1):
            mask = clusters == i
            part_vertices = points[mask]
            assert isinstance(part_vertices, np.ndarray)
            part_mesh = ToadObject._points_to_mesh(part_vertices)
            part_mesh.vertices -= part_vertices.mean(axis=0)  # Center the mesh at the centroid.
            part_mesh.vertices /= scene_scale
            part_mesh_list.append(part_mesh)

        return ToadObject(
            points=torch.tensor(points) / scene_scale,
            clusters=torch.tensor(clusters),
            scene_scale=scene_scale,
            _meshes=part_mesh_list,
        )

    @staticmethod
    def dummy_object() -> ToadObject:
        cylinder_1 = trimesh.creation.cylinder(radius=0.01, height=0.1, sections=20)
        cylinder_2 = trimesh.creation.cylinder(radius=0.01, height=0.1, sections=20, transform=trimesh.transformations.translation_matrix([0.05, 0.0, 0.0]))
        points = np.concatenate([cylinder_1.vertices, cylinder_2.vertices], axis=0)
        clusters = np.concatenate([np.zeros(len(cylinder_1.vertices)), np.ones(len(cylinder_2.vertices))], axis=0).astype(np.int32)
        toad_obj = ToadObject(
            points=torch.tensor(points),
            clusters=torch.tensor(clusters),
            _meshes=[cylinder_1, cylinder_2],
            scene_scale=1.0,
        )
        return toad_obj

    @staticmethod
    def _points_to_mesh(vertices: np.ndarray) -> trimesh.Trimesh:
        """Converts a point cloud to a mesh, using alpha hulls and smoothing."""
        points = o3d.geometry.PointCloud()
        points.points = o3d.utility.Vector3dVector(vertices)
        points.estimate_normals()

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(points, 0.04)
        mesh.compute_vertex_normals()

        mesh = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices),
            faces=np.asarray(mesh.triangles),
        )
        mesh = mesh.simplify_quadric_decimation(50)

        # Correct normals are important for grasp sampling!
        mesh.fix_normals()
        mesh.fill_holes()

        return mesh

    @property
    def meshes(self) -> List[trimesh.Trimesh]:
        """Deepcopy of meshes, one for each part. Meshes are centered at the part centroid."""
        return [mesh.copy() for mesh in self._meshes]

    def to_world_config(self, poses_wxyz_xyz: Optional[List[np.ndarray]] = None, spheres: bool = False) -> Union[List[Mesh], List[Sphere]]:
        """Turn mesh into curobo's WorldConfig, for collision checking.
        Args:
        - poses_wxyz_xyz: List of poses for each part (from part frame), in the format [x, y, z, w, x, y, z].
        - spheres: If True, return spheres instead of meshes.
        """
        if poses_wxyz_xyz is None:
            poses_wxyz_xyz = [np.array([1, 0, 0, 0, 0, 0, 0])] * len(self._meshes)

        object_mesh_list = [
            Mesh(
                name=f'object_{i}',
                vertices=mesh.vertices,
                faces=mesh.faces,  # type: ignore
                pose=[*poses_wxyz_xyz[i][4:], *poses_wxyz_xyz[i][:4]] # xyz, wxyz
            )
            for i, mesh in enumerate(self.meshes)  # duplicate it?
        ]

        if spheres:
            object_sphere_list = [
                mesh.get_bounding_spheres(n_spheres=100)
                for mesh in object_mesh_list
            ]
            # flatten the list of lists
            object_sphere_list = [item for sublist in object_sphere_list for item in sublist]
            return object_sphere_list
        return object_mesh_list
    
    def centroid(self, cluster: int) -> torch.Tensor:
        """Get centroid of a given part. Every frame's part is defined at its centroid."""
        mask = self.clusters == cluster
        return self.points[mask].mean(dim=0)


@dataclasses.dataclass(frozen=True)
class GraspableToadObject(ToadObject):
    grasps: List[torch.Tensor]
    """List of grasps, of length N_clusters. Each element is a tensor of shape (N_grasps, 7), for grasp center and axis (quat)."""

    @staticmethod
    def from_ply(ply_file: Path) -> GraspableToadObject:
        toad = ToadObject.from_ply(ply_file)
        # Compute grasps. :-)
        mesh_list = toad._meshes
        grasp_list = []
        for mesh in mesh_list:
            grasps = GraspableToadObject._compute_grasps(mesh)
            grasp_list.append(grasps)

        return GraspableToadObject(
            points=toad.points,
            clusters=toad.clusters,
            _meshes=toad._meshes,
            scene_scale=toad.scene_scale,
            grasps=grasp_list
        )
    
    @staticmethod
    def from_points_and_clusters(
        points: np.ndarray,
        clusters: np.ndarray,
        scene_scale: float = 1.0
    ) -> GraspableToadObject:
        toad = ToadObject.from_points_and_clusters(points, clusters, scene_scale)
        # Compute grasps. :-)
        mesh_list = toad._meshes
        grasp_list = []
        for mesh in mesh_list:
            grasps = None
            try:
                grasps = GraspableToadObject._compute_grasps(mesh)
            except:
                pass
            if grasps is None:
                # Check if the mesh is inverted...
                mesh.invert()
                grasps = GraspableToadObject._compute_grasps(mesh)
            assert grasps is not None, "No grasps found for the object, is the mesh correct?"
            grasp_list.append(grasps)

        return GraspableToadObject(
            points=toad.points,
            clusters=toad.clusters,
            _meshes=toad._meshes,
            scene_scale=toad.scene_scale,
            grasps=grasp_list
        )

    def __post_init__(self):
        # If grasps are provided, don't compute them.
        if self.grasps is not None:
            return

        # Compute grasps. :-)
        mesh_list = self._meshes
        grasp_list = []
        for mesh in mesh_list:
            grasps = self._compute_grasps(mesh)
            grasp_list.append(grasps)
        self.__setattr__('grasps', grasp_list)

    @staticmethod
    def _compute_grasps(mesh: trimesh.Trimesh, num_grasps: int = 10) -> torch.Tensor:
        """Computes grasps for a single part. It's possible that the number of grasps
        returned is less than num_grasps, if they are filtereed due to collision or other reasons (e.g., too low score).
        
        Note that dexgrasp isn't open sourced -- but it should be possible to rewrite it if required."""

        # Move all imports here, so that they don't interfere with the main code.
        # The logger... setup seems to change globally, so we try to avoid that.
        import logging
        curr_logging_level = logging.getLogger().getEffectiveLevel()

        from autolab_core import RigidTransform

        from dexgrasp import YamlLoader
        from dexgrasp.envs.states.objects import GraspableObject
        from dexgrasp.envs.states.states import GraspingState
        from dexgrasp.policies.parallel_jaw_grasp_sampling_policy import ParallelJawGraspSamplingPolicy
        from dexgrasp.envs.states.grippers import ParallelJawGripper

        package_dir = Path(__file__).parent.parent

        basedir = package_dir / Path('dependencies/dexgrasp/cfg')
        data_prefix = package_dir / Path('dependencies/dexgrasp/')

        yaml_obj_loader = YamlLoader(basedir=str(basedir), data_prefix=str(data_prefix))
        sampler = yaml_obj_loader.load('parallel_jaw_grasp_sampling_policy')
        assert isinstance(sampler, ParallelJawGraspSamplingPolicy)

        sampler.max_approach_angle = None  # No approach angle constraint -- so grasps can be offset from the z-axis.

        gripper = yaml_obj_loader.load('yumi_default')  # i.e., the small, plastic grippers.
        assert isinstance(gripper, ParallelJawGripper)

        pose = RigidTransform(from_frame='obj', to_frame='world')
        obj = GraspableObject("object", mesh, pose, mass=0.1)  # Making the mass reasonable is actually important, because this affects sorting.
        state = GraspingState(graspable_objects=[obj])

        # Get grasp pose, in gripper frame.
        grasps = sampler.ranked_actions(state, num_grasps, grippers=[gripper], perturb=True)
        if len(grasps) == 0:
            # Redo, but with mesh flipped; normals are probably incorrect.
            raise ValueError("No grasps found for the object, is the mesh correct?")

        # Convert grasp pose to tooltip frame.
        grasp_to_gripper = RigidTransform(
            translation=(gripper.tooltip_poses[0].translation + gripper.tooltip_poses[1].translation) / 2,
            rotation=np.eye(3),
            from_frame='grasp',
            to_frame='gripper'
        )

        # Make sure that the grasp tensor is exactly num_grasps long. IK will fail if it's not.
        grasp_tensor = torch.zeros((len(grasps), 7))
        for i, g in enumerate(grasps):
            grasp_pose = g.pose * grasp_to_gripper
            grasp_center = grasp_pose.translation
            grasp_axis = vtf.SO3.from_matrix(grasp_pose.rotation).wxyz
            grasp_tensor[i] = torch.tensor([*grasp_center, *grasp_axis])

        grasp_tensor_padding = grasp_tensor[-1].expand(num_grasps - len(grasps), 7)
        grasp_tensor = torch.cat([grasp_tensor, grasp_tensor_padding], dim=0)

        # Reset the logging level...
        logging.getLogger().setLevel(curr_logging_level)

        return grasp_tensor

    @staticmethod
    def grasp_axis_mesh():
        rotation = trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])
        transform = np.eye(4); transform[:3, :3] = rotation[:3, :3]
        bar = trimesh.creation.cylinder(radius=0.001, height=0.05, transform=transform)
        bar.visual.vertex_colors = [150, 150, 255, 255]  # type: ignore[attr-defined]
        return bar

    @staticmethod
    def to_gripper_frame(
        grasps_tensor: torch.Tensor,
        tooltip_to_gripper: vtf.SE3,
        num_rotations: int = 8,
        num_translations: int = 3,
    ) -> vtf.SE3:
        """Put grasps in the tooltip frame to the gripper frame, using provided tooltip_to_gripper offset.
        , and also augment grasps by rotating them around the z-axis of the tooltip frame."""
        grasps = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3(grasps_tensor[:, 3:].cpu().numpy()),
            translation=grasps_tensor[:, :3].cpu().numpy()
        )
        rot_augs = (
            vtf.SE3.from_rotation(
                rotation=vtf.SO3.from_x_radians(
                    np.linspace(-np.pi, np.pi, num_rotations)
                ),
            ).multiply(tooltip_to_gripper.inverse())
        )
        trans_augs = (
            vtf.SE3.from_translation(
                translation=np.array([
                    [d, 0, 0]
                    for d in np.linspace(-0.005, 0.005, num_translations)
                ])
            )
        )
        augs = vtf.SE3(np.tile(rot_augs.wxyz_xyz, (trans_augs.wxyz_xyz.shape[0], 1))).multiply(
            vtf.SE3(np.repeat(trans_augs.wxyz_xyz, rot_augs.wxyz_xyz.shape[0], axis=0))
        )

        len_grasps = grasps.wxyz_xyz.shape[0]
        len_augs = augs.wxyz_xyz.shape[0]

        augs_expanded = vtf.SE3(np.tile(augs.wxyz_xyz, (len_grasps, 1)))
        grasps_expanded = vtf.SE3(np.repeat(grasps.wxyz_xyz, len_augs, axis=0))

        return grasps_expanded.multiply(augs_expanded)


def main(
    ply_path: Path
):
    """
    Visualize the object and its grasps in viser, assuming scene_scale and cluster_labels are saved with garfield.

    Args:
        ply_path: Path to the ply file of the object, saved from nerfstudio.
    """
    import viser
    server = viser.ViserServer()

    toad = GraspableToadObject.from_ply(ply_path)

    for i, mesh in enumerate(toad.meshes):
        grasps = toad.grasps[i]
        server.add_frame(
            f"object/{i}",
            position=toad.centroid(i).cpu().numpy(),
            wxyz=(1, 0, 0, 0),
            show_axes=True,
            axes_length=0.02,
            axes_radius=.002
        )
        server.add_mesh_trimesh(
            f"object/{i}/mesh", mesh
        )
        for j, g in enumerate(grasps):
            server.add_frame(
                f"object/{i}/grasp/grasp_{j}",
                position=g[:3].cpu().numpy(),
                wxyz=g[3:].cpu().numpy(),
                show_axes=False,
            )
            server.add_mesh_trimesh(
                f"object/{i}/grasp/grasp_{j}/mesh",
                GraspableToadObject.grasp_axis_mesh()
            )

    import time
    while True:
        time.sleep(1000)

if __name__ == "__main__":
    tyro.cli(main)