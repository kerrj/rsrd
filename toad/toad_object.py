from __future__ import annotations
from typing import List, Optional
import torch
import numpy as np
import trimesh
import trimesh.repair
import viser.transforms as vtf
import open3d as o3d
import dataclasses
from pathlib import Path


@dataclasses.dataclass
class ToadObject:
    points: torch.Tensor
    """Gaussian centers of the object. Shape: (N, 3)."""
    clusters: torch.Tensor
    """Cluster labels for each point. Shape: (N,)."""
    scene_scale: float = 1.0
    """Scale of the scene. Used to convert the object to metric scale. Default: 1.0.
    This is defined as: (point in metric) = (point in scene) / scene_scale."""

    @staticmethod
    def from_ply(ply_file: str) -> ToadObject:
        pcd_object = trimesh.load(ply_file)
        assert type(pcd_object) == trimesh.PointCloud
        pcd_object.vertices = pcd_object.vertices - np.mean(pcd_object.vertices, axis=0)
        scene_scale = pcd_object.metadata['_ply_raw']['vertex']['data']['scene_scale'][0]
        
        vertices = pcd_object.vertices / scene_scale
        if (vertices.max(axis=0) - vertices.min(axis=0)).max() > 1.0:
            scene_scale *= 20

        cluster_labels = pcd_object.metadata['_ply_raw']['vertex']['data']['cluster_labels'].astype(np.int32)

        return ToadObject(
            points=torch.tensor(pcd_object.vertices),
            clusters=torch.tensor(cluster_labels),
            scene_scale=scene_scale
        )

    def to_mesh(self) -> List[trimesh.Trimesh]:
        """Returns a list of meshes, one for each part, in metric scale."""
        part_mesh_list = []
        for i in range(self.clusters.max() + 1):
            mask = self.clusters == i
            part_vertices = np.array(self.points[mask])
            part_mesh = self._points_to_mesh(part_vertices)
            part_mesh.vertices /= self.scene_scale
            part_mesh_list.append(part_mesh)

        return part_mesh_list

    def _points_to_mesh(self, vertices: np.ndarray) -> trimesh.Trimesh:
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
        mesh.fix_normals()

        for _ in range(3):
            mesh.subdivide()
            trimesh.smoothing.filter_mut_dif_laplacian(mesh, lamb=0.2, iterations=10)
        
        return mesh


@dataclasses.dataclass
class GraspableToadObject(ToadObject):
    grasps: Optional[List[torch.Tensor]] = None
    """List of grasps, of length N_clusters. Each element is a tensor of shape (N_grasps, 7), for grasp center and axis (quat)."""

    @staticmethod
    def from_ply(ply_file: str) -> GraspableToadObject:
        toad = ToadObject.from_ply(ply_file)
        return GraspableToadObject(
            points=toad.points,
            clusters=toad.clusters,
            scene_scale=toad.scene_scale
        )

    def __post_init__(self):
        # If grasps are provided, don't compute them.
        if self.grasps is not None:
            return

        # Compute grasps. :-)
        mesh_list = self.to_mesh()
        grasp_list = []
        for mesh in mesh_list:
            grasps = self._compute_grasps(mesh)
            grasp_list.append(grasps)
        self.grasps = grasp_list

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
        # print([g.confidence for g in grasps])

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
        bar.visual.vertex_colors = [255, 0, 0, 255]
        return bar


if __name__ == "__main__":
    import viser
    server = viser.ViserServer()

    ply_file = "data/mug.ply"
    toad = GraspableToadObject.from_ply(ply_file)

    for i, mesh in enumerate(toad.to_mesh()):
        grasps = toad.grasps[i]
        server.add_mesh_trimesh(
            f"object/{i}/mesh", mesh
        )
        for j, g in enumerate(grasps):
            server.add_frame(
                f"object/{i}/grasp/grasp_{j}",
                position=g[:3],
                wxyz=g[3:],
                show_axes=False,
            )
            server.add_mesh_trimesh(
                f"object/{i}/grasp/grasp_{j}/mesh",
                GraspableToadObject.grasp_axis_mesh()
            )

    import time
    while True:
        time.sleep(1000)
