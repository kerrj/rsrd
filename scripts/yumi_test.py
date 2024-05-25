"""
Quick interactive demo for yumi IK, with curobo.
"""

import torch
import viser
import viser.transforms as vtf
import time
import numpy as np
import trimesh
from typing import Optional, List
import moviepy.editor as mpy
from copy import deepcopy

from nerfstudio.utils.eval_utils import eval_setup
from pathlib import Path
from threading import Lock
import warp as wp
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.viewer.viewer import Viewer
from nerfstudio.configs.base_config import ViewerConfig
from garfield.garfield_gaussian_pipeline import GarfieldGaussianPipeline
from nerfstudio.utils import writer
from nerfstudio.models.splatfacto import SH2RGB

from autolab_core import RigidTransform

from toad.yumi_curobo import YumiCurobo
from toad.zed import Zed
from toad.optimization.rigid_group_optimizer import RigidGroupOptimizer
from toad.toad_object import GraspableToadObject

try:
    from yumirws.yumi import YuMi
except ImportError:
    YuMi = None
    print("YuMi not available -- won't control the robot.")


class ZedOptimizer:
    group_masks: List[torch.Tensor]
    optimizer: RigidGroupOptimizer
    viewer: Viewer
    MATCH_RESOLUTION: int = 500
    init_cam_pose: torch.Tensor  # This is aligned with the actual cam pose provided.
    graspable_toad: GraspableToadObject

    def __init__(
        self,
        config_path: Path,
        K: torch.tensor,
        width: int,
        height: int,
        init_cam_pose: Optional[torch.Tensor] = None
    ):
        config_path = config_path
        
        # load the viewer, etc.
        train_config, self.pipeline, _, _ = eval_setup(config_path)
        assert isinstance(self.pipeline, GarfieldGaussianPipeline)
        train_config.logging.local_writer.enable = False

        dataset_scale = self.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_scale

        writer.setup_local_writer(train_config.logging, max_iter=train_config.max_num_iterations)
        self.viewer = Viewer(
            ViewerConfig(
                default_composite_depth=False,
                num_rays_per_chunk=-1
            ),
            config_path.parent,
            self.pipeline.datamanager.get_datapath(),
            self.pipeline,
            train_lock=Lock()
        )

        # Wait for the user to set up the crops and groups.
        self.group_labels, self.group_masks = self._setup_crops_and_groups()

        # Set up the initial camera pose.
        if init_cam_pose is None:
            print("Init cam pose was none, creating one now.")
            H = np.eye(4)
            H[:3,:3] = vtf.SO3.from_x_radians(np.pi/2).as_matrix()
            init_cam_pose = torch.from_numpy(H).float()[None,:3,:] #TODO ground truth cam to yumi

        assert init_cam_pose.shape == (1, 3, 4)
        print("Init cam pose: ", init_cam_pose)
        self.init_cam_pose = init_cam_pose

        # convert to opengl format
        init_cam_pose_opengl = torch.cat([
            init_cam_pose[0], torch.tensor([0, 0, 0, 1], dtype=torch.float32).reshape(1, 4)
        ], dim=0) @ (torch.from_numpy(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])).float())
        init_cam_pose_opengl = init_cam_pose_opengl[None, :3, :]
        
        assert init_cam_pose_opengl.shape == (1, 3, 4)
        print("Init cam pose opengl: ", init_cam_pose_opengl)
        init_cam_pose_opengl[:, :3, 3] = init_cam_pose_opengl[:, :3, 3] * dataset_scale  # convert to meters
        self.init_cam_pose_opengl = init_cam_pose_opengl

        cam2world = Cameras(camera_to_worlds=init_cam_pose_opengl,fx = K[0,0],fy = K[1,1],cx = K[0,2],cy = K[1,2],width=width,height=height)
        cam2world.rescale_output_resolution(self.MATCH_RESOLUTION/max(cam2world.width,cam2world.height))

        # Set up the optimizer.
        self.optimizer = RigidGroupOptimizer(
            self.pipeline.model,
            self.pipeline.datamanager.dino_dataloader,
            cam2world,
            self.group_masks,
            group_labels=self.group_labels,
            dataset_scale=dataset_scale,
            render_lock=self.viewer.train_lock,
        )
        self.is_inited = False

    def _setup_crops_and_groups(self) -> List[torch.Tensor]:
        cluster_labels = None
        while cluster_labels is None:
            _ = input("Please setup the crops and groups.... (enter to continue)")
            cluster_labels = self.pipeline.cluster_labels

        labels = self.pipeline.cluster_labels.int().cuda()
        group_masks = [(cid == labels).cuda() for cid in range(labels.max() + 1)]
        return labels, group_masks

    def set_frame(self, rgb, depth) -> None:
        target_frame_rgb = (rgb/255)
        self.optimizer.set_frame(target_frame_rgb, depth=depth)
        
    def init_obj_pose(self):
        xs, ys, outputs, renders = self.optimizer.initialize_obj_pose(render=True)  # retval only matters for visualization
        if len(renders)>1:
            renders = [r.detach().cpu().numpy()*255 for r in renders]
            #save video as test_camopt.mp4
            out_clip = mpy.ImageSequenceClip(renders, fps=30)  
            out_clip.write_videofile("test_camopt.mp4")

        self.graspable_toad = GraspableToadObject.from_points_and_clusters(
            self.optimizer.init_means.detach().cpu().numpy(),
            self.group_labels.detach().cpu().numpy(),
            scene_scale=self.optimizer.dataset_scale,
        )

        self.is_inited=True

    def step_opt(self,niter):
        assert self.is_inited
        self.optimizer.step(niter=niter,metric_depth=True)

    def get_groups_pc(self) -> trimesh.PointCloud:
        # c2w = self.cam2world.camera_to_worlds.squeeze()  # (3, 4)
        # parts2cam = self.optimizer.get_poses_relative_to_camera(c2w)  # --> (N, 4, 4)
        with torch.no_grad():
            self.optimizer.apply_to_model(self.optimizer.pose_deltas, self.optimizer.centroids, self.optimizer.group_labels)
        points = self.optimizer.dig_model.means.clone().detach()
        colors = SH2RGB(self.optimizer.dig_model.colors.clone().detach())
        points = points / self.optimizer.dataset_scale
        pc = trimesh.PointCloud(points.cpu().numpy(), colors=colors.cpu().numpy())  # pointcloud in world frame
        
        cam2world = torch.cat([
            self.init_cam_pose.squeeze(),
            torch.Tensor([[0, 0, 0, 1]]).to(self.init_cam_pose.device)
        ], dim=0)
        pc.vertices = trimesh.transform_points(
            pc.vertices,
            cam2world.inverse().cpu().numpy()
        )  # pointcloud in camera frame.
        # pc.vertices = trimesh.transform_points(
        #     pc.vertices,
        #     trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0])
        # )  # pointcloud in opencv(?) frame.
        return pc
    
    def get_groups_mesh(self) -> List[trimesh.Trimesh]:
        meshes = deepcopy(self.graspable_toad.meshes_orig)
        parts2cam = self.optimizer.get_poses_relative_to_camera(self.init_cam_pose_opengl.squeeze().cuda())
        for i, mesh in enumerate(meshes):
            parts2cam[i][:3,3] /= self.optimizer.dataset_scale
            mesh.vertices -= self.optimizer.init_means[self.optimizer.group_masks[i]].mean(dim=0).detach().cpu().numpy() / self.optimizer.dataset_scale
            mesh.vertices = trimesh.transform_points(mesh.vertices, parts2cam[i].cpu().numpy())

            # mesh.vertices *= self.optimizer.dataset_scale # put mesh in nerfstudio coords/scale
            # # mesh.vertices -= self.optimizer.dig_model.means[self.group_labels[[i]]].mean(0).detach().cpu().numpy() # put mesh in "part" frame
            # mesh.vertices -= ((self.optimizer.init_means[self.optimizer.group_masks[i]].mean(dim=0))).detach().cpu().numpy()
            # mesh.vertices = trimesh.transform_points(mesh.vertices, parts2cam[i].cpu().numpy()) # put mesh in camera frame
            # mesh.vertices /= self.optimizer.dataset_scale
        return meshes


def main():
    server = viser.ViserServer()

    # YuMi robot code must be placed before any curobo code!
    robot_button = server.add_gui_button(
        "Move physical robot",
        disabled=True,
    )
    if YuMi is not None:
        robot = YuMi()
        @robot_button.on_click
        def _(_):
            # robot.left.move_joints_traj(urdf.get_left_joints().view(1, 7).cpu().numpy().astype(np.float64))
            robot.move_joints_sync(
                l_joints=urdf.get_left_joints().view(1, 7).cpu().numpy().astype(np.float64),
                r_joints=urdf.get_right_joints().view(1, 7).cpu().numpy().astype(np.float64),
                speed=(0.1, np.pi)
            )

    # Needs to be called before any warp pose gets called.
    wp.init()

    urdf = YumiCurobo(
        server,
    )

    # Create two handles, one for each end effector.
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

    # Update the joint positions based on the handle positions.
    # Run IK on the fly!\
    def update_joints():
        joints_from_ik = urdf.ik(
            torch.Tensor([*drag_l_handle.wxyz, *drag_l_handle.position]).view(1, 7),
            torch.Tensor([*drag_r_handle.wxyz, *drag_r_handle.position]).view(1, 7),
            initial_js=urdf.joint_pos[:14],
        )[0].js_solution.position
        assert isinstance(joints_from_ik, torch.Tensor)
        urdf.joint_pos = joints_from_ik

    @drag_r_handle.on_update
    def _(_):
        update_joints()
    @drag_l_handle.on_update
    def _(_):
        update_joints()

    # First update to set the initial joint positions.
    update_joints()

    if YuMi is not None:
        robot_button.disabled = False

    try:
        zed = Zed()
    except Exception as e:
        print(e)
        print("Zed not available -- won't show camera feed.")
        zed = None
    
    camera_tf = RigidTransform.load("data/zed_to_world.tf")
    camera_frame = server.add_frame(
        "camera",
        position=camera_tf.translation,  # rough alignment.
        wxyz=camera_tf.quaternion,
        show_axes=True,
        axes_length=0.1,
        axes_radius=0.005,
    )
    zed_mesh = trimesh.load("data/ZED2.stl")
    assert isinstance(zed_mesh, trimesh.Trimesh)
    server.add_mesh_trimesh(
        "camera/mesh",
        mesh=zed_mesh,
        scale=0.001,
        position=(0.06, 0.042, -0.03),
        wxyz=vtf.SO3.from_rpy_radians(np.pi/2, 0.0, 0.0).wxyz,
    )

    opt_init_handle = server.add_gui_button("Set initial frame", disabled=True)
    if zed is not None:
        l,_,_=zed.get_frame(depth=False)
        zed_opt = ZedOptimizer(
            Path("outputs/buddha_balls_poly/dig/2024-05-23_153552/config.yml"),
            zed.get_K(),
            l.shape[1],
            l.shape[0],
            init_cam_pose=torch.from_numpy(vtf.SE3(wxyz_xyz=np.array([*camera_frame.wxyz, *camera_frame.position])).as_matrix()[None,:3,:]).float()
        )
        @opt_init_handle.on_click
        def _(_):
            opt_init_handle.disabled = True
            l,_,depth = zed.get_frame(depth=True)
            zed_opt.set_frame(l,depth)
            zed_opt.init_obj_pose()
            pointcloud = zed_opt.get_groups_pc()
            server.add_point_cloud(
                "camera/groups",
                points=pointcloud.vertices,
                colors=pointcloud.colors[:, :3],
                point_size=0.002
            )
            # then have the zed_optimizer be allowed to run the optimizer steps.
        opt_init_handle.disabled = False

    while True:
        if zed is not None:
            left, right, depth = zed.get_frame()
            if zed_opt.is_inited:
                zed_opt.set_frame(left,depth)
                zed_opt.step_opt(niter=50)
                pointcloud = zed_opt.get_groups_pc()
                server.add_point_cloud(
                    "camera/groups",
                    points=pointcloud.vertices,
                    colors=pointcloud.colors[:, :3],
                    point_size=0.002
                )
                meshes = zed_opt.get_groups_mesh()
                for i, mesh in enumerate(meshes):
                    server.add_mesh_trimesh(
                        f"camera/object/group_{i}",
                        mesh=mesh,
                    )
                
            K = torch.from_numpy(zed.get_K()).float().cuda()

            img_wh = left.shape[:2][::-1]

            grid = (
                torch.stack(torch.meshgrid(torch.arange(img_wh[0],device='cuda'), torch.arange(img_wh[1],device='cuda'), indexing='xy'), 2) + 0.5
            )

            homo_grid = torch.concatenate([grid,torch.ones((grid.shape[0],grid.shape[1],1),device='cuda')],axis=2).reshape(-1,3)
            local_dirs = torch.matmul(torch.linalg.inv(K),homo_grid.T).T
            points = (local_dirs * depth.reshape(-1,1)).float()
            points = points.reshape(-1,3)
            
            mask = depth.reshape(-1, 1) <= 1.0
            points = points.reshape(-1, 3)[mask.flatten()][::4].cpu().numpy()
            left = left.reshape(-1, 3)[mask.flatten()][::4].cpu().numpy()

            server.add_point_cloud("camera/points", points = points.reshape(-1,3), colors=left.reshape(-1,3),point_size=.001)


        # print(camera_frame.position, camera_frame.wxyz)

        else:
            time.sleep(1)


if __name__ == "__main__":
    main()