"""
zed.py
"""

import pyzed.sl as sl
from typing import Optional, Tuple, cast
import torch
import numpy as np
from threading import Lock
import plotly
from plotly import express as px
import trimesh
from pathlib import Path
from raftstereo.raft_stereo import create_raft, raft_inference
from autolab_core import RigidTransform

class Zed():
    width: int
    """Width of the rgb/depth images."""
    height: int
    """Height of the rgb/depth images."""
    raft_lock: Lock
    """Lock for the camera, for raft-stereo depth!"""

    zed_mesh: trimesh.Trimesh
    """Trimesh of the ZED camera."""
    T_cam_zed: RigidTransform
    """Transform from left camera to ZED camera base."""

    def __init__(self, recording_file = None, start_time = 0.0, flip: bool = False):
        init = sl.InitParameters()
        if recording_file is not None:
            init.set_from_svo_file(recording_file)
            init.depth_mode=sl.DEPTH_MODE.NONE
            init.camera_resolution = sl.RESOLUTION.HD1080
            init.sdk_verbose = 1
            init.camera_fps = 30
        else:
            init.camera_resolution = sl.RESOLUTION.HD1080
            init.sdk_verbose = 1
            init.camera_fps = 30
            init.depth_mode=sl.DEPTH_MODE.NONE
            init.depth_minimum_distance = 100#millimeters
        if flip:
            init.camera_image_flip = sl.FLIP_MODE.ON
        self.init_res = 1920 if init.camera_resolution == sl.RESOLUTION.HD1080 else 1280
        print("INIT RES",self.init_res)
        self.width = 1280
        self.height = 720
        self.cam = sl.Camera()
        status = self.cam.open(init)
        if recording_file is not None:
            fps = self.cam.get_camera_information().camera_configuration.fps
            self.cam.set_svo_position(int(start_time*fps))
        if status != sl.ERROR_CODE.SUCCESS: #Ensure the camera has opened succesfully
            print("Camera Open : "+repr(status)+". Exit program.")
            exit()
        else:
            print("Opened camera")

        # Create lock for raft -- gpu threading messes up CUDA memory state, with curobo...
        self.raft_lock = Lock()
        with self.raft_lock:
            self.model = create_raft()

        left_cx = self.get_K(cam='left')[0,2]
        right_cx = self.get_K(cam='right')[0,2]
        self.cx_diff = (right_cx-left_cx)

        # For visualiation.
        zed_path = Path(__file__).parent / "../.." / Path("data/zed/ZED2.stl")
        zed_mesh = cast(
            trimesh.Trimesh,
            trimesh.load(str(zed_path))
        )

        # Center the ZED mesh with the left eye.
        import rsrd.transforms as tf
        T_mesh_zed = tf.SE3.from_rotation_and_translation(
            rotation=tf.SO3.from_x_radians(torch.Tensor([np.pi / 2])),
            translation=torch.Tensor([[0.06, 0.042, -0.035]]) / 0.001,
        ).as_matrix().squeeze().numpy()
        zed_mesh.apply_transform(T_mesh_zed)
        zed_mesh.vertices *= 0.001

        assert isinstance(zed_mesh, trimesh.Trimesh)
        self.zed_mesh = zed_mesh

    def get_frame(
        self, depth=True
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        res = sl.Resolution()
        res.width = self.width
        res.height = self.height
        r = self.width/self.init_res
        if self.cam.grab() == sl.ERROR_CODE.SUCCESS:
            left_rgb = sl.Mat()
            right_rgb = sl.Mat()
            self.cam.retrieve_image(left_rgb, sl.VIEW.LEFT, sl.MEM.CPU, res)
            self.cam.retrieve_image(right_rgb, sl.VIEW.RIGHT, sl.MEM.CPU, res)
            left,right = torch.from_numpy(np.flip(left_rgb.get_data()[...,:3],axis=2).copy()).cuda(), torch.from_numpy(np.flip(right_rgb.get_data()[...,:3],axis=2).copy()).cuda()
            if depth:
                left_torch,right_torch = left.permute(2,0,1),right.permute(2,0,1)
                with self.raft_lock:
                    flow = raft_inference(left_torch,right_torch,self.model)
                fx = self.get_K()[0,0]
                depth = fx*self.get_stereo_transform()[0,3]/(flow.abs()+self.cx_diff)
            else:
                depth = None
            return left, right, depth
        elif self.cam.grab() == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            print("End of recording file")
            return None,None,None
        else:
            raise RuntimeError("Could not grab frame")

    def get_K(self,cam='left') -> np.ndarray:
        calib = self.cam.get_camera_information().camera_configuration.calibration_parameters
        if cam=='left':
            intrinsics = calib.left_cam
        else:
            intrinsics = calib.right_cam
        r = self.width/self.init_res
        K = np.array([[intrinsics.fx*r, 0, intrinsics.cx*r], [0, intrinsics.fy*r, intrinsics.cy*r], [0, 0, 1]])
        return K

    def get_stereo_transform(self):
        transform = self.cam.get_camera_information().camera_configuration.calibration_parameters.stereo_transform.m
        transform[:3,3] /= 1000#convert to meters
        return transform

    def start_record(self, out_path):
        recordingParameters = sl.RecordingParameters()
        recordingParameters.compression_mode = sl.SVO_COMPRESSION_MODE.H264
        recordingParameters.video_filename = out_path
        err = self.cam.enable_recording(recordingParameters)

    def stop_record(self):
        self.cam.disable_recording()

    @staticmethod
    def plotly_render(frame) -> plotly.graph_objs.Figure:
        fig = px.imshow(frame)
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            yaxis_visible=False,
            yaxis_showticklabels=False,
            xaxis_visible=False,
            xaxis_showticklabels=False,
        )
        return fig

    @staticmethod
    def project_depth(
        rgb: torch.Tensor,
        depth: torch.Tensor,
        K: torch.Tensor,
        depth_threshold: float = 1.0,
        subsample: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Deproject RGBD image to point cloud (points, colors), using provided intrinsics.
        Also threshold/subsample pointcloud for visualization speed."""

        img_wh = rgb.shape[:2][::-1]

        grid = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(img_wh[0], device="cuda"),
                    torch.arange(img_wh[1], device="cuda"),
                    indexing="xy",
                ),
                2,
            )
            + 0.5
        )

        homo_grid = torch.concat(
            [grid, torch.ones((grid.shape[0], grid.shape[1], 1), device="cuda")],
            dim=2
        ).reshape(-1, 3)
        local_dirs = torch.matmul(torch.linalg.inv(K),homo_grid.T).T
        points = (local_dirs * depth.reshape(-1,1)).float()
        points = points.reshape(-1,3)

        mask = depth.reshape(-1, 1) <= depth_threshold
        points = points.reshape(-1, 3)[mask.flatten()][::subsample].cpu().numpy()
        colors = rgb.reshape(-1, 3)[mask.flatten()][::subsample].cpu().numpy()

        return (points, colors)

