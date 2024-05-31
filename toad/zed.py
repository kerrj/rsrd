import pyzed.sl as sl
from typing import Optional, Tuple
import torch
import numpy as np
from threading import Lock
import plotly
from plotly import express as px
import trimesh
from pathlib import Path
from raftstereo.raft_stereo import *
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
    cam_to_zed: RigidTransform
    """Transform from left camera to ZED camera base."""

    def __init__(self, recording_file = None, start_time = 0.0):
        init = sl.InitParameters()
        if recording_file is not None:
            init.set_from_svo_file(recording_file)
            # disable depth
            # init.camera_image_flip = sl.FLIP_MODE.ON
            init.depth_mode=sl.DEPTH_MODE.NONE
            init.camera_resolution = sl.RESOLUTION.HD1080
            init.sdk_verbose = 1
            init.camera_fps = 30
        else:
            init.camera_resolution = sl.RESOLUTION.HD1080
            init.sdk_verbose = 1
            init.camera_fps = 30
            # flip camera
            # init.camera_image_flip = sl.FLIP_MODE.ON
            init.depth_mode=sl.DEPTH_MODE.NONE
            init.depth_minimum_distance = 100#millimeters
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
        zed_path = Path(__file__).parent / ".." / Path("data/ZED2.stl")
        zed_mesh = trimesh.load(str(zed_path))
        assert isinstance(zed_mesh, trimesh.Trimesh)
        self.zed_mesh = zed_mesh
        self.cam_to_zed = RigidTransform(
            rotation=RigidTransform.quaternion_from_axis_angle(
                np.array([1, 0, 0]) * (np.pi / 2)
            ),
            translation=np.array([0.06, 0.042, -0.03]),
        )

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
        """Deproject RGBD image to point cloud, using provided intrinsics.
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


if __name__ == "__main__":
    import torch
    from viser import ViserServer
    zed = Zed()
    zed.start_record("/home/chungmin/Documents/please2/toad/motion_vids/redbox_moving.svo2")
    import os
    # os.makedirs(out_dir,exist_ok=True)
    i = 0
    import cv2
    while True:
        left, right, _ = zed.get_frame(depth=False)
        # if left is None:
        #     break
        left,right = left.cpu().numpy(),right.cpu().numpy()
        cv2.imshow("Left Image", left)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        # # save left as jpg with PIL
        # from PIL import Image
        # out_dir = "/home/chungmin/Documents/please2/toad/motion_vids/redbox_ball"
        # Image.fromarray(left).save(os.path.join(out_dir,f"left_{i}.jpg"))
        # #save depth
        # np.save(os.path.join(out_dir,f"depth_{i}.npy"),depth)
        i+=1

    # zed.start_record("/home/justin/lerf/motion_vids/mac_charger_fold.svo2")
    # # s = ViserServer()
    # import cv2
    # # fig=None
    # while True:
    #     left, right, depth = zed.get_frame(depth=False)
    #     # if fig is None:
    #     #     fig = s.add_gui_plotly(plotly_render(depth))
    #     # else:
    #     #     fig.figure = plotly_render(depth)
    #     cv2.imshow("Left Image", left.cpu().numpy())
    #     key = cv2.waitKey(1)
    #     if key == ord('q'):
    #         break

    # cv2.destroyAllWindows()



    # #code for visualizing poincloud
    # import viser
    # from matplotlib import pyplot as plt
    # import viser.transforms as tf
    # v = ViserServer()
    # gui_reset_up = v.add_gui_button(
    #     "Reset up direction",
    #     hint="Set the camera control 'up' direction to the current camera's 'up'.",
    # )

    # @gui_reset_up.on_click
    # def _(event: viser.GuiEvent) -> None:
    #     client = event.client
    #     assert client is not None
    #     client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
    #         [0.0, -1.0, 0.0]
    #     )
    # while True:
    #     left,right,depth = zed.get_frame()
    #     left = left.cpu().numpy()
    #     depth = depth.cpu().numpy()
    #     # import matplotlib.pyplot as plt
    #     # plt.imshow(left)
    #     # plt.show()
    #     K = zed.get_K()
    #     T_world_camera = np.eye(4)

    #     img_wh = left.shape[:2][::-1]

    #     grid = (
    #         np.stack(np.meshgrid(np.arange(img_wh[0]), np.arange(img_wh[1])), 2) + 0.5
    #     )

    #     homo_grid = np.concatenate([grid,np.ones((grid.shape[0],grid.shape[1],1))],axis=2).reshape(-1,3)
    #     local_dirs = np.matmul(np.linalg.inv(K),homo_grid.T).T
    #     points = (local_dirs * depth.reshape(-1,1)).astype(np.float32)
    #     points = points.reshape(-1,3)
    #     v.add_point_cloud("points", points = points.reshape(-1,3), colors=left.reshape(-1,3),point_size=.001)
