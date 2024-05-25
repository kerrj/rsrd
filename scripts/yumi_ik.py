"""
Quick interactive demo for yumi IK, with curobo.
"""

import torch
import viser
import viser.transforms as vtf
import time
import numpy as np
import trimesh
from autolab_core import RigidTransform

from toad.yumi_curobo import YumiCurobo
from toad.zed import Zed

try:
    from yumirws.yumi import YuMi
except ImportError:
    YuMi = None
    print("YuMi not available -- won't control the robot.")

def main():
    # import multiprocessing as mp
    # mp.set_start_method("spawn")
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
    except:
        print("Zed not available -- won't show camera feed.")
        zed = None
    
    # get the camera.
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

    while True:
        if zed is not None:
            left, right, depth = zed.get_frame()

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

        else:
            time.sleep(1)


if __name__ == "__main__":
    main()