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
import plotly.graph_objects as go

from toad.yumi_curobo import YumiCurobo
from toad.zed import Zed

try:
    from yumirws.yumi import YuMi
except ImportError:
    YuMi = None
    print("YuMi not available -- won't control the robot.")

def main():
    server = viser.ViserServer()

    # YuMi robot code must be placed before any curobo code!
    if YuMi is not None:
        robot = YuMi()
        robot_button = server.add_gui_button(
            "Move physical robot",
        )
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
    # Run IK on the fly!
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

    try:
        zed = Zed()
    except:
        print("Zed not available -- won't show camera feed.")
        zed = None

    if zed is not None:
        # get the camera.
        camera_tf = RigidTransform.load("data/zed_to_world.tf")
        server.add_frame(
            "camera",
            position=camera_tf.translation,  # rough alignment.
            wxyz=camera_tf.quaternion,
            show_axes=True,
            axes_length=0.1,
            axes_radius=0.005,
        )
        server.add_mesh_trimesh(
            "camera/mesh",
            mesh=zed.zed_mesh,
            scale=0.001,
            position=zed.cam_to_zed.translation,
            wxyz=zed.cam_to_zed.quaternion,
        )

    rgb_vis_handle = server.add_gui_plotly(
        go.Figure(),
        aspect=9/16,
    )
    depth_vis_handle = server.add_gui_plotly(
        go.Figure(),
        aspect=9/16,
    )

    while True:
        if zed is not None:
            left, right, depth = zed.get_frame()
            K = torch.from_numpy(zed.get_K()).float().cuda()
            assert isinstance(left, torch.Tensor) and isinstance(depth, torch.Tensor)
            points, colors = Zed.project_depth(left, depth, K)

            server.add_point_cloud(
                "camera/points",
                points=points,
                colors=colors,
                point_size=0.001,
            )
            if rgb_vis_handle is not None:
                fig = zed.plotly_render(left[::4, ::4].cpu().numpy())
                rgb_vis_handle.figure = fig
            if depth_vis_handle is not None:
                fig = zed.plotly_render(depth[::4, ::4].cpu().numpy())
                depth_vis_handle.figure = fig

        else:
            time.sleep(1)


if __name__ == "__main__":
    main()
