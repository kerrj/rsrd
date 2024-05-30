import torch
import viser
import viser.transforms as vtf
import time
import numpy as np
from typing import Optional, List
import tyro

from nerfstudio.utils.eval_utils import eval_setup
from pathlib import Path
import warp as wp
from nerfstudio.models.splatfacto import SH2RGB

from autolab_core import RigidTransform

from toad.yumi_curobo import YumiCurobo
from toad.zed import Zed
from toad.toad_optimizer import ToadOptimizer

try:
    from yumirws.yumi import YuMi
except ImportError:
    YuMi = None
    print("YuMi not available -- won't control the robot.")

def main(
    config_path: Path = Path("outputs/buddha_balls_poly/dig/2024-05-23_184345/config.yml"),
):
    """Quick interactive demo for object tracking.

    Args:
        config_path: Path to the nerfstudio config file.
    """
    # Path("outputs/calbear/dig/2024-05-24_160735/config.yml"),
    # Path("outputs/mallet/dig/2024-05-27_180206/config.yml")

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

    if YuMi is not None:
        robot_button.disabled = False

    # Set up the camera.
    opt_init_handle = server.add_gui_button("Set initial frame", disabled=True)
    try:
        zed = Zed()

        # Visualize the camera.
        camera_tf = RigidTransform.load("data/zed_to_world.tf")
        camera_frame = server.add_frame(
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

        l, _, depth = zed.get_frame(depth=True)  # type: ignore
        toad_opt = ToadOptimizer(
            config_path,
            zed.get_K(),
            l.shape[1],
            l.shape[0], 
            # zed.width,
            # zed.height,
            init_cam_pose=torch.from_numpy(
                vtf.SE3(
                    wxyz_xyz=np.array([*camera_frame.wxyz, *camera_frame.position])
                ).as_matrix()[None, :3, :]
            ).float(),
        )

        @opt_init_handle.on_click
        def _(_):
            assert (zed is not None) and (toad_opt is not None)
            opt_init_handle.disabled = True
            l, _, depth = zed.get_frame(depth=True)
            toad_opt.set_frame(l,depth)
            with zed.raft_lock:
                toad_opt.init_obj_pose()
            # then have the zed_optimizer be allowed to run the optimizer steps.
        opt_init_handle.disabled = False

    except:
        print("Zed not available -- won't show camera feed.")
        print("Also won't run the optimizer.")
        zed, toad_opt = None, None

    while True:
        if zed is not None:
            left, right, depth = zed.get_frame()
            assert isinstance(toad_opt, ToadOptimizer)
            if toad_opt.initialized:
                toad_opt.set_frame(left,depth)
                with zed.raft_lock:
                    toad_opt.step_opt(niter=50)

                tf_list = toad_opt.get_parts2cam()
                for idx, tf in enumerate(tf_list):
                    server.add_frame(
                        f"camera/object/group_{idx}",
                        position=tf.translation(),
                        wxyz=tf.rotation().wxyz,
                        show_axes=True,
                        axes_length=0.02,
                        axes_radius=.002
                    )
                    mesh = toad_opt.toad_object.meshes[idx]
                    server.add_mesh_trimesh(
                        f"camera/object/group_{idx}/mesh",
                        mesh=mesh,
                    )
                    grasps = toad_opt.toad_object.grasps[idx] # [N_grasps, 7]
                    grasp_mesh = toad_opt.toad_object.grasp_axis_mesh()
                    for j, grasp in enumerate(grasps):
                        server.add_mesh_trimesh(
                            f"camera/object/group_{idx}/grasp_{j}",
                            grasp_mesh,
                            position=grasp[:3].cpu().numpy(),
                            wxyz=grasp[3:].cpu().numpy(),
                        )

            # Visualize pointcloud.
            K = torch.from_numpy(zed.get_K()).float().cuda()
            assert isinstance(left, torch.Tensor) and isinstance(depth, torch.Tensor)
            points, colors = Zed.project_depth(left, depth, K)
            server.add_point_cloud(
                "camera/points",
                points=points,
                colors=colors,
                point_size=0.001,
            )

        else:
            time.sleep(1)


if __name__ == "__main__":
    tyro.cli(main)
