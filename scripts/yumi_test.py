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

from curobo.types.state import JointState

from autolab_core import RigidTransform

from toad.yumi_curobo import YumiCurobo, createTableWorld
from toad.zed import Zed
from toad.optimization.rigid_group_optimizer import RigidGroupOptimizer
from toad.toad_optimizer import ToadOptimizer

try:
    from yumirws.yumi import YuMi
except ImportError:
    YuMi = None
    print("YuMi not available -- won't control the robot.")

if __name__ == "__main__":
    server = viser.ViserServer()

    # YuMi robot code must be placed before any curobo code!
    with server.add_gui_folder("Robot control"):
        gripper_open_button = server.add_gui_button(
            "Open physical gripper",
            disabled=True,
        )
        gripper_close_button = server.add_gui_button(
            "Close physical gripper",
            disabled=True,
        )
        robot_handler = server.add_gui_button("Move robot", disabled=True)
    if YuMi is not None:
        try:
            robot = YuMi()
            @gripper_open_button.on_click
            def _(_):
                robot.left.open_gripper()
                robot.right.open_gripper()
            @gripper_close_button.on_click
            def _(_):
                robot.left.close_gripper()
                robot.right.close_gripper()
            @robot_handler.on_click
            def _(_):
                robot.move_joints_sync(
                    l_joints=urdf.get_left_joints().view(1, 7).cpu().numpy().astype(np.float64),
                    r_joints=urdf.get_right_joints().view(1, 7).cpu().numpy().astype(np.float64),
                    speed=(0.1, np.pi)
                )
            gripper_close_button.disabled = False
            gripper_open_button.disabled = False
            robot_handler.disabled = False
        except:
            print("YuMi not initialized -- won't control the robot.")
            robot = None

    # Needs to be called before any warp pose gets called.
    wp.init()

    urdf = YumiCurobo(
        server,
        ik_solver_batch_size=240,
        motion_gen_batch_size=240,
    )
    # for i, mesh in enumerate(urdf.get_robot_as_spheres(urdf.joint_pos)):
    #     server.add_mesh_trimesh(f'mesh_{i}', mesh)

    try:
        zed = Zed()
    except Exception as e:
        print(e)
        print("Zed not available -- won't show camera feed.")
        zed = None
    play_zed = True
    
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

    part_handle = None

    if zed is not None:
        l,_,_=zed.get_frame(depth=False)
        zed_opt = ToadOptimizer(
            Path("outputs/buddha_balls_poly/dig/2024-05-23_184345/config.yml"),
            # Path("outputs/calbear/dig/2024-05-24_160735/config.yml"),
            zed.get_K(),
            l.shape[1],
            l.shape[0],
            init_cam_pose=torch.from_numpy(vtf.SE3(wxyz_xyz=np.array([*camera_frame.wxyz, *camera_frame.position])).as_matrix()[None,:3,:]).float(),
        )
        part_handle = server.add_gui_number("Part", 0, 0, zed_opt.num_groups-1, 1, disabled=True)
        @opt_init_handle.on_click
        def _(_):
            global play_zed
            opt_init_handle.disabled = True
            play_zed = False
            l,_,depth = zed.get_frame(depth=True)
            zed_opt.set_frame(l,depth)
            zed_opt.init_obj_pose()
            try:
                zed_opt.optimizer.load_trajectory("renders/buddha_balls_poly/keyframes.pt")
                keyframe_ind.disabled = False
            except FileNotFoundError:
                print("\nCouldn't load trajectory file for rigid group optimizer\n")
            # then have the zed_optimizer be allowed to run the optimizer steps.

            # Set up the robot.
            def curry_mesh(idx, tf):
                server.add_frame(
                    f"camera/object/group_{idx}",
                    position=tf.translation(),
                    wxyz=tf.rotation().wxyz,
                    show_axes=True,
                    axes_length=0.02,
                    axes_radius=.002
                )
                mesh = zed_opt.get_mesh_centered(idx)
                handle = server.add_mesh_trimesh(
                    f"camera/object/group_{idx}/mesh",
                    mesh=mesh,
                )
                grasps = zed_opt.get_grasps_centered(idx) # [N_grasps, 7]
                grasp_mesh = zed_opt.toad_object.grasp_axis_mesh()
                for j, grasp in enumerate(grasps):
                    server.add_mesh_trimesh(
                        f"camera/object/group_{idx}/grasp_{j}",
                        grasp_mesh,
                        position=grasp[:3],
                        wxyz=grasp[3:],
                    )
                @handle.on_click
                def _(_):
                    part_handle.value = idx
                return handle
            for i, tf in enumerate(zed_opt.get_parts2cam()):
                curry_mesh(i, tf)
            play_zed = True
            button_handle.disabled = False
            part_handle.disabled = False

        opt_init_handle.disabled = False

    traj, traj_handle, play_handle = None, None, None
    button_handle = server.add_gui_button("Calculate working grasps", disabled=True)
    keyframe_ind = server.add_gui_slider("Keyframe Index",0.0,1.0,.01,0.0,disabled=True)

    @button_handle.on_click
    def _(_):
        global traj, traj_handle, play_handle, play_zed
        play_zed = False
        button_handle.disabled = True
        start = time.time()

        # Update the object's pose, for collisionbody.
        world_config = createTableWorld()
        assert world_config.mesh is not None  # should not be none after post_init.
        poses_part2cam = zed_opt.get_parts2cam()
        poses_part2world = [vtf.SE3(wxyz_xyz=np.array([*camera_frame.wxyz, *camera_frame.position])).multiply(pose) for pose in poses_part2cam]
        poses_wxyz_xyz = [pose.wxyz_xyz for pose in poses_part2world]
        world_config.mesh.extend(zed_opt.toad_object.to_world_config(poses_wxyz_xyz=poses_wxyz_xyz))

        urdf.update_world(world_config)

        grasps = zed_opt.get_grasps_centered(part_handle.value)  # [N_grasps, 7]
        grasps_gripper = zed_opt.toad_object.to_gripper_frame(grasps, urdf._tooltip_to_gripper)
        grasp_cand_list = poses_part2world[part_handle.value].multiply(grasps_gripper)

        goal_l_wxyz_xyz = torch.Tensor(grasp_cand_list.wxyz_xyz)
        goal_r_wxyz_xyz = torch.Tensor([[0, 1, 0, 0, 0.4, -0.2, 0.5]]).expand(grasp_cand_list.wxyz_xyz.shape[0], 7)            

        start_state = JointState.from_position(urdf.home_pos)
        motiongen_list = urdf.motiongen(
            goal_l_wxyz_xyz=goal_l_wxyz_xyz,
            goal_r_wxyz_xyz=goal_r_wxyz_xyz,
            start_state=start_state
        )
        assert len(motiongen_list) == 1

        print(f"Time taken: {time.time() - start:.2f} s")
        button_handle.disabled = False
        play_zed = True

        # If none succeeded...
        if not motiongen_list[0].success.any():
            return

        traj = motiongen_list[0].interpolated_plan[motiongen_list[0].success].position
        print(motiongen_list[0].ik_time, motiongen_list[0].trajopt_time, motiongen_list[0].finetune_time)

        if zed_opt.optimizer.keyframes is not None:
            prev_js = motiongen_list[0].interpolated_plan[:, -1]
            js_list = []
            start = time.time()
            play_zed = False  # genuinely does seem like a problem to have both zed and curobo running at the same time.
            for i in range(len(zed_opt.optimizer.keyframes)):
                poses_part2cam = zed_opt.get_parts2cam(keyframe=i)
                poses_part2world = [vtf.SE3(wxyz_xyz=np.array([*camera_frame.wxyz, *camera_frame.position])).multiply(pose) for pose in poses_part2cam]
                grasp_cand_list = poses_part2world[part_handle.value].multiply(grasps_gripper)
                goal_l_wxyz_xyz = torch.Tensor(grasp_cand_list.wxyz_xyz)

                joints_from_ik = urdf.ik(
                    goal_l_wxyz_xyz=goal_l_wxyz_xyz,
                    goal_r_wxyz_xyz=goal_r_wxyz_xyz,
                    initial_js=prev_js.position.squeeze()[:, :14],
                )[0].js_solution
                js_list.append(joints_from_ik.position[:, :, :14])
                prev_js = joints_from_ik
            # traj = torch.cat([traj.unsqueeze(0), torch.stack(js_list)])
            print(f"Time taken for keyframes: {time.time() - start:.2f} s")
            traj = torch.cat([motiongen_list[0].interpolated_plan.position[:, :, :14], *js_list], dim=1)
            traj = traj[motiongen_list[0].success]
            play_zed = True

        if traj_handle is not None:
            traj_handle.remove()
            play_handle.remove()
        traj_handle = server.add_gui_slider("trajectory", 0, len(traj)-1, 1, 0)
        play_handle = server.add_gui_slider("play", 0, traj.shape[1]-1, 1, 0)
        move_traj_handle = server.add_gui_button("Play traj")

        @traj_handle.on_update
        def _(_):
            assert traj is not None
            urdf.joint_pos = traj[int(traj_handle.value), int(play_handle.value)]

        @play_handle.on_update
        def _(_):
            assert traj is not None
            urdf.joint_pos = traj[int(traj_handle.value), int(play_handle.value)]

        @move_traj_handle.on_click
        def _(_):
            robot.move_joints_sync(
                l_joints=traj[int(traj_handle.value)][:, :7].cpu().numpy().astype(np.float64),
                r_joints=traj[int(traj_handle.value)][:, 7:].cpu().numpy().astype(np.float64),
                speed=(0.1, np.pi)
            )


    while True:
        if zed is not None and play_zed:
            left, right, depth = zed.get_frame()
            if zed_opt.initialized:
                zed_opt.set_frame(left,depth)
                zed_opt.step_opt(niter=50)
                keyframe = None if keyframe_ind.disabled else int(keyframe_ind.value*(len(zed_opt.optimizer.keyframes)-1))
                tf_list = zed_opt.get_parts2cam(keyframe = keyframe)
                # tf_list = zed_opt.get_parts2cam(keyframe=None)
                for idx, tf in enumerate(tf_list):
                    server.add_frame(
                        f"camera/object/group_{idx}",
                        position=tf.translation(),
                        wxyz=tf.rotation().wxyz,
                        show_axes=True,
                        axes_length=0.02,
                        axes_radius=.002
                    )
                
            # Visualize the camera feed.
            K = torch.from_numpy(zed.get_K()).float().cuda()
            img_wh = left.shape[:2][::-1]
            grid = (torch.stack(torch.meshgrid(torch.arange(img_wh[0],device='cuda'), torch.arange(img_wh[1],device='cuda'), indexing='xy'), 2) + 0.5)
            homo_grid = torch.concatenate([grid,torch.ones((grid.shape[0],grid.shape[1],1),device='cuda')],axis=2).reshape(-1,3)
            local_dirs = torch.matmul(torch.linalg.inv(K),homo_grid.T).T
            points = (local_dirs * depth.reshape(-1,1)).float()
            points = points.reshape(-1,3)
            mask = depth.reshape(-1, 1) <= 1.0
            points = points.reshape(-1, 3)[mask.flatten()][::4].cpu().numpy()
            left = left.reshape(-1, 3)[mask.flatten()][::4].cpu().numpy()
            server.add_point_cloud("camera/points", points = points.reshape(-1,3), colors=left.reshape(-1,3),point_size=.0008,point_shape='rounded')

        else:
            time.sleep(1)

