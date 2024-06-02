from typing import Tuple
from pathlib import Path
import numpy as np
import torch
import time

import viser
import viser.transforms as vtf

import warp as wp

from autolab_core import RigidTransform

from toad.toad_optimizer import ToadOptimizer
from toad.yumi.yumi_robot import YumiRobot
from toad.zed import Zed

import tyro

def create_zed_and_toad(
    server: viser.ViserServer,
    config_path: Path,
    camera_frame_name: str = "camera",
) -> Tuple[Zed, viser.FrameHandle, ToadOptimizer]:
    # Visualize the camera.
    zed = Zed()
    camera_tf = RigidTransform.load("data/zed_to_world.tf")
    camera_frame = server.add_frame(
        f"{camera_frame_name}",
        position=camera_tf.translation,  # rough alignment.
        wxyz=camera_tf.quaternion,
        show_axes=True,
        axes_length=0.1,
        axes_radius=0.005,
    )
    server.add_mesh_trimesh(
        f"{camera_frame_name}/mesh",
        mesh=zed.zed_mesh,
        scale=0.001,
        position=zed.cam_to_zed.translation,
        wxyz=zed.cam_to_zed.quaternion,
    )    

    # Create the ToadOptimizer.
    toad_opt = ToadOptimizer(
        config_path,
        zed.get_K(),
        zed.width,
       zed.height,
        # zed.width,
        # zed.height,
        init_cam_pose=torch.from_numpy(
            vtf.SE3(
                wxyz_xyz=np.array([*camera_frame.wxyz, *camera_frame.position])
            ).as_matrix()[None, :3, :]
        ).float(),
    )
    return zed, camera_frame, toad_opt


def replay_traj(
    config_path: Path = Path("outputs/nerfgun/dig/2024-05-30_210410/config.yml"),
    keyframe_path: Path = Path("renders/nerfgun/keyframes.pt")
    # config_path: Path = Path("outputs/mallet/dig/2024-05-27_180206/config.yml"),
    # keyframe_path: Path = Path("renders/mallet/keyframes.pt")
):
    """Quick interactive demo for object traj following.

    Args:
        config_path: Path to the nerfstudio config file.
        keyframe_path: Path to the keyframe file.
    """
    server = viser.ViserServer()
    wp.init()

    robot = YumiRobot(
        server,
        batch_size=120,
    )

    zed, camera_frame, toad_opt = create_zed_and_toad(
        server,
        config_path,
        camera_frame_name="camera",
    )

    # Quick hack to make object move around in object frame.
    toad_opt.optimizer.objreg2objinit = torch.zeros((4, 4)).float().cuda()
    toad_opt.optimizer.objreg2objinit[:3, :3] = torch.eye(3).float().cuda()

    # Load the keyframes.
    toad_opt.optimizer.load_trajectory(keyframe_path)

    object_frame = server.add_transform_controls(
        "camera/object",
        scale=0.2,
    )

    # Keep track of:
    # "moving" part -- the part that is being moved.
    # "anchor" part -- the part that is being anchored.
    # One arm should be in charge of moving, the other in charge of anchoring.
    part_handle = server.add_gui_number(
        "Moving", -1, -1, toad_opt.num_groups - 1, 1, disabled=True
    )
    anchor_handle = server.add_gui_number(
        "Anchor", -1, -1, toad_opt.num_groups - 1, 1, disabled=True
    )
    reset_part_anchor_handle = server.add_gui_button("Reset part and anchor")
    @reset_part_anchor_handle.on_click
    def _(_):
        part_handle.value = -1
        anchor_handle.value = -1
        for i in range(toad_opt.num_groups):
            curry_mesh(i)

    # Make the meshes s.t. that if you double-click, it becomes the anchor.
    def curry_mesh(idx):
        mesh = toad_opt.toad_object.meshes[idx]
        if part_handle.value == idx:
            mesh.visual.vertex_colors = [100, 255, 100, 255]
        elif anchor_handle.value == idx:
            mesh.visual.vertex_colors = [255, 100, 100, 255]

        handle = server.add_mesh_trimesh(
            f"camera/object/group_{idx}/mesh",
            mesh=mesh,
        )
        @handle.on_click
        def _(_):
            container_handle = server.add_3d_gui_container(
                f"camera/object/group_{idx}/container",
            )
            with container_handle:
                anchor_button = server.add_gui_button("Anchor")
                @anchor_button.on_click
                def _(_):
                    anchor_handle.value = idx
                    if part_handle.value == idx:
                        part_handle.value = -1
                    curry_mesh(idx)
                move_button = server.add_gui_button("Move")
                @move_button.on_click
                def _(_):
                    part_handle.value = idx
                    if anchor_handle.value == idx:
                        anchor_handle.value = -1
                    curry_mesh(idx)
                close_button = server.add_gui_button("Close")
                @close_button.on_click
                def _(_):
                    container_handle.remove()

        return handle

    for i, tf in enumerate(toad_opt.get_parts2cam(keyframe=0)):
        server.add_frame(
            f"camera/object/group_{i}",
            position=tf.translation(),
            wxyz=tf.rotation().wxyz,
            show_axes=True,
            axes_length=0.02,
            axes_radius=.002
        )
        curry_mesh(i)
        grasps = toad_opt.toad_object.grasps[i].numpy() # [N_grasps, 7]
        grasp_mesh = toad_opt.toad_object.grasp_axis_mesh()
        for j, grasp in enumerate(grasps):
            server.add_mesh_trimesh(
                f"camera/object/group_{i}/grasp_{j}",
                grasp_mesh,
                position=grasp[:3],
                wxyz=grasp[3:],
            )

    # Visualize the keyframes.
    keyframe_ind = server.add_gui_slider(
        "Keyframe Index", min=0, max=len(toad_opt.optimizer.keyframes) - 1, step=1, initial_value=0
    )
    def update_keyframe(keyframe: int):
        for idx, tf in enumerate(toad_opt.get_parts2cam(keyframe=keyframe)):
            server.add_frame(
                f"camera/object/group_{idx}",
                position=tf.translation(),
                wxyz=tf.rotation().wxyz,
                show_axes=True,
                axes_length=0.02,
                axes_radius=.002
            )
    @keyframe_ind.on_update
    def _(_):
        update_keyframe(keyframe_ind.value)

    # Create the trajectories!
    traj, traj_handle, play_handle = None, None, None
    button_handle = server.add_gui_button("Calculate working grasps")
    @button_handle.on_click
    def _(_):
        nonlocal traj, traj_handle, play_handle

        if part_handle.value == -1:
            print("Please select a part to move.")
            return

        button_handle.disabled = True

        # Get the object part poses in world frame.
        # This isn't the way it should be done with object registration!
        camera_vtf = vtf.SE3(
            wxyz_xyz=np.array([*camera_frame.wxyz, *camera_frame.position])
        )
        obj_tf_vtf = vtf.SE3(
            wxyz_xyz=np.array([*object_frame.wxyz, *object_frame.position])
        )
        poses_part2world = [
            camera_vtf.multiply(obj_tf_vtf).multiply(pose)
            for pose in toad_opt.get_parts2cam(keyframe=0)
        ]

        # Update the object's pose, for collisionbody.
        poses_wxyz_xyz = [pose.wxyz_xyz for pose in poses_part2world]
        mesh_list = toad_opt.toad_object.to_world_config(poses_wxyz_xyz=poses_wxyz_xyz)
        robot.plan.update_world_objects(mesh_list)

        # Get grasps in world frame.
        grasps = toad_opt.toad_object.grasps[part_handle.value]  # [N_grasps, 7]
        grasps_gripper = toad_opt.toad_object.to_gripper_frame(
            grasps, robot.tooltip_to_gripper
        )
        grasp_cand_list = poses_part2world[part_handle.value].multiply(grasps_gripper)

        # Find the anchor grasp pose, if it exists.
        # ...... this is not correct!
        if anchor_handle.value != -1:
            anchor_pose_l = anchor_pose_r = torch.from_numpy(
                poses_part2world[anchor_handle.value].wxyz_xyz
            ).flatten()
        else:
            anchor_pose_l = torch.Tensor([[0, 1, 0, 0, 0.4, 0.2, 0.5]])
            anchor_pose_r = torch.Tensor([[0, 1, 0, 0, 0.4, -0.2, 0.5]])

        goal_l_wxyz_xyz = torch.cat([
            torch.Tensor(grasp_cand_list.wxyz_xyz),
            anchor_pose_l.expand(grasp_cand_list.wxyz_xyz.shape[0], 7)
        ])
        goal_r_wxyz_xyz = torch.cat([
            anchor_pose_r.expand(grasp_cand_list.wxyz_xyz.shape[0], 7),
            torch.Tensor(grasp_cand_list.wxyz_xyz),
        ])

        approach_traj, approach_success = robot.plan.gen_motion_from_goal(
            goal_l_wxyz_xyz=goal_l_wxyz_xyz,
            goal_r_wxyz_xyz=goal_r_wxyz_xyz,
            initial_js=robot.plan.home_pos,
        )
        if approach_traj is None:
            print("Failed to approach grasp for object.")
            button_handle.disabled = False
            return
        assert isinstance(approach_traj, torch.Tensor) and isinstance(approach_success, torch.Tensor)

        # Get the object part poses in world frame.
        # This isn't the way it should be done with object registration!
        grasp_path_wxyz_xyz = torch.cat([
            torch.Tensor(
                camera_vtf.multiply(obj_tf_vtf).multiply(
                    toad_opt.get_parts2cam(keyframe_idx)[part_handle.value]
                )
                .multiply(grasps_gripper)
                .wxyz_xyz
            ).unsqueeze(1)  # [num_grasps, 7] -> [num_grasps, 1, 7]
            for keyframe_idx in range(len(toad_opt.optimizer.keyframes))
        ], dim=1) # [num_grasps, num_keyframes, 7]

        path_l_wxyz_xyz = torch.cat([
            grasp_path_wxyz_xyz,
            anchor_pose_l.expand(grasp_cand_list.wxyz_xyz.shape[0], len(toad_opt.optimizer.keyframes), 7)
        ], dim=0)
        path_r_wxyz_xyz = torch.cat([
            anchor_pose_r.expand(grasp_cand_list.wxyz_xyz.shape[0], len(toad_opt.optimizer.keyframes), 7),
            grasp_path_wxyz_xyz
        ], dim=0)

        assert len(path_l_wxyz_xyz.shape) == 3
        assert path_l_wxyz_xyz.shape[0] == 2*grasp_cand_list.wxyz_xyz.shape[0]  # batch
        assert path_l_wxyz_xyz.shape[1] == len(toad_opt.optimizer.keyframes)  # time
        assert path_l_wxyz_xyz.shape[2] == 7  # wxyz_xyz

        # Let's subsample the keyframes.
        waypoint_subsample = 10
        path_l_wxyz_xyz = path_l_wxyz_xyz[:, ::waypoint_subsample, :]
        path_r_wxyz_xyz = path_r_wxyz_xyz[:, ::waypoint_subsample, :]

        waypoint_traj, waypoint_success = robot.plan.gen_motion_from_ik_chain(
            path_l_wxyz_xyz=path_l_wxyz_xyz,
            path_r_wxyz_xyz=path_r_wxyz_xyz,
            initial_js=approach_traj[:, -1, :]
        )
        if waypoint_traj is None:
            print("Following the object part waypoints failed.")
            button_handle.disabled = False
            return

        assert isinstance(waypoint_traj, torch.Tensor) and isinstance(waypoint_success, torch.Tensor)
        waypoint_success = waypoint_success.all(dim=1)  # [batch, time] -> [batch,]

        traj = torch.cat([approach_traj, waypoint_traj], dim=1)  # [batch, time, dof]
        success = approach_success & waypoint_success  # [batch,]

        if not success.any():
            print("No successful trajectory found.")
            button_handle.disabled = False
            return

        traj = traj[success]
        assert len(traj.shape) == 3, "Should be [batch, time, dof]."

        if traj_handle is not None:
            traj_handle.remove()
        traj_handle = server.add_gui_slider("Trajectory Index", 0, len(traj) - 1, 1, 0)

        if play_handle is not None:
            play_handle.remove()
        play_handle = server.add_gui_slider("play", min=0, max=traj.shape[1]-1, step=1, initial_value=0)

        button_handle.disabled = False

        def move_to_traj_position():
            assert traj is not None and traj_handle is not None and play_handle is not None
            robot.joint_pos = traj[traj_handle.value][play_handle.value].view(1, 14)
            if play_handle.value >= approach_traj.shape[1]:
                update_keyframe((play_handle.value - approach_traj.shape[1])*waypoint_subsample)

        @traj_handle.on_update
        def _(_):
            move_to_traj_position()
        @play_handle.on_update
        def _(_):
            move_to_traj_position()

    export_button = server.add_gui_button("Export trajectory")
    @export_button.on_click
    def _(_):
        if traj is None:
            print("No trajectory to export.")
            return
        if traj_handle is None:
            print("No trajectory handle.")
            return
        np.save("trajectory.npy", traj[traj_handle.value].cpu().numpy())


    while True:
        time.sleep(1)


if __name__ == "__main__":
    tyro.cli(replay_traj)
