"""
Interactive demo for finding (some) non-colliding grasp on a specified object part, even as the object moves around.
"""

import viser
import viser.transforms as vtf
import time
import numpy as np
import torch
from curobo.geom.types import Cuboid, Mesh, WorldConfig

from toad.toad_object import GraspableToadObject
from toad.yumi_curobo import YumiCurobo

if __name__ == "__main__":
    server = viser.ViserServer()

    # ply_file = "data/mug.ply"
    # ply_file = "data/mug_new.ply"
    ply_file = "data/painter_sculpture.ply"

    toad = GraspableToadObject.from_ply(ply_file)

    part_handle = server.add_gui_number("Part", 0, 0, len(toad.grasps)-1, 1)
    object_tf_handle = server.add_transform_controls(
        name='object',
        scale=0.1,
    )
    def curry_mesh(i, mesh):
        handle = server.add_mesh_trimesh(
            name=f'object/part_{i}',
            mesh=mesh
        )
        @handle.on_click
        def _(_):
            print(f"Clicked on part {i}")
            part_handle.value = i
        return handle
    for i, mesh in enumerate(toad.meshes):
        curry_mesh(i, mesh)

    # Create object for collision, and let the poses be updatable using the tf handle.
    world_config = WorldConfig(
        mesh=toad.to_world_config()
    )

    urdf = YumiCurobo(
        server,
        world_config=world_config,
        ik_solver_batch_size=240,
        motion_gen_batch_size=4
    )  # ... can take a while to load...

    mesh_list = toad.meshes
    # @profile
    def calculate_working_grasps():
        with server.atomic():
            global traj

            start = time.time()

            # Update the object's pose, for collisionbody.
            poses_wxyz_xyz = [np.array([*object_tf_handle.wxyz, *object_tf_handle.position])] * len(toad.meshes)
            urdf.update_world(WorldConfig(mesh=toad.to_world_config(poses_wxyz_xyz)))

            grasps = toad.grasps[part_handle.value]  # [N_grasps, 7]
            grasps_gripper = toad.to_gripper_frame(grasps, urdf._tooltip_to_gripper)
            grasp_cand_list = vtf.SE3.from_rotation_and_translation(
                rotation=vtf.SO3(object_tf_handle.wxyz),
                translation=object_tf_handle.position
            ).multiply(grasps_gripper)

            goal_l_wxyz_xyz = torch.Tensor(grasp_cand_list.wxyz_xyz)
            goal_r_wxyz_xyz = torch.Tensor([[0, 1, 0, 0, 0.4, -0.2, 0.5]]).expand(grasp_cand_list.wxyz_xyz.shape[0], 7)            

            ik_result_list = urdf.ik(
                goal_l_wxyz_xyz=goal_l_wxyz_xyz,
                goal_r_wxyz_xyz=goal_r_wxyz_xyz,
            )

            ik_result_list = list(filter(lambda x: x.success.any(), ik_result_list))
            if len(ik_result_list) == 0:
                print("No IK solution found.")
                return

            traj_all = []
            for ik_results in ik_result_list:
                traj = ik_results.js_solution[ik_results.success].position
                assert isinstance(traj, torch.Tensor) and len(traj.shape) == 2
                if traj.shape[0] == 0:
                    continue
                d_world, d_self = urdf._robot_world.get_world_self_collision_distance_from_joint_trajectory(traj.unsqueeze(1))
                traj = traj[(d_world.squeeze() <= 0) & (d_self.squeeze() <= 0)]
                if len(traj) > 0:
                    traj_all.append(traj.squeeze(1))

            if len(traj_all) == 0:
                print("No collision-free IK solution found.")
                drag_slider.disabled = False
                urdf.joint_pos = urdf.home_pos
                return

            traj = torch.cat(traj_all, dim=0)
            print(f"Time taken: {time.time() - start:.2f}s")

            drag_slider.disabled = False
            urdf.joint_pos = traj[0]

    traj = None
    drag_slider = server.add_gui_slider("Time", 0, 1, 0.01, 0, disabled=True)
    import trimesh
    @drag_slider.on_update
    def _(_):
        assert traj is not None
        idx = int(drag_slider.value * (len(traj)-1))
        urdf.joint_pos = traj[idx]

    @part_handle.on_update
    def _(_):
        calculate_working_grasps()

    @object_tf_handle.on_update
    def _(_):
        calculate_working_grasps()

    while True:
        time.sleep(1)