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

    ply_file = "data/mug.ply"
    # ply_file = "data/painter_sculpture.ply"

    toad = GraspableToadObject.from_ply(ply_file)

    part_handle = server.add_gui_number("Part", 0, 0, len(toad.grasps)-1, 1)

    object_tf_handle = server.add_transform_controls(
        name='object',
        scale=0.1,
    )
    for i, mesh in enumerate(toad.to_mesh()):
        server.add_mesh_trimesh(
            name=f'object/part_{i}',
            mesh=mesh
        )

    # Create object for collision, and let the poses be updatable using the tf handle.
    object_mesh_list = [
        Mesh(
            name=f'object_{i}',
            vertices=mesh.vertices,
            faces=mesh.faces,
            pose=[0.0, 0.0, 0.0, 1, 0, 0, 0.0],  # x, y, z, qw, qx, qy, qz
        )
        for i, mesh in enumerate(toad.to_mesh())
    ]
    @object_tf_handle.on_update
    def _(_):
        for mesh in object_mesh_list:
            mesh.pose = [*object_tf_handle.position, *object_tf_handle.wxyz]

    # from curobo.geom.sphere_fit import SphereFitType, fit_spheres_to_mesh
    # object_mesh_list = sum([mesh.get_bounding_spheres(n_spheres=100, voxelize_method=SphereFitType.VOXEL_VOLUME) for mesh in object_mesh_list], [])
    # world_config = WorldConfig.create_obb_world(WorldConfig(sphere=object_mesh_list))
    world_config = WorldConfig(mesh=object_mesh_list)

    urdf = YumiCurobo(
        server,
        world_config=world_config
    )  # ... can take a while to load...

    def calculate_working_grasps():
        with server.atomic():
            global traj
            # ... only like a hundred or so is viable as a batch...
            grasps = toad.grasps[part_handle.value][:10] # [N_grasps, 7]
            grasps_in_world = (
                vtf.SE3.from_rotation_and_translation(
                    rotation=vtf.SO3(object_tf_handle.wxyz),
                    translation=object_tf_handle.position
                ).multiply(vtf.SE3.from_rotation_and_translation(
                    rotation=vtf.SO3(grasps[:, 3:]),
                    translation=grasps[:, :3]
                ))
            )
            grasp_augmentations = (
                vtf.SE3.from_rotation(
                    rotation=vtf.SO3.from_x_radians(torch.linspace(-np.pi, np.pi, 6))
                ).multiply(urdf._tooltip_to_gripper.inverse())
            )

            len_grasps = grasps_in_world.wxyz_xyz.shape[0]
            len_augs = grasp_augmentations.wxyz_xyz.shape[0]

            grasps_in_world = vtf.SE3(np.repeat(grasps_in_world.wxyz_xyz, len_augs, axis=0))
            grasp_augmentations = vtf.SE3(np.tile(grasp_augmentations.wxyz_xyz, (len_grasps, 1)))

            grasp_cand_list = grasps_in_world.multiply(grasp_augmentations)

            ik_results = urdf.ik_batch(
                goal_l_wxyz_xyz=torch.Tensor(grasp_cand_list.wxyz_xyz),
                goal_r_wxyz_xyz=torch.Tensor([[0, 1, 0, 0, 0.4, -0.2, 0.5]]).expand(grasp_cand_list.wxyz_xyz.shape[0], 7)
            )
            if not ik_results.success.any():
                print("No IK solution found.")
                return

            traj = ik_results.js_solution[ik_results.success].position

            d_world, d_self = urdf._robot_world.get_world_self_collision_distance_from_joint_trajectory(traj.unsqueeze(1))
            traj = traj[(d_world.squeeze() <= 0) & (d_self.squeeze() <= 0)]
            
            if len(traj) == 0:
                print("No collision-free IK solution found.")
                return

            drag_slider.disabled = False
            urdf.joint_pos = traj[0]

    traj = None
    drag_slider = server.add_gui_slider("Time", 0, 1, 0.01, 0, disabled=True)
    @drag_slider.on_update
    def _(_):
        assert traj is not None
        idx = int(drag_slider.value * (len(traj)-1))
        urdf.joint_pos = traj[idx]


    # @part_handle.on_update
    # def _(_):
    #     calculate_working_grasps()

    # @object_tf_handle.on_update
    # def _(_):
    #     calculate_working_grasps()

    create_button = server.add_gui_button("Calculate Working Grasps")
    @create_button.on_click
    def _(_):
        create_button.disabled = True
        start = time.time()
        calculate_working_grasps()
        print(f"Time taken: {time.time() - start:.2f}s")
        create_button.disabled = False



    # goal is to find the list of ik that works.



    # # Visualize the grasps, make them clickable.
    # def curry_add_handle(grasp: torch.Tensor):
    #     handle = server.add_mesh_trimesh(
    #         name=f'object/grasp/{i}_{j}',
    #         mesh=toad.grasp_axis_mesh(),
    #         position=grasp[:3],
    #         wxyz=grasp[3:],
    #     ) 
    #     @handle.on_click
    #     def _(_):
    #         global traj
    #         grasp_pose_in_world = (
    #             vtf.SE3.from_rotation_and_translation(
    #                 rotation=vtf.SO3(object_tf_handle.wxyz),
    #                 translation=object_tf_handle.position
    #             ).multiply(vtf.SE3.from_rotation_and_translation(
    #                 rotation=vtf.SO3(wxyz=grasp[3:].numpy()),
    #                 translation=grasp[:3].numpy()
    #             ))
    #         )
    #         grasp_cand_list = [
    #             grasp_pose_in_world.multiply(
    #                 vtf.SE3.from_rotation(
    #                     rotation=vtf.SO3.from_x_radians(delta_theta),
    #                 ).multiply(urdf._tooltip_to_gripper.inverse())
    #             )
    #             for delta_theta in torch.linspace(-np.pi, np.pi, 18)
    #         ]

    #         # find the list of IK that works.
    #         working_grasp_list = []
    #         working_ik_list = []
    #         for _grasp in grasp_cand_list:
    #             _left_ik = urdf.ik(
    #                 goal_l_pos=torch.from_numpy(_grasp.translation()),
    #                 goal_l_wxyz=torch.from_numpy(_grasp.rotation().wxyz),
    #                 goal_r_pos=torch.Tensor([0.4, -0.2, 0.5]),
    #                 goal_r_wxyz=torch.Tensor([0, 1, 0, 0])
    #             )
    #             if _left_ik.success:
    #                 working_grasp_list.append(_grasp)
    #                 working_ik_list.append(_left_ik)
    #         if len(working_ik_list) == 0:
    #             print("No IK solution found.")
    #             return
    #         print("...")

    #         # find the motion planning that works.
    #         working_mg = None
    #         for _grasp in working_grasp_list:
    #             mg = urdf.motiongen(
    #                 start_l_pos=torch.Tensor([0.4, 0.2, 0.5]),
    #                 start_l_wxyz=torch.Tensor([0, 1, 0, 0]),
    #                 goal_l_pos=torch.from_numpy(_grasp.translation()),
    #                 goal_l_wxyz=torch.from_numpy(_grasp.rotation().wxyz),
    #                 start_r_pos=torch.Tensor([0.4, -0.2, 0.5]),
    #                 start_r_wxyz=torch.Tensor([0, 1, 0, 0]),
    #                 goal_r_pos=torch.Tensor([0.4, -0.2, 0.5]),
    #                 goal_r_wxyz=torch.Tensor([0, 1, 0, 0])
    #             )
    #             if mg.success:
    #                 working_mg = mg
    #                 break
            
    #         if working_mg is None:
    #             print("No motion planning found.")
    #             return

    #         traj = working_mg.get_interpolated_plan().position
    #         drag_slider.disabled = False

    #     return handle

    # grasp_handles = []
    # for i, part_grasp_list in enumerate(toad.grasps):
    #     for j, grasp in enumerate(part_grasp_list):
    #         handle = curry_add_handle(grasp)
    #         grasp_handles.append(handle)

    # traj = None
    # drag_slider = server.add_gui_slider("Time", 0, 1, 0.01, 0, disabled=True)
    # @drag_slider.on_update
    # def _(_):
    #     assert traj is not None
    #     idx = int(drag_slider.value * (len(traj)-1))
    #     urdf.joint_pos = traj[idx]

    while True:
        time.sleep(1)