"""
Interactive demo for moving a mug to another position.
"""

import viser
import viser.transforms as vtf
import time
import numpy as np
import trimesh
import torch
from curobo.geom.types import Cuboid, Mesh, WorldConfig
from curobo.types.state import JointState

from toad.toad_object import GraspableToadObject
from toad.yumi_curobo import YumiCurobo, createTableWorld

if __name__ == "__main__":
    server = viser.ViserServer()

    ply_file = "data/mug_new.ply"
    toad = GraspableToadObject.from_ply(ply_file)

    # Create the starting object.
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
            part_handle.value = i
        return handle
    for i, mesh in enumerate(toad.meshes):
        curry_mesh(i, mesh)

    # # Add the goal object.
    # object_goal_handle = server.add_transform_controls(
    #     name='goal',
    #     scale=0.1,
    # )
    # mug_mesh = sum(toad.meshes, trimesh.Trimesh())
    # mug_mesh.visual.vertex_colors = [100, 100, 100, 255]  # type: ignore
    # server.add_mesh_trimesh(
    #     name='goal/mug',
    #     mesh=mug_mesh,
    # )

    # Create object for collision, and let the poses be updatable using the tf handle.
    # Note that only the starting object is used for collision.
    world_config = createTableWorld()
    assert world_config.mesh is not None  # should not be none after post_init.
    # world_config.sphere.extend(toad.to_world_config_spheres())
    world_config.mesh.extend(toad.to_world_config())

    urdf = YumiCurobo(
        server,
        world_config=world_config,
        # ik_solver_batch_size=240,
        motion_gen_batch_size=240,
    )  # ... can take a while to load...

    traj, traj_handle, play_handle = None, None, None
    button_handle = server.add_gui_button("Calculate working grasps")
    @button_handle.on_click
    def _(_):
        global traj, traj_handle, play_handle
        button_handle.disabled = True
        start = time.time()

        # Update the object's pose, for collisionbody.
        poses_wxyz_xyz = [np.array([*object_tf_handle.wxyz, *object_tf_handle.position])] * len(toad.meshes)

        world_config = createTableWorld()
        assert world_config.mesh is not None  # should not be none after post_init.
        # world_config.sphere.extend(toad.to_world_config_spheres(poses_wxyz_xyz=poses_wxyz_xyz))
        world_config.mesh.extend(toad.to_world_config(poses_wxyz_xyz=poses_wxyz_xyz))
        urdf.update_world(world_config)

        grasps = toad.grasps[part_handle.value]  # [N_grasps, 7]
        grasps_gripper = toad.to_gripper_frame(grasps, urdf._tooltip_to_gripper)
        grasp_cand_list = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3(object_tf_handle.wxyz),
            translation=object_tf_handle.position
        ).multiply(grasps_gripper)

        goal_l_wxyz_xyz = torch.Tensor(grasp_cand_list.wxyz_xyz)
        goal_r_wxyz_xyz = torch.Tensor([[0, 1, 0, 0, 0.4, -0.2, 0.5]]).expand(grasp_cand_list.wxyz_xyz.shape[0], 7)            

        start_state = JointState.from_position(urdf.home_pos)
        motiongen_list = urdf.motiongen(
            goal_l_wxyz_xyz=goal_l_wxyz_xyz,
            goal_r_wxyz_xyz=goal_r_wxyz_xyz,
            start_state=start_state
        )
        assert len(motiongen_list) == 1
        traj = motiongen_list[0].interpolated_plan[motiongen_list[0].success].position

        print(f"Time taken: {time.time() - start:.2f} s")
        button_handle.disabled = False

        if traj_handle is not None:
            traj_handle.remove()
            play_handle.remove()
        traj_handle = server.add_gui_slider("trajectory", 0, len(traj)-1, 1, 0)
        play_handle = server.add_gui_slider("play", 0, traj.shape[1]-1, 1, 0)

        @play_handle.on_update
        def _(_):
            assert traj is not None
            urdf.joint_pos = traj[int(traj_handle.value), int(play_handle.value)]


    while True:
        time.sleep(10)