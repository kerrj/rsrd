"""
Interactive demo for moving a mug to another position.
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
            print(f"Clicked on part {i}")
            part_handle.value = i
        return handle
    for i, mesh in enumerate(toad.meshes):
        curry_mesh(i, mesh)

    # 
    object_goal_handle = server.add_transform_controls(
        name='goal',
        scale=0.1,
    )
