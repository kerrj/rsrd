"""
Quick interactive demo for yumi IK, with curobo.
"""

import torch
import viser
import time

from toad.yumi_curobo import YumiCurobo

def main():
    server = viser.ViserServer()
    urdf = YumiCurobo(
        server,
        world_config = {
            "cuboid": {
                "table": {
                    "dims": [1.0, 1.0, 0.2],  # x, y, z
                    "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0.0],  # x, y, z, qw, qx, qy, qz
                },
            },
        }
    )  # ... can take a while to load...

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
        pos_l, quat_l = drag_l_handle.position, drag_l_handle.wxyz
        pos_r, quat_r = drag_r_handle.position, drag_r_handle.wxyz
        ik_result = urdf.ik(
            torch.tensor(pos_l), torch.tensor(quat_l),
            torch.tensor(pos_r), torch.tensor(quat_r),
        )
        urdf.joint_pos = ik_result.js_solution.position

    @drag_r_handle.on_update
    def _(_):
        update_joints()
    @drag_l_handle.on_update
    def _(_):
        update_joints()

    # First update to set the initial joint positions.
    update_joints()

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()