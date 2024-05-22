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
    server.add_grid(
        name="grid",
        width=1,
        height=1,
        position=(0.5, 0, 0),
        section_size=0.05,
    )

    # Update the joint positions based on the handle positions.
    # Run IK on the fly!
    def update_joints():
        joints_from_ik = urdf.ik(
            torch.Tensor([*drag_l_handle.wxyz, *drag_l_handle.position]).view(1, 7),
            torch.Tensor([*drag_r_handle.wxyz, *drag_r_handle.position]).view(1, 7),
        ).js_solution.position
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

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()