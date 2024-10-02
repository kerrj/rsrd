"""
Articulated gaussians, visible in the 3D viser viewer.
"""

import numpy as np
import torch
import viser
import viser.transforms as vtf
from toad.optimization.rigid_group_optimizer import RigidGroupOptimizer


class ViserRSRD:
    """Visualize RSRD's multi-part objects in Viser.
    We learn the residuals for object and part poses, so the structure is:

    ```
    /object
    /object/delta
    /object/delta/group_0
    /object/delta/group_0/delta
    ...
    ```
    """

    server: viser.ViserServer
    optimizer: RigidGroupOptimizer
    base_frame_name: str
    base_frame: viser.FrameHandle
    base_delta_frame: viser.FrameHandle
    part_frames: list[viser.FrameHandle]
    part_delta_frames: list[viser.FrameHandle]

    def __init__(
        self,
        server: viser.ViserServer,
        optimizer: RigidGroupOptimizer,
        base_frame_name: str = "/object",
    ):
        self.server = server
        self.optimizer = optimizer
        self.optimizer.reset_transforms()

        self.base_frame_name = base_frame_name
        self.base_frame = self.server.scene.add_frame(self.base_frame_name, show_axes=False)
        self.base_delta_frame = self.server.scene.add_frame(
            self.base_frame_name + "/delta", show_axes=False
        )

        self.part_frames = []
        self.part_delta_frames = []
        self.num_groups = len(self.optimizer.group_masks)
        for group_idx in range(self.num_groups):
            self._create_frames_and_gaussians(
                group_idx, self.base_frame_name + "/delta/group_" + str(group_idx)
            )

    def _create_frames_and_gaussians(self, group_idx: int, frame_name: str):
        dig_model = self.optimizer.dig_model
        group_mask = self.optimizer.group_masks[group_idx].detach().cpu()

        Rs = vtf.SO3(dig_model.quats[group_mask].cpu().numpy()).as_matrix()
        covariances = np.einsum(
            "nij,njk,nlk->nil",
            Rs,
            np.eye(3)[None, :, :]
            * dig_model.scales[group_mask].detach().exp().cpu().numpy()[:, None, :]
            ** 2,
            Rs,
        )

        centered_means = (
            (
                dig_model.means.detach()[group_mask]
                - dig_model.means.detach()[group_mask].mean(dim=0)
            )
            .detach().cpu().numpy()
        )
        rgbs = torch.clamp(dig_model.colors, 0.0, 1.0).detach()[group_mask].cpu().numpy()
        opacities = dig_model.opacities.sigmoid()[group_mask].detach().cpu().numpy()

        # Add the frame to the scene. Convention is xyz_wxyz.
        p2o_7vec = self.optimizer.init_p2o[group_idx].cpu().numpy()
        self.part_frames.append(
            self.server.scene.add_frame(
                frame_name, position=p2o_7vec[4:], wxyz=p2o_7vec[:4], show_axes=False
            )
        )
        self.part_delta_frames.append(
            self.server.scene.add_frame(
                frame_name + "/delta", # show_axes=False
            )
        )

        self.server.scene._add_gaussian_splats(
            frame_name + "/delta/gaussians",
            centers=centered_means,
            rgbs=rgbs,
            opacities=opacities,
            covariances=covariances,
        )

    def update_cfg(self, obj_delta: torch.Tensor, part_deltas: torch.Tensor):
        """Update the configuration of the objects in the scene. Assumes *_delta tensors are wxyz_xyz."""
        obj_delta_np = obj_delta.detach().cpu().numpy().flatten()
        self.base_delta_frame.position = vtf.SE3(obj_delta_np).translation()
        self.base_delta_frame.wxyz = vtf.SE3(obj_delta_np).rotation().wxyz
        for group_idx in range(self.num_groups):
            group_delta = part_deltas[group_idx].detach().cpu().numpy().flatten()
            self.part_delta_frames[group_idx].position = vtf.SE3(group_delta).translation()
            self.part_delta_frames[group_idx].wxyz = vtf.SE3(group_delta).rotation().wxyz
