"""
Articulated gaussians, visible in the 3D viser viewer.
"""

import numpy as np
import torch
import viser
import viser.transforms as vtf
import trimesh
from rsrd.motion.motion_optimizer import RigidGroupOptimizer
from rsrd.util.common import MANO_KEYPOINTS


class ViserRSRD:
    """Visualize RSRD's multi-part objects in Viser.
    We learn the residuals for object and part poses, so the structure is:

    ```
    /object
    /object/group_0
    /object/group_0/delta
    ...
    ```
    """

    _server: viser.ViserServer
    optimizer: RigidGroupOptimizer
    base_frame_name: str
    part_frames: list[viser.FrameHandle]
    part_delta_frames: list[viser.FrameHandle]
    _scale: float
    hand_handles: list[viser.GlbHandle]
    show_hands: bool
    show_finger_keypoints: bool

    def __init__(
        self,
        server: viser.ViserServer,
        optimizer: RigidGroupOptimizer,
        root_node_name: str = "/object",
        scale: float = 1.0,
        show_hands: bool = True,
        show_finger_keypoints: bool = True,
    ):
        self._server = server
        self.optimizer = optimizer
        self.optimizer.reset_transforms()
        self.show_hands = show_hands
        self.show_finger_keypoints = show_finger_keypoints
        self.hand_handles = []

        self.base_frame_name = root_node_name
        self._scale = scale

        self.part_frames = []
        self.part_delta_frames = []
        self.num_groups = len(self.optimizer.group_masks)
        for group_idx in range(self.num_groups):
            self._create_frames_and_gaussians(
                group_idx,
                self.base_frame_name + "/group_" + str(group_idx),
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
            .detach()
            .cpu()
            .numpy()
        )
        rgbs = (
            torch.clamp(dig_model.colors, 0.0, 1.0).detach()[group_mask].cpu().numpy()
        )
        opacities = dig_model.opacities.sigmoid()[group_mask].detach().cpu().numpy()

        # Add the frame to the scene. Convention is xyz_wxyz.
        p2o_7vec = self.optimizer.init_p2o[group_idx].cpu().numpy()
        self.part_frames.append(
            self._server.scene.add_frame(
                frame_name,
                position=p2o_7vec[4:] * self._scale,
                wxyz=p2o_7vec[:4],
                show_axes=False,
            )
        )
        self.part_delta_frames.append(
            self._server.scene.add_frame(frame_name + "/delta", show_axes=False)
        )

        self._server.scene._add_gaussian_splats(
            frame_name + "/delta/gaussians",
            centers=centered_means * self._scale,
            rgbs=rgbs,
            opacities=opacities,
            covariances=covariances * (self._scale**2),
        )

    def update_cfg(self, part_deltas: torch.Tensor):
        """Update the configuration of the objects in the scene. Assumes *_delta tensors are wxyz_xyz."""
        for group_idx in range(self.num_groups):
            group_delta = part_deltas[group_idx].detach().cpu().numpy().flatten()
            self.part_delta_frames[group_idx].position = (
                vtf.SE3(group_delta).translation() * self._scale
            )
            self.part_delta_frames[group_idx].wxyz = (
                vtf.SE3(group_delta).rotation().wxyz
            )

    def update_hands(self, tstep: int): 
        for handle in self.hand_handles:
            handle.remove()
        self.hand_handles = []

        if (
            not self.show_hands
            or self.optimizer.hands_info is None
            or self.optimizer.hands_info.get(tstep, None) is None
        ):
            return

        hand_idx = 0
        keypoint_mesh = trimesh.creation.uv_sphere(radius=0.01 * self._scale)
        keypoint_mesh.visual.vertex_colors = [255, 0, 0, 255]

        left_hand, right_hand = self.optimizer.hands_info[tstep]
        for hand in [left_hand, right_hand]:
            if hand is None:
                continue
            for idx in range(hand["verts"].shape[0]):
                self.hand_handles.append(
                    self._server.scene.add_mesh_trimesh(
                        self.base_frame_name + f"/hand_{hand_idx}",
                        trimesh.Trimesh(
                            vertices=hand["verts"][idx] * self._scale,
                            faces=hand["faces"].astype(np.int32),
                        )
                    )
                )
                for keypoint_idx in MANO_KEYPOINTS.values():
                    keypoint = hand["keypoints_3d"][idx][keypoint_idx]
                    if self.show_finger_keypoints:
                        self.hand_handles.append(
                            self._server.scene.add_mesh_trimesh(
                                self.base_frame_name + f"/hand_{hand_idx}/keypoint_{keypoint_idx}",
                                keypoint_mesh,
                                position=keypoint * self._scale,
                            )
                        )
                hand_idx += 1