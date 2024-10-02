"""
Known constants for RSRD, e.g., camera intrinsics.
"""

import numpy as np
import viser.transforms as vtf
from dataclasses import dataclass

import torch
from nerfstudio.cameras.cameras import Cameras

@dataclass
class CameraIntr:
    name: str
    fx: float
    fy: float
    cx: float
    cy: float
    width: int

@dataclass
class IPhoneIntr(CameraIntr):
    name: str = "iphone"
    fx: float = 1137.0
    fy: float = 1137.0
    cx: float = 1280.0 / 2
    cy: float = 720 / 2
    width: int = 1280

class MXIPhoneIntr(CameraIntr):
    name: str = "mx_iphone"
    fx = 1085.0
    fy = 1085.0
    cx = 644.0
    cy = 361.0
    width = 1280

class IPhoneVerticalIntr(CameraIntr):
    name: str = "iphone_vertical"
    fy = 1137.0
    fx = 1137.0
    cy = 1280 / 2
    cx = 720 / 2
    height = 1280
    width = 720

class GoProIntr(CameraIntr):
    name: str = "gopro"
    fx = 2.55739580e03
    fy = 2.55739580e03
    cx = 1.92065792e03
    cy = 1.07274675e03
    width = 3840
    height = 2160

def get_ns_camera_at_origin(cam_intr: CameraIntr) -> Cameras:
    """Initialize a nerfstudio camera at the origin, with known intrinsics."""
    H = np.eye(4)
    H[:3, :3] = vtf.SO3.from_x_radians(np.pi / 4).as_matrix()
    cam_pose = torch.from_numpy(H).float()[None, :3, :]
    assert cam_pose.shape == (1, 3, 4)

    return Cameras(
        camera_to_worlds=cam_pose,
        fx=cam_intr.fx,
        fy=cam_intr.fy,
        cx=cam_intr.cx,
        cy=cam_intr.cy,
        width=cam_intr.width
    )
