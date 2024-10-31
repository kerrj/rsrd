"""
Known constants for RSRD, e.g., camera intrinsics.
"""

from typing import Optional
import numpy as np
from dataclasses import dataclass
import cv2

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
    height: int


@dataclass
class IPhoneIntr(CameraIntr):
    # TODO(cmk) get iphone model.
    name: str = "iphone"
    fx: float = 1137.0
    fy: float = 1137.0
    cx: float = 1280.0 / 2
    cy: float = 720 / 2
    width: int = 1280
    height: int = 720


@dataclass
class MXIPhoneIntr(CameraIntr):
    name: str = "mx_iphone"
    fx: float = 1085.0
    fy: float = 1085.0
    cx: float = 644.0
    cy: float = 361.0
    width: int = 1280
    height: int = 720


@dataclass
class IPhoneVerticalIntr(CameraIntr):
    name: str = "iphone_vertical"
    fy: float = 1137.0
    fx: float = 1137.0
    cy: float = 1280 / 2
    cx: float = 720 / 2
    height: int = 1280
    width: int = 720


@dataclass
class GoProIntr(CameraIntr):
    name: str = "gopro"
    fx: float = 2.55739580e03
    fy: float = 2.55739580e03
    cx: float = 1.92065792e03
    cy: float = 1.07274675e03
    width: int = 3840
    height: int = 2160


def get_ns_camera_at_origin(
    cam_intr: Optional[CameraIntr] = None,
    K: Optional[np.ndarray] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Cameras:
    """
    Initialize a nerfstudio camera centered at the origin, opengl conventions (z backwards).
    """
    cam_pose = torch.eye(4).float()[None, :3, :]
    assert cam_pose.shape == (1, 3, 4)

    if cam_intr is None and K is None:
        raise ValueError("Must provide either cam_intr or K.")

    if K is not None and cam_intr is None:
        assert width is not None
        assert height is not None
        cam_intr = CameraIntr(
            name="custom",
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
            width=width,
            height=height,
        )
    assert cam_intr is not None

    return Cameras(
        camera_to_worlds=cam_pose,
        fx=cam_intr.fx,
        fy=cam_intr.fy,
        cx=cam_intr.cx,
        cy=cam_intr.cy,
        width=cam_intr.width,
        height=cam_intr.height,
    )


def get_vid_frame(
    cap: cv2.VideoCapture,
    timestamp: Optional[float] = None,
    frame_idx: Optional[int] = None,
) -> np.ndarray:
    """Get frame from video at timestamp (in seconds)."""
    if frame_idx is None:
        assert timestamp is not None
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            raise ValueError("Video has unknown FPS.")
        frame_idx = min(
            int(timestamp * fps),
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
        )
    assert frame_idx is not None and frame_idx >= 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = cap.read()
    if not success:
        raise ValueError(f"Failed to read frame at {timestamp} s, or frame {frame_idx}.")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame
