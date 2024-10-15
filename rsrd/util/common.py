import torch
from rsrd.transforms import SE3, SO3
from nerfstudio.cameras.cameras import Cameras
from typing import TypeVar, Generic
from typing import TypedDict

class MANOKeypoints(TypedDict):
    thumb: int
    index: int
    middle: int
    ring: int
    pinky: int

MANO_KEYPOINTS: MANOKeypoints = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}


T = TypeVar('T')
class Future(Generic[T]):
    """
    A simple wrapper for deferred execution of a callable until retrieved
    """
    def __init__(self,callable):
        self.callable = callable
        self.executed = False

    def retrieve(self):
        if not self.executed:
            self.result = self.callable()
            self.executed = True
        return self.result


def identity_7vec(device="cuda"):
    """
    Returns a 7-tensor of identity pose, as wxyz_xyz.
    """
    return torch.tensor([[1, 0, 0, 0, 0, 0, 0]], dtype=torch.float32, device=device)


def extrapolate_poses(p1_7v, p2_7v, lam):
    r1 = SO3(p1_7v[..., :4])
    t1 = SE3.from_rotation_and_translation(r1, p1_7v[..., 4:])
    r2 = SO3(p2_7v[..., :4])
    t2 = SE3.from_rotation_and_translation(r2, p2_7v[..., 4:])
    t_2_1 = t1.inverse() @ t2
    delta_pos = t_2_1.translation() * lam
    delta_rot = SO3.exp((t_2_1.rotation().log() * lam))
    new_t = t2 @ SE3.from_rotation_and_translation(delta_rot, delta_pos)
    return new_t.wxyz_xyz


def mnn_matcher(feat_a, feat_b):
    """
    Returns mutual nearest neighbors between two sets of features

    feat_a: NxD
    feat_b: MxD
    return: K, K (indices in feat_a and feat_b)
    """
    device = feat_a.device
    sim = feat_a.mm(feat_b.t())
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = ids1 == nn21[nn12]
    return ids1[mask], nn12[mask]


def crop_camera(camera: Cameras, xmin, xmax, ymin, ymax):
    height = torch.tensor(ymax - ymin, device="cuda").view(1, 1).int()
    width = torch.tensor(xmax - xmin, device="cuda").view(1, 1).int()
    cx = (camera.cx - xmin).view(1, 1)
    cy = (camera.cy - ymin).view(1, 1)
    fx = camera.fx.clone()
    fy = camera.fy.clone()
    return Cameras(camera.camera_to_worlds.clone(), fx, fy, cx, cy, width, height)
