import trimesh
import torch
from typing import Optional, Callable, TYPE_CHECKING
from nerfstudio.cameras.cameras import Cameras
from torchvision.transforms.functional import resize
from copy import deepcopy

from toad.util.common import Future
from toad.util.common import crop_camera
from toad.util.frame_detectors import Hand2DDetector, Hand3DDetector, MonoDepthEstimator

if TYPE_CHECKING:
    from toad.optimization.rigid_group_optimizer import RigidGroupOptimizer


class Frame:
    camera: Cameras
    rgb: torch.Tensor
    has_metric_depth: bool
    metric_depth: Optional[torch.Tensor]
    _depth: Future[torch.Tensor]
    _dino_feats: Future[torch.Tensor]
    _hand_mask: Future[torch.Tensor]

    @property
    def depth(self):
        return self._depth.retrieve().cuda()

    @property
    def dino_feats(self):
        return self._dino_feats.retrieve().cuda()

    @property
    def hand_mask(self):
        return self._hand_mask.retrieve().cuda()

    def __init__(
        self,
        rgb: torch.Tensor,
        camera: Cameras,
        dino_fn: Callable,
        metric_depth: Optional[torch.Tensor] = None,
    ):
        self.camera = deepcopy(camera.to("cuda"))
        self.rgb = resize(
            rgb.permute(2, 0, 1),
            (camera.height, camera.width),
            antialias=True,
        ).permute(1, 2, 0)
        self.metric_depth = metric_depth
        self.has_metric_depth = metric_depth is not None
        self.dino_fn = dino_fn

        self._depth = Future(self._get_depth)
        self._dino_feats = Future(self._get_dino)
        self._hand_mask = Future(self._get_hand_mask)

    @torch.no_grad()
    def _get_depth(self) -> torch.Tensor:
        if self.has_metric_depth:
            depth = self.metric_depth
        else:
            depth = MonoDepthEstimator.get_depth((self.rgb * 255).to(torch.uint8))
        depth = (
            resize(
                depth.unsqueeze(0),
                (self.camera.height, self.camera.width),
                antialias=True,
            )
            .squeeze()
            .unsqueeze(-1)
        )
        return depth.cpu().pin_memory()

    @torch.no_grad()
    def _get_dino(self) -> torch.Tensor:
        dino_feats = self.dino_fn(self.rgb.permute(2, 0, 1).unsqueeze(0)).squeeze()
        dino_feats = resize(
            dino_feats.permute(2, 0, 1),
            (self.camera.height, self.camera.width),
            antialias=True,
        ).permute(1, 2, 0)
        return dino_feats.cpu().pin_memory()

    @torch.no_grad()
    def _get_hand_mask(self) -> torch.Tensor:
        hand_mask = Hand2DDetector.get_hand_mask((self.rgb * 255).to(torch.uint8))
        hand_mask = (
            torch.nn.functional.max_pool2d(
                hand_mask[None, None], 3, padding=1, stride=1
            ).squeeze()
            == 0.0
        )
        return hand_mask.cpu().pin_memory()

    @torch.no_grad()
    def get_hand_3d(self, rendered_scaled_depth: torch.Tensor) -> list[trimesh.Trimesh]:
        focal = self.camera.fx.item() * (
            max(self.rgb.shape[0], self.rgb.shape[1])
            / PosedObservation.rasterize_resolution
        )
        left, right = Hand3DDetector.detect_hands(self.rgb, focal)
        if left is not None or right is not None:
            breakpoint()
        hands = Hand3DDetector.get_aligned_hands_3d(
            [left, right],
            self.depth,
            self.hand_mask,
            rendered_scaled_depth,
            focal,
        )
        return hands


class PosedObservation:
    """
    Class for computing relevant data products for a frame and storing them
    """

    rasterize_resolution: int = 490
    _frame: Frame
    _raw_rgb: torch.Tensor
    _original_camera: Cameras
    _original_depth: Optional[torch.Tensor] = None
    _roi_frame: Optional[Frame] = None

    def __init__(
        self,
        rgb: torch.Tensor,
        camera: Cameras,
        dino_fn: Callable,
        metric_depth_img: Optional[torch.Tensor] = None,
    ):
        """
        Initialize the frame

        rgb: HxWx3 tensor of the rgb image, normalized to [0,1]
        camera: Cameras object for the camera intrinsics and extrisics to render the frame at
        dino_fn: callable taking in 3HxW RGB image and outputting dino features CxHxW
        metric_depth_img: HxWx1 tensor of metric depth, if desired
        """
        assert (
            rgb.shape[0] == camera.height and rgb.shape[1] == camera.width
        ), f"Input image should be the same size as the camera, got {rgb.shape} and {camera.height}x{camera.width}"
        self._dino_fn = dino_fn
        assert rgb.shape[-1] == 3, rgb.shape
        self._raw_rgb = rgb
        if metric_depth_img is not None:
            self._original_depth = metric_depth_img
        self._original_camera = deepcopy(camera.to("cuda"))
        cam = deepcopy(camera.to("cuda"))
        cam.rescale_output_resolution(
            self.rasterize_resolution / max(camera.width.item(), camera.height.item())
        )
        self._frame = Frame(rgb, cam, dino_fn, metric_depth_img)

    @property
    def frame(self):
        return self._frame

    @property
    def roi_frame(self):
        if self._roi_frame is None:
            raise RuntimeError("ROI not set")
        return self._roi_frame

    def set_roi(self, xmin, xmax, ymin, ymax):
        assert xmin < xmax and ymin < ymax
        assert xmin >= 0 and ymin >= 0
        assert xmax <= 1.0 and ymax <= 1.0, "xmin and ymin should be normalized"
        # convert normalized to pixels in original image
        xmin, xmax, ymin, ymax = (
            int(xmin * (self._original_camera.width - 1)),
            int(xmax * (self._original_camera.width - 1)),
            int(ymin * (self._original_camera.height - 1)),
            int(ymax * (self._original_camera.height - 1)),
        )
        # adjust these value to be multiples of 14, dino patch size
        xlen = ((xmax - xmin) // 14) * 14
        ylen = ((ymax - ymin) // 14) * 14
        xmax = xmin + xlen
        ymax = ymin + ylen
        rgb = self._raw_rgb[ymin:ymax, xmin:xmax].clone()
        camera = crop_camera(self._original_camera, xmin, xmax, ymin, ymax)
        camera.rescale_output_resolution(
            self.rasterize_resolution / max(camera.width.item(), camera.height.item())
        )
        self._roi_frame = Frame(rgb, camera, self._dino_fn, self._original_depth)

    def compute_roi(self, optimizer: "RigidGroupOptimizer"):
        """
        Calculate the ROI for the object given a certain camera pose
        """
        with torch.no_grad():
            outputs = optimizer.dig_model.get_outputs(self.frame.camera)
            object_mask = outputs["accumulation"] > optimizer.config.mask_threshold
            valids = torch.where(object_mask)
            valid_xs = valids[1] / object_mask.shape[1]
            valid_ys = valids[0] / object_mask.shape[0]  # normalize to 0-1
            inflate_amnt = (
                optimizer.config.roi_inflate * (valid_xs.max() - valid_xs.min()).item(),
                optimizer.config.roi_inflate * (valid_ys.max() - valid_ys.min()).item(),
            )  # x, y
            xmin, xmax, ymin, ymax = (
                max(0, valid_xs.min().item() - inflate_amnt[0]),
                min(1, valid_xs.max().item() + inflate_amnt[0]),
                max(0, valid_ys.min().item() - inflate_amnt[1]),
                min(1, valid_ys.max().item() + inflate_amnt[1]),
            )
        return xmin, xmax, ymin, ymax

    def compute_and_set_roi(self, optimizer):
        roi = self.compute_roi(optimizer)
        self.set_roi(*roi)


class VideoSequence:
    def __init__(self):
        self.obs = []

    def add_obs(self, frame: PosedObservation, idx: int = -1):
        self.obs.append(frame)

    def get_obs(self, idx: int) -> PosedObservation:
        return self.obs[idx]

    def __len__(self) -> int:
        return len(self.obs)
