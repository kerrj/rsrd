import torch
from typing import List, Optional, Tuple, Callable, Generic, TypeVar
from nerfstudio.cameras.cameras import Cameras
from torchvision.transforms.functional import resize
from toad.utils import *
from nerfstudio.model_components.losses import depth_ranking_loss
from toad.optimization.utils import *
from copy import deepcopy
from dataclasses import dataclass

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
    
class Frame:
    camera: Cameras
    rgb: torch.Tensor
    metric_depth: bool
    _depth: Future[torch.Tensor]
    _dino_feats: Future[torch.Tensor]
    _hand_mask: Future[torch.Tensor]

    @property
    def depth(self):
        return self._depth.retrieve()
    
    @property
    def dino_feats(self):
        return self._dino_feats.retrieve()

    @property
    def hand_mask(self):
        return self._hand_mask.retrieve()
    
    def __init__(self, rgb: torch.Tensor, camera: Cameras, dino_fn: Callable, metric_depth_img: Optional[torch.Tensor]):
        self.camera = deepcopy(camera.to('cuda'))
        self.rgb = resize(
                rgb.permute(2, 0, 1),
                (camera.height, camera.width),
                antialias=True,
            ).permute(1, 2, 0)
        self.metric_depth = metric_depth_img is not None
        @torch.no_grad()
        def _get_depth():
            if metric_depth_img is not None:
                depth = metric_depth_img
            else:
                depth = get_depth((rgb*255).to(torch.uint8))
            depth = resize(
                            depth.unsqueeze(0),
                            (camera.height, camera.width),
                            antialias=True,
                        ).squeeze().unsqueeze(-1)
            return depth
        self._depth = Future(_get_depth)
        @torch.no_grad()
        def _get_dino():
            dino_feats = dino_fn(
                rgb.permute(2, 0, 1).unsqueeze(0)
            ).squeeze()
            dino_feats = resize(
                dino_feats.permute(2, 0, 1),
                (camera.height, camera.width),
                antialias=True,
            ).permute(1, 2, 0)
            return dino_feats
        self._dino_feats = Future(_get_dino)
        @torch.no_grad()
        def _get_hand_mask():
            hand_mask = get_hand_mask((self.rgb * 255).to(torch.uint8))
            hand_mask = (
                torch.nn.functional.max_pool2d(
                    hand_mask[None, None], 3, padding=1, stride=1
                ).squeeze()
                == 0.0
            )
        self._hand_mask = Future(_get_hand_mask)
        

    
class PosedObservation:
    """
    Class for computing relevant data products for a frame and storing them
    """
    rasterize_resolution: int = 504
    _frame: Frame
    _raw_rgb: torch.Tensor
    _original_camera: Cameras
    _original_depth: Optional[torch.Tensor] = None
    _roi_frame: Optional[Frame] = None

    """
    TODO for roi; 
    1. make rgb, depth, camera properties which are computed LAZILY when roi is changed
    """
    
    def __init__(self, rgb: torch.Tensor, camera: Cameras, dino_fn: Callable, metric_depth_img: Optional[torch.Tensor] = None):
        """
        Initialize the frame

        rgb: HxWx3 tensor of the rgb image, normalized to [0,1]
        camera: Cameras object for the camera intrinsics and extrisics to render the frame at
        dino_fn: callable taking in 3HxW RGB image and outputting dino features CxHxW
        metric_depth_img: HxWx1 tensor of metric depth, if desired
        """
        assert rgb.shape[0] == camera.height and rgb.shape[1] == camera.width, f"Input image should be the same size as the camera, got {rgb.shape} and {camera.height}x{camera.width}"
        self._dino_fn = dino_fn
        assert rgb.shape[-1] == 3, rgb.shape
        self._raw_rgb = rgb
        if metric_depth_img is not None:
            self._original_depth = metric_depth_img
        self._original_camera = deepcopy(camera.to('cuda'))
        cam = deepcopy(camera.to('cuda'))
        cam.rescale_output_resolution(self.rasterize_resolution/max(camera.width.item(),camera.height.item()))
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
        xmin,xmax,ymin,ymax = int(xmin*(self._original_camera.width-1)), int(xmax*(self._original_camera.width-1)),\
              int(ymin*(self._original_camera.height-1)), int(ymax*(self._original_camera.height-1))
        # adjust these value to be multiples of 14, dino patch size
        xlen = ((xmax - xmin)//14) * 14
        ylen = ((ymax - ymin)//14) * 14
        xmax = xmin + xlen
        ymax = ymin + ylen
        rgb = self._raw_rgb[ymin:ymax, xmin:xmax].clone()
        camera = crop_camera(self._original_camera, xmin, xmax, ymin, ymax)
        camera.rescale_output_resolution(self.rasterize_resolution/max(camera.width.item(),camera.height.item()))
        self._roi_frame = Frame(rgb, camera, self._dino_fn, self._original_depth)


class VideoSequence:
    def __init__(self):
        self.frames = []

    def add_frame(self, frame:PosedObservation, idx:int =-1):
        self.frames.append(frame)

    def get_last_frame(self) -> PosedObservation:
        return self.frames[-1]

    def get_frame(self, idx:int) -> PosedObservation:
        return self.frames[idx]

    def __len__(self) -> int:
        return len(self.frames)