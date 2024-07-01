import torch
from typing import List, Optional, Literal, Callable
from nerfstudio.cameras.cameras import Cameras
from torchvision.transforms.functional import resize
from toad.utils import *
from nerfstudio.model_components.losses import depth_ranking_loss
from toad.optimization.utils import *

class Frame:
    """
    Class for computing relevant data products for a frame and storing them
    """
    rgb: torch.Tensor
    depth: torch.Tensor
    camera: Cameras
    metric_depth: bool
    dino_feats: torch.Tensor
    hand_mask: torch.Tensor

    def __init__(self, rgb: torch.Tensor, camera: Cameras, dino_fn: Callable, metric_depth_img: Optional[torch.Tensor] = None):
        """
        Initialize the frame

        rgb: HxWx3 tensor of the rgb image, normalized to [0,1]
        camera: Cameras object for the camera intrinsics and extrisics to render the frame at
        dino_fn: callable taking in 3HxW RGB image and outputting dino features CxHxW
        metric_depth_img: HxWx1 tensor of metric depth, if desired
        """
        assert rgb.shape[-1] == 3, rgb.shape
        self.rgb = resize(
                rgb.permute(2, 0, 1),
                (camera.height, camera.width),
                antialias=True,
            ).permute(1, 2, 0)
        self.camera = camera.to('cuda')
        self.metric_depth = metric_depth_img is not None
        
        if metric_depth_img is not None:
            depth = metric_depth_img
        else:
            depth = get_depth((rgb*255).to(torch.uint8))
        self.depth = resize(
                            depth.unsqueeze(0),
                            (camera.height, camera.width),
                            antialias=True,
                        ).squeeze().unsqueeze(-1)

        self.dino_feats = dino_fn(
            rgb.permute(2, 0, 1).unsqueeze(0)
        ).squeeze()
        self.dino_feats = resize(
            self.dino_feats.permute(2, 0, 1),
            (camera.height, camera.width),
            antialias=True,
        ).permute(1, 2, 0)
            
        self.hand_mask = get_hand_mask((self.rgb * 255).to(torch.uint8))
        self.hand_mask = (
            torch.nn.functional.max_pool2d(
                self.hand_mask[None, None], 3, padding=1, stride=1
            ).squeeze()
            == 0.0
        )