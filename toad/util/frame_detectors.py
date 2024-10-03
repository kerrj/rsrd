import torch
from torchvision.transforms.functional import resize
from typing import Union, cast, Iterable
import numpy as np
from PIL import Image
from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
    AutoModelForDepthEstimation,
)
import trimesh

from hamer_helper import HamerHelper, HandOutputsWrtCamera
from toad.util.common import Future


class Hand2DDetector:
    hand_processor = Future(lambda: AutoImageProcessor.from_pretrained(
        "facebook/mask2former-swin-base-coco-panoptic")
    )
    hand_model = Future(lambda: Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-base-coco-panoptic"
    ).to("cuda"))

    @classmethod
    def get_hand_mask(cls, img: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        assert img.shape[2] == 3
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        image = Image.fromarray(img)

        # prepare image for the model
        inputs = cls.hand_processor.retrieve()(images=image, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].cuda()

        with torch.no_grad():
            outputs = cls.hand_model.retrieve()(**inputs)

        # Perform post-processing to get panoptic segmentation map
        seg_ids = cls.hand_processor.retrieve().post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
        hand_mask = (seg_ids == cls.hand_model.retrieve().config.label2id["person"]).float()
        return hand_mask


class Hand3DDetector:
    hamer_helper = Future(lambda: HamerHelper())

    @classmethod
    def detect_hands(
        cls, image: torch.Tensor, focal: float
    ) -> tuple[HandOutputsWrtCamera | None, HandOutputsWrtCamera | None]:
        """
        Detects hands in the image and aligns them with the ground truth depth image.
        """
        left, right = cast(
            HamerHelper,
            cls.hamer_helper.retrieve()
        ).look_for_hands(image.cpu().numpy(), focal_length=focal)
        return left, right

    @classmethod
    def get_aligned_hands_3d(
        cls, 
        hand_outputs: Iterable[HandOutputsWrtCamera | None], 
        monodepth: torch.Tensor, 
        object_mask: torch.Tensor, 
        rendered_scaled_depth: torch.Tensor, 
        focal_length: float
    ) -> list[trimesh.Trimesh]:
        obj_rendered_depth = monodepth[object_mask]

        # Get shift/scale for matching monodepth to object depth.
        monodepth_scale = rendered_scaled_depth[object_mask].std() / monodepth[object_mask].std()
        monodepth_aligned = (
            monodepth - monodepth[object_mask].mean()
        ) * monodepth_scale + obj_rendered_depth.mean()

        hands = []
        for hand_output in hand_outputs:
            if hand_output is not None:
                num_hands = hand_output["verts"].shape[0]
                for hand_idx in range(num_hands):
                    hands.append(
                        cls._get_aligned_hand_3d(
                            hand_output,
                            hand_idx,
                            monodepth_aligned,
                            focal_length,
                        )
                    )
        return hands
    
    @classmethod
    def _get_aligned_hand_3d(
        cls,
        hand_output: HandOutputsWrtCamera,
        hand_idx: int,
        monodepth_aligned: torch.Tensor,
        focal_length: float,
    ) -> trimesh.Trimesh:
        # Get the hand depth.
        rgb, hand_depth, hand_mask = cast(
            HamerHelper, cls.hamer_helper.retrieve()
        ).render_detection(
            hand_output,
            hand_idx,
            monodepth_aligned.shape[1],
            monodepth_aligned.shape[0],
            focal_length,
        )
        hand_depth = torch.from_numpy(hand_depth).cuda().float()
        hand_mask = torch.from_numpy(hand_mask).cuda()

        # Get shift (no scale!) to match hand depth to monodepth.
        hand_shift = hand_depth[hand_mask].mean() - monodepth_aligned[hand_mask].mean()
        hand_mesh = trimesh.Trimesh(
            vertices=hand_output["verts"][hand_idx] - hand_shift,
            faces=hand_output["faces"],
        )
        return hand_mesh


class MonoDepthEstimator:
    image_processor = Future(lambda: AutoImageProcessor.from_pretrained(
        "depth-anything/Depth-Anything-V2-Base-hf"
    ))
    model = Future(lambda: AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Base-hf"
    ).to("cuda"))

    @classmethod
    def get_depth(cls, img: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        assert img.shape[2] == 3
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        image = Image.fromarray(img)

        # prepare image for the model
        inputs = cls.image_processor.retrieve()(images=image, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].cuda()

        with torch.no_grad():
            outputs = cls.model.retrieve()(**inputs)
            predicted_depth = outputs.predicted_depth

        # interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        return prediction.squeeze()
