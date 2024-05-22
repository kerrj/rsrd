from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from typing import Union
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

da_image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
da_model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")
da_model.to('cuda')
def get_depth(img: Union[torch.tensor,np.ndarray]):
    assert img.shape[2] == 3
    if isinstance(img,torch.Tensor):
        img = img.cpu().numpy()
    image = Image.fromarray(img)

    # prepare image for the model
    inputs = da_image_processor(images=image, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'].cuda()

    with torch.no_grad():
        outputs = da_model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    return prediction.squeeze()

hand_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
hand_model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
hand_model.to('cuda')
def get_hand_mask(img: Union[torch.tensor,np.ndarray]):
    assert img.shape[2] == 3
    if isinstance(img,torch.Tensor):
        img = img.cpu().numpy()
    image = Image.fromarray(img)

    # prepare image for the model
    inputs = hand_processor(images=image, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'].cuda()

    with torch.no_grad():
        outputs = hand_model(**inputs)

    # Perform post-processing to get panoptic segmentation map
    seg_ids = hand_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]
    hand_mask = (seg_ids == hand_model.config.label2id['person']).float()
    return hand_mask