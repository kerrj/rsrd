from toad.hamer_helper import HamerHelper
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision.transforms.functional import resize

class HandRegistration:
    def __init__(self):
        self.hamer_helper = HamerHelper()

    def detect_hands(self, image, focal):
        """
        Detects hands in the image and aligns them with the ground truth depth image.
        """
        left,right = self.hamer_helper.look_for_hands(image,focal_length = focal)
        return left,right
    
    def align_hands(self, hand_dict, gauss_depth, global_depth, obj_mask, focal):
        #THESE ARE ALL CUDA TENSORS
        n_hands = hand_dict['verts'].shape[0]
        for i in range(n_hands):
            _, hand_depth, hand_mask = self.hamer_helper.render_detection(hand_dict, i, global_depth.shape[1],global_depth.shape[0], focal)
            hand_depth = torch.from_numpy(hand_depth).cuda().float()
            hand_mask = torch.from_numpy(hand_mask).cuda()
            masked_hand_depth = hand_depth[hand_mask]#this line needs to be before the resize
            hand_mask = resize(
                    hand_mask.unsqueeze(0),
                    (obj_mask.shape[0], obj_mask.shape[1]),
                    antialias = True,
                ).permute(1, 2, 0).bool()
            obj_mask &= ~hand_mask
            obj_gauss_depth = gauss_depth[obj_mask]
            obj_global_depth = global_depth[obj_mask]
            
            # calculate affine that aligns the two
            scale = obj_gauss_depth.std() / obj_global_depth.std()
            scaled_global = (global_depth - obj_global_depth.mean()) * scale + obj_gauss_depth.mean()

            scaled_hand = scaled_global[hand_mask]
            hand_offset = (masked_hand_depth.mean() - scaled_hand.mean()).item()
            hand_dict['verts'][i][:, 2] -= hand_offset