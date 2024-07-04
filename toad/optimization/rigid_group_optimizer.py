import torch
import matplotlib.pyplot as plt
from nerfstudio.utils.eval_utils import eval_setup
from pathlib import Path
import numpy as np
from nerfstudio.viewer.viewer import Viewer
from nerfstudio.configs.base_config import ViewerConfig
import cv2
from torchvision.transforms import ToTensor
from PIL import Image
from typing import List, Optional, Literal, Callable
from nerfstudio.utils import writer
import time
from threading import Lock
import kornia
from lerf.dig import DiGModel
from lerf.data.utils.dino_dataloader import DinoDataloader
from nerfstudio.cameras.cameras import Cameras
from copy import deepcopy
from torchvision.transforms.functional import resize
from contextlib import nullcontext
from nerfstudio.engine.schedulers import (
    ExponentialDecayScheduler,
    ExponentialDecaySchedulerConfig,
)
import warp as wp
from toad.optimization.atap_loss import ATAPLoss
from toad.utils import *
import toad.transforms as vtf
import trimesh
from typing import Tuple
from nerfstudio.model_components.losses import depth_ranking_loss
from toad.optimization.utils import *
from toad.optimization.observation import PosedObservation, VideoSequence

class RigidGroupOptimizer:
    use_depth: bool = True
    rank_loss_mult: float = 0.2
    rank_loss_erode: int = 5
    depth_ignore_threshold: float = 0.1  # in meters
    use_atap: bool = True
    pose_lr: float = 0.005
    pose_lr_final: float = 0.001
    mask_hands: bool = True
    do_obj_optim: bool = True
    blur_kernel_size: int = 5

    init_p2o: torch.Tensor
    """From: part, To: object. in current world frame. Part frame is centered at part centroid, and object frame is centered at object centroid."""

    def __init__(
        self,
        dig_model: DiGModel,
        dino_loader: DinoDataloader,
        group_masks: List[torch.Tensor],
        group_labels: torch.Tensor,
        dataset_scale: float,
        render_lock = nullcontext(),
    ):
        """
        This one takes in a list of gaussian ID masks to optimize local poses for
        Each rigid group can be optimized independently, with no skeletal constraints
        """
        self.dataset_scale = dataset_scale
        self.tape = None
        self.is_initialized = False
        self.hand_lefts = []#list of bools for each hand frame
        self.dig_model = dig_model
        # detach all the params to avoid retain_graph issue
        self.dig_model.gauss_params["means"] = self.dig_model.gauss_params[
            "means"
        ].detach().clone()
        self.dig_model.gauss_params["quats"] = self.dig_model.gauss_params[
            "quats"
        ].detach().clone()
        self.dino_loader = dino_loader
        self.group_labels = group_labels
        self.group_masks = group_masks
        # store a 7-vec of trans, rotation for each group
        self.part_deltas = torch.zeros(
            1, len(group_masks), 7, dtype=torch.float32, device="cuda"
        )
        self.part_deltas[0, :, 3:] = torch.tensor(
            [1, 0, 0, 0], dtype=torch.float32, device="cuda"
        )
        self.part_deltas = torch.nn.Parameter(self.part_deltas)
        k = self.blur_kernel_size
        s = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
        self.blur = kornia.filters.GaussianBlur2d((k, k), (s, s))
        self.part_optimizer = torch.optim.Adam([self.part_deltas], lr=self.pose_lr)
        #hand_frames stores a list of hand vertices and faces for each keyframe, stored in the OBJECT COORDINATE FRAME
        self.hand_frames = []
        # lock to prevent blocking the render thread if provided
        self.render_lock = render_lock
        if self.use_atap:
            self.atap = ATAPLoss(dig_model, group_masks, group_labels, self.dataset_scale)
        self.init_means = self.dig_model.gauss_params["means"].detach().clone()
        self.init_quats = self.dig_model.gauss_params["quats"].detach().clone()
        # Save the initial object to world transform, and initial part to object transforms
        self.init_o2w = vtf.SE3.from_rotation_and_translation(
            vtf.SO3.identity(dtype=torch.float32,device='cuda'), self.init_means.mean(dim=0).squeeze()
        ).as_matrix()
        self.init_o2w_7vec = torch.tensor([[0,0,0,1,0,0,0]],dtype=torch.float32,device='cuda')
        self.init_o2w_7vec[0,:3] = self.init_means.mean(dim=0).squeeze()
        self.init_p2o = torch.empty(len(self.group_masks), 4, 4, dtype=torch.float32, device="cuda")
        self.init_p2o_7vec = torch.zeros(len(self.group_masks), 7, dtype=torch.float32, device="cuda")
        self.init_p2o_7vec[:,3] = 1.0
        for i,g in enumerate(self.group_masks):
            gp_centroid = self.init_means[g].mean(dim=0)
            self.init_p2o_7vec[i,:3] = gp_centroid
            self.init_p2o[i,:,:] = vtf.SE3.from_rotation_and_translation(
                vtf.SO3.identity(dtype=torch.float32,device='cuda'), (gp_centroid - self.init_means.mean(dim=0))
            ).as_matrix()
        self.sequence = VideoSequence()

    def initialize_obj_pose(self, first_obs: PosedObservation, niter=200, n_seeds=6, render=False):
        self.sequence.add_frame(first_obs)
        renders = []
        assert not self.is_initialized, "Can only initialize once"

        def try_opt(start_pose_adj, niter, use_depth):
            "tries to optimize the pose, returns None if failed, otherwise returns outputs and loss"
            self.reset_transforms()
            whole_pose_adj = start_pose_adj.detach().clone()
            whole_pose_adj = torch.nn.Parameter(whole_pose_adj)
            optimizer = torch.optim.Adam([whole_pose_adj], lr=0.003)
            for _ in range(niter):
                with torch.no_grad():
                    whole_pose_adj[:, 3:] = whole_pose_adj[:, 3:] / whole_pose_adj[:, 3:].norm(dim=-1, keepdim=True)
                tape = wp.Tape()
                optimizer.zero_grad()
                with tape:
                    loss = self.get_optim_loss(self.sequence.get_last_frame(), whole_pose_adj, identity_7vec().repeat(len(self.group_masks), 1), use_depth, False, False, False, False)
                loss.backward()
                tape.backward()
                optimizer.step()
                if render:
                    with torch.no_grad():
                        dig_outputs = self.dig_model.get_outputs(self.sequence.get_last_frame().camera)
                    renders.append(dig_outputs["rgb"].detach())
            self.is_initialized = True
            return loss, whole_pose_adj.data.detach()

        best_loss = float("inf")

        def find_pixel(n_gauss=10000):
            """
            returns the y,x coord and box size of the object in the video frame, based on the dino features
            and mutual nearest neighbors
            """
            samps = torch.randint(
                0, self.dig_model.num_points, (n_gauss,), device="cuda"
            )
            nn_inputs = self.dig_model.gauss_params["dino_feats"][samps]
            # dino_feats = self.dig_model.nn(nn_inputs.half()).float()  # NxC
            dino_feats = self.dig_model.nn(nn_inputs)  # NxC
            downsamp_factor = 4
            downsamp_frame_feats = self.sequence.get_last_frame().dino_feats[
                ::downsamp_factor, ::downsamp_factor, :
            ]
            frame_feats = downsamp_frame_feats.reshape(
                -1, downsamp_frame_feats.shape[-1]
            )  # (H*W) x C
            _, match_ids = mnn_matcher(dino_feats, frame_feats)
            x, y = (match_ids % (self.sequence.get_last_frame().camera.width / downsamp_factor)).float(), (
                match_ids // (self.sequence.get_last_frame().camera.width / downsamp_factor)
            ).float()
            x, y = x * downsamp_factor, y * downsamp_factor
            return y, x, torch.tensor([y.mean().item(), x.mean().item()], device="cuda")

        ys, xs, best_pix = find_pixel()
        obj_centroid = self.dig_model.means.mean(dim=0, keepdim=True)  # 1x3
        ray = self.sequence.get_last_frame().camera.generate_rays(0, best_pix)
        dist = 1.0
        point = ray.origins + ray.directions * dist
        for z_rot in np.linspace(0, np.pi * 2, n_seeds):
            whole_pose_adj = torch.zeros(1, 7, dtype=torch.float32, device="cuda")
            # x y z qw qx qy qz
            # (x,y,z) = something along ray - centroid
            quat = vtf.SO3.from_z_radians(torch.tensor(z_rot)).wxyz.float().cuda()
            whole_pose_adj[:, :3] = point - obj_centroid
            whole_pose_adj[:, 3:] = quat
            loss, final_pose = try_opt(whole_pose_adj, niter, False)
            if loss is not None and loss < best_loss:
                best_loss = loss
                best_pose = final_pose
        _, best_pose = try_opt(best_pose, 150, True)# do a few optimization steps with depth
        with self.render_lock:
            self.apply_to_model(
                best_pose,
                identity_7vec().repeat(len(self.group_masks), 1),
                self.group_labels,
            )
        best_outputs = self.dig_model.get_outputs(self.sequence.get_last_frame().camera)
        self.obj_delta = best_pose[None].contiguous()
        if self.do_obj_optim:
            self.obj_delta.requires_grad_(True)
            self.obj_optimizer = torch.optim.Adam([self.obj_delta], lr=self.pose_lr)
        return xs, ys, best_outputs, renders
    
    @property
    def objreg2objinit(self):
        return torch_posevec_to_mat(self.obj_delta[-1]).squeeze()
    
    def get_poses_relative_to_camera(self, c2w: torch.Tensor, keyframe: Optional[int] = None):
        """
        Returns the current group2cam transform as defined by the specified camera pose in world coords
        c2w: 3x4 tensor of camera to world transform

        Coordinate origin of the object aligns with world axes and centered at centroid

        returns:
        Nx4x4 tensor of obj2camera transform for each of the N groups, in the same ordering as the cluster labels
        """
        with torch.no_grad():
            assert c2w.shape == (3, 4)
            c2w = torch.cat(
                [
                    c2w,
                    torch.tensor([0, 0, 0, 1], dtype=torch.float32, device="cuda").view(
                        1, 4
                    ),
                ],
                dim=0,
            )
            obj2cam_physical_batch = torch.empty(
                len(self.group_masks), 4, 4, dtype=torch.float32, device="cuda"
            )
            for i in range(len(self.group_masks)):
                if keyframe is None:
                    obj2world_physical = self.get_part2world_transform(i)
                else:
                    obj2world_physical = self.get_keyframe_part2world_transform(i, keyframe)
                obj2world_physical[:3,3] /= self.dataset_scale
                obj2cam_physical_batch[i, :, :] = c2w.inverse().matmul(obj2world_physical)
        return obj2cam_physical_batch
    
    def get_partdelta_transform(self,i):
        """
        returns the transform from part_i to parti_i init at keyframe index given
        """
        return torch_posevec_to_mat(self.part_deltas[-1,i].unsqueeze(0)).squeeze()
    
    def get_keyframe_part2world_transform(self,i,keyframe):
        """
        returns the transform from part_i to world at keyframe index given
        """
        part2part = torch_posevec_to_mat(self.part_deltas[keyframe,i].detach().unsqueeze(0))
        return self.get_registered_o2w.matmul(self.init_p2o[i]).matmul(part2part[i])
    
    def get_registered_o2w(self):
        return self.init_o2w.matmul(self.objreg2objinit)
    
    def get_optim_loss(self, frame: PosedObservation, obj_delta, part_deltas, use_depth, use_rgb, use_atap, use_hand_mask, do_obj_optim):
        """
        Returns a backpropable loss for the given frame
        """
        with self.render_lock:
            self.dig_model.eval()
            self.apply_to_model(
                obj_delta, part_deltas, self.group_labels
            )
            dig_outputs = self.dig_model.get_outputs(frame.camera)
        if "dino" not in dig_outputs:
            self.reset_transforms()
            raise RuntimeError("Lost tracking")
        with torch.no_grad():
            object_mask = dig_outputs["accumulation"] > 0.8

        dino_feats = (
            self.blur(dig_outputs["dino"].permute(2, 0, 1)[None])
            .squeeze()
            .permute(1, 2, 0)
        )
        dino_feats = torch.where(object_mask,dig_outputs["dino"],dino_feats)
        if use_hand_mask:
            loss = (frame.dino_feats - dino_feats)[frame.hand_mask].norm(dim=-1).mean()
        else:
            loss = (frame.dino_feats - dino_feats).norm(dim=-1).mean()
        # THIS IS BAD WE NEED TO FIX THIS (because resizing makes the image very slightly misaligned)
        if use_depth:
            if frame.metric_depth:
                physical_depth = dig_outputs["depth"] / self.dataset_scale
                valids = object_mask & (~frame.depth.isnan())
                if use_hand_mask:
                    valids = valids & frame.hand_mask.unsqueeze(-1)
                pix_loss = (physical_depth - frame.depth) ** 2
                pix_loss = pix_loss[
                    valids & (pix_loss < self.depth_ignore_threshold**2)
                ]
                loss = loss + pix_loss.mean()
            else:
                # This is ranking loss for monodepth (which is disparity)
                frame_depth = 1 / frame.depth # convert disparity to depth
                N = 30000
                # erode the mask by like 10 pixels
                object_mask = object_mask & (~frame_depth.isnan())
                object_mask = object_mask & (dig_outputs['depth'] > .05)
                object_mask = kornia.morphology.erosion(
                    object_mask.squeeze().unsqueeze(0).unsqueeze(0).float(), torch.ones((self.rank_loss_erode, self.rank_loss_erode), device='cuda')
                ).squeeze().bool()
                if use_hand_mask:
                    object_mask = object_mask & frame.hand_mask
                valid_ids = torch.where(object_mask)
                rand_samples = torch.randint(
                    0, valid_ids[0].shape[0], (N,), device="cuda"
                )
                rand_samples = (
                    valid_ids[0][rand_samples],
                    valid_ids[1][rand_samples],
                )
                rend_samples = dig_outputs["depth"][rand_samples]
                mono_samples = frame_depth[rand_samples]
                rank_loss = depth_ranking_loss(rend_samples, mono_samples)
                loss = loss + self.rank_loss_mult*rank_loss
        if use_rgb:
            loss = loss + 0.05 * (dig_outputs["rgb"] - frame.rgb).abs().mean()
        if use_atap:
            weights = torch.ones(len(self.group_masks), len(self.group_masks),dtype=torch.float32,device='cuda')
            atap_loss = self.atap(weights)
            loss = loss + atap_loss
        if do_obj_optim:
            # add regularizer on the parts to not move much
            loss = loss + 0.01 * part_deltas[:,:3].norm(dim=-1).mean() + .01 * 2 * torch.acos(0.99*part_deltas[:,3]).mean()
        return loss
    
    def get_smoothness_loss(self, deltas: torch.Tensor, position_lambda: float, rotation_lambda: float, active_timesteps = slice(None)):
        """
        Returns the smoothness loss for the given deltas
        """
        loss = torch.empty(deltas.shape[0], deltas.shape[1], dtype=torch.float32, device="cuda", requires_grad=True)
        wp.launch(
            kernel=traj_smoothness_loss,
            dim=(deltas.shape[0], deltas.shape[1]),
            inputs=[wp.from_torch(deltas), position_lambda, rotation_lambda],
            outputs = [wp.from_torch(loss)],
        )
        return loss[active_timesteps].sum()
    
    def step(self, niter=1, use_depth=True, use_rgb=False, all_frames = False):
        lr_init = self.pose_lr
        n_warmup = 5 if all_frames else 0
        if all_frames:
            # reset the optimizers
            self.part_optimizer = torch.optim.Adam([self.part_deltas], lr=self.pose_lr)
            if self.do_obj_optim:
                self.obj_optimizer = torch.optim.Adam([self.obj_delta], lr=self.pose_lr)
        part_scheduler = ExponentialDecayScheduler(
            ExponentialDecaySchedulerConfig(
                lr_final=self.pose_lr_final, max_steps=niter, warmup_steps = n_warmup, ramp='linear', lr_pre_warmup = 1e-5
            )
        ).get_scheduler(self.part_optimizer, lr_init)
        if self.do_obj_optim:
            obj_scheduler = ExponentialDecayScheduler(
                ExponentialDecaySchedulerConfig(
                    lr_final=self.pose_lr_final, max_steps=niter, warmup_steps = n_warmup, ramp='linear', lr_pre_warmup = 1e-5
                )
            ).get_scheduler(self.obj_optimizer, lr_init)
        for _ in range(niter):
            # renormalize rotation representation
            with torch.no_grad():
                self.part_deltas[..., 3:] = self.part_deltas[..., 3:] / self.part_deltas[..., 3:].norm(dim=-1, keepdim=True)
                self.obj_delta[..., 3:] = self.obj_delta[..., 3:] / self.obj_delta[..., 3:].norm(dim=-1,keepdim=True)
            self.part_optimizer.zero_grad()
            if self.do_obj_optim:
                self.obj_optimizer.zero_grad()
            # Compute loss
            frame_ids_to_optimize = list(range(len(self.sequence))) if all_frames else [-1]
            for idx in frame_ids_to_optimize:
                tape = wp.Tape()
                with tape:
                    this_obj_delta = self.obj_delta[idx]
                    this_part_deltas = self.part_deltas[idx]
                    loss = self.get_optim_loss(self.sequence.get_frame(idx), this_obj_delta, this_part_deltas, 
                        use_depth, use_rgb, self.use_atap, self.mask_hands, self.do_obj_optim)
                loss.backward()
                #tape backward needs to be after loss backward since loss backward propagates gradients to the outputs of warp kernels
                tape.backward()
                # tape backward only propagates up to the slice, so we need to call autograd again to keep going beyond
                torch.autograd.backward([this_obj_delta,this_part_deltas], grad_tensors=[this_obj_delta.grad,this_part_deltas.grad])
            if all_frames:
                tape = wp.Tape()
                with tape:
                    part_smoothness= 0.1 * self.get_smoothness_loss(self.part_deltas, 1.0, 1.0, frame_ids_to_optimize)
                    obj_smoothness = 0.1 * self.get_smoothness_loss(self.obj_delta, 1.0, 1.0, frame_ids_to_optimize)
                print("Traj smoothness penalty (pos, rot)", part_smoothness.item(), obj_smoothness.item())
                (part_smoothness + obj_smoothness).backward()
                tape.backward()
            self.part_optimizer.step()
            part_scheduler.step()
            if self.do_obj_optim:
                self.obj_optimizer.step()
                obj_scheduler.step()
        with torch.no_grad():
            with self.render_lock:
                self.dig_model.eval()
                self.apply_to_model(
                        self.obj_delta[-1], self.part_deltas[-1], self.group_labels
                    )
                full_outputs = self.dig_model.get_outputs(self.sequence.get_last_frame().camera)
        return {k:i.detach() for k,i in full_outputs.items()}

    def apply_to_model(self, objdelta, part_deltas, group_labels):
        """
        Takes the current part_deltas and applies them to each of the group masks
        """
        self.reset_transforms()
        new_quats = torch.empty_like(
            self.dig_model.gauss_params["quats"], requires_grad=False
        )
        new_means = torch.empty_like(
            self.dig_model.gauss_params["means"], requires_grad=True
        )
        assert objdelta.shape == (1,7), objdelta.shape
        wp.launch(
            kernel=apply_to_model,
            dim=self.dig_model.num_points,
            inputs = [
                wp.from_torch(self.init_o2w_7vec),
                wp.from_torch(self.init_p2o_7vec),
                wp.from_torch(objdelta),
                wp.from_torch(part_deltas),
                wp.from_torch(group_labels),
                wp.from_torch(self.dig_model.gauss_params["means"], dtype=wp.vec3),
                wp.from_torch(self.dig_model.gauss_params["quats"]),
            ],
            outputs=[wp.from_torch(new_means, dtype=wp.vec3), wp.from_torch(new_quats)],
        )
        self.dig_model.gauss_params["quats"] = new_quats
        self.dig_model.gauss_params["means"] = new_means

    @torch.no_grad()
    def register_keyframe(self, lhands: List[trimesh.Trimesh], rhands: List[trimesh.Trimesh]):
        """
        Saves the current pose_deltas as a keyframe
        """
        # hand vertices are given in world coordinates
        w2o = self.get_registered_o2w().inverse().cpu().numpy()
        all_hands = lhands + rhands
        is_lefts = [True]*len(lhands) + [False]*len(rhands)
        if len(all_hands)>0:
            all_hands = [hand.apply_transform(w2o) for hand in all_hands]
            self.hand_frames.append(all_hands)
            self.hand_lefts.append(is_lefts)
        else:
            self.hand_frames.append([])
            self.hand_lefts.append([])

    @torch.no_grad()
    def apply_keyframe(self, i):
        """
        Applies the ith keyframe to the pose_deltas
        """
        self.apply_to_model(self.obj_delta[i], self.part_deltas[i], self.group_labels)
    
    def save_trajectory(self, path: Path):
        """
        Saves the trajectory to a file
        """
        torch.save({
            "part_deltas": self.part_deltas,
            "obj_delta": self.obj_delta,
            "hand_frames": self.hand_frames,
            "hand_lefts": self.hand_lefts
        }, path)

    def load_trajectory(self, path: Path):
        """
        Loads the trajectory from a file. Sets keyframes and hand_frames.
        """
        data = torch.load(path)
        self.part_deltas = data['part_deltas'].cuda()
        self.obj_delta = data['obj_delta'].cuda()
        self.hand_frames = data['hand_frames']
        self.hand_lefts = data['hand_lefts']
    
    def compute_single_hand_assignment(self) -> List[int]:
        """
        returns the group index closest to the hand

        list of increasing distance. list[0], list[1] second best etc
        """
        sum_part_dists = [0]*len(self.group_masks)
        for frame_id,hands in enumerate(self.hand_frames):
            if len(hands) == 0:
                continue
            self.apply_keyframe(frame_id)
            for h_id,h in enumerate(hands):
                h = h.copy()
                h.apply_transform(self.get_registered_o2w().cpu().numpy())
                # compute distance to fingertips for each group
                for g in range(len(self.group_masks)):
                    group_mask = self.group_masks[g]
                    means = self.dig_model.gauss_params["means"][group_mask].detach()
                    # compute nearest neighbor distance from index finger to the gaussians
                    finger_position = h.vertices[349]
                    thumb_position = h.vertices[745]
                    finger_dist = (means - torch.from_numpy(np.array(finger_position)).cuda()).norm(dim=1).min().item()
                    thumb_dist = (means - torch.from_numpy(np.array(thumb_position)).cuda()).norm(dim=1).min().item()
                    closest_dist = (finger_dist + thumb_dist)/2
                    sum_part_dists[g] += closest_dist
        ids = list(range(len(self.group_masks)))
        zipped = list(zip(ids,sum_part_dists))
        zipped.sort(key=lambda x: x[1])
        return [z[0] for z in zipped]

    def compute_two_hand_assignment(self) -> List[Tuple[int,int]]:
        """
        tuple of left_group_id, right_group_id
        list of increasing distance. list[0], list[1] second best etc
        """
        # store KxG tensors storing the minimum distance to left, right hands at each frame
        left_part_dists = torch.zeros(len(self.hand_frames), len(self.group_masks),device='cuda')
        right_part_dists = torch.zeros(len(self.hand_frames), len(self.group_masks),device='cuda')
        for frame_id,hands in enumerate(self.hand_frames):
            if len(hands) == 0:
                continue
            self.apply_keyframe(frame_id)
            for h_id,h in enumerate(hands):
                h = h.copy()
                h.apply_transform(self.get_registered_o2w().cpu().numpy())
                # compute distance to fingertips for each group
                for g in range(len(self.group_masks)):
                    group_mask = self.group_masks[g]
                    means = self.dig_model.gauss_params["means"][group_mask].detach()
                    # compute nearest neighbor distance from index finger to the gaussians
                    finger_position = h.vertices[349]
                    thumb_position = h.vertices[745]
                    finger_dist = (means - torch.from_numpy(np.array(finger_position)).cuda()).norm(dim=1).min().item()
                    thumb_dist = (means - torch.from_numpy(np.array(thumb_position)).cuda()).norm(dim=1).min().item()
                    closest_dist = (finger_dist + thumb_dist)/2
                    if self.hand_lefts[frame_id][h_id]:
                        left_part_dists[frame_id,g] = closest_dist
                    else:
                        right_part_dists[frame_id,g] = closest_dist
        # Next brute force all hand-part assignments and pick the best one
        assignments = []
        for li in range(len(self.group_masks)):
            for ri in range(len(self.group_masks)):
                if li == ri:
                    continue
                dist = left_part_dists[:,li].sum() + right_part_dists[:,ri].sum()
                assignments.append((li,ri,dist))
        assignments.sort(key=lambda x: x[2])
        return [(a[0],a[1]) for a in assignments]

    def reset_transforms(self):
        with torch.no_grad():
            self.dig_model.gauss_params["means"] = self.init_means.detach().clone()
            self.dig_model.gauss_params["quats"] = self.init_quats.detach().clone()

    def add_frame(self, frame: PosedObservation, extrapolate_velocity = True):
        """
        Sets the rgb_frame to optimize the pose for
        rgb_frame: HxWxC tensor image
        """
        assert self.is_initialized, "Must initialize first with the first frame"
        self.sequence.add_frame(frame)
        # add another timestep of pose to the part and object poses
        if extrapolate_velocity and self.obj_delta.shape[0] > 1:
            with torch.no_grad():
                new_obj = extrapolate_poses(self.obj_delta[-2], self.obj_delta[-1])
                new_parts = extrapolate_poses(self.part_deltas[-2], self.part_deltas[-1])
                self.obj_delta = torch.nn.Parameter(torch.cat([self.obj_delta, new_obj.unsqueeze(0)], dim=0))
                self.part_deltas = torch.nn.Parameter(torch.cat([self.part_deltas, new_parts.unsqueeze(0)], dim=0))
        else:
            self.obj_delta = torch.nn.Parameter(torch.cat([self.obj_delta, self.obj_delta[-1].unsqueeze(0)], dim=0))
            self.part_deltas = torch.nn.Parameter(torch.cat([self.part_deltas, self.part_deltas[-1].unsqueeze(0)], dim=0))
        append_in_optim(self.part_optimizer, [self.part_deltas])
        append_in_optim(self.obj_optimizer, [self.obj_delta])
        zero_optim_state(self.part_optimizer, [-2])
        zero_optim_state(self.obj_optimizer, [-2])