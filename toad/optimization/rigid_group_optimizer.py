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
from typing import List, Optional, Literal
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
import viser.transforms as vtf
import trimesh
from typing import Tuple

def quatmul(q0: torch.Tensor, q1: torch.Tensor):
    w0, x0, y0, z0 = torch.unbind(q0, dim=-1)
    w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
    return torch.stack(
        [
            -x0 * x1 - y0 * y1 - z0 * z1 + w0 * w1,
            x0 * w1 + y0 * z1 - z0 * y1 + w0 * x1,
            -x0 * z1 + y0 * w1 + z0 * x1 + w0 * y1,
            x0 * y1 - y0 * x1 + z0 * w1 + w0 * z1,
        ],
        dim=-1,
    )


def depth_ranking_loss(rendered_depth, gt_depth):
    """
    Depth ranking loss as described in the SparseNeRF paper
    Assumes that the layout of the batch comes from a PairPixelSampler, so that adjacent samples in the gt_depth
    and rendered_depth are from pixels with a radius of each other
    """
    m = 1e-4
    if rendered_depth.shape[0] % 2 != 0:
        # chop off one index
        rendered_depth = rendered_depth[:-1, :]
        gt_depth = gt_depth[:-1, :]
    dpt_diff = gt_depth[::2, :] - gt_depth[1::2, :]
    out_diff = rendered_depth[::2, :] - rendered_depth[1::2, :] + m
    differing_signs = torch.sign(dpt_diff) != torch.sign(out_diff)
    loss = out_diff[differing_signs] * torch.sign(out_diff[differing_signs])
    return loss.mean()

@wp.func
def poses_7vec_to_transform(poses: wp.array(dtype=float, ndim=2), i: int):
    position = wp.vector(poses[i,0], poses[i,1], poses[i,2])
    quaternion = wp.quaternion(poses[i,4], poses[i,5], poses[i,6], poses[i,3])
    return wp.transformation(position, quaternion)

@wp.kernel
def apply_to_model(
    init_o2w: wp.array(dtype=float, ndim=2),
    init_p2os: wp.array(dtype=float, ndim=2),
    o_delta: wp.array(dtype=float, ndim=2),
    p_deltas: wp.array(dtype=float, ndim=2),
    group_labels: wp.array(dtype=int),
    means: wp.array(dtype=wp.vec3),
    quats: wp.array(dtype=float, ndim=2),
    #outputs
    means_out: wp.array(dtype=wp.vec3),
    quats_out: wp.array(dtype=float, ndim=2),
):
    tid = wp.tid()
    group_id = group_labels[tid]
    o2w_T = poses_7vec_to_transform(init_o2w,0)
    p2o_T = poses_7vec_to_transform(init_p2os,group_id)
    odelta_T = poses_7vec_to_transform(o_delta,0)
    pdelta_T = poses_7vec_to_transform(p_deltas,group_id)
    g2w_T = wp.transformation(means[tid], wp.quaternion(quats[tid, 1], quats[tid, 2], quats[tid, 3], quats[tid, 0]))
    g2p_T = wp.transform_inverse(p2o_T) * wp.transform_inverse(o2w_T) * g2w_T
    new_g2w_T = o2w_T * odelta_T * p2o_T * pdelta_T * g2p_T
    means_out[tid] = wp.transform_get_translation(new_g2w_T)
    new_quat = wp.transform_get_rotation(new_g2w_T)
    quats_out[tid, 0] = new_quat[3]
    quats_out[tid, 1] = new_quat[0]
    quats_out[tid, 2] = new_quat[1]
    quats_out[tid, 3] = new_quat[2]

def mnn_matcher(feat_a, feat_b):
    """
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

class RigidGroupOptimizer:
    use_depth: bool = False
    rank_loss_mult: float = 0.1
    rank_loss_erode: int = 5
    depth_ignore_threshold: float = 0.1  # in meters
    use_atap: bool = True
    pose_lr: float = 0.003
    pose_lr_final: float = 0.0005
    mask_hands: bool = False
    use_roi: bool = False

    init_p2o: torch.Tensor
    """From: part, To: object. in current world frame. Part frame is centered at part centroid, and object frame is centered at object centroid."""

    def __init__(
        self,
        dig_model: DiGModel,
        dino_loader: DinoDataloader,
        init_c2w: Cameras,
        group_masks: List[torch.Tensor],
        group_labels: torch.Tensor,
        dataset_scale: float,
        render_lock=nullcontext(),
    ):
        """
        This one takes in a list of gaussian ID masks to optimize local poses for
        Each rigid group can be optimized independently, with no skeletal constraints
        """
        assert self.use_roi is False, "ROI seems to be cursed, dont use"
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
        self.pose_deltas = torch.zeros(
            len(group_masks), 7, dtype=torch.float32, device="cuda"
        )
        self.pose_deltas[:, 3:] = torch.tensor(
            [1, 0, 0, 0], dtype=torch.float32, device="cuda"
        )
        self.pose_deltas = torch.nn.Parameter(self.pose_deltas)
        k = 13
        s = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
        self.blur = kornia.filters.GaussianBlur2d((k, k), (s, s))
        self.optimizer = torch.optim.Adam([self.pose_deltas], lr=self.pose_lr)
        self.keyframes = []
        #hand_frames stores a list of hand vertices and faces for each keyframe, stored in the OBJECT COORDINATE FRAME
        self.hand_frames = []
        # lock to prevent blocking the render thread if provided
        self.render_lock = render_lock
        if self.use_atap:
            self.atap = ATAPLoss(dig_model, group_masks, group_labels, self.dataset_scale)
        self.init_c2w = deepcopy(init_c2w).to("cuda")._apply_fn_to_fields(torch.detach)
        self.init_means = self.dig_model.gauss_params["means"].detach().clone()
        self.init_quats = self.dig_model.gauss_params["quats"].detach().clone()
        # Save the initial object to world transform, and initial part to object transforms
        self.init_o2w = torch.from_numpy(vtf.SE3.from_rotation_and_translation(
            vtf.SO3.identity(), self.init_means.mean(dim=0).cpu().numpy().squeeze()
        ).as_matrix()).float().cuda()
        self.init_o2w_7vec = torch.tensor([[0,0,0,1,0,0,0]],dtype=torch.float32,device='cuda')
        self.init_o2w_7vec[0,:3] = self.init_means.mean(dim=0).squeeze()
        self.init_p2o = torch.empty(len(self.group_masks), 4, 4, dtype=torch.float32, device="cuda")
        self.init_p2o_7vec = torch.zeros(len(self.group_masks), 7, dtype=torch.float32, device="cuda")
        self.init_p2o_7vec[:,3] = 1.0
        for i,g in enumerate(self.group_masks):
            gp_centroid = self.init_means[g].mean(dim=0)
            self.init_p2o_7vec[i,:3] = gp_centroid
            self.init_p2o[i,:,:] = torch.from_numpy(vtf.SE3.from_rotation_and_translation(
                vtf.SO3.identity(), (gp_centroid - self.init_means.mean(dim=0)).cpu().numpy()
            ).as_matrix()).float().cuda()

    def initialize_obj_pose(self, niter=200, n_seeds=6, render=False, metric_depth = True):
        renders = []
        assert not self.is_initialized, "Can only initialize once"

        def try_opt(start_pose_adj,niter, depth = False):
            "tries to optimize the pose, returns None if failed, otherwise returns outputs and loss"
            self.reset_transforms()
            whole_obj_gp_labels = torch.zeros(self.dig_model.num_points).int().cuda()
            whole_pose_adj = start_pose_adj.detach().clone()
            whole_pose_adj = torch.nn.Parameter(whole_pose_adj)
            optimizer = torch.optim.Adam([whole_pose_adj], lr=0.005)
            for i in range(niter):
                with torch.no_grad():
                    whole_pose_adj[:, 3:] = whole_pose_adj[:, 3:] / whole_pose_adj[
                        :, 3:
                    ].norm(dim=1, keepdim=True)
                tape = wp.Tape()
                optimizer.zero_grad()
                with self.render_lock:
                    self.dig_model.eval()
                    with tape:
                        self.apply_to_model(
                            whole_pose_adj, whole_obj_gp_labels
                        )
                    dig_outputs = self.dig_model.get_outputs(self.init_c2w)
                    if "dino" not in dig_outputs:
                        return None, None,None
                dino_feats = (
                    self.blur(dig_outputs["dino"].permute(2, 0, 1)[None])
                    .squeeze()
                    .permute(1, 2, 0)
                )
                pix_loss = self.frame_pca_feats - dino_feats
                loss = pix_loss.norm(dim=-1).mean()
                if depth and self.use_depth:
                    object_mask = dig_outputs["accumulation"] > 0.9
                    if metric_depth:
                        physical_depth = dig_outputs["depth"] / self.dataset_scale
                        valids = object_mask & (~self.frame_depth.isnan())
                        if self.mask_hands:
                            valids = valids & self.hand_mask.unsqueeze(-1)
                        pix_loss = (physical_depth - self.frame_depth) ** 2
                        pix_loss = pix_loss[
                            valids & (pix_loss < self.depth_ignore_threshold**2)
                        ]
                        loss = loss + pix_loss.mean()
                    else:
                        # This is ranking loss for monodepth (which is disparity)
                        frame_depth = 1 / self.frame_depth # convert disparity to depth
                        N = 30000
                        # erode the mask by like 10 pixels
                        object_mask = object_mask & (~self.frame_depth.isnan())
                        object_mask = object_mask & (dig_outputs['depth'] > .05)
                        object_mask = kornia.morphology.erosion(
                            object_mask.squeeze().unsqueeze(0).unsqueeze(0).float(), torch.ones((self.rank_loss_erode, self.rank_loss_erode), device='cuda')
                        ).squeeze().bool()
                        if self.mask_hands:
                            object_mask = object_mask & self.hand_mask
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
                loss.backward()
                tape.backward()
                optimizer.step()
                if render:
                    renders.append(dig_outputs["rgb"].detach())
            self.is_initialized = True
            return dig_outputs, loss, whole_pose_adj.data.detach().clone()

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
            downsamp_frame_feats = self.frame_pca_feats[
                ::downsamp_factor, ::downsamp_factor, :
            ]
            frame_feats = downsamp_frame_feats.reshape(
                -1, downsamp_frame_feats.shape[-1]
            )  # (H*W) x C
            _, match_ids = mnn_matcher(dino_feats, frame_feats)
            x, y = (match_ids % (self.init_c2w.width / downsamp_factor)).float(), (
                match_ids // (self.init_c2w.width / downsamp_factor)
            ).float()
            x, y = x * downsamp_factor, y * downsamp_factor
            return y, x, torch.tensor([y.mean().item(), x.mean().item()], device="cuda")

        ys, xs, best_pix = find_pixel()
        obj_centroid = self.dig_model.means.mean(dim=0, keepdim=True)  # 1x3
        ray = self.init_c2w.generate_rays(0, best_pix)
        dist = 1.0
        point = ray.origins + ray.directions * dist
        for z_rot in np.linspace(0, np.pi * 2, n_seeds):
            whole_pose_adj = torch.zeros(1, 7, dtype=torch.float32, device="cuda")
            # x y z qw qx qy qz
            # (x,y,z) = something along ray - centroid
            quat = torch.from_numpy(vtf.SO3.from_z_radians(z_rot).wxyz).cuda()
            whole_pose_adj[:, :3] = point - obj_centroid
            whole_pose_adj[:, 3:] = quat
            dig_outputs, loss, final_pose = try_opt(whole_pose_adj,niter)
            if loss is not None and loss < best_loss:
                best_loss = loss
                best_outputs = dig_outputs
                best_pose = final_pose
        _,_,best_pose = try_opt(best_pose,200,depth=True)
        self.reset_transforms()
        with self.render_lock:
            self.apply_to_model(
                best_pose,
                torch.zeros(self.dig_model.num_points).int().cuda(),
            )
            self.objreg2objinit = torch.from_numpy(vtf.SE3.from_rotation_and_translation(
                vtf.SO3(best_pose[0,3:].cpu().numpy()), best_pose[0,:3].cpu().numpy().squeeze()
                ).as_matrix()).float().cuda()
            self.objreg2objinit_7vec = torch.tensor([[0,0,0,1,0,0,0]],dtype=torch.float32,device='cuda')
            self.objreg2objinit_7vec[0,:3] = best_pose[0,:3]
            self.objreg2objinit_7vec[0,3:] = torch.from_numpy(vtf.SO3.from_matrix(self.objreg2objinit[:3,:3].cpu().numpy()).wxyz).cuda()
            with torch.no_grad():
                self.pose_deltas[:, 3:] = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device="cuda")
                self.pose_deltas[:, :3] = 0
        return xs, ys, best_outputs, renders

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
        p_delta = self.pose_deltas.detach().clone()
        R_delta = torch.from_numpy(vtf.SO3(p_delta[i, 3:].cpu().numpy()).as_matrix()).float().cuda()
        delta = torch.eye(4, device="cuda")
        delta[:3, :3] = R_delta
        delta[:3, 3] = p_delta[i, :3]
        return delta
        # initial_part2world = self.get_initial_part2world(i)
        # part2world = self.get_part2world_transform(i)
        # return initial_part2world.inverse().matmul(part2world)
    
    def get_keyframe_part2world_transform(self,i,keyframe):
        """
        returns the transform from part_i to world at keyframe index given
        """
        part2part = self.keyframes[keyframe]
        return self.get_registered_o2w.matmul(self.init_p2o[i]).matmul(part2part[i])
    
    def get_registered_o2w(self):
        return self.init_o2w.matmul(self.objreg2objinit)
    
    def step(self, niter=1, use_depth=True, use_rgb=False, metric_depth=False):
        scheduler = ExponentialDecayScheduler(
            ExponentialDecaySchedulerConfig(
                lr_final=self.pose_lr_final, max_steps=niter
            )
        ).get_scheduler(self.optimizer, self.pose_lr)
        roi_cam = self.get_ROI_cam(self.get_ROI()) if self.use_roi else self.init_c2w
        for i in range(niter):
            # renormalize rotation representation
            with torch.no_grad():
                self.pose_deltas[:, 3:] = self.pose_deltas[:, 3:] / self.pose_deltas[
                    :, 3:
                ].norm(dim=1, keepdim=True)
            tape = wp.Tape()
            self.optimizer.zero_grad()
            with self.render_lock:
                self.dig_model.eval()
                with tape:
                    self.apply_to_model(
                        self.pose_deltas, self.group_labels
                    )
                dig_outputs = self.dig_model.get_outputs(roi_cam)
            if "dino" not in dig_outputs:
                self.reset_transforms()
                raise RuntimeError("Lost tracking")
            with torch.no_grad():
                object_mask = dig_outputs["accumulation"] > 0.9
            dino_feats = (
                self.blur(dig_outputs["dino"].permute(2, 0, 1)[None])
                .squeeze()
                .permute(1, 2, 0)
            )
            if self.mask_hands:
                mse_loss = (self.frame_pca_feats - dino_feats)[self.hand_mask].norm(dim=-1)
            else:
                mse_loss = (self.frame_pca_feats - dino_feats).norm(dim=-1)
            # THIS IS BAD WE NEED TO FIX THIS (because resizing makes the image very slightly misaligned)
            loss = mse_loss.mean()
            if use_depth and self.use_depth:
                if metric_depth:
                    physical_depth = dig_outputs["depth"] / self.dataset_scale
                    valids = object_mask & (~self.frame_depth.isnan())
                    if self.mask_hands:
                        valids = valids & self.hand_mask.unsqueeze(-1)
                    pix_loss = (physical_depth - self.frame_depth) ** 2
                    pix_loss = pix_loss[
                        valids & (pix_loss < self.depth_ignore_threshold**2)
                    ]
                    loss = loss + pix_loss.mean()
                else:
                    # This is ranking loss for monodepth (which is disparity)
                    frame_depth = 1 / self.frame_depth # convert disparity to depth
                    N = 30000
                    # erode the mask by like 10 pixels
                    object_mask = object_mask & (~self.frame_depth.isnan())
                    object_mask = object_mask & (dig_outputs['depth'] > .05)
                    object_mask = kornia.morphology.erosion(
                        object_mask.squeeze().unsqueeze(0).unsqueeze(0).float(), torch.ones((self.rank_loss_erode, self.rank_loss_erode), device='cuda')
                    ).squeeze().bool()
                    if self.mask_hands:
                        object_mask = object_mask & self.hand_mask
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
                loss = loss + 0.05 * (dig_outputs["rgb"] - self.rgb_frame).abs().mean()
            if self.use_atap:
                weights = torch.ones(len(self.group_masks), len(self.group_masks),dtype=torch.float32,device='cuda')
                with tape:
                    atap_loss = self.atap(weights)
                loss = loss + atap_loss
            loss.backward()
            tape.backward()
            self.optimizer.step()
            # self.weights_optimizer.step()
            scheduler.step()
        # reset lr
        self.optimizer.param_groups[0]["lr"] = self.pose_lr
        with torch.no_grad():
            with self.render_lock:
                self.dig_model.eval()
                self.apply_to_model(
                        self.pose_deltas.detach(), self.group_labels
                    )
                full_outputs = self.dig_model.get_outputs(self.init_c2w)
        return {k:i.detach() for k,i in full_outputs.items()}

    def apply_to_model(self, part_deltas, group_labels):
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
        
        if hasattr(self,'objreg2objinit'):
            objdelta = self.objreg2objinit_7vec
        else:
            objdelta = torch.zeros(1, 7, dtype=torch.float32, device="cuda")
            objdelta[0,3] = 1.0
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

        partdeltas = torch.empty(len(self.group_masks), 4, 4, dtype=torch.float32, device="cuda")
        for i in range(len(self.group_masks)):
            partdeltas[i] = self.get_partdelta_transform(i)
        self.keyframes.append(partdeltas)
    @torch.no_grad()
    def apply_keyframe(self, i):
        """
        Applies the ith keyframe to the pose_deltas
        """
        deltas_to_apply = torch.empty(len(self.group_masks), 7, dtype=torch.float32, device="cuda")
        for j in range(len(self.group_masks)):
            delta = self.keyframes[i][j]
            deltas_to_apply[j,:3] = delta[:3,3]
            deltas_to_apply[j,3:] = torch.from_numpy(vtf.SO3.from_matrix(delta[:3,:3].cpu().numpy()).wxyz).cuda()
        self.apply_to_model(deltas_to_apply, self.group_labels)
    
    def save_trajectory(self, path: Path):
        """
        Saves the trajectory to a file
        """
        torch.save({
            "keyframes": self.keyframes,
            "hand_frames": self.hand_frames,
            "hand_lefts": self.hand_lefts
        }, path)

    def load_trajectory(self, path: Path):
        """
        Loads the trajectory from a file. Sets keyframes and hand_frames.
        """
        data = torch.load(path)
        self.keyframes = [d.cuda() for d in data["keyframes"]]
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
    
    def get_ROI(self, inflate = 0.3):
        """
        returns the bounding box of the object in the current frame

        returns: ymin, ymax, xmin, xmax
        """
        assert self.use_roi
        # assert inflate < 1
        with torch.no_grad():
            with self.render_lock:
                self.dig_model.eval()
                self.apply_to_model(self.pose_deltas.detach(), self.group_labels)
                object_mask = self.dig_model.get_outputs(self.init_c2w)["accumulation"] > 0.8
            valids = torch.where(object_mask)
            inflate_amnt = (inflate*(valids[0].max() - valids[0].min()).item(),
                            inflate*(valids[1].max() - valids[1].min()).item())
            candidate_roi = (valids[0].min().item() - inflate_amnt[0], inflate_amnt[0] + valids[0].max().item(), 
                    valids[1].min().item() - inflate_amnt[1], inflate_amnt[1] + valids[1].max().item())
            #clip ROI to image bounds
            candidate_roi = (int(candidate_roi[0]),int(candidate_roi[1]),int(candidate_roi[2]),int(candidate_roi[3]))
            candidate_roi = (max(0,candidate_roi[0]),min(self.init_c2w.height-1,candidate_roi[1]),
                            max(0,candidate_roi[2]),min(self.init_c2w.width-1,candidate_roi[3]))
            candidate_roi = (int(candidate_roi[0]),int(candidate_roi[1]),int(candidate_roi[2]),int(candidate_roi[3]))
        return candidate_roi
        
    def get_ROI_cam(self, roi):
        """
        returns the intrinsics of the camera for the current ROI
        """
        assert self.use_roi
        ymin, ymax, xmin, xmax = roi
        height = torch.tensor(ymax - ymin,device='cuda').view(1,1).int()
        width = torch.tensor(xmax - xmin,device='cuda').view(1,1).int()
        cx = torch.tensor(self.init_c2w.cx.detach().clone() - xmin,device='cuda').view(1,1)
        cy = torch.tensor(self.init_c2w.cy.detach().clone() - ymin,device='cuda').view(1,1)
        newcam = Cameras(self.init_c2w.camera_to_worlds.detach().clone(),
                         self.init_c2w.fx.detach().clone(),self.init_c2w.fy.detach().clone(),
                         cx.detach().clone(),cy.detach().clone(),
                         width.detach().clone(),height.detach().clone())
        newcam.rescale_output_resolution(500/max(newcam.width,newcam.height))
        return newcam
    
    def set_frame(self, rgb_frame: torch.Tensor, depth: Optional[torch.Tensor] = None):
        """
        Sets the rgb_frame to optimize the pose for
        rgb_frame: HxWxC tensor image
        init_c2w: initial camera to object transform (given whatever coordinates the self.dig_model is in)
        """
        if self.use_roi and self.is_initialized:
            ymin,ymax,xmin,xmax = self.get_ROI()
            roi_cam = self.get_ROI_cam((ymin,ymax,xmin,xmax))
            # ROI is determined inside the global camera, so we need to convert it to the right resolution
            ymin,ymax = int((ymin/(self.init_c2w.height-1))*(rgb_frame.shape[0]-1)),int((ymax/(self.init_c2w.height-1))*(rgb_frame.shape[0]-1))
            xmin,xmax = int((xmin/(self.init_c2w.width-1))*(rgb_frame.shape[1]-1)),int((xmax/(self.init_c2w.width-1))*(rgb_frame.shape[1]-1))
            rgb_frame = rgb_frame[ymin:ymax,xmin:xmax].detach().clone()
            if self.use_depth and depth is not None:
                depth = depth[ymin:ymax,xmin:xmax].detach().clone()
        else:
            roi_cam = self.init_c2w

        with torch.no_grad():
            self.rgb_frame = resize(
                rgb_frame.permute(2, 0, 1),
                (roi_cam.height, roi_cam.width),
                antialias=True,
            ).permute(1, 2, 0)
            self.frame_pca_feats = self.dino_loader.get_pca_feats(
                rgb_frame.permute(2, 0, 1).unsqueeze(0), keep_cuda=True
            ).squeeze()
            self.frame_pca_feats = resize(
                self.frame_pca_feats.permute(2, 0, 1),
                (roi_cam.height, roi_cam.width),
                antialias=True,
            ).permute(1, 2, 0)
            # HxWxC
            if self.use_depth:
                if depth is None:
                    depth = get_depth((rgb_frame*255).to(torch.uint8))
                self.frame_depth = (
                    resize(
                        depth.unsqueeze(0),
                        (roi_cam.height, roi_cam.width),
                        antialias=True,
                    )
                    .squeeze()
                    .unsqueeze(-1)
                )
            if self.mask_hands:
                self.hand_mask = get_hand_mask((self.rgb_frame * 255).to(torch.uint8))
                self.hand_mask = (
                    torch.nn.functional.max_pool2d(
                        self.hand_mask[None, None], 3, padding=1, stride=1
                    ).squeeze()
                    == 0.0
                )