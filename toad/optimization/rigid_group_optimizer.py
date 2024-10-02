import torch
import matplotlib.pyplot as plt
from nerfstudio.utils.eval_utils import eval_setup
from pathlib import Path
import numpy as np
from nerfstudio.viewer.viewer import Viewer
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.configs.base_config import ViewerConfig
import cv2
from torchvision.transforms import ToTensor
from PIL import Image
from typing import List, Optional, Literal, Callable, cast
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
from toad.optimization.atap_loss import ATAPLoss, ATAPConfig
from toad.utils import *
import toad.transforms as vtf
import trimesh
from typing import Tuple
from nerfstudio.model_components.losses import depth_ranking_loss
from toad.optimization.utils import (
    apply_to_model_warp,
    traj_smoothness_loss_warp,
    append_in_optim,
    zero_optim_state,
    extrapolate_poses,
    identity_7vec,
    mnn_matcher,
)
from toad.optimization.observation import PosedObservation, VideoSequence, Frame
from tqdm import tqdm
from dataclasses import dataclass
from jaxtyping import Float
from loguru import logger

@dataclass
class RigidGroupOptimizerConfig:
    use_depth: bool = True
    rank_loss_mult: float = 0.2
    rank_loss_erode: int = 3
    depth_ignore_threshold: float = 0.1  # in meters
    atap_config: ATAPConfig = ATAPConfig()
    use_roi: bool = True
    roi_inflate: float = 0.25
    pose_lr: float = 0.005
    pose_lr_final: float = 0.001
    mask_hands: bool = False
    do_obj_optim: bool = False
    blur_kernel_size: int = 5
    mask_threshold: float = 0.8
    rgb_loss_weight: float = 0.05
    part_still_weight: float = 0.01

class RigidGroupOptimizer:
    dig_model: DiGModel
    dino_loader: DinoDataloader
    dataset_scale: float

    num_groups: int
    group_labels: torch.Tensor
    group_masks: List[torch.Tensor]

    # Poses, as scanned.
    init_means: torch.Tensor
    init_quats: torch.Tensor
    init_o2w: torch.Tensor
    init_p2o: Float[torch.Tensor, "group 7"]

    is_initialized: bool
    obj_delta: Float[torch.nn.Parameter, "time 7"]
    part_deltas: Float[torch.nn.Parameter, "time group 7"]

    def __init__(
        self,
        config: RigidGroupOptimizerConfig,
        pipeline: Pipeline,
        render_lock: Union[Lock, nullcontext] = nullcontext(),
    ):
        """
        This one takes in a list of gaussian ID masks to optimize local poses for
        Each rigid group can be optimized independently, with no skeletal constraints
        """
        self.config = config
        self.dig_model = cast(DiGModel, pipeline.model)
        self.dino_loader = pipeline.datamanager.dino_dataloader

        assert pipeline.datamanager.train_dataset is not None
        self.dataset_scale = pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_scale
        self.render_lock = render_lock

        k = self.config.blur_kernel_size
        s = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
        self.blur = kornia.filters.GaussianBlur2d((k, k), (s, s))

        # Detach all the params to avoid retain_graph issue.
        for k, v in self.dig_model.gauss_params.items():
            self.dig_model.gauss_params[k] = v.detach().clone()

        # Store the initial means and quats, for state restoration later on.
        self.init_means = self.dig_model.gauss_params["means"].detach().clone()
        self.init_quats = self.dig_model.gauss_params["quats"].detach().clone()

        # Initialize parts + optimizers.
        if pipeline.cluster_labels is None:
            labels = torch.zeros(pipeline.model.num_points).int().cuda()
        else:
            labels = pipeline.cluster_labels.int().cuda()
        self.configure_from_clusters(labels)

        self.sequence = VideoSequence()
        self.is_initialized = False

    def configure_from_clusters(self, group_labels: torch.Tensor):
        """
        Given `group_labels`, set the group masks and labels.

        Affects all attributes affected by # of parts:
        - `self.num_groups
        - `self.group_labels`
        - `self.group_masks`
        - `self.part_deltas`
        - `self.init_p2o`
        - `self.atap`
        , as well as `self.init_o2w`.

        NOTE(cmk) why do you need to store both `self.group_labels` and `self.group_masks`?
        """
        # Get group / cluster label info.
        self.group_labels = group_labels.cuda()
        self.num_groups = int(self.group_labels.max().item() + 1)
        self.group_masks = [(self.group_labels == cid).cuda() for cid in range(self.group_labels.max() + 1)]

        # Store pose of each part, as wxyz_xyz.
        part_deltas = torch.zeros(1, self.num_groups, 7, dtype=torch.float32, device="cuda")
        part_deltas[:, :, 0] = 1  # wxyz = [1, 0, 0, 0] is identity.
        self.part_deltas = torch.nn.Parameter(part_deltas)

        # Initialize the object pose. Centered at object centroid, and identity rotation.
        self.init_o2w = identity_7vec()
        self.init_o2w[0, 4:] = self.init_means.mean(dim=0).squeeze()

        # Initialize the part poses to identity. Again, wxyz_xyz.
        # Parts are initialized at the centroid of the part cluster.
        self.init_p2o = identity_7vec().repeat(self.num_groups, 1)
        for i, g in enumerate(self.group_masks):
            gp_centroid = self.init_means[g].mean(dim=0)
            self.init_p2o[i, 4:] = gp_centroid - self.init_means.mean(dim=0)

        self.atap = ATAPLoss(
            self.config.atap_config,
            self.dig_model,
            self.group_masks,
            self.group_labels,
            self.dataset_scale,
        )

    def initialize_obj_pose(
        self,
        first_obs: PosedObservation,
        niter=200,
        n_seeds=6,
        use_depth=False,
        render=False,
    ):
        """
        Initializes object pose w/ observation. Also sets:
        - `self.obj_delta`
        """
        assert not self.is_initialized, "Can only initialize once"
        self.sequence.add_frame(first_obs)

        renders = []

        # Initial guess for 3D object location.
        est_dist_to_obj = 0.4 * self.dataset_scale  # 0.4m in nerfstudio world.
        xs, ys, est_loc_2d = self._find_object_pixel_location(first_obs)
        ray = first_obs.frame.camera.generate_rays(0, est_loc_2d)
        est_loc_3d = ray.origins + ray.directions * est_dist_to_obj

        # Take `n_seed` rotations around the object centroid, optimize, then pick the best one.
        # Don't use ROI for this step.
        best_pose, best_loss = identity_7vec(), float("inf")
        obj_centroid = self.dig_model.means.mean(dim=0, keepdim=True)  # 1x3
        for z_rot in tqdm(np.linspace(0, np.pi * 2, n_seeds), "Trying seeds..."):
            candidate_pose = torch.zeros(1, 7, dtype=torch.float32, device="cuda")
            candidate_pose[:, :4] = (
                vtf.SO3.from_z_radians(torch.tensor(z_rot)).wxyz.float().cuda()
            )
            candidate_pose[:, 4:] = est_loc_3d - obj_centroid
            loss, final_pose, rend = self._try_opt(
                candidate_pose, first_obs.frame, niter, use_depth, render=render
            )
            renders.extend(rend)

            if loss is not None and loss < best_loss:
                best_loss = loss
                best_pose = final_pose

        # Extra optimization steps, with the best pose.
        # Use ROI for this step, since we're close to the GT object pose.
        first_obs.compute_and_set_roi(self)

        # Note the lower LR, for this fine-tuning step.
        _, best_pose, rend = self._try_opt(
            best_pose, first_obs.roi_frame, niter, use_depth, lr=0.005, render=render
        )

        assert best_pose.shape == (1, 7), best_pose.shape
        self.obj_delta = torch.nn.Parameter(best_pose)  # [1, 7]

        if self.config.do_obj_optim:
            self.obj_delta.requires_grad_(True)

        self.is_initialized = True
        logger.info("Initialized object pose")
        return renders

    def fit(self, frame_indices: list[int] = [], niter=1, all_frames=False):
        # TODO(cmk) temporarily removed all_frames)
        lr_init = self.config.pose_lr

        assert len(frame_indices) == 1
        idx = frame_indices[0]
        part_optimizer = torch.optim.Adam([self.part_deltas], lr=lr_init)
        part_scheduler = ExponentialDecayScheduler(
            ExponentialDecaySchedulerConfig(
                lr_final=self.config.pose_lr_final,
                max_steps=niter,
                ramp="linear",
                lr_pre_warmup=1e-5,
            )
        ).get_scheduler(part_optimizer, lr_init)

        if self.config.do_obj_optim:
            obj_optimizer = torch.optim.Adam([self.obj_delta], lr=lr_init)
            obj_scheduler = ExponentialDecayScheduler(
                ExponentialDecaySchedulerConfig(
                    lr_final=self.config.pose_lr_final,
                    max_steps=niter,
                    ramp="linear",
                    lr_pre_warmup=1e-5,
                )
            ).get_scheduler(obj_optimizer, lr_init)

        for _ in range(niter):
            # renormalize rotation representation
            with torch.no_grad():
                self.part_deltas[..., :4] = self.part_deltas[..., :4] / self.part_deltas[..., :4].norm(dim=-1, keepdim=True)
                self.obj_delta[..., :4] = self.obj_delta[..., :4] / self.obj_delta[..., :4].norm(dim=-1, keepdim=True)

            part_optimizer.zero_grad()
            if self.config.do_obj_optim:
                obj_optimizer.zero_grad()

            # Compute loss
            tape = wp.Tape()
            with tape:
                this_obj_delta = self.obj_delta[idx].view(1, 7)
                this_part_deltas = self.part_deltas[idx]
                observation = self.sequence.get_frame(idx)
                frame = (
                    observation.frame
                    if not self.config.use_roi
                    else observation.roi_frame
                )
                loss = self._get_loss(
                    frame,
                    this_obj_delta,
                    this_part_deltas,
                    self.config.use_depth,
                    False,
                    self.config.atap_config.use_atap,
                    self.config.do_obj_optim,
                )

            assert loss is not None
            loss.backward()
            tape.backward()  # torch, then tape backward, to propagate gradients to warp kernels.

            # tape backward only propagates up to the slice, so we need to call autograd again to continue.
            assert this_obj_delta.grad is not None
            assert this_part_deltas.grad is not None
            torch.autograd.backward(
                [this_obj_delta, this_part_deltas],
                grad_tensors=[this_obj_delta.grad, this_part_deltas.grad],
            )

            part_optimizer.step()
            part_scheduler.step()

            if self.config.do_obj_optim:
                obj_optimizer.step()
                obj_scheduler.step()

    def _try_opt(
        self,
        pose: torch.Tensor,
        frame: Frame,
        niter: int,
        use_depth: bool,
        lr: float = 0.01,
        render: bool = False
    ) -> Tuple[float, torch.Tensor, List[torch.Tensor]]:
        "tries to optimize the pose, returns None if failed, otherwise returns outputs and loss"
        pose = torch.nn.Parameter(pose.detach().clone())

        optimizer = torch.optim.Adam([pose], lr=lr)
        scheduler = ExponentialDecayScheduler(
            ExponentialDecaySchedulerConfig(
                lr_final=0.005,
                max_steps=niter,
            )
        ).get_scheduler(optimizer, lr)

        loss = torch.inf
        renders = []
        for _ in tqdm(range(niter), "Optimizing pose...", leave=False):
            with torch.no_grad():
                pose[..., :4] = pose[..., :4] / pose[..., :4].norm(dim=-1, keepdim=True)

            tape = wp.Tape()
            optimizer.zero_grad()
            with tape:
                loss = self._get_loss(
                    frame,
                    pose,
                    identity_7vec().repeat(len(self.group_masks), 1),
                    use_depth,
                    False,
                    False,
                    False,
                )
            if loss is None:
                return torch.inf, pose.data.detach(), renders

            loss.backward()
            tape.backward()  # torch, then tape backward, to propagate gradients to warp kernels.
            optimizer.step()
            scheduler.step()

            loss = loss.item()
            if render:
                with torch.no_grad():
                    dig_outputs = self.dig_model.get_outputs(frame.camera)
                renders.append((dig_outputs["rgb"].detach() * 255).int().cpu().numpy())

        return loss, pose.data.detach(), renders

    def _find_object_pixel_location(self, obs: PosedObservation, n_gauss=20000):
        """
        returns the y,x coord and box size of the object in the video frame, based on the dino features
        and mutual nearest neighbors
        """
        samps = torch.randint(0, self.dig_model.num_points, (n_gauss,), device="cuda")
        nn_inputs = self.dig_model.gauss_params["dino_feats"][samps]
        dino_feats = self.dig_model.nn(nn_inputs)  # NxC
        downsamp_frame_feats = obs.frame.dino_feats
        frame_feats = downsamp_frame_feats.reshape(
            -1, downsamp_frame_feats.shape[-1]
        )  # (H*W) x C
        downsamp = 4
        frame_feats = frame_feats[::downsamp]
        _, match_ids = mnn_matcher(dino_feats, frame_feats)
        x, y = (match_ids*downsamp % (obs.frame.camera.width)).float(), (
            match_ids*downsamp // (obs.frame.camera.width)
        ).float()
        return x, y, torch.tensor([y.median().item(), x.median().item()], device="cuda")

    def _get_loss(
        self,
        frame: Frame,
        obj_delta: torch.Tensor,
        part_deltas: torch.Tensor,
        use_depth: bool,
        use_rgb: bool,
        use_atap: bool,
        do_obj_optim: bool
    ) -> Optional[torch.Tensor]:
        """
        Returns a backpropable loss for the given frame
        """
        with self.render_lock:
            self.dig_model.eval()
            self.apply_to_model(obj_delta, part_deltas)
            outputs = cast(
                dict[str, torch.Tensor],
                self.dig_model.get_outputs(frame.camera)
            )

        assert "accumulation" in outputs, outputs.keys()
        with torch.no_grad():
            object_mask = outputs["accumulation"] > self.config.mask_threshold
        if not object_mask.any():
            logger.warning("No object detected in frame")
            return None

        loss = torch.Tensor([0.0]).cuda()

        dino_loss = self._get_dino_loss(outputs, frame, object_mask)
        loss = loss + dino_loss

        if use_depth:
            depth_loss = self._get_depth_loss(outputs, frame, object_mask)
            loss = loss + depth_loss

        if use_rgb:
            rgb_loss = 0.05 * (outputs["rgb"] - frame.rgb).abs().mean()
            loss = loss + rgb_loss

        if use_atap:
            weights = torch.full(
                (self.num_groups, self.num_groups),
                1,
                dtype=torch.float32,
                device="cuda",
            )
            atap_loss = self.atap(weights)
            loss = loss + atap_loss

        if do_obj_optim:
            part_still_loss = 0.01 * part_deltas[:,4:].norm(dim=-1).mean()
            loss = loss + part_still_loss

        return loss

    def _get_dino_loss(
        self,
        outputs: dict[str, torch.Tensor],
        frame: Frame,
        object_mask: torch.Tensor,
    ) -> torch.Tensor:
        assert "dino" in outputs and isinstance(outputs["dino"], torch.Tensor)

        blurred_dino_feats = (
            self.blur(outputs["dino"].permute(2, 0, 1)[None]).squeeze().permute(1, 2, 0)
        )
        dino_feats = torch.where(object_mask, outputs["dino"], blurred_dino_feats)

        if frame.hand_mask is not None:
            loss = (frame.dino_feats - dino_feats)[frame.hand_mask].norm(dim=-1).mean()
        else:
            loss = (frame.dino_feats - dino_feats).norm(dim=-1).mean()

        return loss

    def _get_depth_loss(
        self,
        outputs: dict[str, torch.Tensor],
        frame: Frame,
        object_mask: torch.Tensor,
        n_samples_for_ranking: int = 20000,
    ) -> torch.Tensor:
        if frame.metric_depth:
            physical_depth = outputs["depth"] / self.dataset_scale

            valids = object_mask & (~frame.depth.isnan())
            if frame.hand_mask is not None:
                valids = valids & frame.hand_mask.unsqueeze(-1)

            pix_loss = (physical_depth - frame.depth) ** 2
            pix_loss = pix_loss[
                valids & (pix_loss < self.config.depth_ignore_threshold**2)
            ]
            return pix_loss.mean()

        # Otherwise, we're using disparity.
        frame_depth = 1 / frame.depth # convert disparity to depth
        # erode the mask by like 10 pixels
        object_mask = object_mask & (~frame_depth.isnan())
        object_mask = kornia.morphology.erosion(
            object_mask.squeeze().unsqueeze(0).unsqueeze(0).float(),
            torch.ones((self.config.rank_loss_erode, self.config.rank_loss_erode), device='cuda')
        ).squeeze().bool()
        if frame.hand_mask is not None:
            object_mask = object_mask & frame.hand_mask
        valid_ids = torch.where(object_mask)

        if len(valid_ids[0]) > 0:
            rand_samples = torch.randint(
                0, valid_ids[0].shape[0], (n_samples_for_ranking,), device="cuda"
            )
            rand_samples = (
                valid_ids[0][rand_samples],
                valid_ids[1][rand_samples],
            )
            rend_samples = outputs["depth"][rand_samples]
            mono_samples = frame_depth[rand_samples]
            rank_loss = depth_ranking_loss(rend_samples, mono_samples)
            return self.config.rank_loss_mult*rank_loss

        return torch.Tensor([0.0])

    def apply_to_model(self, obj_delta, part_deltas):
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
        assert obj_delta.shape == (1,7), obj_delta.shape
        wp.launch(
            kernel=apply_to_model_warp,
            dim=self.dig_model.num_points,
            inputs = [
                wp.from_torch(self.init_o2w),
                wp.from_torch(self.init_p2o),
                wp.from_torch(obj_delta),
                wp.from_torch(part_deltas),
                wp.from_torch(self.group_labels),
                wp.from_torch(self.dig_model.gauss_params["means"], dtype=wp.vec3),
                wp.from_torch(self.dig_model.gauss_params["quats"]),
            ],
            outputs=[wp.from_torch(new_means, dtype=wp.vec3), wp.from_torch(new_quats)],
        )
        self.dig_model.gauss_params["quats"] = new_quats
        self.dig_model.gauss_params["means"] = new_means

    @torch.no_grad()
    def apply_keyframe(self, i):
        """
        Applies the ith keyframe to the pose_deltas
        """
        self.apply_to_model(self.obj_delta[i], self.part_deltas[i])

    def load_deltas(self, path: Path):
        """
        Loads the trajectory from a file. Sets keyframes and hand_frames.
        """
        data = torch.load(path)
        self.part_deltas = data['part_deltas'].cuda()
        self.obj_delta = data['obj_delta'].cuda()
        self.is_initialized = True

    def save_deltas(self, path: Path):
        """
        Saves the trajectory to a file
        """
        torch.save({
            "part_deltas": self.part_deltas,
            "obj_delta": self.obj_delta,
        }, path)

    def reset_transforms(self):
        with torch.no_grad():
            self.dig_model.gauss_params["means"] = self.init_means.detach().clone()
            self.dig_model.gauss_params["quats"] = self.init_quats.detach().clone()

    def add_observation(self, obs: PosedObservation, extrapolate_velocity = True):
        """
        Sets the rgb_frame to optimize the pose for
        rgb_frame: HxWxC tensor image
        """
        assert self.is_initialized, "Must initialize first with the first frame"

        if self.config.use_roi:
            obs.compute_and_set_roi(self)
        self.sequence.add_frame(obs)

        # add another timestep of pose to the part and object poses
        self.obj_delta = torch.nn.Parameter(
            torch.cat([self.obj_delta, self.obj_delta[-1].unsqueeze(0)], dim=0)
        )

        if extrapolate_velocity and self.part_deltas.shape[0] > 1:
            with torch.no_grad():
                next_part_delta = extrapolate_poses(
                    self.part_deltas[-2], self.part_deltas[-1], 0.2
                )
        else:
            next_part_delta = self.part_deltas[-1]
        self.part_deltas = torch.nn.Parameter(
            torch.cat([self.part_deltas, next_part_delta.unsqueeze(0)], dim=0)
        )

    def create_observation_from_rgb_and_camera(
        self, rgb: np.ndarray, camera: Cameras
    ) -> PosedObservation:
        target_frame_rgb = ToTensor()(Image.fromarray(rgb)).permute(1, 2, 0).cuda()
        dino_fn = lambda x: self.dino_loader.get_pca_feats(x, keep_cuda=True)
        frame = PosedObservation(target_frame_rgb, camera, dino_fn)
        return frame

    @property
    def objreg2objinit(self):
        """
        Transform:
        - from: `objreg`, in camera frame from which it was registered (e.g., robot camera).
        - from: `objinit`, in original frame from which it was scanned
        """
        return self.obj_delta[0]
