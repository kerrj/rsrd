import torch
import numpy as np
from typing import List
from dig.dig import DiGModel
import warp as wp
from dataclasses import dataclass
from loguru import logger

from rsrd.util.warp_kernels import atap_loss_warp


@dataclass
class ATAPConfig:
    use_atap: bool = True
    touch_radius: float = 0.002
    N: int = 500
    loss_mult: float = 0.2
    loss_alpha: float = (
        0.1  # rule: for jointed, use 1.0 alpha, for non-jointed use 0.1 ish
    )


class ATAPLoss:
    def __init__(
        self,
        config: ATAPConfig,
        dig_model: DiGModel,
        group_masks: List[torch.Tensor],
        group_labels: torch.Tensor,
        dataset_scale: float = 1.0,
    ):
        """
        Initializes the data structure to compute the loss between groups touching
        """
        self.config = config
        if not self.config.use_atap:
            return
        self.touch_radius = config.touch_radius * dataset_scale
        logger.info(f"Touch radius is {self.touch_radius}")
        self.dig_model = dig_model
        self.group_masks = group_masks
        self.group_labels = group_labels
        self.nn_info = []
        for grp in self.group_masks:
            with torch.no_grad():
                dists, ids, match_ids, group_ids1, group_ids = self._radius_nn(
                    grp, self.touch_radius
                )
                self.nn_info.append((dists, ids, match_ids, group_ids1, group_ids))
                logger.info(f"Group {len(self.nn_info)} has {len(ids)} neighbors")
        self.dists = torch.cat([x[0] for x in self.nn_info]).cuda()
        self.ids = torch.cat([x[1] for x in self.nn_info]).cuda().int()
        self.match_ids = torch.cat([x[2] for x in self.nn_info]).cuda().int()
        self.group_ids1 = torch.cat([x[3] for x in self.nn_info]).cuda().int()
        self.group_ids2 = torch.cat([x[4] for x in self.nn_info]).cuda().int()
        self.num_pairs = (
            torch.cat([torch.tensor(len(x[1])).repeat(len(x[1])) for x in self.nn_info])
            .cuda()
            .float()
        )

    def __call__(self, connectivity_weights: torch.Tensor):
        """
        Computes the loss between groups touching
        connectivity_weights: a tensor of shape (num_groups,num_groups) representing the weights between each group

        returns: a differentiable loss
        """
        if len(self.group_masks) == 1 or not self.config.use_atap:
            return torch.tensor(0.0, device="cuda")
        if self.dists.shape[0] == 0:
            return torch.tensor(0.0, device="cuda")
        assert connectivity_weights.shape == (
            len(self.group_masks),
            len(self.group_masks),
        ), "connectivity weights must be a square matrix of size num_groups"
        loss = wp.empty(
            self.dists.shape[0], dtype=wp.float32, requires_grad=True, device="cuda"
        )
        wp.launch(
            dim=self.dists.shape[0],
            kernel=atap_loss_warp,
            inputs=[
                wp.from_torch(self.dig_model.gauss_params["means"], dtype=wp.vec3),
                wp.from_torch(self.dists),
                wp.from_torch(self.ids),
                wp.from_torch(self.match_ids),
                wp.from_torch(self.group_ids1),
                wp.from_torch(self.group_ids2),
                wp.from_torch(connectivity_weights),
                loss,
                self.config.loss_alpha,
            ],
        )
        return (wp.to_torch(loss) / self.num_pairs).sum() * self.config.loss_mult

    def _radius_nn(self, group_mask: torch.Tensor, r: float):
        """
        returns the nearest neighbors to gaussians in a group within a certain radius (and outside that group)
        returns -1 indices for neighbors outside the radius or within the same group
        """
        global_group_ids = torch.zeros(
            self.dig_model.num_points, dtype=torch.long, device="cuda"
        )
        for i, grp in enumerate(self.group_masks):
            global_group_ids[grp] = i
        from cuml.neighbors import NearestNeighbors

        model = NearestNeighbors(n_neighbors=self.config.N)
        means = self.dig_model.means.detach().cpu().numpy()
        model.fit(means)
        dists, match_ids = model.kneighbors(means)
        dists, match_ids = (
            torch.tensor(dists, dtype=torch.float32, device="cuda"),
            torch.tensor(match_ids, dtype=torch.long, device="cuda"),
        )
        dists, match_ids = dists[group_mask], match_ids[group_mask]
        # filter matches outside the radius
        match_ids[dists > r] = -1
        # filter out ones within same group mask
        match_ids[group_mask[match_ids]] = -1
        ids = (
            torch.arange(self.dig_model.num_points, dtype=torch.long, device="cuda")[
                group_mask
            ]
            .unsqueeze(-1)
            .repeat(1, self.config.N)
        )
        # flatten all the ids/dists/match_ids
        ids = ids[match_ids != -1].flatten()
        dists = dists[match_ids != -1].flatten()
        match_ids = match_ids[match_ids != -1].flatten()
        return dists, ids, match_ids, global_group_ids[ids], global_group_ids[match_ids]
