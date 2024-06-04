import yourdfpy
from typing import Literal
import torch
import viser.transforms as vtf
from jax import numpy as jnp
import numpy as np

from robot_descriptions.loaders.yourdfpy import load_robot_description
from toad.yumi.brent_jax_trajsmooth import JaxUrdf, motion_plan_yumi

class YumiJaxPlanner:
    jax_urdf: JaxUrdf

    def __init__(
        self,
        yourdf: yourdfpy.URDF,
    ):
        # yourdf = load_robot_description("yumi_description")
        self.jax_urdf = JaxUrdf.from_urdf(yourdf)

    def plan_from_waypoints(
        self,
        left_poses: torch.Tensor,
        right_poses: torch.Tensor,
    ) -> torch.Tensor:
        """
        poses in shape (batch, timesteps, 7) -- wxyz_xyz.
        returns joint angles in shape (batch, timesteps, 14).
        unfortunately, this can't be batched, or with batching the perf will be worse because of 2nd order opt.
        """
        # # all z poses should be offset by 0.1...
        # right_poses[:, :, 6] += 0.1
        # left_poses[:, :, 6] += 0.1
        # This can't be batched, or with batching the perf will be worse because of 2nd order opt.
        opt_traj_list = []
        device = right_poses.device
        for i in range(right_poses.shape[0]):
            opt_traj = motion_plan_yumi(
                self.jax_urdf,
                pose_target_from_joint_idx={
                    # These poses should have shape (timesteps, 7).
                    6: jnp.array(right_poses[i].cpu()),
                    13: jnp.array(left_poses[i].cpu()),
                },
            )
            opt_traj_tensor = torch.from_numpy(np.array(opt_traj))
            opt_traj_list.append(opt_traj_tensor)
        opt_traj_tensor = torch.stack(opt_traj_list, dim=0).to(device)
        assert opt_traj_tensor.shape == (right_poses.shape[0], right_poses.shape[1], 16)
        return opt_traj_tensor