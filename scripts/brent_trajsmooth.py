from __future__ import annotations

import functools
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
import viser
import viser.extras
import viser.transforms as vtf
import yourdfpy
from jaxtyping import Float
from robot_descriptions.loaders.yourdfpy import load_robot_description
from toad.transforms import SE3
from torch import Tensor
from tqdm.auto import tqdm


@dataclass(frozen=True)
class TorchUrdf:
    """A differentiable robot kinematics model."""

    num_joints: int
    joint_twists: Float[Tensor, "joints 6"]
    Ts_parent_joint: Float[Tensor, "joints 4 4"]
    limits_lower: Float[Tensor, "joints"]
    limits_upper: Float[Tensor, "joints"]
    parent_indices: tuple[int, ...]

    @staticmethod
    def from_urdf(
        urdf: yourdfpy.URDF,
        device: torch.device | str = "cpu",
    ) -> TorchUrdf:
        """Build a differentiable robot model from a URDF."""

        joint_from_child = {joint.child: joint for joint in urdf.joint_map.values()}

        # Get the parent indices + joint twist parameters.
        joint_twists = list[np.ndarray]()
        Ts_parent_joint = list[np.ndarray]()
        joint_lim_lower = list[float]()
        joint_lim_upper = list[float]()
        parent_indices = list[int]()
        for joint in urdf.actuated_joints:
            assert joint.origin.shape == (4, 4)
            assert joint.axis.shape == (3,)
            assert (
                joint.limit is not None
                and joint.limit.lower is not None
                and joint.limit.upper is not None
            ), "We currently assume there are joint limits!"
            joint_lim_lower.append(joint.limit.lower)
            joint_lim_upper.append(joint.limit.upper)

            # We use twists in the (v, omega) convention.
            if joint.type == "revolute":
                joint_twists.append(np.concatenate([np.zeros(3), joint.axis]))
            elif joint.type == "prismatic":
                joint_twists.append(np.concatenate([joint.axis, np.zeros(3)]))
            else:
                assert False

            # Get the transform from the parent joint to the current joint.
            # The loop is required to take unactuated joints into account.
            T_parent_joint = joint.origin
            parent_joint = joint_from_child[joint.parent]
            root = False
            while parent_joint not in urdf.actuated_joints:
                T_parent_joint = parent_joint.origin @ T_parent_joint
                if parent_joint.parent not in joint_from_child:
                    root = True
                    break
                parent_joint = joint_from_child[parent_joint.parent]
            Ts_parent_joint.append(T_parent_joint)
            parent_indices.append(
                -1 if root else urdf.actuated_joints.index(parent_joint)
            )

        joint_twists = torch.tensor(
            np.array(joint_twists), dtype=torch.float32, device=device
        )
        return TorchUrdf(
            num_joints=len(parent_indices),
            joint_twists=joint_twists,
            Ts_parent_joint=joint_twists.new_tensor(
                np.array(Ts_parent_joint),
            ),
            limits_lower=joint_twists.new_tensor(
                np.array(joint_lim_lower),
            ),
            limits_upper=joint_twists.new_tensor(
                np.array(joint_lim_upper),
            ),
            parent_indices=tuple(parent_indices),
        )

    def forward_kinematics(
        self,
        cfg: Float[Tensor, "*batch num_joints"],
    ) -> Float[Tensor, "*batch num_joints 4 4"]:
        batch_axes = cfg.shape[:-1]
        assert cfg.shape == (*batch_axes, self.num_joints)

        device = self.Ts_parent_joint.device
        dtype = self.Ts_parent_joint.dtype

        list_Ts_world_joint = list[Tensor]()
        for i in range(self.num_joints):
            if self.parent_indices[i] == -1:
                T_world_parent = torch.broadcast_to(
                    torch.eye(4, device=device, dtype=dtype), (*batch_axes, 4, 4)
                )
            else:
                T_world_parent = list_Ts_world_joint[self.parent_indices[i]]
            list_Ts_world_joint.append(
                torch.einsum(
                    "...ij,jk,...kl->...il",
                    T_world_parent,
                    self.Ts_parent_joint[i],
                    SE3.exp(
                        torch.broadcast_to(self.joint_twists[i], (*batch_axes, 6))
                        * cfg[..., i, None]
                    ).as_matrix(),
                )
            )

        Ts_world_joint = torch.stack(list_Ts_world_joint, dim=-3)
        assert Ts_world_joint.shape == (*batch_axes, self.num_joints, 4, 4)
        return Ts_world_joint


def smooth_trajectory(
    torch_urdf: TorchUrdf,
    traj: Float[Tensor, "timesteps num_joints"],
    hold_indices: tuple[int, ...],
    *,
    smooth_weight: float = 5.0,
    rest_prior_weight: float = 0.001,
    hold_pos_weight: float = 50.0,
    hold_ori_weight: float = 0.2,
    limits_weight: float = 500.0,
    limits_pad: float = 0.05,
    iterations: int = 20,
) -> Float[Tensor, "timesteps num_joints"]:
    """Smooth a trajectory, while holding the output frames of some set of
    joints fixed."""
    traj = traj.detach().clone().requires_grad_(False)
    (timesteps, num_joints) = traj.shape
    assert num_joints == torch_urdf.num_joints

    # Parameters to optimize.
    # We're going to hold the first and final joint positions the same.
    # offsets = traj.new_zeros((timesteps - 2, num_joints)).requires_grad_(True)
    updated_traj = traj.new_zeros((timesteps, num_joints)).requires_grad_(True)
    optim = torch.optim.LBFGS(
        [updated_traj],
        history_size=20,
        max_iter=10,
        tolerance_grad=1e-4,
        tolerance_change=1e-5,
        line_search_fn="strong_wolfe",
    )

    # Original forward kinematics. We'll use this to hold some joint poses still.
    orig_Ts_world_joint = torch_urdf.forward_kinematics(traj).detach()
    assert orig_Ts_world_joint.shape == (timesteps, torch_urdf.num_joints, 4, 4)

    def closure():
        optim.zero_grad(set_to_none=True)
        assert updated_traj.shape == (timesteps, num_joints)
        Ts_world_joint = torch_urdf.forward_kinematics(updated_traj)

        loss_terms = dict[str, Tensor]()

        # Smoothness term in joint space.
        loss_terms["smooth"] = smooth_weight * torch.sum(
            (updated_traj[:-1, :] - updated_traj[1:, :]) ** 2
        )
        loss_terms["prior"] = rest_prior_weight * torch.sum(updated_traj**2)
        loss_terms["lim_low"] = limits_weight * torch.sum(
            torch.minimum(
                torch.zeros(1),
                updated_traj - (torch_urdf.limits_lower - limits_pad),
            )
            ** 2
        )
        loss_terms["lim_up"] = limits_weight * torch.sum(
            torch.maximum(
                torch.zeros(1),
                updated_traj - (torch_urdf.limits_upper + limits_pad),
            )
            ** 2
        )

        # Hold some set of joints in place.
        for hold_joint in hold_indices:
            loss_terms[f"pos_{hold_joint}"] = hold_pos_weight * torch.sum(
                (
                    orig_Ts_world_joint[:, hold_joint, :3, 3]
                    - Ts_world_joint[:, hold_joint, :3, 3]
                )
                ** 2
            )
            loss_terms[f"ori_{hold_joint}"] = hold_ori_weight * torch.sum(
                (
                    orig_Ts_world_joint[:, hold_joint, :3, :3]
                    - Ts_world_joint[:, hold_joint, :3, :3]
                )
                ** 2
            )

        pbar.set_description(
            " ".join(f"{k}: {v.item():.5f}" for k, v in loss_terms.items())
        )
        loss = functools.reduce(torch.add, loss_terms.values())
        loss.backward()
        return loss

    for _ in (pbar := tqdm(range(iterations))):
        optim.step(closure)  # type: ignore

    return updated_traj.detach()


def main(
    traj_npy: Path,
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load the URDF.
    yourdf = load_robot_description("yumi_description")
    torch_urdf = TorchUrdf.from_urdf(yourdf, device=device)

    # Load the trajectory.
    raw_traj = np.load(traj_npy)
    traj = np.zeros((raw_traj.shape[0], 16))
    traj[:, :7] = raw_traj[:, 7:14]
    traj[:, 7:14] = raw_traj[:, :7]
    del raw_traj

    # Visualize two robots: original and smoothed.
    server = viser.ViserServer()
    urdf_orig = viser.extras.ViserUrdf(
        server, yourdf, mesh_color_override=(220, 100, 100), root_node_name="/urdf_orig"
    )
    urdf_smoothed = viser.extras.ViserUrdf(
        server, yourdf, root_node_name="/urdf_smoothed"
    )

    # Smooth the trajectory.
    smoothed_traj = smooth_trajectory(
        torch_urdf,
        torch.from_numpy(traj.astype(np.float32)).to(device),
        hold_indices=(6, 13),
    ).numpy(force=True)

    # Visualize!
    timesteps = traj.shape[0]
    slider = server.gui.add_slider(
        "Timestep", min=0, max=timesteps - 1, step=1, initial_value=0
    )

    @slider.on_update
    def _(_) -> None:
        urdf_orig.update_cfg(traj[slider.value])
        urdf_smoothed.update_cfg(smoothed_traj[slider.value])

    playing = server.gui.add_checkbox("Playing", initial_value=True)

    while True:
        if playing.value:
            slider.value = (slider.value + 1) % timesteps
        time.sleep(1.0 / 10.0)


if __name__ == "__main__":
    tyro.cli(main)
