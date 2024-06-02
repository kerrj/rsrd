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

        list_Ts_world_joint = list[Tensor]()
        Ts_joint_child = SE3.exp(
            torch.broadcast_to(self.joint_twists, (*batch_axes, self.num_joints, 6))
            * cfg[..., None]
        ).as_matrix()
        assert Ts_joint_child.shape == (*batch_axes, self.num_joints, 4, 4)
        for i in range(self.num_joints):
            if self.parent_indices[i] == -1:
                list_Ts_world_joint.append(
                    torch.einsum(
                        "jk,...kl->...jl",
                        self.Ts_parent_joint[i],
                        Ts_joint_child[..., i, :, :],
                    )
                )
            else:
                T_world_parent = list_Ts_world_joint[self.parent_indices[i]]
                list_Ts_world_joint.append(
                    torch.einsum(
                        "...ij,jk,...kl->...il",
                        T_world_parent,
                        self.Ts_parent_joint[i],
                        Ts_joint_child[..., i, :, :],
                    )
                )

        Ts_world_joint = torch.stack(list_Ts_world_joint, dim=-3)
        assert Ts_world_joint.shape == (*batch_axes, self.num_joints, 4, 4)
        return Ts_world_joint


def motion_plan_yumi(
    yumi_urdf: TorchUrdf,
    pose_target_from_joint_idx: dict[int, Float[Tensor, "timesteps 4 4"]],
    *,
    lbfgs_iterations: int = 40,
    smooth_weight: float = 5.0,
    rest_prior_weight: float = 0.001,
    pos_weight: float = 50.0,
    ori_weight: float = 0.2,
    limits_weight: float = 500.0,
    limits_pad: float = 0.05,
) -> Float[Tensor, "timesteps num_joints"]:
    """Smooth a trajectory, while holding the output frames of some set of
    joints fixed."""
    proto = next(iter(pose_target_from_joint_idx.values()))
    timesteps = proto.shape[0]
    device = proto.device
    dtype = proto.dtype
    del proto

    # Parameters to optimize.
    num_joints = yumi_urdf.num_joints
    recovered_traj = torch.zeros(
        (timesteps, num_joints), device=device, dtype=dtype
    ).requires_grad_(True)
    with torch.no_grad():
        recovered_traj += torch.tensor(
            [
                1.21442839,
                -1.03205606,
                -1.10072738,
                0.2987352,
                -1.85257716,
                1.25363652,
                -2.42181893,
                -1.24839656,
                -1.09802876,
                1.06634394,
                0.31386161,
                1.90125141,
                1.3205139,
                2.43563939,
                0.0,
                0.0,
            ],
            device=device,
            dtype=dtype,
        )

    optim = torch.optim.LBFGS(
        [recovered_traj],
        max_iter=10,
        tolerance_grad=1e-4,
        tolerance_change=1e-5,
        line_search_fn="strong_wolfe",
    )

    loss_terms = dict[str, Float[Tensor, "num_seeds"]]()

    def closure():
        optim.zero_grad(set_to_none=True)
        assert recovered_traj.shape == (timesteps, num_joints)
        Ts_world_joint = yumi_urdf.forward_kinematics(recovered_traj)

        nonlocal loss_terms
        loss_terms = dict[str, Float[Tensor, "num_seeds"]]()

        # Smoothness term in joint space.
        loss_terms["smooth"] = smooth_weight * torch.sum(
            (recovered_traj[:-1, :] - recovered_traj[1:, :]) ** 2
        )
        loss_terms["prior"] = rest_prior_weight * torch.sum(recovered_traj**2)
        loss_terms["lim_low"] = limits_weight * torch.sum(
            torch.minimum(
                recovered_traj.new_zeros(1),
                recovered_traj - (yumi_urdf.limits_lower + limits_pad)[None, :],
            )
            ** 2
        )
        loss_terms["lim_up"] = limits_weight * torch.sum(
            torch.maximum(
                recovered_traj.new_zeros(1),
                recovered_traj - (yumi_urdf.limits_upper - limits_pad)[None, :],
            )
            ** 2
        )

        # Pose target losses.
        for joint_idx, target_pose in pose_target_from_joint_idx.items():
            loss_terms[f"pos_{joint_idx}"] = pos_weight * torch.sum(
                (target_pose[:, :3, 3] - Ts_world_joint[:, joint_idx, :3, 3]) ** 2
            )
            loss_terms[f"ori_{joint_idx}"] = ori_weight * torch.sum(
                (target_pose[:, :3, :3] - Ts_world_joint[:, joint_idx, :3, :3]) ** 2
            )

        loss = functools.reduce(torch.add, map(torch.sum, loss_terms.values()))
        assert loss.shape == ()
        pbar.set_description(str(loss.item()))
        loss.backward()
        return loss

    for _ in (pbar := tqdm(range(lbfgs_iterations))):
        optim.step(closure)  # type: ignore

        for label, loss in loss_terms.items():
            print(f"{label.ljust(20)} {loss.numpy(force=True)}")

    return recovered_traj.detach()


def main(
    traj_npy: Path,
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load the URDF.
    yourdf = load_robot_description("yumi_description")
    torch_urdf = TorchUrdf.from_urdf(yourdf, device=device)

    # Load the trajectory.
    raw_traj = np.load(traj_npy)
    traj = np.zeros((raw_traj.shape[0], 16), dtype=np.float32)
    traj[:, :7] = torch.from_numpy(raw_traj[:, 7:14])
    traj[:, 7:14] = torch.from_numpy(raw_traj[:, :7])
    del raw_traj
    timesteps = traj.shape[0]

    # Forward kinematics for original trajectory.
    orig_Ts_world_joint = torch_urdf.forward_kinematics(
        torch.from_numpy(traj).to(device)
    )
    assert orig_Ts_world_joint.shape == (traj.shape[0], torch_urdf.num_joints, 4, 4)

    # Run motion planning optimization.
    opt_traj = motion_plan_yumi(
        torch_urdf,
        pose_target_from_joint_idx={
            # These poses should have shape (timesteps, 4, 4).
            6: orig_Ts_world_joint[:, 6, :, :],
            13: orig_Ts_world_joint[:, 13, :, :],
        },
    )
    assert opt_traj.shape == (timesteps, torch_urdf.num_joints)
    opt_traj = opt_traj.numpy(force=True)

    # Visualize two robots: original and smoothed.
    server = viser.ViserServer()
    urdf_orig = viser.extras.ViserUrdf(
        server, yourdf, mesh_color_override=(220, 100, 100), root_node_name="/urdf_orig"
    )
    urdf_smoothed = viser.extras.ViserUrdf(
        server, yourdf, root_node_name="/urdf_smoothed"
    )

    # Visualize!
    slider = server.gui.add_slider(
        "Timestep", min=0, max=timesteps - 1, step=1, initial_value=0
    )

    @slider.on_update
    def _(_) -> None:
        urdf_orig.update_cfg(traj[slider.value])
        urdf_smoothed.update_cfg(opt_traj[slider.value])

    playing = server.gui.add_checkbox("Playing", initial_value=True)

    while True:
        if playing.value:
            slider.value = (slider.value + 1) % timesteps
        time.sleep(1.0 / 10.0)


if __name__ == "__main__":
    tyro.cli(main)
