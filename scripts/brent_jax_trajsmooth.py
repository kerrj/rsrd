from __future__ import annotations

import functools
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax_dataclasses as jdc
import jaxlie
import numpy as onp
import tyro
import viser
import viser.extras
import viser.transforms as vtf
import yourdfpy
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float, Int
from robot_descriptions.loaders.yourdfpy import load_robot_description
from toad.transforms import SE3
from tqdm.auto import tqdm


@jdc.pytree_dataclass
class JaxUrdf:
    """A differentiable robot kinematics model."""

    num_joints: jdc.Static[int]
    joint_twists: Float[Array, "joints 6"]
    Ts_parent_joint: Float[Array, "joints 7"]
    limits_lower: Float[Array, "joints"]
    limits_upper: Float[Array, "joints"]
    parent_indices: Int[Array, "joints"]

    @staticmethod
    def from_urdf(
        urdf: yourdfpy.URDF,
    ) -> JaxUrdf:
        """Build a differentiable robot model from a URDF."""

        joint_from_child = {joint.child: joint for joint in urdf.joint_map.values()}

        # Get the parent indices + joint twist parameters.
        joint_twists = list[onp.ndarray]()
        Ts_parent_joint = list[onp.ndarray]()
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
                joint_twists.append(onp.concatenate([onp.zeros(3), joint.axis]))
            elif joint.type == "prismatic":
                joint_twists.append(onp.concatenate([joint.axis, onp.zeros(3)]))
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
            Ts_parent_joint.append(vtf.SE3.from_matrix(T_parent_joint).wxyz_xyz)  # type: ignore
            parent_indices.append(
                -1 if root else urdf.actuated_joints.index(parent_joint)
            )

        joint_twists = jnp.array(joint_twists)
        return JaxUrdf(
            num_joints=len(parent_indices),
            joint_twists=joint_twists,
            Ts_parent_joint=jnp.array(Ts_parent_joint),
            limits_lower=jnp.array(joint_lim_lower),
            limits_upper=jnp.array(joint_lim_upper),
            parent_indices=jnp.array(parent_indices),
        )

    @jdc.jit
    def forward_kinematics(
        self,
        cfg: Float[Array, "*batch num_joints"],
    ) -> Float[Array, "*batch num_joints 7"]:
        batch_axes = cfg.shape[:-1]
        assert cfg.shape == (*batch_axes, self.num_joints)

        Ts_joint_child = jaxlie.SE3.exp(self.joint_twists * cfg[..., None]).wxyz_xyz
        assert Ts_joint_child.shape == (*batch_axes, self.num_joints, 7)

        def compute_joint(i: int, Ts_world_joint: Array) -> Array:
            T_world_parent = jnp.where(
                self.parent_indices[i] == -1,
                jnp.broadcast_to(jaxlie.SE3.identity().wxyz_xyz, (*batch_axes, 7)),
                Ts_world_joint[..., self.parent_indices[i], :],
            )
            return Ts_world_joint.at[..., i, :].set(
                (
                    jaxlie.SE3(T_world_parent)
                    @ jaxlie.SE3(self.Ts_parent_joint[i])
                    @ jaxlie.SE3(Ts_joint_child[..., i, :])
                ).wxyz_xyz
            )

        Ts_world_joint = jax.lax.fori_loop(
            lower=0,
            upper=self.num_joints,
            body_fun=compute_joint,
            init_val=jnp.zeros((*batch_axes, self.num_joints, 7)),
        )
        assert Ts_world_joint.shape == (self.num_joints, 7)
        return Ts_world_joint


def main(
    traj_npy: Path,
) -> None:
    # Load the URDF.
    yourdf = load_robot_description("yumi_description")
    jax_urdf = JaxUrdf.from_urdf(yourdf)

    # Load the trajectory.
    raw_traj = onp.load(traj_npy)
    traj = onp.zeros((raw_traj.shape[0], 16))
    traj[:, :7] = raw_traj[:, 7:14]
    traj[:, 7:14] = raw_traj[:, :7]
    del raw_traj

    print("Running FK...")
    Ts_world_joint = onp.array(jax_urdf.forward_kinematics(traj[0]))
    print(Ts_world_joint.shape)
    # Visualize two robots: original and smoothed.
    server = viser.ViserServer()
    urdf_orig = viser.extras.ViserUrdf(
        server, yourdf, mesh_color_override=(220, 100, 100), root_node_name="/urdf_orig"
    )

    # Visualize!
    timesteps = traj.shape[0]
    slider = server.gui.add_slider(
        "Timestep", min=0, max=timesteps - 1, step=1, initial_value=0
    )

    @slider.on_update
    def _(_) -> None:
        urdf_orig.update_cfg(traj[slider.value])

        start = time.time()
        Ts_world_joint = onp.array(jax_urdf.forward_kinematics(traj[slider.value]))
        print(time.time() - start)
        for i in range(Ts_world_joint.shape[0]):
            server.scene.add_frame(
                f"/joints/{i}",
                wxyz=Ts_world_joint[i, :4],
                position=Ts_world_joint[i, 4:7],
                axes_length=0.1,
                axes_radius=0.01,
            )

    playing = server.gui.add_checkbox("Playing", initial_value=True)

    while True:
        if playing.value:
            slider.value = (slider.value + 1) % timesteps
        time.sleep(1.0 / 10.0)


if __name__ == "__main__":
    tyro.cli(main)
