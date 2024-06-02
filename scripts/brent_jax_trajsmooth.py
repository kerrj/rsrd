"""Motion planning implementation for Yumi, in JAX.

Requires jaxfg:

    conda install -c conda-forge suitesparse

    cd ~
    git clone https://github.com/brentyi/jaxfg.git
    cd jaxfg
    pip install -e .
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import cast

import jax
import jax_dataclasses as jdc
import jaxfg
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
from tqdm.auto import tqdm
from typing_extensions import override

YUMI_REST_POSE = [
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
]


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
        assert Ts_world_joint.shape == (*batch_axes, self.num_joints, 7)
        return Ts_world_joint


@jdc.jit
def motion_plan_yumi(
    yumi_urdf: JaxUrdf,
    pose_target_from_joint_idx: dict[int, Float[Array, "timesteps 7"]],
    *,
    smooth_weight: float = 0.01,
    rest_prior_weight: float = 0.001,
    pos_weight: float = 50.0,
    ori_weight: float = 0.5,
    limits_weight: float = 100_000.0,
    limits_pad: float = 0.1,
) -> Array:
    """Smooth a trajectory, while holding the output frames of some set of
    joints fixed."""

    Vec16Variable = jaxfg.core.RealVectorVariable[16]

    timesteps, pose_dim = next(iter(pose_target_from_joint_idx.values())).shape
    assert pose_dim == 7  # wxyz, xyz
    qs_variables = [Vec16Variable() for _ in range(timesteps)]

    factors = list[jaxfg.core.FactorBase]()

    # Per-joint cost terms.
    yumi_rest_pose = jnp.array(YUMI_REST_POSE)

    @jdc.pytree_dataclass
    class RegFactor(jaxfg.core.FactorBase):
        @override
        def compute_residual_vector(self, variable_values: tuple[Array]) -> Array:
            (q_t,) = variable_values
            return q_t - yumi_rest_pose

    @jdc.pytree_dataclass
    class UpperLimitFactor(jaxfg.core.FactorBase):
        @override
        def compute_residual_vector(self, variable_values: tuple[Array]) -> Array:
            (q_t,) = variable_values
            return jnp.maximum(
                # We don't pad the gripper joint limits!
                0.0,
                q_t - (yumi_urdf.limits_upper.at[:14].add(-limits_pad)),
            )

    @jdc.pytree_dataclass
    class LowerLimitFactor(jaxfg.core.FactorBase):
        @override
        def compute_residual_vector(self, variable_values: tuple[Array]) -> Array:
            (q_t,) = variable_values
            return jnp.minimum(
                # We don't pad the gripper joint limits!
                0.0,
                q_t - (yumi_urdf.limits_lower.at[:14].add(limits_pad)),
            )

    for qs_var in qs_variables:
        factors.extend(
            [
                RegFactor(
                    variables=(qs_var,),
                    noise_model=jaxfg.noises.DiagonalGaussian(
                        jnp.ones(16) * rest_prior_weight
                    ),
                ),
                UpperLimitFactor(
                    variables=(qs_var,),
                    noise_model=jaxfg.noises.DiagonalGaussian(
                        jnp.ones(16) * limits_weight
                    ),
                ),
                LowerLimitFactor(
                    variables=(qs_var,),
                    noise_model=jaxfg.noises.DiagonalGaussian(
                        jnp.ones(16) * limits_weight
                    ),
                ),
            ]
        )

    # Smoothness term.
    @jdc.pytree_dataclass
    class SmoothnessFactor(jaxfg.core.FactorBase):
        @override
        def compute_residual_vector(
            self, variable_values: tuple[Array, Array]
        ) -> Array:
            (q_tm1, q_t) = variable_values
            return q_tm1 - q_t

    for i in range(1, len(qs_variables)):
        factors.append(
            SmoothnessFactor(
                variables=(qs_variables[i - 1], qs_variables[i]),
                noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(16) * smooth_weight),
            ),
        )

    # Inverse kinematics term.
    @jdc.pytree_dataclass
    class InverseKinematicsFactor(jaxfg.core.FactorBase):
        joint_index: Int[Array, ""]
        pose_target: Float[Array, "7"]

        @override
        def compute_residual_vector(self, variable_values: tuple[Array]) -> Array:
            (joints,) = variable_values
            assert self.joint_index.shape == ()
            Ts_world_joint = yumi_urdf.forward_kinematics(joints)
            assert Ts_world_joint.shape == (yumi_urdf.num_joints, 7)
            assert self.pose_target.shape == (7,)
            return (
                jaxlie.SE3(Ts_world_joint[self.joint_index]).inverse()
                @ jaxlie.SE3(self.pose_target)
            ).log()

    ik_weight = jnp.array([pos_weight] * 3 + [ori_weight] * 3)
    for joint_idx, pose_target in pose_target_from_joint_idx.items():
        assert pose_target.shape == (timesteps, 7)
        for timestep, qs_var in enumerate(qs_variables):
            factors.append(
                InverseKinematicsFactor(
                    variables=(qs_var,),
                    noise_model=jaxfg.noises.DiagonalGaussian(ik_weight),
                    joint_index=jnp.array(joint_idx),
                    pose_target=pose_target[timestep],
                ),
            )

    graph = jaxfg.core.StackedFactorGraph.make(factors, use_onp=False)
    solver = jaxfg.solvers.LevenbergMarquardtSolver(
        lambda_initial=0.1, gradient_tolerance=1e-5, parameter_tolerance=1e-5
    )
    assignments = jaxfg.core.VariableAssignments.make_from_dict(
        {q: yumi_rest_pose for q in qs_variables}
    )
    solved_assignments = solver.solve(graph, assignments)
    out = solved_assignments.get_stacked_value(Vec16Variable)
    assert out.shape == (timesteps, 16)
    return cast(Array, out)


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

    timesteps = traj.shape[0]
    orig_Ts_world_joint = jax_urdf.forward_kinematics(jnp.array(traj))
    assert orig_Ts_world_joint.shape == (timesteps, jax_urdf.num_joints, 7)

    start_time = time.time()
    opt_traj = onp.array(
        motion_plan_yumi(
            jax_urdf,
            pose_target_from_joint_idx={
                # These poses should have shape (timesteps, 7).
                6: orig_Ts_world_joint[:, 6, :],
                13: orig_Ts_world_joint[:, 13, :],
            },
        )
    )
    print(time.time() - start_time, "!!!!")
    start_time = time.time()

    opt_traj = onp.array(
        motion_plan_yumi(
            jax_urdf,
            pose_target_from_joint_idx={
                # These poses should have shape (timesteps, 7).
                6: orig_Ts_world_joint[:, 6, :],
                13: orig_Ts_world_joint[:, 13, :],
            },
        )
    )
    print(time.time() - start_time, "!!!!")

    if onp.any(opt_traj < jax_urdf.limits_lower) or onp.any(
        opt_traj > jax_urdf.limits_upper
    ):
        print("Violated joint limits!!")
    else:
        print("Trajectory is within joint limits!")

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

        Ts_world_joint = onp.array(jax_urdf.forward_kinematics(traj[slider.value]))
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
