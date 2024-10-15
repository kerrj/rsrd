"""Motion planning implementation for Yumi, in JAX.

Requires jaxfg:

    conda install -c conda-forge suitesparse

    cd ~
    git clone https://github.com/brentyi/jaxfg.git
    cd jaxfg
    pip install -e .
"""

from __future__ import annotations

from typing import cast
import jax_dataclasses as jdc
import jaxfg
import jaxlie
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float, Int
from typing_extensions import override

from rsrd.robot.kinematics import JaxKinTree

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

@jdc.jit
def ik_yumi(
    yumi_urdf: JaxKinTree,
    ind_targets: Int[Array, "inds"],
    pose_target: Float[Array, "inds 7"],
    *,
    rest_prior_weight: float = 0.001,
    pos_weight: float = 50.0,
    ori_weight: float = 0.5,
    limits_weight: float = 100_000.0,
    limits_pad: float = 0.1,
) -> Array:
    Vec16Variable = jaxfg.core.RealVectorVariable[16]

    _, pose_dim = pose_target.shape
    assert pose_dim == 7  # wxyz, xyz
    qs_var = Vec16Variable()

    factors = list[jaxfg.core.FactorBase]()

    # Per-joint cost terms.
    yumi_rest_pose = jnp.array(YUMI_REST_POSE)

    factors.extend(
        [
            RegFactor(
                variables=(qs_var,),
                noise_model=jaxfg.noises.DiagonalGaussian(
                    jnp.ones(16) * rest_prior_weight
                ),
                yumi_rest_pose=yumi_rest_pose,
            ),
            UpperLimitFactor(
                variables=(qs_var,),
                noise_model=jaxfg.noises.DiagonalGaussian(
                    jnp.ones(16) * limits_weight
                ),
                limits_pad=limits_pad,
                limits_upper=yumi_urdf.limits_upper,
            ),
            LowerLimitFactor(
                variables=(qs_var,),
                noise_model=jaxfg.noises.DiagonalGaussian(
                    jnp.ones(16) * limits_weight
                ),
                limits_pad=limits_pad,
                limits_lower=yumi_urdf.limits_lower,
            ),
        ]
    )

    ik_weight = jnp.array([pos_weight] * 3 + [ori_weight] * 3)
    for joint_idx, pose_target in zip(ind_targets, pose_target):
        assert pose_target.shape == (7,)
        factors.append(
            InverseKinematicsFactor(
                variables=(qs_var,),
                noise_model=jaxfg.noises.DiagonalGaussian(ik_weight),
                joint_index=jnp.array(joint_idx),
                pose_target=pose_target,
                yumi_urdf=yumi_urdf,
            ),
        )

    graph = jaxfg.core.StackedFactorGraph.make(factors, use_onp=False)
    solver = jaxfg.solvers.LevenbergMarquardtSolver(
        lambda_initial=0.1, gradient_tolerance=1e-5, parameter_tolerance=1e-5
    )
    assignments = jaxfg.core.VariableAssignments.make_from_dict(
        {qs_var: yumi_rest_pose}
    )
    solved_assignments = solver.solve(graph, assignments)
    out = solved_assignments.get_stacked_value(Vec16Variable)
    assert out.shape == (1, 16)
    return cast(Array, out)



@jdc.jit
def motion_plan_yumi(
    yumi_urdf: JaxKinTree,
    ind_targets: Int[Array, "inds"],
    pose_target: Float[Array, "inds timesteps 7"],
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

    _, timesteps, pose_dim = pose_target.shape
    assert pose_dim == 7  # wxyz, xyz
    qs_variables = [Vec16Variable() for _ in range(timesteps)]

    factors = list[jaxfg.core.FactorBase]()

    # Per-joint cost terms.
    yumi_rest_pose = jnp.array(YUMI_REST_POSE)

    for qs_var in qs_variables:
        factors.extend(
            [
                RegFactor(
                    variables=(qs_var,),
                    noise_model=jaxfg.noises.DiagonalGaussian(
                        jnp.ones(16) * rest_prior_weight
                    ),
                    yumi_rest_pose=yumi_rest_pose,
                ),
                UpperLimitFactor(
                    variables=(qs_var,),
                    noise_model=jaxfg.noises.DiagonalGaussian(
                        jnp.ones(16) * limits_weight
                    ),
                    limits_pad=limits_pad,
                    limits_upper=yumi_urdf.limits_upper,
                ),
                LowerLimitFactor(
                    variables=(qs_var,),
                    noise_model=jaxfg.noises.DiagonalGaussian(
                        jnp.ones(16) * limits_weight
                    ),
                    limits_pad=limits_pad,
                    limits_lower=yumi_urdf.limits_lower,
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

    ik_weight = jnp.array([pos_weight] * 3 + [ori_weight] * 3)
    for joint_idx, pose_target in zip(ind_targets, pose_target):
        assert pose_target.shape == (timesteps, 7)
        for timestep, qs_var in enumerate(qs_variables):
            factors.append(
                InverseKinematicsFactor(
                    variables=(qs_var,),
                    noise_model=jaxfg.noises.DiagonalGaussian(ik_weight),
                    joint_index=jnp.array(joint_idx),
                    pose_target=pose_target[timestep],
                    yumi_urdf=yumi_urdf,
                ),
            )

    graph = jaxfg.core.StackedFactorGraph.make(factors, use_onp=False)
    solver = jaxfg.solvers.LevenbergMarquardtSolver(
        lambda_initial=0.1, gradient_tolerance=1e-5, parameter_tolerance=1e-5, verbose=False
    )
    assignments = jaxfg.core.VariableAssignments.make_from_dict(
        {q: yumi_rest_pose for q in qs_variables}
    )
    solved_assignments = solver.solve(graph, assignments)
    out = solved_assignments.get_stacked_value(Vec16Variable)
    assert out.shape == (timesteps, 16)
    return cast(Array, out)


@jdc.pytree_dataclass
class RegFactor(jaxfg.core.FactorBase):
    yumi_rest_pose: Array
    @override
    def compute_residual_vector(self, variable_values: tuple[Array]) -> Array:
        (q_t,) = variable_values
        return q_t - self.yumi_rest_pose

@jdc.pytree_dataclass
class UpperLimitFactor(jaxfg.core.FactorBase):
    limits_pad: float
    limits_upper: Array

    @override
    def compute_residual_vector(self, variable_values: tuple[Array]) -> Array:
        (q_t,) = variable_values
        return jnp.maximum(
            # We don't pad the gripper joint limits!
            0.0,
            q_t - (self.limits_upper.at[:14].add(-self.limits_pad)),
        )

@jdc.pytree_dataclass
class LowerLimitFactor(jaxfg.core.FactorBase):
    limits_pad: float
    limits_lower: Array

    @override
    def compute_residual_vector(self, variable_values: tuple[Array]) -> Array:
        (q_t,) = variable_values
        return jnp.minimum(
            # We don't pad the gripper joint limits!
            0.0,
            q_t - (self.limits_lower.at[:14].add(self.limits_pad)),
        )

# Inverse kinematics term.
@jdc.pytree_dataclass
class InverseKinematicsFactor(jaxfg.core.FactorBase):
    yumi_urdf: JaxKinTree
    joint_index: Int[Array, ""]
    pose_target: Float[Array, "7"]

    @override
    def compute_residual_vector(self, variable_values: tuple[Array]) -> Array:
        (joints,) = variable_values
        assert self.joint_index.shape == ()
        Ts_world_joint = self.yumi_urdf.forward_kinematics(joints)
        assert Ts_world_joint.shape == (self.yumi_urdf.num_joints, 7)
        assert self.pose_target.shape == (7,)
        return (
            jaxlie.SE3(Ts_world_joint[self.joint_index]).inverse()
            @ jaxlie.SE3(self.pose_target)
        ).log()

# Smoothness term.
@jdc.pytree_dataclass
class SmoothnessFactor(jaxfg.core.FactorBase):
    @override
    def compute_residual_vector(
        self, variable_values: tuple[Array, Array]
    ) -> Array:
        (q_tm1, q_t) = variable_values
        return q_tm1 - q_t