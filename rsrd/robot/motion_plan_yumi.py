"""Motion planning implementation for Yumi, in JAX.

Requires jaxfg:

    conda install -c conda-forge suitesparse

    cd ~
    git clone https://github.com/brentyi/jaxfg.git
    cd jaxfg
    pip install -e .
"""

from __future__ import annotations

import jax
import jaxlie
import jax_dataclasses as jdc
from jax import Array, numpy as jnp
from jaxmp.kinematics import JaxKinTree
from jaxmp.jaxls.robot_factors import RobotFactors
import jaxls
from jaxtyping import Float

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
    0.025,
    0.025,
]


@jdc.jit
def motion_plan_yumi(
    kin: JaxKinTree,
    target_joint_inds: Float[Array, "inds"],
    target_ee_per_ind: Float[Array, "inds timestep 7"],
    rest_pose: Array,
    pos_weight: jdc.Static[float] = 5.0,
    rot_weight: jdc.Static[float] = 1.0,
    limit_weight: jdc.Static[float] = 100.0,
    rest_weight: jdc.Static[float] = 0.01,
    smoothness_weight: jdc.Static[float] = 1.0,
) -> tuple[Array, Array]:
    _, timesteps, pose_dim = target_ee_per_ind.shape
    assert pose_dim == 7

    # Create factor graph.
    JointVar = RobotFactors.get_var_class(kin, default_val=rest_pose)
    traj_vars = [JointVar(id=i) for i in range(timesteps)]

    factors = []
    for tstep in range(timesteps):
        for idx in target_joint_inds:
            factors.extend(
                [
                    jaxls.Factor.make(
                        RobotFactors.ik_cost,
                        (
                            kin,
                            traj_vars[tstep],
                            jaxlie.SE3(target_ee_per_ind[idx][tstep]),
                            idx,
                            jnp.array([pos_weight] * 3 + [rot_weight] * 3),
                        ),
                    ),
                ]
            )
        factors.extend(
            [
                jaxls.Factor.make(
                    RobotFactors.limit_cost,
                    (
                        kin,
                        traj_vars[tstep],
                        jnp.array([limit_weight] * kin.num_actuated_joints),
                    ),
                ),
                jaxls.Factor.make(
                    RobotFactors.rest_cost,
                    (
                        traj_vars[tstep],
                        jnp.array([rest_weight] * kin.num_actuated_joints),
                    ),
                ),
            ]
        )
        if tstep > 0:
            factors.append(
                jaxls.Factor.make(
                    RobotFactors.smoothness_cost,
                    (
                        traj_vars[tstep],
                        traj_vars[tstep - 1],
                        jnp.array([smoothness_weight] * kin.num_actuated_joints),
                    ),
                )
            )

    graph = jaxls.FactorGraph.make(
        factors,
        traj_vars,
        use_onp=False,
    )
    solution = graph.solve(
        initial_vals=jaxls.VarValues.make(traj_vars),
        linear_solver="conjugate_gradient",
        trust_region=jaxls.TrustRegionConfig(lambda_initial=0.1),
        termination=jaxls.TerminationConfig(
            gradient_tolerance=1e-5, parameter_tolerance=1e-5, max_iterations=50,
        ),
        verbose=False,
    )
    traj = jnp.array([solution[var] for var in traj_vars])

    # Check success for all joint indices.
    def check_success(traj, target_joint_inds, target_ee_per_ind, kin):
        def body_fun(i, success):
            idx = target_joint_inds[i]
            pose_from_joints = kin.forward_kinematics(traj)[:, idx]
            pose_target = target_ee_per_ind[i]
            succ = jnp.all(
                jnp.isclose(
                    jaxlie.SE3(pose_target).as_matrix(),
                    jaxlie.SE3(pose_from_joints).as_matrix(),
                    atol=1e-2
                )
            )
            return jnp.logical_and(success, succ)

        success = jnp.ones((), dtype=bool)
        success = jax.lax.fori_loop(0, len(target_joint_inds), body_fun, success)
        return success

    success = check_success(traj, target_joint_inds, target_ee_per_ind, kin)

    return traj, jnp.array([success])