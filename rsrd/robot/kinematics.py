"""
Differentiable robot kinematics model, implemented in JAX.
Includes:
 - URDF parsing
 - Forward kinematics
"""

# pylint: disable=invalid-name

from __future__ import annotations

import jax
import jax_dataclasses as jdc
import jaxlie
import yourdfpy
from loguru import logger

from jax import Array
from jax import numpy as jnp
from jaxtyping import Float, Int


@jdc.pytree_dataclass
class JaxKinTree:
    """A differentiable robot kinematics tree."""

    num_joints: jdc.Static[int]
    """Number of joints in the robot."""

    num_actuated_joints: jdc.Static[int]
    """Number of actuated joints in the robot."""

    joint_twists: Float[Array, "act_joints 6"]
    """Twist parameters for each actuated joint, for revolute and prismatic joints."""

    Ts_parent_joint: Float[Array, "joints 7"]
    """Transform from parent joint to current joint, in the format `joints 7`."""

    idx_parent_joint: Int[Array, "joints"]
    """Parent joint index for each joint. -1 for root."""

    idx_actuated_joint: Int[Array, "joints"]
    """Index of actuated joint for each joint, for handling mimic joints. -1 otherwise."""

    limits_lower: Float[Array, "act_joints"]
    """Lower joint limits for each actuated joint."""

    limits_upper: Float[Array, "act_joints"]
    """Upper joint limits for each actuated joint."""

    joint_names: jdc.Static[tuple[str]]
    """Names of the joints, in shape `joints`."""

    joint_vel_limit: Float[Array, "act_joints"]
    """Joint limit velocities for each actuated joint."""

    @staticmethod
    def from_urdf(urdf: yourdfpy.URDF) -> JaxKinTree:
        """Build a differentiable robot model from a URDF."""

        # Get the parent indices + joint twist parameters.
        joint_twists = list[Array]()
        Ts_parent_joint = list[Array]()
        idx_parent_joint = list[int]()
        idx_actuated_joint = list[int]()
        limits_lower = list[float]()
        limits_upper = list[float]()
        joint_names = list[str]()
        joint_vel_limits = list[float]()

        for joint_idx, joint in enumerate(urdf.joint_map.values()):
            # Get joint names.
            joint_names.append(joint.name)

            # Get the actuated joint index.
            act_idx = JaxKinTree._get_act_joint_idx(urdf, joint, joint_idx)
            idx_actuated_joint.append(act_idx)

            # Get the twist parameters for all actuated joints.
            if joint in urdf.actuated_joints:
                twist = JaxKinTree._get_act_joint_twist(joint)
                joint_twists.append(twist)

                # Get the joint limits.
                lower, upper = JaxKinTree._get_joint_limits(joint)
                limits_lower.append(lower)
                limits_upper.append(upper)

                # Get the joint velocities.
                joint_vel_limit = JaxKinTree._get_joint_limit_vel(joint)
                joint_vel_limits.append(joint_vel_limit)

            # Get the parent joint index and transform for each joint.
            parent_idx, T_parent_joint = JaxKinTree._get_T_parent_joint(urdf, joint, joint_idx)
            idx_parent_joint.append(parent_idx)
            Ts_parent_joint.append(T_parent_joint)

        joint_twists = jnp.array(joint_twists)
        Ts_parent_joint = jnp.array(Ts_parent_joint)
        idx_parent_joint = jnp.array(idx_parent_joint)
        idx_actuated_joint = jnp.array(idx_actuated_joint)
        num_joints = len(urdf.joint_map)
        num_actuated_joints = len(urdf.actuated_joints)
        limits_lower = jnp.array(limits_lower)
        limits_upper = jnp.array(limits_upper)
        joint_names = tuple[str](joint_names)
        joint_vel_limits = jnp.array(joint_vel_limits)

        assert idx_actuated_joint.shape == (len(urdf.joint_map),)
        assert joint_twists.shape == (num_actuated_joints, 6)
        assert Ts_parent_joint.shape == (num_joints, 7)
        assert idx_parent_joint.shape == (num_joints,)
        assert idx_actuated_joint.max() == num_actuated_joints - 1
        assert limits_lower.shape == (num_actuated_joints,)
        assert limits_upper.shape == (num_actuated_joints,)
        assert len(joint_names) == num_joints
        assert joint_vel_limits.shape == (num_actuated_joints,)

        return JaxKinTree(
            num_joints=num_joints,
            num_actuated_joints=num_actuated_joints,
            idx_actuated_joint=idx_actuated_joint,
            joint_twists=joint_twists,
            Ts_parent_joint=Ts_parent_joint,
            idx_parent_joint=idx_parent_joint,
            limits_lower=limits_lower,
            limits_upper=limits_upper,
            joint_names=joint_names,
            joint_vel_limit=joint_vel_limits,
        )

    @staticmethod
    def _get_act_joint_idx(urdf: yourdfpy.URDF, joint: yourdfpy.Joint, joint_idx: int) -> int:
        """Get the actuated joint index for a joint, checking for mimic joints."""
        # Check if this joint is a mimic joint -- assume multiplier=1.0, offset=0.0.
        if joint.mimic is not None:
            mimicked_joint = urdf.joint_map[joint.mimic.joint]
            mimicked_joint_idx = urdf.actuated_joints.index(mimicked_joint)
            assert mimicked_joint_idx < joint_idx, "Code + fk `fori_loop` assumes this!"
            logger.warning("Mimic joint detected.")
            act_joint_idx = urdf.actuated_joints.index(mimicked_joint)

        # Track joint twists for actuated joints.
        elif joint in urdf.actuated_joints:
            assert joint.axis.shape == (3,)
            act_joint_idx = urdf.actuated_joints.index(joint)

        # Not actuated.
        else:
            act_joint_idx = -1

        return act_joint_idx

    @staticmethod
    def _get_act_joint_twist(joint: yourdfpy.Joint) -> Array:
        """Get the twist parameters for an actuated joint."""
        if joint.type in ("revolute", "continuous"):
            twist = jnp.concatenate([jnp.zeros(3), joint.axis])
        elif joint.type == "prismatic":
            twist = jnp.concatenate([joint.axis, jnp.zeros(3)])
        else:
            raise ValueError(f"Unsupported joint type {joint.type}!")
        return twist

    @staticmethod
    def _get_T_parent_joint(
        urdf: yourdfpy.URDF,
        joint: yourdfpy.Joint,
        joint_idx: int,
    ) -> tuple[int, Array]:
        """Get the transform from the parent joint to the current joint,
        as well as the parent joint index."""
        assert joint.origin.shape == (4, 4)

        joint_from_child = {joint.child: joint for joint in urdf.joint_map.values()}

        T_parent_joint = joint.origin
        if joint.parent not in joint_from_child:
            # Must be root node.
            parent_index = -1
        else:
            parent_joint = joint_from_child[joint.parent]
            parent_index = urdf.joint_names.index(parent_joint.name)
            if parent_index >= joint_idx:
                logger.warning(
                    f"Parent index {parent_index} >= joint index {joint_idx}! " +
                    "Assuming that parent is root."
                )
                if parent_joint.parent != urdf.scene.graph.base_frame:
                    raise ValueError("Parent index >= joint_index, but parent is not root!")
                T_parent_joint = parent_joint.origin @ T_parent_joint  # T_root_joint.
                parent_index = -1

        return (
            parent_index,
            jaxlie.SE3.from_matrix(T_parent_joint).wxyz_xyz
        )

    @staticmethod
    def _get_joint_limits(joint: yourdfpy.Joint) -> tuple[float, float]:
        """Get the joint limits for an actuated joint, returns (lower, upper)."""
        assert joint.limit is not None
        if (
            joint.limit.lower is not None and
            joint.limit.upper is not None
        ):
            lower = joint.limit.lower
            upper = joint.limit.upper
        elif joint.type == "continuous":
            logger.warning("Continuous joint detected, cap to [-pi, pi] limits.")
            lower = -jnp.pi
            upper = jnp.pi
        else:
            raise ValueError("We currently assume there are joint limits!")
        return lower, upper

    @staticmethod
    def _get_joint_limit_vel(joint: yourdfpy.Joint) -> float:
        """Get the joint velocity for an actuated joint."""
        if joint.limit is not None and joint.limit.velocity is not None:
            return joint.limit.velocity
        logger.warning("Joint velocity not specified, defaulting to 1.0.")
        return 1.0

    @jdc.jit
    def forward_kinematics(
        self,
        cfg: Float[Array, "*batch num_act_joints"],
    ) -> Float[Array, "*batch num_joints 7"]:
        """
        Run forward kinematics on the robot, in the provided configuration.

        Args:
            cfg: The configuration of the actuated joints, in the format `(*batch num_act_joints)`.
        
        Returns:
            The SE(3) transforms of the joints, in the format `(*batch num_joints wxyz_xyz)`.
        """
        batch_axes = cfg.shape[:-1]
        assert cfg.shape == (*batch_axes, self.num_actuated_joints)

        Ts_joint_child = jaxlie.SE3.exp(self.joint_twists * cfg[..., None]).wxyz_xyz
        assert Ts_joint_child.shape == (*batch_axes, self.num_actuated_joints, 7)

        def compute_joint(i: int, Ts_world_joint: Array) -> Array:
            T_world_parent = jnp.where(
                self.idx_parent_joint[i] == -1,
                jnp.broadcast_to(jaxlie.SE3.identity().wxyz_xyz, (*batch_axes, 7)),
                Ts_world_joint[..., self.idx_parent_joint[i], :],
            )

            T_joint_child = jnp.where(
                self.idx_actuated_joint[i] != -1,
                Ts_joint_child[..., self.idx_actuated_joint[i], :],
                jnp.broadcast_to(jaxlie.SE3.identity().wxyz_xyz, (*batch_axes, 7)),
            )
            return Ts_world_joint.at[..., i, :].set(
                (
                    jaxlie.SE3(T_world_parent)
                    @ jaxlie.SE3(self.Ts_parent_joint[i])
                    @ jaxlie.SE3(T_joint_child)
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
