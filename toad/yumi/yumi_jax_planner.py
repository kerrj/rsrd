import yourdfpy
from typing import Literal, Tuple
import torch
import viser.transforms as vtf
from jax import numpy as jnp
import numpy as np
from yourdfpy import URDF
from lxml import etree
from pathlib import Path
from copy import deepcopy
import tqdm

from toad.yumi.brent_jax_trajsmooth import JaxUrdf, motion_plan_yumi_arm
from toad.yumi.yumi_arm_planner import YUMI_REST_POSE_LEFT, YUMI_REST_POSE_RIGHT

class YumiJaxPlanner:
    jax_urdf_left: JaxUrdf
    jax_urdf_right: JaxUrdf

    def __init__(
        self,
        yourdf_path: Path,
    ):
        # yourdf = load_robot_description("yumi_description")
        parser = etree.XMLParser(remove_blank_text=True)
        tree = etree.parse(yourdf_path, parser=parser)
        xml_root = tree.getroot()

        # Remove comments
        etree.strip_tags(xml_root, etree.Comment)
        etree.cleanup_namespaces(xml_root) # type: ignore
        robot = URDF._parse_robot(xml_element=xml_root)

        # basically, go through all the joints, and if they're either left or right, set them to fixed.
        _robot = deepcopy(robot)
        for joints in _robot.joints:
            if joints.name in YUMI_REST_POSE_LEFT.keys():
                joints.type = "fixed"
        self.jax_urdf_right = JaxUrdf.from_urdf(URDF(robot=_robot, load_meshes=False))

        _robot = deepcopy(robot)
        for joints in _robot.joints:
            if joints.name in YUMI_REST_POSE_RIGHT.keys():
                joints.type = "fixed"
        self.jax_urdf_left = JaxUrdf.from_urdf(URDF(robot=_robot, load_meshes=False))


    def plan_from_waypoints(
        self,
        poses: torch.Tensor,
        arm: Literal["left", "right"]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        opt_succ_list = []
        device = poses.device
        urdf = self.jax_urdf_right if arm == "right" else self.jax_urdf_left

        # for i in tqdm.trange(poses.shape[0], desc="Waypoint opt"):
        for i in range(poses.shape[0]):
            # smooth the poses first.
            mat = vtf.SE3(poses[i].cpu().numpy()).as_matrix()  # (timesteps, 4, 4)
            _mat = mat.copy()
            for i in range(4):
                _mat = (
                    1*np.roll(_mat, 2, axis=0) + 
                    4*np.roll(_mat, 1, axis=0) + 
                    6*_mat + 
                    4*np.roll(_mat, -1, axis=0) +
                    1*np.roll(_mat, -2, axis=0)
                )/16
                _mat[0], _mat[-1] = mat[0], mat[-1]
                # make the rotations normal again, using svd.
                rotmat = _mat[:, :3, :3]
                u, s, vh = np.linalg.svd(rotmat)
                rotmat = u @ vh
                _mat[:, :3, :3] = rotmat
            pose = vtf.SE3.from_matrix(_mat).wxyz_xyz

            opt_traj, opt_succ = motion_plan_yumi_arm(
                urdf,
                pose_target_from_joint_idx={
                    # These poses should have shape (timesteps, 7).
                    # 6: jnp.array(poses[i].cpu()),
                    6: jnp.array(pose)
                },
                arm=arm,
            )
            opt_traj_tensor = torch.from_numpy(np.array(opt_traj))
            opt_succ_tensor = torch.tensor(np.array(opt_succ)).squeeze()
            opt_traj_list.append(opt_traj_tensor)
            opt_succ_list.append(opt_succ_tensor)
        opt_traj_tensor = torch.stack(opt_traj_list, dim=0).to(device)
        opt_succ_tensor = torch.stack(opt_succ_list, dim=0).to(device)
        assert opt_traj_tensor.shape == (poses.shape[0], poses.shape[1], 8)
        assert opt_succ_tensor.shape == (poses.shape[0],), f"Success tensor should be of shape (batch,), got {opt_succ_tensor.shape}"
        return opt_traj_tensor, opt_succ_tensor