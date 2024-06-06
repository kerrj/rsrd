from pathlib import Path

import numpy as np
import torch
import tyro
import viser
import viser.transforms as vtf


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def main(output_dir: Path) -> None:
    # Load all of the data.
    print("Loading data...")
    zed_data = dict(**np.load(output_dir / "data.npz"))
    assert zed_data.keys() == {
        "init_pose",
        "keyframes",
        "moving_part_id",
        "nerf_c2w",
        "max_resolution",
        "dataset_scale",
        "zed_rgb",
        "zed_depth",
        "zed_cam_k",
    }, zed_data.keys()

    gaussian_path = list(output_dir.glob("*.pt"))
    assert len(gaussian_path) == 1
    gaussian_path = gaussian_path[0]
    splat_data = {
        k: v.numpy(force=True)
        for k, v in torch.load(gaussian_path, map_location="cpu").items()
    }
    assert splat_data.keys() == {
        "features_dc",
        "features_rest",
        "means",
        "opacities",
        "quats",
        "scales",
        "dino_feats",
        "cluster_labels",
    }

    # Undo the nerfstudio dataparser scale.
    print("Unapplying nerfstudio dataparser scale...")
    zed_data["keyframes"][:, :, :3, 3] /= zed_data["dataset_scale"]
    zed_data["nerf_c2w"][:, :3, 3] /= zed_data["dataset_scale"]
    splat_data["scales"] -= np.log(zed_data["dataset_scale"] * 2.0)
    splat_data["means"] /= zed_data["dataset_scale"]
    zed_data["init_pose"][:3, 3] /= zed_data["dataset_scale"]

    num_gaussians = splat_data["means"].shape[0]
    assert splat_data["quats"].shape == (num_gaussians, 4)
    assert splat_data["scales"].shape == (num_gaussians, 3)

    # Compute Gaussian things.
    print("Computing gaussian covariances...")
    gaussian_rotmat = vtf.SO3(splat_data["quats"]).as_matrix()
    assert gaussian_rotmat.shape == (num_gaussians, 3, 3)
    scaleR = np.exp(splat_data["scales"])[:, None, :] * gaussian_rotmat
    covariances = np.einsum("nij,nkj->nik", scaleR, scaleR)
    del scaleR
    del gaussian_rotmat

    # Start visualization server.
    server = viser.ViserServer()
    part_offset_control = server.scene.add_transform_controls(
        "/object/part/part_offset",
        depth_test=True,
        scale=0.05,
    )
    means_wrt_part: np.ndarray | None = None

    # Populate GUI.
    with server.gui.add_folder("Control"):
        time_slider = server.gui.add_slider(
            "Timestep",
            min=0,
            max=zed_data["zed_rgb"].shape[0] - 1,
            step=1,
            initial_value=0,
        )
        reset_button = server.gui.add_button(
            "Reset transform offset", icon=viser.Icon.HOME_MOVE
        )

    with server.gui.add_folder("Record"):
        save_button = server.gui.add_button(
            "Save timestep", icon=viser.Icon.DEVICE_FLOPPY
        )
        remove_button = server.gui.add_button("Remove timestep", icon=viser.Icon.TRASH)

    with server.gui.add_folder("Status"):
        status_add = server.gui.add_text("ADD", "0.0")
        status_rot = server.gui.add_text("Rotation err", "0.0")
        status_pos = server.gui.add_text("Position err", "0.0")

    @reset_button.on_click
    def _(_) -> None:
        part_offset_control.wxyz = (1.0, 0.0, 0.0, 0.0)
        part_offset_control.position = (0.0, 0.0, 0.0)

    @part_offset_control.on_update
    def _(_) -> None:
        T_partcorrected_part = vtf.SE3.from_rotation_and_translation(
            vtf.SO3(part_offset_control.wxyz),
            part_offset_control.position,
        )
        assert means_wrt_part is not None
        errors = get_errors(T_partcorrected_part, means_wrt_part)
        status_add.value = str(errors["ADD"])
        status_rot.value = str(errors["rot_err"])
        status_pos.value = str(errors["pos_err"])

    annotation_data: dict[int, dict[str, float]] = {}

    def get_errors(
        T_partcorrected_part: vtf.SE3, means_wrt_part: np.ndarray
    ) -> dict[str, float]:
        return {
            "ADD": float(
                np.mean(
                    np.linalg.norm(
                        means_wrt_part - T_partcorrected_part @ means_wrt_part, axis=-1
                    )
                )
            )
            * 1000.0,
            "rot_err": float(np.linalg.norm(T_partcorrected_part.rotation().log()))
            * 180.0
            / np.pi,
            "pos_err": float(np.linalg.norm(T_partcorrected_part.translation()))
            * 1000.0,
        }

    for prev_anno_path in output_dir.glob("_annotation_*.npz"):
        prev_anno = dict(**np.load(prev_anno_path))
        t = int(prev_anno_path.stem.split("_")[-1])
        T_partcorrected_part = vtf.SE3.from_matrix(prev_anno["T_partcorrected_part"])
        annotation_data[t] = get_errors(
            T_partcorrected_part, prev_anno["means_wrt_part"]
        )

    with server.gui.add_folder("Results"):
        results_markdown = server.gui.add_markdown(
            "Annotation summary will be displayed here."
        )

    @remove_button.on_click
    def _(_) -> None:
        """Remove the annotation for the current timestep."""
        t = time_slider.value
        annotation_data.pop(t)
        update_markdown()
        annotation_out_path = output_dir / f"_annotation_{t}.npz"
        annotation_out_path.unlink()
        print("Deleted", annotation_out_path)
        remove_button.disabled = True

    @save_button.on_click
    def _(_) -> None:
        """Save the current annotation."""
        assert means_wrt_part is not None
        t = time_slider.value
        annotation_out_path = output_dir / f"_annotation_{t}.npz"

        T_partcorrected_part = vtf.SE3.from_rotation_and_translation(
            vtf.SO3(part_offset_control.wxyz),
            part_offset_control.position,
        )
        np.savez(
            annotation_out_path,
            means_wrt_part=means_wrt_part,  # This is actually the same for every timestep. But that's fine.
            T_partcorrected_part=T_partcorrected_part.as_matrix(),
        )
        print("Saved:", annotation_out_path)
        assert len(means_wrt_part.shape) == 2 and means_wrt_part.shape[1] == 3
        annotation_data[t] = get_errors(T_partcorrected_part, means_wrt_part)
        update_markdown()
        remove_button.disabled = False

    def update_markdown() -> None:
        """Update the result summary markdown."""
        lines = list[str]()
        agg_results = dict[str, list[float]]()
        for k in sorted(annotation_data):
            timestep_metrics = annotation_data[k]
            lines.append(
                f"`k={k}:"
                + " ".join([f"{k}={v:.4f}" for k, v in timestep_metrics.items()])
                + "`"
            )
            for k, v in timestep_metrics.items():
                if k not in agg_results:
                    agg_results[k] = []
                agg_results[k].append(v)

        results_markdown.content = (
            "(units: mm / deg)<br /> **"
            + " ".join([f"{k}={float(np.mean(v)):.4f}" for k, v in agg_results.items()])
            + "**"
            + "\n\n"
            + "\n\n".join(lines)
        )

    def slider_updated_callback() -> None:
        """When the slider is updated, we need to update the visualization."""

        remove_button.disabled = time_slider.value not in annotation_data

        part_offset_control.wxyz = (1.0, 0.0, 0.0, 0.0)
        part_offset_control.position = (0.0, 0.0, 0.0)

        part_mask = (
            splat_data["cluster_labels"].astype(np.int64) == zed_data["moving_part_id"]
        )

        t = time_slider.value

        means = splat_data["means"] - np.mean(
            splat_data["means"], axis=0, keepdims=True
        )
        means = means[part_mask]
        assert len(means.shape) == 2 and means.shape[1] == 3
        T_object_part = vtf.SE3.from_translation(np.mean(means, axis=0))
        means = means - T_object_part.translation()

        nonlocal means_wrt_part
        means_wrt_part = means

        if t > 0:
            assert zed_data["keyframes"].shape[0] == zed_data["zed_rgb"].shape[0] - 2
            T_part0_partt = zed_data["keyframes"][
                t - 1, zed_data["moving_part_id"], :, :
            ]
            T_object_part = T_object_part @ vtf.SE3.from_matrix(T_part0_partt)

        T_world_objectinit = vtf.SE3.from_matrix(zed_data["init_pose"])
        server.scene.add_frame(
            "/object",
            wxyz=T_world_objectinit.rotation().wxyz,
            position=T_world_objectinit.translation(),
            show_axes=False,
        )
        server.scene.add_frame(
            "/object/part",
            wxyz=T_object_part.rotation().wxyz,
            position=T_object_part.translation(),
            show_axes=False,
        )
        server.scene.add_gaussian_splats(
            "/object/part/part_offset/gaussians",
            centers=means,
            covariances=covariances[part_mask],
            rgbs=SH2RGB(splat_data["features_dc"][part_mask]),
            opacities=splat_data["opacities"][part_mask],
        )

        assert zed_data["nerf_c2w"].shape == (1, 3, 4)
        T_world_cam = vtf.SE3.from_matrix(
            zed_data["nerf_c2w"].squeeze(0)
        ) @ vtf.SE3.from_rotation(vtf.SO3.from_x_radians(np.pi))

        H, W = zed_data["zed_depth"].shape[1:3]
        fy = zed_data["zed_cam_k"][1, 1]

        # Put the camera in the scene...
        server.scene.add_camera_frustum(
            "/camera",
            fov=2 * np.arctan2(H / 2, fy),
            aspect=W / H,
            wxyz=T_world_cam.rotation().wxyz,
            position=T_world_cam.translation(),
            scale=0.1,
        )

        # Set the scene up direction to the camera up direction.
        server.scene.set_up_direction(
            T_world_cam.rotation() @ np.array([0.0, -1.0, 0.0])
        )

        depths = zed_data["zed_depth"][t]
        assert depths.shape == (H, W)

        uv = np.mgrid[:H, :W][::-1, :, :]
        points = (
            (
                np.einsum(
                    "ij,jhw->ihw",
                    np.linalg.inv(zed_data["zed_cam_k"]),
                    np.concatenate([uv, np.ones((1, H, W))], axis=0),
                )
                * depths[None, :, :]
            )
            .reshape((3, -1))
            .T
        )
        rgbs = zed_data["zed_rgb"][t].reshape((-1, 3))

        server.scene.add_point_cloud(
            "/camera/rgbd",
            points=points[::2],
            colors=rgbs[::2],
            point_size=0.001,
            point_shape="sparkle",
        )

    update_markdown()
    slider_updated_callback()
    time_slider.on_update(lambda _: slider_updated_callback())

    breakpoint()


if __name__ == "__main__":
    tyro.cli(main)
