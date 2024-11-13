"""
Script for running the tracker.
"""

import json
from typing import Optional, cast
import time
from pathlib import Path
from threading import Lock
import moviepy.editor as mpy
import plotly.express as px

import cv2
import numpy as np
import torch
import tqdm
import warp as wp

import viser
import tyro
from loguru import logger

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.utils import writer
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.viewer.viewer import Viewer

from rsrd.motion.motion_optimizer import (
    RigidGroupOptimizer,
    RigidGroupOptimizerConfig,
)
import rsrd.transforms as tf
from rsrd.motion.atap_loss import ATAPConfig
from rsrd.extras.cam_helpers import (
    CameraIntr,
    IPhoneIntr,
    get_ns_camera_at_origin,
    get_vid_frame,
)
from rsrd.extras.viser_rsrd import ViserRSRD

torch.set_float32_matmul_precision("high")


def main(
    output_dir: Path,
    is_obj_jointed: Optional[bool] = None,
    dig_config_path: Optional[Path] = None,
    video_path: Optional[Path] = None,
    camera_intr_type: CameraIntr = IPhoneIntr(),
    save_hand: bool = True,
):
    """Track objects in video using RSRD.

    If a `cache_info.json` file is found in the output directory,
    the tracker will load using the cached data + paths and skip tracking.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the paths to the cache file.
    if (output_dir / "cache_info.json").exists():
        cache_data = json.loads((output_dir / "cache_info.json").read_text())
    
        if is_obj_jointed is None:
            is_obj_jointed = bool(cache_data["is_obj_jointed"])
        if video_path is None:
            video_path = Path(cache_data["video_path"])
        if dig_config_path is None:
            dig_config_path = Path(cache_data["dig_config_path"])
    
    assert is_obj_jointed is not None, "Must provide whether the object is jointed."
    assert dig_config_path is not None, "Must provide a dig config path."
    assert video_path is not None, "Must provide a video path."

    cache_data = {
        "is_obj_jointed": is_obj_jointed,
        "video_path": str(video_path),
        "dig_config_path": str(dig_config_path),
    }
    (output_dir / "cache_info.json").write_text(json.dumps(cache_data))

    # Load video.
    assert video_path.exists(), f"Video path {video_path} does not exist."
    video = cv2.VideoCapture(str(video_path.absolute()))

    # Load DIG model, create viewer.
    train_config, pipeline, _, _ = eval_setup(dig_config_path)
    del pipeline.garfield_pipeline
    pipeline.eval()
    viewer_lock = Lock()
    Viewer(
        ViewerConfig(default_composite_depth=False, num_rays_per_chunk=-1),
        dig_config_path.parent,
        pipeline.datamanager.get_datapath(),
        pipeline,
        train_lock=viewer_lock,
    )
    # Need to set up the writer to track number of rays, otherwise the viewer will not calculate the resolution correctly.
    writer.setup_local_writer(
        train_config.logging, max_iter=train_config.max_num_iterations
    )

    try:
        pipeline.load_state()
        pipeline.reset_colors()
    except FileNotFoundError:
        print("No state found, starting from scratch")

    # Initialize tracker.
    wp.init()  # Must be called before any other warp API call.
    optimizer_config = RigidGroupOptimizerConfig(
        atap_config=ATAPConfig(
            loss_alpha=(1.0 if is_obj_jointed else 0.1),
        )
    )
    optimizer = RigidGroupOptimizer(
        optimizer_config,
        pipeline,
        render_lock=viewer_lock,
    )
    logger.info("Initialized tracker.")

    # Generate + load keyframes.
    camopt_render_path = output_dir / "camopt_render.mp4"
    frame_opt_path = output_dir / "frame_opt.mp4"
    track_data_path = output_dir / "keyframes.txt"
    if not track_data_path.exists():
        track_and_save_motion(
            optimizer,
            video,
            camera_intr_type,
            camopt_render_path,
            frame_opt_path,
            track_data_path,
            save_hand,
        )

    optimizer.load_tracks(track_data_path)

    # Load camera and hands info, in the object frame.
    assert optimizer.T_objreg_objinit is not None
    T_cam_obj = optimizer.T_objreg_world.inverse()
    T_cam_obj = (
        T_cam_obj @
        tf.SE3.from_rotation(tf.SO3.from_x_radians(torch.Tensor([torch.pi]).cuda()))
        @ tf.SE3.from_rotation(tf.SO3.from_z_radians(torch.Tensor([torch.pi]).cuda()))
    )
    hands = optimizer.hands_info
    assert hands is not None

    overlay_video = cv2.VideoCapture(str(frame_opt_path))

    # Before visualizing, reset colors...
    optimizer.reset_transforms()

    server = viser.ViserServer()
    viser_rsrd = ViserRSRD(
        server, optimizer, root_node_name="/object", show_finger_keypoints=False
    )

    height, width = camera_intr_type.height, camera_intr_type.width
    aspect = height / width

    camera_handle = server.scene.add_camera_frustum(
        "camera",
        fov=80,
        aspect=width / height,
        scale=0.1,
        position=T_cam_obj.translation().detach().cpu().numpy().squeeze(),
        wxyz=T_cam_obj.rotation().wxyz.detach().cpu().numpy().squeeze(),
    )
    @camera_handle.on_click
    def _(event: viser.GuiEvent):
        client = event.client
        if client is None:
            return
        client.camera.position = T_cam_obj.translation().detach().cpu().numpy().squeeze()
        client.camera.wxyz = T_cam_obj.rotation().wxyz.detach().cpu().numpy().squeeze()

    timesteps = len(optimizer.part_deltas)
    track_slider = server.gui.add_slider("timestep", 0, timesteps - 1, 1, 0)
    play_checkbox = server.gui.add_checkbox("play", True)
    show_overlay_checkbox = server.gui.add_checkbox("Show Demo Vid (slow)", False)
    
    video_handle = server.gui.add_plotly(px.imshow(np.zeros((1, 1, 3))), aspect)
    overlay_handle = server.gui.add_plotly(px.imshow(np.zeros((1, 1, 3))), aspect)

    while True:
        if play_checkbox.value:
            track_slider.value = (track_slider.value + 1) % timesteps
        tstep = track_slider.value
        vid_frame = get_vid_frame(video, frame_idx=tstep)
        part_deltas = optimizer.part_deltas[tstep]
        viser_rsrd.update_cfg(part_deltas)
        viser_rsrd.update_hands(tstep)
        camera_handle = server.scene.add_camera_frustum(
            "camera",
            fov=80,
            aspect=width / height,
            scale=0.05,
            position=T_cam_obj.translation().detach().cpu().numpy().squeeze(),
            wxyz=T_cam_obj.rotation().wxyz.detach().cpu().numpy().squeeze(),
            image = vid_frame
        )
        if show_overlay_checkbox.value:
            video_handle.visible = True
            overlay_handle.visible = True
            fig = px.imshow(vid_frame)
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
            )
            video_handle.figure = fig

            fig = px.imshow(vid_frame)
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
            )
            overlay_handle.figure = fig
        else:
            video_handle.visible = False
            overlay_handle.visible = False
            time.sleep(1/30)


def render_video(
    optimizer: RigidGroupOptimizer,
    motion_clip: cv2.VideoCapture,
    num_frames: int
):
    renders = []
    # Create a render for the video.
    for frame_id in tqdm.trange(0, num_frames):
        rgb = get_vid_frame(motion_clip, frame_idx=frame_id)
        optimizer.apply_keyframe(frame_id)
        with torch.no_grad():
            outputs = cast(
                dict[str, torch.Tensor],
                optimizer.dig_model.get_outputs(optimizer.sequence[frame_id].frame.camera)
            )
        render = (outputs["rgb"].cpu() * 255).numpy().astype(np.uint8)
        rgb = cv2.resize(rgb, (render.shape[1], render.shape[0]))
        render = (render * 0.8 + rgb * 0.2).astype(np.uint8)
        renders.append(render)
    return renders
    

# But this should _really_ be in the rigid optimizer.
def track_and_save_motion(
    optimizer: RigidGroupOptimizer,
    motion_clip: cv2.VideoCapture,
    camera_type: CameraIntr,
    camopt_render_path: Path,
    frame_opt_path: Path,
    track_data_path: Path,
    save_hand: bool = False,
):
    """Get part poses for each frame in the video, ad save the keyframes to a file."""
    camera = get_ns_camera_at_origin(camera_type)
    num_frames = int(motion_clip.get(cv2.CAP_PROP_FRAME_COUNT))

    rgb = get_vid_frame(motion_clip, frame_idx=0)
    obs = optimizer.create_observation_from_rgb_and_camera(rgb, camera)

    # Initialize.
    renders = optimizer.initialize_obj_pose(obs, render=True, niter=150, n_seeds=6)

    # Save the frames.
    out_clip = mpy.ImageSequenceClip(renders, fps=30)
    out_clip.write_videofile(str(camopt_render_path), codec="libx264",bitrate='5000k')
    out_clip.write_videofile(str(camopt_render_path).replace('.mp4','_mac_compat.mp4'),codec='mpeg4',bitrate='5000k')

    # Add each frame, optimize them separately.
    renders = []
    for frame_id in tqdm.trange(0, num_frames):
        try:
            rgb = get_vid_frame(motion_clip, frame_idx=frame_id)
        except ValueError:
            num_frames = frame_id
            break

        obs = optimizer.create_observation_from_rgb_and_camera(rgb, camera)
        optimizer.add_observation(obs)
        optimizer.fit([frame_id], 50)
        if num_frames > 300:
            obs.clear_cache() # Clear the cache to save memory (can overflow on very long videos)

        if save_hand:
            optimizer.detect_hands(frame_id)
    
    # Save a pre-smoothing video
    renders = render_video(optimizer, motion_clip, num_frames)
    out_clip = mpy.ImageSequenceClip(renders, fps=30)
    out_clip.write_videofile(str(frame_opt_path).replace(".mp4","_pre_smooth.mp4"), codec="libx264",bitrate='5000k')
    out_clip.write_videofile(str(frame_opt_path).replace('.mp4','_pre_smooth_mac_compat.mp4'),codec='mpeg4',bitrate='5000k')
    
    # Smooth all frames, together.
    logger.info("Performing temporal smoothing...")
    optimizer.fit(list(range(num_frames)), 50)
    logger.info("Finished temporal smoothing.")

    # Save part trajectories.
    optimizer.save_tracks(track_data_path)

    renders = render_video(optimizer, motion_clip, num_frames)
    # Save the final video
    out_clip = mpy.ImageSequenceClip(renders, fps=30)
    out_clip.write_videofile(str(frame_opt_path), codec="libx264",bitrate='5000k')
    out_clip.write_videofile(str(frame_opt_path).replace('.mp4','_mac_compat.mp4'),codec='mpeg4',bitrate='5000k')


if __name__ == "__main__":
    tyro.cli(main)
