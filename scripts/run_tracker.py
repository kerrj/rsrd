import torch
import matplotlib.pyplot as plt
from nerfstudio.utils.eval_utils import eval_setup
from pathlib import Path
import numpy as np
from nerfstudio.viewer.viewer import Viewer
from nerfstudio.configs.base_config import ViewerConfig
import cv2
from torchvision.transforms import ToTensor,ToPILImage
from PIL import Image
from nerfstudio.utils import writer
import time
from threading import Lock
from nerfstudio.cameras.cameras import Cameras
from copy import deepcopy
from typing import Tuple,Optional,List,Literal
from torchvision.transforms.functional import resize
from toad.zed import Zed
import warp as wp
from toad.optimization.rigid_group_optimizer import RigidGroupOptimizer,RigidGroupOptimizerConfig
from toad.optimization.atap_loss import ATAPLoss
from toad.utils import *
from toad.hand_registration import HandRegistration
from toad.optimization.observation import PosedObservation
import moviepy.editor as mpy
import tqdm
import moviepy.editor as mpy
import plotly.express as px
import viser
import trimesh
import tyro
from dataclasses import dataclass
import datetime
from viser import transforms as vtf

def animate_traj(optimizer:RigidGroupOptimizer, output_path: Optional[Path] = None):
    from viser import ViserServer
    gs_viser = ViserServer()
    optimizer.reset_transforms()
    dig_model = optimizer.dig_model
    Rs = vtf.SO3(dig_model.quats.cpu().numpy()).as_matrix()
    covariances = np.einsum(
        "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * dig_model.scales.detach().exp().cpu().numpy()[:, None, :] ** 2, Rs
    )
    rec = gs_viser._start_scene_recording()
    gs_viser.scene.world_axes.remove()
    p = optimizer.obj_delta[0,0].detach().cpu().numpy()
    init_obj_se3 = vtf.SE3.from_rotation_and_translation(vtf.SO3(p[3:]),p[:3])
    obj_frame = gs_viser.scene.add_frame("/object",show_axes=True, axes_length=.1,axes_radius=.01)
    delta_frames = []
    for i in range(optimizer.part_deltas.shape[1]):
        p2o_7vec = optimizer.init_p2o_7vec[i].cpu().numpy()
        gs_viser.scene.add_frame(f"/object/part{i}",position=p2o_7vec[:3],wxyz=p2o_7vec[3:],show_axes=False)
        frame = gs_viser.scene.add_frame(f"/object/part{i}/delta",show_axes=True,axes_length=.1,axes_radius=.005)
        delta_frames.append(frame)
        group_mask = optimizer.group_masks[i].cpu().numpy()
        shifted_centers = dig_model.means.detach()[group_mask]-dig_model.means.detach()[group_mask].mean(dim=0)
        gs_viser.scene._add_gaussian_splats(
                f"/object/part{i}/delta/gaussians",
                centers=shifted_centers.cpu().numpy(),
                rgbs=torch.clamp(dig_model.colors, 0.0, 1.0).detach()[group_mask].cpu().numpy(),
                opacities=dig_model.opacities.sigmoid().detach()[group_mask].cpu().numpy(),
                covariances=covariances[group_mask],
            )
    rec.set_loop_start()
    for t in range(optimizer.part_deltas.shape[0]):
        for i in range(optimizer.part_deltas.shape[1]):
            p = optimizer.obj_delta[t,0].detach().cpu().numpy()
            obj_delta = init_obj_se3.inverse() @ vtf.SE3.from_rotation_and_translation(vtf.SO3(p[3:]),p[:3])
            delta = optimizer.part_deltas[t,i].detach().cpu().numpy()
            with gs_viser.atomic():
                obj_frame.wxyz = obj_delta.rotation().wxyz
                obj_frame.position = obj_delta.translation()
                delta_frames[i].position = delta[:3]
                delta_frames[i].wxyz = delta[3:]
        rec.insert_sleep(1/30)
    bs = rec.end_and_serialize()
    if output_path is not None:
        (output_path/"animation.viser").write_bytes(bs)
    animate_button = gs_viser.gui.add_button("Download Animation")
    @animate_button.on_click
    def _(_):
        gs_viser.send_file_download("animation.viser",bs)
    while True:
        for t in range(optimizer.part_deltas.shape[0]):
            for i in range(optimizer.part_deltas.shape[1]):
                p = optimizer.obj_delta[t,0].detach().cpu().numpy()
                obj_delta = init_obj_se3.inverse() @ vtf.SE3.from_rotation_and_translation(vtf.SO3(p[3:]),p[:3])
                delta = optimizer.part_deltas[t,i].detach().cpu().numpy()
                with gs_viser.atomic():
                    obj_frame.wxyz = obj_delta.rotation().wxyz
                    obj_frame.position = obj_delta.translation()
                    delta_frames[i].position = delta[:3]
                    delta_frames[i].wxyz = delta[3:]
            time.sleep(1/30)
        time.sleep(.5)

def get_hands(handreg, frame, framedepth, optimizer, outputs) -> Tuple[Optional[List[trimesh.Trimesh]],Optional[List[trimesh.Trimesh]]]:
    left_hand,right_hand = handreg.detect_hands(frame,optimizer.init_c2w.fx.item()*(max(frame.shape[0],frame.shape[1])/PosedObservation.rasterize_resolution))
    for i,hands in enumerate([left_hand,right_hand]):
        if hands is None:continue
        hands['trimeshes'] = []
        #Compute hand shift to align with gaussians
        #resize framedepth to the same size as the rendered frame
        framedepth = resize(
                    framedepth.permute(2, 0, 1),
                    (outputs['rgb'].shape[0], outputs['rgb'].shape[1]),
                    antialias = True,
                ).permute(1, 2, 0)
        handreg.align_hands(hands,outputs['depth'].detach()/optimizer.dataset_scale, framedepth, outputs['accumulation'].detach()>.8,optimizer.init_c2w.fx.item())
        # visualize result
        for j in range(hands['verts'].shape[0]):
            vertices = hands['verts'][j]
            faces = hands['faces']
            mesh = trimesh.Trimesh(vertices, faces)
            cam_pose = torch.eye(4)
            cam_pose[:3,:] = optimizer.init_c2w.camera_to_worlds
            cam_pose[1:3,:] *= -1
            mesh.apply_transform(cam_pose.cpu().numpy())

            mesh.vertices = mesh.vertices*optimizer.dataset_scale
            v.viser_server.add_mesh_trimesh(f"hand{i}_{j}",mesh,scale=10)
            hands['trimeshes'].append(mesh)
    return [] if left_hand is None else left_hand['trimeshes'],[] if right_hand is None else right_hand['trimeshes']

def get_vid_frame(cap,timestamp):
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate the frame number based on the timestamp and fps
    frame_number = min(int(timestamp * fps),int(cap.get(cv2.CAP_PROP_FRAME_COUNT)-1))
    
    # Set the video position to the calculated frame number
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the frame
    success, frame = cap.read()
    # convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

@dataclass
class ExperimentConfig:
    load_keyframes: bool = False
    camera_input: Literal['iphone_vertical','iphone','gopro','mx_iphone'] = 'iphone'
    video_file: Path = Path("motion_vids/buddha_empty_close.MOV")
    time_bounds: Tuple[float,float] = (0,3.5)
    fps: int = 30
    dig_config: Path = Path("outputs/buddha_empty/dig/2024-07-03_195352/config.yml")#vit-l with 64 dim
    #Path("outputs/sunglasses3/dig/2024-07-03_225001/config.yml")#vit-l with 64 dim
    #Path("outputs/sunglasses3/dig/2024-07-17_215116/config.yml")#no antialiasing
    #Path("outputs/sunglasses3/dig/2024-07-03_225001/config.yml")#vit-l with 64 dim
    #Path("outputs/articulated_objects/dig/2024-07-15_205720/config.yml")
    # Path("outputs/nerfgun_poly_far/dig/2024-07-03_211551/config.yml")#vit-l with 64 dim
    # dig_config = Path("outputs/purple_flower/dig/2024-07-03_200636/config.yml")#vit-l with 64 dim
    # dig_config = Path("outputs/lens_cleaner/dig/2024-07-04_183148/config.yml")
    # dig_config = Path("outputs/left_shoe/dig/2024-07-08_211329/config.yml")
    base_output_folder: Path = Path("renders/tyro_streamlined")
    output_name: Optional[str] = None
    """Stores the output experiment name, otherwise set to current datetime plus data name"""
    detect_hands: bool = False
    optimizer_config: RigidGroupOptimizerConfig = RigidGroupOptimizerConfig()
    
    @property
    def output_folder(self):
        return self.base_output_folder / self.output_name
    
    def __post_init__(self):
        assert self.video_file.exists()
        if self.output_name is None:
            # search for the folder name before 'dig'
            id_end = str(self.dig_config).find("/dig/")
            id_start = str(self.dig_config).rfind("/",0,id_end)
            data_name = str(self.dig_config)[id_start+1:id_end]
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            self.output_name = f"{data_name}_{timestamp}"

def main(exp: ExperimentConfig):
    exp.output_folder.mkdir(exist_ok=True,parents=True)
    train_config,pipeline,_,_ = eval_setup(exp.dig_config)

    dino_loader = pipeline.datamanager.dino_dataloader
    train_config.logging.local_writer.enable = False
    # We need to set up the writer to track number of rays, otherwise the viewer will not calculate the resolution correctly
    writer.setup_local_writer(train_config.logging, max_iter=train_config.max_num_iterations)
    v = Viewer(ViewerConfig(default_composite_depth=False,num_rays_per_chunk=-1),exp.dig_config.parent,pipeline.datamanager.get_datapath(),pipeline,train_lock=Lock())
    try:
        pipeline.load_state()
        pipeline.reset_colors()
    except FileNotFoundError:
        print("No state found, starting from scratch")
    if exp.detect_hands:
        handreg = HandRegistration()


    """
    INITIALIZE THE VIDEO and camera intrinsics
    """

    cam_pose = None
    if cam_pose is None:
        H = np.eye(4)
        H[:3,:3] = vtf.SO3.from_x_radians(np.pi/4).as_matrix()
        cam_pose = torch.from_numpy(H).float()[None,:3,:]
    if exp.camera_input == 'iphone':
        init_cam = Cameras(camera_to_worlds=cam_pose,fx = 1137.0,fy = 1137.0,cx = 1280.0/2,cy = 720/2,width=1280,height=720)
    elif exp.camera_input == 'mx_iphone':
        init_cam = Cameras(camera_to_worlds=cam_pose,fx = 1085.,fy = 1085.,cx = 644.,cy = 361.,width=1280,height=720)
        # init_cam.rescale_output_resolution(1920/1280)
        # init_cam = Cameras(camera_to_worlds=cam_pose,fx = 884.72,fy = 884.72,cx = 955.7,cy = 534.2,width=1920,height=1080)
    elif exp.camera_input == 'iphone_vertical':
        init_cam = Cameras(camera_to_worlds=cam_pose,fy = 1137.0,fx = 1137.0,cy = 1280/2,cx = 720/2,height=1280,width=720)
        init_cam.rescale_output_resolution(1920/1280)
    elif exp.camera_input == 'gopro':
        init_cam = Cameras(camera_to_worlds=cam_pose,fx = 2.55739580e+03,fy = 2.55739580e+03,cx = 1.92065792e+03,cy = 1.07274675e+03,width=3840,height=2160)
    else:
        raise ValueError("Unknown camera type")
    
    if pipeline.cluster_labels is not None:
        labels = pipeline.cluster_labels.int().cuda()
        group_masks = [(cid == labels).cuda() for cid in range(labels.max() + 1)]
    else:
        labels = torch.zeros(pipeline.model.num_points).int().cuda()
        group_masks = [torch.ones(pipeline.model.num_points).bool().cuda()]


    """
    INITIALIZE THE TRACKER
    """
    @torch.no_grad
    def composite_vis_frame(target_frame_rgb,outputs):
        target_vis_frame = resize(target_frame_rgb.permute(2,0,1),(outputs["rgb"].shape[0],outputs["rgb"].shape[1]),antialias=True).permute(1,2,0)
        # composite the outputs['rgb'] on top of target_vis frame
        target_vis_frame = target_vis_frame*0.3 + outputs["rgb"].detach()*0.7
        return target_vis_frame

    optimizer = RigidGroupOptimizer(exp.optimizer_config,pipeline.model,dino_loader,group_masks, group_labels = labels, dataset_scale = pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_scale, render_lock = v.train_lock)
    rgb_renders = [] 
    dino_fn = lambda x: dino_loader.get_pca_feats(x, keep_cuda=True)
    
    motion_clip = cv2.VideoCapture(str(exp.video_file.absolute()))
    if not exp.load_keyframes:
        frame = get_vid_frame(motion_clip,exp.time_bounds[0])
        target_frame_rgb = ToTensor()(Image.fromarray(frame)).permute(1,2,0).cuda()
        frame = PosedObservation(target_frame_rgb, init_cam, dino_fn)
        xs,ys,outputs,renders = optimizer.initialize_obj_pose(frame, render=False, niter=300, n_seeds=5)
        if len(renders)>1:
            print("Saving Initialization video...")
            renders = [r.detach().cpu().numpy()*255 for r in renders]
            # save video as test_camopt.mp4
            out_clip = mpy.ImageSequenceClip(renders, fps=30)
            out_clip.write_videofile(str(exp.output_folder/"test_camopt.mp4"))

        
        def plotly_render(frame):
            fig = px.imshow(frame)
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),showlegend=False,yaxis_visible=False, yaxis_showticklabels=False,xaxis_visible=False, xaxis_showticklabels=False
            )
            return fig
        target_vis_frame = composite_vis_frame(target_frame_rgb,outputs)
        fig = plotly_render(target_vis_frame.cpu().numpy())
        frame_vis = pipeline.viewer_control.viser_server.add_gui_plotly(fig, 9/16)
        target_vis_frame_pil = Image.fromarray((target_vis_frame.cpu().numpy()*255).astype(np.uint8))
        #convert to uint8 
        target_vis_frame_pil.save(str(exp.output_folder / 'initialized_pose.jpg'))

        if exp.detect_hands:
            depth = get_depth((target_frame_rgb*255).to(torch.uint8))
            lh,rh = get_hands(handreg, target_frame_rgb.cpu().numpy(),1/depth,optimizer,outputs)
            optimizer.register_keyframe(lhands = lh, rhands = rh)
        for t in tqdm.tqdm(np.linspace(exp.time_bounds[0],exp.time_bounds[1],int((exp.time_bounds[1]-exp.time_bounds[0])*exp.fps))):
            frame = get_vid_frame(motion_clip,t)
            target_frame_rgb = ToTensor()(Image.fromarray(frame)).permute(1,2,0).cuda()
            optim_frame = PosedObservation(target_frame_rgb, init_cam, dino_fn)
            optimizer.add_observation(optim_frame)
            outputs = optimizer.step(50)
            if exp.detect_hands:
                lhands,rhands = get_hands(handreg, frame,1/get_depth((target_frame_rgb*255).to(torch.uint8))[...,None],optimizer,outputs)
            else:
                lhands,rhands = [],[]
            optimizer.register_keyframe(lhands = lhands, rhands = rhands)
            v._trigger_rerender()
            target_vis_frame = composite_vis_frame(target_frame_rgb,outputs)
            vis_frame = torch.concatenate([outputs["rgb"].detach(),target_vis_frame],dim=1).detach().cpu().numpy()
            fig = plotly_render(target_vis_frame.detach().cpu().numpy())
            frame_vis.figure = fig
            rgb_renders.append(vis_frame*255)
        #save as an mp4
        out_clip = mpy.ImageSequenceClip(rgb_renders, fps=exp.fps)

        #save rendering video
        fname = str(exp.output_folder / "optimizer_out.mp4")
        out_clip.write_videofile(fname, fps=exp.fps,codec='libx264')
        out_clip.write_videofile(fname.replace('.mp4','_mac.mp4'),fps=exp.fps,codec='mpeg4',bitrate='5000k')

        #save part trajectories
        optimizer.save_trajectory(exp.output_folder / "keyframes.pt")
        print("Saved keyframes to",exp.output_folder / "keyframes.pt")
    if exp.load_keyframes:
        optimizer.load_trajectory(exp.output_folder / "keyframes.pt")
    # Populate some viewer elements to visualize the animation
    animate_button = v.viser_server.gui.add_button("Play Animation")
    frame_slider = v.viser_server.gui.add_slider("Frame",0,optimizer.part_deltas.shape[0]-1,1,0)
    smooth_traj_button = v.viser_server.gui.add_button("Smooth Traj (50 Steps)")
    filename_input = v.viser_server.gui.add_text("File Name","render")
    render_video_view = v.viser_server.gui.add_checkbox("From Video View",False)
    status_mkdown = v.viser_server.gui.add_markdown(" ")
    render_button = v.viser_server.gui.add_button("Render Animation",color='green',icon=viser.Icon.MOVIE)
    @render_button.on_click
    def render(_):
        render_button.disabled = True
        render_frames = []
        camera = pipeline.viewer_control.get_camera(1080,1920,0)
        for i in tqdm.tqdm(range(len(optimizer.sequence))):
            status_mkdown.content = f"Rendering...{i/len(optimizer.sequence):.01f}"
            pipeline.model.eval()
            optimizer.apply_keyframe(i)
            with torch.no_grad():
                if render_video_view.value:
                    obs = optimizer.sequence.get_frame(i)
                    cam = deepcopy(obs.frame.camera)
                    cam.rescale_output_resolution(1920/cam.width)
                    frame_outputs = pipeline.model.get_outputs_for_camera(cam)
                    target_vis_frame = composite_vis_frame(obs._raw_rgb,frame_outputs)
                    vis_frame = torch.cat([frame_outputs["rgb"],target_vis_frame],dim=1).detach().cpu().numpy()*255
                    render_frames.append(vis_frame)
                else:
                    outputs = pipeline.model.get_outputs_for_camera(camera)
                    render_frames.append(outputs["rgb"].detach().cpu().numpy()*255)
        status_mkdown.content = "Saving..."
        out_clip = mpy.ImageSequenceClip(render_frames, fps=exp.fps)
        fname = filename_input.value
        (exp.output_folder / 'posed_renders').mkdir(exist_ok=True)
        render_folder = exp.output_folder / 'posed_renders'
        out_clip.write_videofile(f"{render_folder}/{fname}.mp4", fps=exp.fps,codec='libx264')
        out_clip.write_videofile(f"{render_folder}/{fname}_mac.mp4", fps=exp.fps,codec='mpeg4',bitrate='5000k')
        v.viser_server.send_file_download(f"{fname}_mac.mp4",open(f"{render_folder}/{fname}_mac.mp4",'rb').read())
        status_mkdown.content = "Done!"
        render_button.disabled = False
    @animate_button.on_click
    def play_animation(_):
        for i in range(optimizer.part_deltas.shape[0]):
            optimizer.apply_keyframe(i)
            hands = optimizer.hand_frames[i]
            for ih,h in enumerate(hands):
                h_world = h.copy()
                h_world.apply_transform(optimizer.get_registered_o2w().cpu().numpy())
                v.viser_server.add_mesh_trimesh(f"hand{ih}",h_world,scale=10)
            v._trigger_rerender()
            time.sleep(1/exp.fps)
    @frame_slider.on_update
    def apply_keyframe(_):
        optimizer.apply_keyframe(frame_slider.value)
        hands = optimizer.hand_frames[frame_slider.value]
        for ih,h in enumerate(hands):
            h_world = h.copy()
            h_world.apply_transform(optimizer.get_registered_o2w().cpu().numpy())
            v.viser_server.add_mesh_trimesh(f"hand{ih}", h_world, scale=10)
        v._trigger_rerender()
    @smooth_traj_button.on_click
    def smooth_traj(_):
        if len(optimizer.sequence) == optimizer.part_deltas.shape[0]:
            optimizer.step(all_frames=True,niter=50)
            optimizer.save_trajectory(exp.output_folder / "keyframes.pt")
        else:
            print("Please load all frames first!")
    animate_traj(optimizer,exp.output_folder)

    print("Finished tracking!")
    while True:
        time.sleep(.1)

if __name__ == "__main__":
    wp.init()
    tyro.cli(main)