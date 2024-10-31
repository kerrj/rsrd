<h1 align="left">
    <img alt="rsrd logo" src="https://robot-see-robot-do.github.io/data/favicon.png" width="auto" height="30" />
    Robot See Robot Do
    <img alt="rsrd logo" src="https://robot-see-robot-do.github.io/data/favicon.png" width="auto" height="30" />
</h1>

[Project Page](https://robot-see-robot-do.github.io/), [Paper](https://arxiv.org/abs/2409.18121)

- [ ] Try creating environment from scratch.
- [ ] Make sure the dependencies install correctly

## Installation
Clone this repo with `git clone --recursive https://github.com/kerrj/rsrd`, which will clone submodules into `dependencies/`
### Outside Dependencies
First please install PyTorch 2.1.2 in a python 3.10 conda env with cuda version 12.0 (should also work with different torch versions but this is what we've tested). Next, install [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) and [gsplat](https://github.com/nerfstudio-project/gsplat) using the instructions provided in their documentation. We use `nerfstudio` version 1.1.4 and `gsplat` version 1.4.0.

Once these are installed, install [GARField](https://github.com/chungmin99/garfield) (included inside `dependencies/`), which should simply be pip installable except for [cuML](https://docs.rapids.ai/install/), which can be pip installed with 
`pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12==24.10.* cuml-cu12==24.10.*`

Finally, for robot trajectory optimization, install [JAX](https://jax.readthedocs.io/en/latest/installation.html), with GPU if possible; trajectory remapping optimization heavily relies on batch IK + trajectory solves. Please see [`jaxmp`](https://github.com/chungmin99/jaxmp/tree/main) for installation details.

The system was tested with a RTX 4090, with `jax==0.4.35`, but should work for other versions as well.

### Install RSRD
Simply `pip install -e .` inside your cloned `rsrd` repo to install. 

### Submodule Dependencies
Please make sure you `pip` install RSRD before submodules, as it installs some dependencies. There are a number of submodules inside the `dependencies/` folder, which can all be pip installed via
```cd dependencies/ 
pip install -e dig -e garfield -e raftstereo -e hamer_helper -e jaxls -e jaxmp
``` 

### Testing the install
To catch most install issues, after installation you should be able to run `ns-train garfield -h` and `ns-train dig -h` and see a formatted help output of the two training pipelines.

## Running 4D-DPM
We have published data to reproduce the paper results [on our site]([TODO](https://robot-see-robot-do.github.io/)). They consist of a multi-view scan in nerfstudio format, and a `.mp4` video of the input demonstration.

### Custom data
To capture your own objects, please first scan an object of interest as described [here](https://docs.nerf.studio/quickstart/custom_dataset.html).

### Training 4D-DPM
4D-DPM consists of two overlaid models: a GARField and a dino-embedded gaussian model (which we call DiG)

1) *Train GARField* with `ns-train garfield --data <path/to/datasetname>`. This will produce an output config file inside `output/datasetname/garfield/<timestamp>/config.yml`
2) *Train DiG* with `ns-train dig --data <path/to/data/directory> --pipeline.garfield-ckpt <path/to/config.yml>`, using the output config file from the GARField training.
3) *Segment the model*: inside the viewer for DiG, you should see the following GUI:
<img src="assets/dig_gui.png" width="50%" alt="image">
First we need to segment the model. To do this, click on the "Click" button, then click inside the viewer window to select a point in 3D. Next, click "Crop to Click". You should see a result like this:
<img src="assets/dig_crop1.png" width="50%" alt="image">
Next adjust the scale until only the object is segmented, like this:
<img src="assets/dig_crop2.png" width="50%" alt="image">
Finally, switch to "Cluster" mode, then click "Cluster Scene". For best results pick a scale such that the fewest parts are segmented, with just the part of interest remaining. You can cluster as many times as you want, the final clusters are saved for use in the motion recovery stage of the pipeline.
<img src="assets/dig_cluster.png" width="50%" alt="image">


### Reconstructing Video Motion
After the 4D-DPM is completed, the script `scripts/run_tracker.py` executes the motion reconstruction from a video. To see a full list of options run `python scripts/run_tracker.py -h`:
```
╭─ options ─────────────────────────────────────────────────────────────╮
│ -h, --help              show this help message and exit               │
│ --is-obj-jointed {True,False}                                         │
│                         (required)                                    │
│ --dig-config-path PATH  (required)                                    │
│ --video-path PATH       (required)                                    │
│ --output-dir PATH       (required)                                    │
│ --save-hand, --no-save-hand                                           │
│                         (default: True)                               │
╰───────────────────────────────────────────────────────────────────────╯
```
* `--is-obj-jointed`: False for objects that contain removable parts or prismatic joints. Setting False makes the ARAP loss weaker
* `--dig-config-path`: the path to the dig config.yml file. (NOT GARField), which usually looks like `outputs/datasetname/dig/timestamp/config.yml`
* `--video-path`: the path to the demonstration .mp4 video
* `--output-dir`: location to output track trajectory
* `--save-hand`: if specified will compute hand poses with HaMer, otherwise if `--no-save-hand` will not

After tracking executes you can visualize the 4D reconstruction in a viser window (usually `localhost:8080`), which should look like this!

<img src="assets/4d_viewer.png" width="50%" alt="image">

In addition, the output folder you specified will contain:

* `camopt_render.mp4`: a file showing an animation of the object pose initialization (including all random seeds)
* `frame_opt.mp4`: an mp4 file showing the rendered object trajectory from the camera perspective, overlaid on top of the video like so:

https://github.com/user-attachments/assets/343e0176-975e-4ece-b32e-20815b2053f0

* `keyframes.txt`: a loadable representation of the tracked part poses

### Running the robot visualizer
**TODO** CMK fill in

## Bibtex
If you find this useful, please cite the paper!
<pre id="codecell0">@inproceedings{kerr2024rsrd,
&nbsp;title={Robot See Robot Do: Imitating Articulated Object Manipulation with Monocular 4D Reconstruction},
&nbsp;author={Justin Kerr and Chung Min Kim and Mingxuan Wu and Brent Yi and Qianqian Wang and Ken Goldberg and Angjoo Kanazawa},
&nbsp;booktitle={8th Annual Conference on Robot Learning},
&nbsp;year = {2024},
} </pre>
