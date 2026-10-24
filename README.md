
# FreeTimeGSVanilla

### Gsplat-based 4D Gaussian Splatting for Dynamic Scenes

<img src="assets/demo.gif" width="100%" alt="FreeTimeGS Demo">

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)

A vanilla minimal implementation of **FreeTimeGS** built on [gsplat](https://github.com/nerfstudio-project/gsplat) for reconstructing dynamic scenes from multi-view video.

## Model scope

This repository uses normalized time, exponential durations, the original motion
and opacity equations, and the two original training presets.

Correctness and compatibility fixes cover frame-gap velocity estimates,
consistent NPZ time and coordinate conversion, per-point durations, small sampling
budgets, Adam resume settings, duration gradients, LPIPS input normalization, relocation
statistics, and matching training/viewer/export math. MP4 output handles odd
image dimensions and export failures do not interrupt a saved training run.

For isolated objects, use `--alpha-mode transparent --random-bkgd` with 8-bit
**straight RGBA** frames. Alpha is a soft foreground target, including zero
opacity outside the object. The trainer composites both target and rendered
RGB on the same random background and adds full-image alpha L1
(`--lambda-alpha 0.1` by default). Do not premultiply the saved RGB or infer
alpha from black pixels: black objects still need opaque alpha. Missing alpha
or frames fail explicitly in this mode. Resizing, undistortion and cropping
transform RGB and alpha together. Evaluation composites the target on black
and reports alpha L1 alongside RGB metrics. The default `--alpha-mode ignore`
preserves RGB-only training for existing scene datasets.

This independently implemented PyTorch loss follows the transparent-alpha
approach in [Brush](https://github.com/ArthurBrussee/brush/tree/6378a76add3b93501abb55c2dc08d71688537679),
with uniform random background colors in [0, 1]. No Brush code is bundled.

## Training policy

Temporal duration is Gaussian **sigma**, rather than a hard visibility cutoff.
`--min-duration-frames` and `--max-duration-frames` express sigma bounds in source
frames, converted with `max(end_frame - start_frame - 1, 1)`. The default retains
the legacy normalized minimum of 0.02 and no maximum. Checkpoints record the
resolved bounds; checkpoint loading, the native viewer, FTGS PLY and TSOG input
conversion preserve them. A half-frame minimum therefore remains a half frame
when exporting a 121-frame sequence, instead of widening to 2.4 frames.

`--freeze-times` anchors temporal centers while velocities remain learnable.
`--lambda-foreground` adds image-area-independent RGB MSE over target alpha > 0.5
in transparent mode. It supplements full-image RGB and alpha supervision; it
does not remove background errors. Empty foreground masks contribute zero.
`--lpips-interval N` evaluates LPIPS every N iterations with weight multiplied
by N. This changes the optimization trajectory, so sparse LPIPS should be
evaluated as a recipe choice rather than assumed equivalent to dense LPIPS.

The Python API supports explicit phases. Each phase inherits fields it does not
specify; learning-rate factors multiply the current continuous schedule. Adam
momentum is retained, and frozen parameters receive no optimizer updates.
Setting `allow_relocation=False` stops relocation, densification and pruning
during polishing. For example:

```python
from simple_trainer_freetime_4d_pure_relocation import Config
from training_policy import TrainingPhase

cfg = Config(
    alpha_mode="transparent", random_bkgd=True,
    freeze_times=True, lambda_foreground=1.0,
    phases=[TrainingPhase(
        start_step=24_000,
        freeze_durations=True, allow_relocation=False,
        position_lr_factor=0.25, velocity_lr_factor=0.2,
        appearance_lr_factor=0.25, lpips_interval=4,
    )],
)
```

This illustrates the API, not a validated universal preset. Narrow temporal
widths require compatible temporal seed coverage. In particular, reducing
sigma while keeping widely spaced keyframes can leave unsupported intermediate
frames. Initialization and recipe selection belong to the calling pipeline;
FreeTimeGsVanilla supplies the reusable optimization controls.

New checkpoints retain the training policy, Adam state, fixed learning-rate
horizon, allocation statistics, strategy state and Python/NumPy/Torch RNG state.
Iteration sampling uses a separate RNG and the consumed iteration number, so
DataLoader prefetch cannot advance resumed sampling. `--resume-mode resume`
restores this policy, including when extending `--max-steps`; the frame range,
alpha mode, image factor and LPIPS network must match. Use
`--resume-mode finetune` for a new recipe and fresh Adam states at step zero.
Legacy checkpoints retain their original 0.02 duration interpretation and can
load model/Adam state, but cannot recover training state they never recorded.
CUDA rasterization may produce small numerical differences after restart.

## Installation

Use Python 3.12+, a CUDA-capable PyTorch environment, and a CUDA toolkit for the
gsplat, fused-ssim, and torch-scatter extensions. The lockfile pins dependencies:

```bash
uv sync --locked
source .venv/bin/activate
```

### Cloud GPU setup (Runpod, PKU-DyMVHumans)

Tested on an H100 pod (NGC image, `nvcc` 12.8) with a network volume at `/workspace`.

- The network volume does not support `chmod`, so the repo, `.venv`, `~/roma-env`
  and the uv cache live on the container disk under `/root`. Only data, results,
  `.env` and the two setup scripts live on `/workspace`.
- `/root` is wiped when the pod stops; rebuild it with `bash /workspace/setup_env.sh`
  and restore data from `/workspace/backup_20261024` (see
  [Status on Oct 24 and how to resume](#status-on-oct-24-and-how-to-resume)).
- `/workspace/bootstrap.sh` sets `UV_CACHE_DIR`, `XDG_CACHE_HOME` and `TORCH_HOME`
  under `/root/.cache`, `TORCH_CUDA_ARCH_LIST=9.0`, installs uv if missing and
  loads the R2 keys from `/workspace/.env`. Source it in every new shell.
- `/workspace/setup_env.sh` clones the repo, runs `uv sync --locked` without
  `torch-scatter`, `gsplat` and `fused-ssim`, then builds those three with
  `uv pip install --no-build-isolation` against the locked `torch==2.10.0+cu128`.
  With build isolation, uv puts the newest PyTorch (CUDA 13) in the build
  environment and the extensions fail with a CUDA version mismatch. After this,
  call `.venv/bin/python` directly rather than `uv sync` or `uv run`.
- RoMa (`romatch`) is installed in a separate `~/roma-env` so it cannot change
  the trainer's PyTorch.

Dataset preparation and dense point clouds: `scripts/pku_dymvhumans/extract_and_convert.py`
(R2 tar to `data_FreeTimeGS/`) and `scripts/pku_dymvhumans/roma_pointclouds.py`
(RoMa matches to `pointclouds/*.npy` and the init NPZ).

**Key Features**

- **4D Gaussian Primitives** - Each Gaussian has position, velocity, time, and duration
- **Temporal Motion Model** - `x(t) = x + v * (t - t_canonical)`
- **gsplat Backend** - Efficient CUDA kernels for fast rendering
- **Flexible Optimization** - MCMC and DefaultStrategy densification
- **Keyframe Processing** - Smart sampling for large video sequences
- **Portable Playback** - Full-model `.ftgs.ply` export and a browser player for
  `.ftgs.ply` and `.tsog`, with audio, playback metadata, embedding, WASD
  navigation, and WebXR AR


**Based on the paper:** 
*FreeTimeGS: Free Gaussian Primitives at Anytime Anywhere for Dynamic Scene Reconstruction* Yifan Wang, Peishan Yang, Zhen Xu, Jiaming Sun, Zhanhua Zhang, Yong Chen, Hujun Bao, Sida Peng, Xiaowei Zhou **CVPR 2025** [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_FreeTimeGS_Free_Gaussian_Primitives_at_Anytime_Anywhere_for_Dynamic_Scene_CVPR_2025_paper.pdf) [[Project Page]](https://zju3dv.github.io/freetimegs/)


---


## Repository Structure

```
FreeTimeGsVanilla/
│
├── src/                          # Core source code
│   ├── simple_trainer_freetime_4d_pure_relocation.py   # Main 4D GS trainer
│   ├── combine_frames_fast_keyframes.py                # Keyframe point cloud combiner
│   ├── viewer_4d.py                                    # Interactive 4D Gaussian viewer
│   └── utils.py                                        # Utility functions (KNN, colormap, etc.)
│
├── datasets/                     # Data loading & processing
│   ├── __init__.py               # Package exports
│   ├── FreeTime_dataset.py       # Dataset class (COLMAP poses, images)
│   ├── normalize.py              # Scene normalization utilities
│   ├── traj.py                   # Camera trajectory generation
│   └── read_write_model.py       # COLMAP binary/text I/O
│
├── player/                       # Bundled ftgs-player, canvas/iframe APIs, and format docs
│   └── tools/package-tsog.mjs    # Add audio and playback defaults to a .tsog package
│
├── run_pipeline.sh               # Full pipeline (combine + train)
├── run_small.sh                  # Example using the 5M-point budget
├── run_full.sh                   # Example using all input points
│
├── LICENSE                       # AGPL-3.0 license
└── README.md                     # This file
```

## Pipeline Overview

The training pipeline consists of two main steps:

1. **Point Cloud Preparation** (`src/combine_frames_fast_keyframes.py`):
   - Loads per-frame triangulated 3D points
   - Extracts keyframes at specified intervals
   - Estimates velocity using k-NN matching between consecutive keyframes
   - Outputs an NPZ file with positions, velocities, colors, and timestamps

2. **4D Gaussian Training** (`src/simple_trainer_freetime_4d_pure_relocation.py`):
   - Initializes 4D Gaussians from the NPZ file
   - Trains with temporal parameters (position, velocity, time, duration)
   - Outputs PLY sequences and trajectory videos

## Keyframes vs All Frames (Stride/Step)

### Why Keyframes?

Processing every single frame of a video is computationally expensive and often redundant. Adjacent frames are typically very similar. Instead, we use **keyframes** - frames sampled at regular intervals.

### Keyframe Step (Stride)

The `--keyframe-step` parameter controls how many frames to skip between keyframes:

- **Step = 1**: Use ALL frames (no skipping) - most accurate but slowest
- **Step = 5**: Use every 5th frame (0, 5, 10, 15, ...) - good balance
- **Step = 10**: Use every 10th frame - faster but less temporal detail

**Example**: For a 60-frame video with `--keyframe-step 5`:
```
Frames:    0  1  2  3  4  5  6  7  8  9  10 11 12 ... 55 56 57 58 59
Keyframes: *              *              *              *
           0              5              10             55
```

The final frame is also included when its point cloud exists, so the sequence
retains endpoint coverage even when the stride does not divide the frame range.

### Velocity Estimation

Velocity uses the next frame when available, otherwise the next keyframe or
final endpoint. Displacement is divided by the actual frame gap:
```
v = (matched_position[next_frame] - position[t]) / (next_frame - t)
```

This gives the average velocity over the keyframe interval.

## NPZ File Format

The NPZ file contains the initial 4D Gaussian data:

| Field | Shape | Description |
|-------|-------|-------------|
| `positions` | [N, 3] | 3D coordinates (x, y, z) |
| `velocities` | [N, 3] | Velocity vectors in world units per frame |
| `colors` | [N, 3] | RGB colors normalized to [0, 1] |
| `times` | [N, 1] | Normalized timestamps in [0, 1] |
| `durations` | [N, 1] | Temporal duration (visibility window) |
| `has_velocity` | [N] | Boolean mask for valid velocity estimates |

**Metadata fields:**
- `frame_start`, `frame_end`: Inclusive start and exclusive end in new NPZ files
- `time_denominator`: `frame_end - frame_start - 1` (at least 1); converts normalized time to frame offsets
- `n_keyframes`: Number of keyframes used
- `keyframe_step`: Step between keyframes
- `mode`: Processing mode identifier

Legacy files from this repo's combiner are still accepted with their original
inclusive end and frame-count denominator. When loading a subrange, timestamps
and durations are renormalized together and velocities retain their physical
displacement. Valid per-point durations are preserved in automatic mode.

### Example NPZ Creation

```python
import numpy as np

# Your triangulated point clouds (one per frame)
points_frame_0 = np.load("points3d_frame000000.npy")  # [M, 3]
colors_frame_0 = np.load("colors_frame000000.npy")    # [M, 3], values 0-255

# Combine and save
np.savez(
    "init_points.npz",
    positions=positions,      # [N, 3] float32
    velocities=velocities,    # [N, 3] float32
    colors=colors / 255.0,    # [N, 3] float32, normalized to [0, 1]
    times=times,              # [N, 1] float32, normalized to [0, 1]
    durations=durations,      # [N, 1] float32
    has_velocity=has_velocity, # [N] bool
    frame_start=0,
    frame_end=61,              # exclusive
    time_denominator=60,
)
```

## Input Requirements

### Per-Frame Point Cloud Files

The `src/combine_frames_fast_keyframes.py` script expects:

```
input_dir/
├── points3d_frame000000.npy   # [M, 3] float32 - 3D positions
├── colors_frame000000.npy     # [M, 3] float32 - RGB colors (0-255)
├── points3d_frame000001.npy
├── colors_frame000001.npy
├── ...
└── points3d_frameXXXXXX.npy
```

These are typically generated by triangulating matched features across camera views.

### COLMAP Data

The trainer expects a COLMAP sparse reconstruction:

```
data_dir/
├── images/                    # Or images_Nx/ for downsampled
│   ├── cam01_frame000000.jpg
│   └── ...
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

## Usage

### Full Pipeline

```bash
bash run_pipeline.sh \
    /path/to/triangulation/output \
    /path/to/colmap/data \
    /path/to/results \
    0 61 5 0 default_keyframe_small
```

The arguments after the paths are start frame, exclusive end frame, keyframe
step, GPU ID, and preset. Training uses one image and timestamp per step
(`--batch-size 1`).

### Step by Step

**Step 1: Combine keyframes**

The combiner's `--frame-end` is inclusive; the trainer's `--end-frame` is exclusive.

```bash
python src/combine_frames_fast_keyframes.py \
    --input-dir /path/to/triangulation/output \
    --output-path /path/to/keyframes.npz \
    --frame-start 0 \
    --frame-end 60 \
    --keyframe-step 5
```

**Step 2: Train 4D Gaussians**

```bash
CUDA_VISIBLE_DEVICES=0 python src/simple_trainer_freetime_4d_pure_relocation.py default_keyframe \
    --data-dir /path/to/colmap/data \
    --init-npz-path /path/to/keyframes.npz \
    --result-dir /path/to/results \
    --start-frame 0 \
    --end-frame 61 \
    --max-steps 30000
```

### Available Configs

| Config | Points | Description |
|--------|--------|-------------|
| `default_keyframe` | All input points | No initialization downsampling |
| `default_keyframe_small` | Up to 5M | Reduced initialization budget |

## Outputs

### Single-file FTGS export

Export a trained Vanilla checkpoint to one `.ftgs.ply` on CPU, without the
dataset or a GPU:

```bash
python src/export_ftgs_ply.py \
    --ckpt /path/to/results/ckpts/ckpt_29999.pt \
    --output /path/to/scene.ftgs.ply
```

Omitting `--output` saves next to the checkpoint with the `.ftgs.ply` suffix.
For older checkpoints, `--n-frames 61` supplies frame-count metadata and
`--no-velocity` records a model trained with motion disabled. Otherwise these
settings come from the checkpoint; legacy models default to motion enabled and
omit the frame count when it is unknown.

The trainer accepts `--export-ply --export-ply-format ftgs` to save
`result_dir/ckpt_<step>.ftgs.ply` at the existing PLY export steps. To export an
existing checkpoint through the trainer CLI:

```bash
python src/simple_trainer_freetime_4d_pure_relocation.py default_keyframe \
    --export-only --export-ply-format ftgs \
    --ckpt-path /path/to/results/ckpts/ckpt_29999.pt \
    --result-dir /path/to/export
```

This mode exports only the complete model and reads frame count and motion
settings from the checkpoint. It bypasses dataset loading, GPU initialization,
and video rendering. FTGS export keeps every Gaussian and SH coefficient;
the per-frame opacity threshold, frame stride, and compact-export setting do
not filter this file.

### FTGS PLY layout, version 1

This is this repository's documented format for Vanilla FreeTimeGS. It is a
binary little-endian PLY with one `vertex` per Gaussian; every property is a
32-bit float. Spatial coordinates remain in the trainer's model space.

| Properties | Stored values |
| --- | --- |
| `x`, `y`, `z` | Canonical position at that Gaussian's `time` |
| `nx`, `ny`, `nz` | Zeros, following the usual 3DGS PLY layout |
| `f_dc_0..2` | RGB DC spherical harmonic coefficients |
| `f_rest_0..` | Remaining SH coefficients, all red coefficients then green then blue |
| `opacity` | Base opacity logit; apply sigmoid |
| `scale_0..2` | Spatial log-scales; apply exp |
| `rot_0..3` | Quaternion in wxyz order; normalize when rendering |
| `time` | Canonical time on the normalized sequence timeline |
| `log_duration` | Log temporal standard deviation; apply exp |
| `velocity_0..2` | Displacement per unit of normalized time |

Header comments record `ftgs_version 1`, `time_units normalized`, `sh_degree`,
`use_velocity` (0 or 1), `min_duration 0.02`, `opacity_floor 0.0001`, and
`n_frames` when known. `log_duration` includes Vanilla's minimum-duration
projection. Raw velocity values are retained even when `use_velocity` is 0.

For a time `t`, an importer reconstructs the model using:

```text
dt = t - time
position = [x, y, z] + [velocity_0, velocity_1, velocity_2] * dt
           # use [x, y, z] directly when use_velocity is 0
alpha = max(sigmoid(opacity) * exp(-0.5 * (dt / exp(log_duration))**2), 0.0001)
```

The sequence spans `t=0..1`; Gaussian canonical times may lie outside that
interval. Relative frame `i` maps to `i / max(n_frames - 1, 1)`. This file stores
rendering parameters, not optimizer state. The browser player below reads this
layout directly; the Python/CUDA viewer loads `.pt` checkpoints.

### Existing training outputs

After training, you'll find:

```
results/
├── ckpts/
│   └── ckpt_30000.pt              # Model checkpoint
├── videos/
│   ├── traj_4d_step30000.mp4      # RGB trajectory video
│   ├── traj_duration_step30000.mp4    # Duration heatmap
│   └── traj_velocity_step30000.mp4    # Velocity heatmap
├── ply_sequence_step30000/
│   ├── frame_000000.ply           # Per-frame PLY exports
│   └── ...
└── tb/                            # TensorBoard logs
```

## Browser player for FTGS files

The [FTGS Player](player/README.md) opens `.ftgs.ply` and packaged `.tsog` files
locally or from a URL. It runs entirely in the browser with WebGL2, without a
build step or package installation. Use the
[hosted player](https://opsiclear-4dgs.github.io/ftgs-player/) or serve it locally:

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory player
```

Open <http://localhost:8765> and drop a file onto the page. Local files stay in
the browser. The scene fills the window, with compact borderless controls below
it that fade during playback. Canvas and iframe examples share speed controls
and bottom-right fullscreen. Play, pause, seek, change speed, orbit/pan/zoom, or
move with WASD (Q/E down/up; Shift moves faster). The default preview loads up
to 1 million Gaussians; use `?points=all` for full detail when GPU memory allows.

| File | Playback support |
| --- | --- |
| `.ftgs.ply` | Complete Vanilla model: normalized time, temporal opacity, linear motion, and SH color. |
| `.tsog` | Version-4 WebP attribute packages with continuous, discrete, or static scenes; optional embedded audio and playback defaults. |

### Audio and playback defaults

A `.tsog` may store clip duration, FPS, playback speed, looping, and audio
volume in `meta.json`, alongside one optional audio file in the same ZIP.
The player applies those defaults and keeps audio aligned through play, pause,
seek, and loop. The sound button (M) enables or mutes audio; desktop users can
adjust volume. Audio starts enabled; if the browser blocks sound, press **Play**
to start the scene and its audio together.

For an existing continuous TSOG animation, package a soundtrack from this
repository's root with Node.js 22+:

```bash
node player/tools/package-tsog.mjs scene.tsog scene-with-audio.tsog \
  --audio soundtrack.mp3 --duration 10 --fps 30 --speed 1.25 --loop --volume 0.8
```

The command copies the scene attributes and audio without re-encoding them.
Omit `--audio` to update defaults while retaining the track; `--remove-audio`
removes it. Use a new output filename. See the
[metadata and packaging guide](player/docs/TSOG.md#audio-and-playback-metadata)
for field definitions, discrete/static timing, supported audio types, and API usage.

Use `?src=scene.tsog&speed=1.5&loop=0` to override playback for a link.
`speed` multiplies playback speed; `fps` changes the animation's frame timing.
Older packages keep their existing defaults. Audio and playback metadata are
optional TSOG extensions; the `.ftgs.ply` layout is unchanged.

The extensions are now specified as
[TSOG Playback Profile 1](player/docs/TSOG-SPEC.md), with a
[metadata schema](player/tsog.schema.json) and
[synthetic conformance corpus](player/tests/fixtures/tsog/CONFORMANCE.md).
New packages declare the profile; older v4 files remain readable. Check metadata,
referenced entries, checksums and image headers with
`node player/tools/validate-tsog.mjs scene.tsog --require-profile`.
Omit the flag for legacy files; full attribute validation is documented in the spec.

### AR and embedding

On compatible devices, the **AR** button opens a WebXR session with tap-to-place
and animated playback. Use HTTPS (or localhost); see the
[AR controls and setup](player/docs/PLAYER.md#view-in-ar).

The [embedding API](player/docs/API.md) supports independent canvas instances and
iframe controls, including `load`, `play`, `pause`, `seek`, `setPlaybackRate`,
`setMuted`, `setVolume`, events, and `destroy`. It includes canvas, iframe, and
React examples. An iframe with audio and AR should use
`allow="autoplay; xr-spatial-tracking"` and keep controls available for user gestures.

The [canvas example](https://opsiclear-4dgs.github.io/ftgs-player/examples/embedding.html?mode=canvas) and
[iframe example](https://opsiclear-4dgs.github.io/ftgs-player/examples/embedding.html?mode=iframe) show copyable
integration code beneath the player. The iframe snippet includes the player's
built-in controls; the canvas snippet uses modules served by your own project.

The [hosted embedding example](https://opsiclear-4dgs.github.io/ftgs-player/examples/embedding.html?mode=iframe)
and player include static X large-image cards and Open Graph tags with a
1200×675 thumbnail and no personal account attribution. Share the dedicated
[Slop Example scene page](https://opsiclear-4dgs.github.io/ftgs-player/examples/jensen-2m.html)
for an image link to the interactive 4DGS player. It opens the full scene,
paused with audio enabled; press Play to start playback. X controls preview
display and whether links open in its in-app browser. Actual previews require
checking a real post in X.

TSOG loading shares color centroids and uses a decoding worker where supported.
Sorting caches conservative temporal bounds while retaining all Gaussians and
SH coefficients. See the measured full-quality
[loading improvements](player/docs/PERFORMANCE.md).

See the [player documentation](player/docs/PLAYER.md) for controls, URL options,
rendering limits, and tests. The repository-owned [TSOG module](player/docs/TSOG.md)
uses the format specified by the independent
[Apache-2.0 TSOG project](https://github.com/OpsiClear-4DGS/tsog). That project
contains the canonical specification, schema and optimized codec examples.
The bundled guide also documents the
[TSOG paper](https://arxiv.org/abs/2607.28049),
[Xiaomi Research's repository](https://github.com/xiaomi-research/tsog), and
[license notices](player/docs/THIRD_PARTY.md).

## Python/CUDA 4D Viewer

An interactive viewer for visualizing trained 4D Gaussian Splatting models with temporal animation.

### Installation

The viewer uses the same environment installed above.

**Verify installation:**
```bash
python -c "import viser; import nerfview; import gsplat; print('All dependencies installed!')"
```

### Quick Start

```bash
CUDA_VISIBLE_DEVICES=0 python src/viewer_4d.py \
    --ckpt /path/to/results/ckpts/ckpt_30000.pt \
    --port 8080 \
    --total-frames 60 \
    --temporal-threshold 0.05 \
    --spatial-percentile 95
```

Then open `http://localhost:8080` in your browser.

New checkpoints record the frame count and velocity switch, which the viewer
loads automatically. `--total-frames` supplies the count for legacy checkpoints.
Checkpoints must use the documented parameters, normalized time, and exponential
durations. Incompatible checkpoints are rejected.

### Checkpoint File Format (.pt)

The checkpoint file contains all trained 4D Gaussian parameters:

```python
checkpoint = {
    "splats": {
        "means": tensor[N, 3],       # Canonical 3D positions
        "scales": tensor[N, 3],      # Log-scale parameters
        "quats": tensor[N, 4],       # Rotation quaternions (wxyz)
        "opacities": tensor[N],      # Logit opacities
        "sh0": tensor[N, 1, 3],      # DC spherical harmonics
        "shN": tensor[N, K, 3],      # Higher-order SH coefficients
        # 4D temporal parameters:
        "times": tensor[N, 1],       # Canonical time (when Gaussian is most visible)
        "durations": tensor[N, 1],   # Log temporal duration (visibility window width)
        "velocities": tensor[N, 3],  # Linear velocity vectors
    },
    "step": int,                     # Training step
    ...
}
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--ckpt` | **required** | Path to trained checkpoint `.pt` file |
| `--port` | 8080 | HTTP port for the viewer |
| `--device` | cuda | Device to use (cuda, cuda:0, cuda:1, etc.) |
| `--total-frames` | 300 | Total number of frames in the sequence |
| `--temporal-threshold` | 0.01 | Minimum temporal opacity to render a Gaussian |
| `--spatial-percentile` | 95 | Percentile of points to keep (removes outliers) |
| `--no-spatial-filter` | False | Disable spatial filtering |
| `--no-precompute` | False | Disable precomputing visibility masks |
| `--sh-degree` | 3 | Spherical harmonics degree |

### Understanding Key Parameters

#### `--temporal-threshold`

Controls which Gaussians are rendered at each frame based on their temporal opacity.

Each Gaussian has a temporal opacity computed as:
```
temporal_opacity(t) = exp(-0.5 * ((t - t_canonical) / duration)^2)
```

- **Lower threshold (0.01)**: More Gaussians visible, smoother but slower
- **Higher threshold (0.1)**: Fewer Gaussians, faster but may show gaps

```
Temporal opacity vs time for a Gaussian centered at t=0.5:

    1.0 |       ****
        |      *    *
    0.5 |     *      *
        |    *        *
  0.05 -|---*----------*--- threshold
        |  *            *
    0.0 +-------------------> time
        0.0    0.5    1.0
              ^
          Gaussian visible when opacity > threshold
```

#### `--spatial-percentile`

Removes outlier Gaussians that are far from the scene center.

- **95%**: Keep Gaussians within the 95th percentile distance from center (removes 5% outliers)
- **99%**: Keep more Gaussians (removes only 1% outliers)
- **100%**: Keep all Gaussians (no spatial filtering)

This is useful when training produces "floater" artifacts far from the main scene.

```
Example with 5M Gaussians:
┌─────────────────────────────────────┐
│  · ·                            · · │  <- outliers (removed)
│      ┌───────────────────────┐      │
│      │  * * * * * * * * * *  │      │  <- 95% kept
│      │  * * * SCENE * * * *  │      │
│      │  * * * * * * * * * *  │      │
│      └───────────────────────┘      │
│  ·                              ·   │  <- outliers (removed)
└─────────────────────────────────────┘
```

### Viewer UI Controls

Once the viewer is running, you can control it through the web interface:

**Animation Panel:**
- **Frame Slider**: Manually scrub through time
- **Auto Play**: Toggle automatic playback
- **Play Speed (FPS)**: Control playback speed (1-60 FPS)

**Visibility Filtering Panel:**
- **Temporal Opacity Threshold**: Adjust visibility threshold in real-time
- **Use Visibility Mask**: Toggle efficient rendering on/off

**Camera Controls (in browser):**
- Left-click + drag: Rotate camera
- Right-click + drag: Pan camera
- Scroll: Zoom in/out

### Efficiency: Visibility Masking

The viewer uses multi-level filtering for efficient rendering:

| Filter Stage | Purpose | Typical Reduction |
|--------------|---------|-------------------|
| Spatial filter | Remove outliers | 100% → 96% |
| Base opacity filter | Remove transparent Gaussians | 96% → 95% |
| Temporal filter | Only render temporally-visible | 95% → **8%** |

**Result**: Only ~8% of Gaussians are rendered per frame, enabling interactive framerates with millions of Gaussians.

### Example Usage

**Basic viewing:**
```bash
python src/viewer_4d.py --ckpt results/ckpts/ckpt_30000.pt --total-frames 60
```

**High-quality (show more Gaussians):**
```bash
python src/viewer_4d.py \
    --ckpt results/ckpts/ckpt_30000.pt \
    --total-frames 60 \
    --temporal-threshold 0.01 \
    --spatial-percentile 99
```

**Fast preview (fewer Gaussians):**
```bash
python src/viewer_4d.py \
    --ckpt results/ckpts/ckpt_30000.pt \
    --total-frames 60 \
    --temporal-threshold 0.1 \
    --spatial-percentile 90
```

**Debug mode (no filtering):**
```bash
python src/viewer_4d.py \
    --ckpt results/ckpts/ckpt_30000.pt \
    --total-frames 60 \
    --no-spatial-filter \
    --temporal-threshold 0.0
```

## Key Parameters

### Point Cloud Preparation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--keyframe-step` | 5 | Frames between keyframes |
| `--max-velocity-distance` | 0.5 | Max k-NN match distance |
| `--sample-ratio` | 1.0 | Point subsampling ratio |

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-steps` | 60000 | Training iterations |
| `--init-duration` | 0.1 | Initial temporal duration |
| `--velocity-lr-start` | 5e-3 | Initial velocity learning rate |
| `--velocity-lr-end` | 1e-4 | Final velocity learning rate |
| `--lambda-4d-reg` | 1e-3 | 4D regularization weight |

## 4D Gaussian Parameters

Each Gaussian has 8 learnable parameter groups:

1. **Position (x)**: [N, 3] - Canonical 3D position
2. **Time (t)**: [N, 1] - When the Gaussian is most visible
3. **Duration (s)**: [N, 1] - Temporal width
4. **Velocity (v)**: [N, 3] - Linear velocity
5. **Scale**: [N, 3] - 3D scale
6. **Quaternion**: [N, 4] - Rotation
7. **Opacity**: [N] - Base opacity
8. **Spherical Harmonics**: [N, K, 3] - View-dependent color

### Motion Model

Position at time t:
```
x(t) = x + v * (t - t_canonical)
```

Temporal opacity (Gaussian falloff):
```
opacity(t) = exp(-0.5 * ((t - t_canonical) / duration)^2)
```
## Regression checks

With the runtime dependencies installed, run the CPU tests with:

```bash
python -m unittest discover -s tests -v
```

These exercise NPZ/combiner round trips, duration gradients, optimizer resume,
relocation, checkpoint compatibility, PLY/NPZ export, and MP4 output. They do not
replace a full scene training and quality comparison on a GPU.

## Citation

If you find this work useful, please cite the original paper:

```bibtex
@InProceedings{Wang_2025_CVPR,
    author    = {Wang, Yifan and Yang, Peishan and Xu, Zhen and Sun, Jiaming and Zhang, Zhanhua and Chen, Yong and Bao, Hujun and Peng, Sida and Zhou, Xiaowei},
    title     = {FreeTimeGS: Free Gaussian Primitives at Anytime Anywhere for Dynamic Scene Reconstruction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {21750-21760}
}
```

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Run notes: PKU-DyMVHumans Kungfu Weapon Pair (Oct 2026)

Scene `part1/1080_Kungfu_Weapon_Pair_m12m13`: 60 cameras, 1920x1080, 25 fps,
250 frames, two performers with spear and staff on a dark stage. Full RGB
frames (`per_view/cam_N/images`), `--alpha-mode ignore`.

### Status on Oct 24 and how to resume

The pod `fl2mg6zf0psc6v` ("FreetimeGS - Pku-dymvhumans", 1x H100 80GB HBM3,
Secure Cloud, US-MO-1, $3.99/h, image `runpod/pytorch:1.4.0-cu1281-torch280-ubuntu2404`)
was stopped on Oct 24. Everything that matters is on the global volume "50Labs"
(`cmvhe45sk000007jt0lui58r0`), mounted at `/workspace`; the container disk
(`/root`) and `/dev/shm` are wiped by a stop.

What is on `/workspace`:

| Path | Contents |
|---|---|
| `bootstrap.sh`, `setup_env.sh`, `.env` | Shell setup, environment build, R2 keys |
| `data/zxyun/PKU-DyMVHumans/part1/1080_Kungfu_Weapon_Pair_m12m13` | Extracted scene: `per_view/cam_N/{images,pha}`, `data_FreeTimeGS/{sparse,manifest.json,pointclouds,*.npz}` |
| `backup_20261024/generated_data.tar` | `/root/data_local` outputs: full-frame RoMa clouds for all 250 frames (`pointclouds/`), same-camera-velocity clouds for 0-60 (`pointclouds_flow/`), NPZs for C/E (`keyframes_61frames_step1`), F1 (`flow_61frames_step1`), F2 (`flow_speeddur_61frames_step1`), the step-5 NPZs |
| `backup_20261024/scripts_logs.tar` | Job scripts `run_{A,C,D,E,F,...}.sh`, `move_data.sh`, diagnostic scripts, `/root/logs` |
| `backup_20261024/extra_results.tar` | Frame 54 renders for F1/F2, `staff_vel`, `count_f54`, `kungfu_full_v2test` |
| `backup_20261024/pod_code.diff`, `combine_stage_performer.py` | Pod code changes (also committed to the `pku-dymvhumans` branch) |
| `results/` | Pilot, full run (100k), A, C, D, E, F1, F2 with checkpoints (5.85 GB each at 7.5M Gaussians) |
| `logs/` | Training, RoMa and combine logs; `staff_vel.py`, `staff_flow_diag*.py`, `render_f54.py`, `count_f54*.py` |

Turning it back on:

1. Start the pod in the Runpod console (or create a new H100 pod with the same
   image and the "50Labs" volume at `/workspace`). Take the SSH command from the
   console: the port changes on restart. Check `nvidia-smi` first; a restarted pod
   can come back without a GPU if the host's H100 was taken, in which case
   create a new pod on the same volume.
2. Rebuild `/root` (about 15-20 min, mostly the gsplat build):
   ```bash
   source /workspace/bootstrap.sh && bash /workspace/setup_env.sh
   ```
   `setup_env.sh` clones `damimorley/FreetimegsVanilla`, branch `pku-dymvhumans`.
   Push local changes before relying on it.
3. Restore data and scripts (`/root` has 50 GB; the restored data is about 15 GB):
   ```bash
   tar -C /root -xf /workspace/backup_20261024/scripts_logs.tar
   bash /root/move_data.sh            # images, mattes, sparse/0 from /workspace (minutes)
   tar -C /root/data_local -xf /workspace/backup_20261024/generated_data.tar
   mkdir -p /root/results /root/logs
   ```
   `move_data.sh` also copies `data_FreeTimeGS/pointclouds`; the tar then
   overwrites them with the full-frame clouds for all 250 frames.
4. Check: re-score E (expect 30.08 dB, 18.87 dB inside mattes):
   ```bash
   cd ~/FreetimegsVanilla && .venv/bin/python src/simple_trainer_freetime_4d_pure_relocation.py default_keyframe \
     --data-dir /root/data_local --init-npz-path /root/data_local/keyframes_61frames_step1.npz \
     --ckpt-path /workspace/results/E_reg1e-2_0to60/ckpts/ckpt_6999.pt --eval-only \
     --result-dir /root/results/E_check --start-frame 0 --end-frame 61 \
     --no-auto-init-duration --init-duration 0.0333 --min-duration-frames 1.0 \
     --eval-sample-every 7 --eval-matte-dir /root/data_local/masks
   ```
5. Write results to `/root/results`, run long jobs in `tmux`, and copy results
   to `/workspace/results` when a job finishes.

What works so far (frames 0-60, 7k steps, about 7 min on the H100):

- The pipeline end to end: R2 extraction, poses from `cams/`, RoMa point
  clouds, combine, training, matte-based evaluation, frame 54 diagnostics.
- RoMa on every frame with a 2-frame sigma (C) beats keyframe step 5 (pilot)
  by +0.49 dB inside the mattes, on 67 of 70 images.
- The paper's opacity regularization `--lambda-4d-reg 1e-2` (E) adds about
  +0.17 dB inside the mattes over C (55 of 70 images). E is the current best
  baseline: 30.08 dB whole image, 18.87 dB inside the mattes.
- Same-camera RoMa velocities (F1, F2) place the staff Gaussians much better
  at neighbouring frames (40% versus 15% inside the mattes one frame later).

What does not work yet:

- Thin, fast weapons (upper staffs at frame 54) are missing in every run,
  in training and held-out cameras alike. Better starting points (masked RoMa,
  more performer points, same-camera velocities, shorter durations) did not
  fix it; the staff Gaussians are darkened or relocated during training.
- Performers score about 11 dB below the frame average; whole-image PSNR is
  inflated by the black background, so compare `psnr_fg`.
- The 250-frame, 100k-step run is softer than the 61-frame pilot (velocity
  and time learning rates act in normalized time) and damps fast motion.
- Separate stage and performer initialization (D) was worse: the static
  stage Gaussians faded and were relocated.
- No-relocation after 7k steps (v2 test) and removing the duration penalty
  (A) did not help.

Untried next steps, in order: transparency supervision (performer mattes plus
the floor, where the median frame is brighter than 0.05, as alpha, with
`--alpha-mode transparent --random-bkgd`) so that darkening is no longer free;
the FreeTimeGS++ SelfCap settings (1-2M Gaussians, initial scale 0.1, time LR
1.6e-4, velocity LR 1.6e-3); per-camera colour correction; a second seed of E
to measure run-to-run noise; MCMC-style relocation and a persistence gate.

Mistakes and pitfalls so far:

- Results and generated data written to the container disk are lost on a
  stop; earlier runs only survived because they were copied to `/workspace`.
  Validation metrics of the first full run were not copied back.
- The `/workspace` geesefs mount disconnected three times and lost its
  directory symlinks across a pod restart; data symlinked from `/workspace`
  broke running jobs. Copy data to `/root/data_local` instead, and write many
  small files to `/workspace` as one tar (it writes about 5 MB/s).
- `/root` filled to 97% (49 of 50 GB) with data copies and 5.85 GB
  checkpoints. Delete old checkpoints from `/root/results` once copied.
- `uv sync` with build isolation pulls the newest PyTorch (CUDA 13) and the
  CUDA extensions fail; caches default to `/workspace/.cache`, which has no
  `chmod` (see below).
- `--init-duration` is ignored unless `--no-auto-init-duration` is passed, so
  the first runs had a 15-frame sigma instead of the intended one.
- `--eval-sample-every 60` on 250 frames put every validation image on a
  keyframe; use `--eval-sample-every 7` and `--eval-matte-dir`.
- The presets ship `lambda_4d_reg=1e-4`, 100x below the paper; nothing showed
  that the reduction had been tested.
- RoMa `--num-matches 100000` runs out of GPU memory (pairwise distances
  over 4x the matches); `--use-mask` alone filters after sampling and lowers
  performer coverage, so use `--mask-before-sample`.
- Nearest-neighbour velocities between point clouds are useless for the
  staffs (they move more than their own width per frame); some staff points
  were also triangulated at the wrong depth, probably by matching one staff
  to the other.
- Single 7k-step runs differ by up to about 0.36 dB from seed noise alone
  (FreeTimeGS++), so differences below that need a second seed.

### Setup that worked (Runpod H100 80GB, NGC image, `nvcc` 12.8, driver CUDA 13.2)

- Network volume at `/workspace` has no `chmod`: git, uv caches and virtual
  environments fail there. Repo, `.venv`, `~/roma-env` and caches live in `/root`;
  data, results, logs, `.env`, `bootstrap.sh` and `setup_env.sh` live in `/workspace`.
  See [Cloud GPU setup](#cloud-gpu-setup-runpod-pku-dymvhumans).
- The pod sets caches to `/workspace/.cache`; `bootstrap.sh` must redirect
  `UV_CACHE_DIR`, `XDG_CACHE_HOME` and `TORCH_HOME` to `/root/.cache`.
- `uv sync --locked` fails building `torch-scatter` with a CUDA 12.8 vs 13.0
  mismatch because the isolated build environment gets the newest PyTorch. Sync
  without the three CUDA extensions, then build them with
  `uv pip install --no-build-isolation` against the locked `torch 2.10.0+cu128`.
  `uv sync --build-constraint` is not available in the installed uv.
- `romatch` defaults to `use_custom_corr=True`, which needs the optional
  `local_corr` kernel (`romatch[fused-local-corr]`). `roma_pointclouds.py` falls
  back to native PyTorch correlation when it is missing (~58 s per frame on H100).

### Data preparation

- `extract_and_convert.py`: one streamed pass over the 17.4 GiB tar from R2 (about
  30k files written). Camera poses come from `cams/` (MVSNet world-to-camera,
  per-camera `PINHOLE`); `per_frame/000000/sparse/0` tracks are re-triangulated
  with those poses as seed points: median reprojection error 1.03 px (p90 2.18 px),
  confirming `000000NN_cam.txt` maps to `cam_N`.
- `roma_pointclouds.py` (outdoor model, 3 neighbours per camera, 103 pairs per
  frame): every frame hit the 1M point cap, median reprojection error ~0.28 px.
  Keyframe step 5 over 250 frames needs 101 frames (keyframes plus `t+1`).

### Temporal settings

The default keyframe path gives every Gaussian a temporal sigma of 3x the
keyframe gap (15 frames at step 5) because `auto_init_duration` keeps the NPZ
durations, and `--init-duration` alone is ignored in that mode. The legacy
0.02 normalized floor is 1.2 frames over 61 frames but 5 frames over 250.
Both runs used a 5-frame sigma (1x gap) and a 1-frame floor:

```bash
--no-auto-init-duration --init-duration <5 / (frames - 1)> --min-duration-frames 1.0 \
--velocity-lr-start 1e-2
```

| | Pilot | Full |
|---|---|---|
| Frames | 0-60 | 0-249 |
| Combine budget, `--center-weight` | 3M, 1.0 | 15M, 1.0 |
| Points with valid velocity | 92.4% (mean 7.2 mm/frame) | |
| `--init-duration` | 0.0833 | 0.0201 |
| Preset, steps | `default_keyframe`, 7k | `default_keyframe`, 100k |

### Results

Review files are kept locally in `pilot_review/` and `fullrun_review/` (git-ignored):
validation render `renders/val_step*_0000.png` (ground truth left, render right)
and `videos/traj_4d_step{24999,49999,74999,99999}.mp4` (novel camera path,
50 frames).

- Pilot (7k steps): close match to ground truth on the held-out view, with
  sharp spear, staff and tassel.
- Full run (100k steps): correct geometry and motion over the whole clip, but
  on the same view and frame it is softer than the pilot. The right performer's
  face is washed out and the spear tip shows a magenta colour smear. In the
  trajectory videos, 100k has fewer dark speckle holes than 25k but fast arm and
  weapon motion remains smeared from novel viewpoints.
- Validation metrics (`stats/val_step*.json`) were not copied back; compare
  them before drawing conclusions from one frame.

### Open items for the next full run

- Initial velocities are capped at 10 normalized units per unit time. Over 250
  frames that is about 3.7 m/s (about 15 m/s over the 61-frame pilot), so fast
  weapon tips start with clipped velocities.
- `times_lr` and `velocity_lr_*` act in normalized time, so their effect in frames
  depends on sequence length: over 250 frames temporal centres move about 4x more
  frames per step and velocities change 4x less in metres than in the pilot.
  Matching the pilot's per-frame behaviour means `--times-lr` divided by about 4
  and `--velocity-lr-start`/`--velocity-lr-end` multiplied by about 4 (249/60);
  this may recover pilot sharpness.
- Alternatives: train the full clip as overlapping 60-frame segments, or use
  keyframe step 3 for the weapons.

### Follow-up diagnostics and tests (Oct 21)

Checkpoint statistics, full run at 100k versus pilot at 7k:

- Ruled out: temporal centres do not drift late in training (95% unchanged
  between 75k and 100k), only 0.6% of initial velocities exceed the full-run
  cap, and sigma does not widen (p95 5.0 frames, the initial value).
- Fast motion is damped in the full run: trained speed p99 is about a third of
  the initial p99, while the pilot keeps its initial p99.
- Relocation is continuous: 31.5M relocations over 895 rounds (2.1 per
  Gaussian) versus 0.82M over 58 rounds in the pilot. Per round it was limited
  by the number of dead Gaussians, never by `relocation_max_ratio`.
- On the frame both runs share (test camera 0, frame 0) the full run scores
  30.45 dB versus the pilot's 30.84 dB. The lower full-run average comes mostly
  from frames 61-249, whose initial speed p99 is about twice the pilot range's.
- With `eval_sample_every=60`, every full-run validation image falls on a
  multiple of 10 frames, so the full-run validation set contains only keyframes.

Tests (validation PSNR / SSIM / LPIPS):

| Run | Changes from its baseline | Result | Baseline |
|---|---|---|---|
| v2 test, 0-249, 25k steps | `--relocation-stop-iter 7000`, velocity LR 4e-2 to 4e-4, `--lr-schedule-steps 100000` | 25.50 / 0.935 / 0.159 | full run at 25k: 25.99 / 0.940 / 0.142 |
| A, 0-60, 7k steps | `--lambda-duration-reg 0` | 28.44 / 0.952 / 0.104 | pilot: 28.50 / 0.952 / 0.104 |
| C, 0-60, 7k steps | RoMa on every frame, keyframe step 1, `--init-duration 0.0333` (2-frame sigma), budget 7.5M | 28.90 / 0.9525 / 0.100 | pilot: 28.50 / 0.952 / 0.104 |

- v2 test: worse. Without relocation after step 7k, 78.5% of Gaussians end
  below 0.005 opacity. The higher velocity LR preserves fast motion (speed p99
  0.035 versus 0.010 per frame) but doubles median speed on static points.
  Keep relocation on and velocity LR at 1e-2.
- A: no change. Per-image differences are within +/-0.4 dB, sigma p95 rises
  only from 7.2 to 7.8 frames, and opacity and velocity statistics match the
  pilot. The duration penalty does not limit the 61-frame run.
- C: +0.40 dB. The 7.5M budget keeps the number of Gaussians visible at a
  given time equal to the pilot's (budget x sigma / span). Gains are largest
  between keyframes (+0.55 to +1.05 dB on frames 54, 56, 57, 58); frame 0 is
  unchanged (30.83 versus 30.84) and frame 59 on test camera 8 drops 0.19 dB.
  Sigma p50/p95 1.6/3.5 frames, 4.7% dead, speed p99 0.017 per frame
  (pilot 0.015). RoMa on the 36 new frames took about 45 s per frame (1M
  points, 0.28 px median); combine kept 98.4% valid velocities. Thin vertical
  weapons in fast motion are still largely missing (test camera 48, frame 54).

Run C commands (`L=/root/data_local`):

```bash
~/roma-env/bin/python scripts/pku_dymvhumans/roma_pointclouds.py --data-dir $L \
    --start 0 --end 61 --keyframe-step 1
.venv/bin/python src/combine_frames_fast_keyframes.py \
    --input-dir $L/pointclouds --output-path $L/keyframes_61frames_step1.npz \
    --frame-start 0 --frame-end 60 --keyframe-step 1 \
    --use-smart-sampling --total-budget 7500000 --center-weight 1.0
.venv/bin/python src/simple_trainer_freetime_4d_pure_relocation.py default_keyframe \
    --data-dir $L --init-npz-path $L/keyframes_61frames_step1.npz \
    --result-dir /root/results/C_step1_sigma2_0to60 --start-frame 0 --end-frame 61 \
    --no-auto-init-duration --init-duration 0.0333 --min-duration-frames 1.0 \
    --velocity-lr-start 1e-2 --max-steps 7000 --eval-steps 7000 --save-steps 7000
```

Outputs: `/workspace/results/C_step1_sigma2_0to60` (with checkpoint), logs
`/workspace/logs/{roma,combine,train}_C.log`, local review copy `C_review/`.
Point clouds now exist for 137 frames (all of 0-61 plus the step-5 frames).

Scaling C to frames 0-249: 113 more RoMa frames (about 85 min). Matching the
full run's per-moment density at a 2-frame sigma needs a 30-37M budget
(about 40-47 GB of GPU memory, 22-27 GB per checkpoint, so save only the
final one) and about 3-3.5 h for 60k steps.

### Option D (planned): separate stage and performer initialization

C confirms that temporal density limits fast motion, but at a fixed budget a
short sigma spends most Gaussians on re-adding the static stage at every
frame, and thin weapons remain under-sampled because most RoMa matches land on
the stage. Option D changes only the initialization; the trainer stays vanilla
(every Gaussian keeps position, velocity, time and duration).

- Stage: a few million points from a handful of full-image frames, kept only
  where they project outside the performer mattes (`masks/`) in most cameras.
  Velocity 0, time 0.5, sigma spanning the whole clip.
- Performers: RoMa with `--use-mask` on every frame, so all matches fall on
  bodies and weapons, plus `--neighbors 4` for more views of thin objects.
  Velocity from frame t to t+1 with `--k-neighbors 4`, sigma 1.5-2 frames.
- Budget split: about 2-3M stage, the rest to performers.
- Training: keep per-point NPZ durations (auto init duration) and set
  `--lambda-duration-reg 0` so long stage durations are not penalized; keep
  relocation and velocity LR 1e-2.
- Needs: a new combine script (about 150 lines) that merges the two sets into
  the existing NPZ format, and masked RoMa written to a separate data
  directory (for example `/root/data_local_fg`) so the full-image clouds stay
  intact. Masked RoMa for frames 0-60 takes about 45 min.
- Test: frames 0-60, 7k steps, compared against C (28.90 dB) and the pilot,
  with attention to the weapons on test camera 48, frame 54.

### Checks before Option D (Oct 21, evening)

Evaluation: `--eval-only --ckpt-path <ckpt>` re-scores a checkpoint without
training, and `--eval-matte-dir <data>/masks` adds `psnr_fg` (PSNR over pixels
inside the performer mattes) plus a per-image JSON. With
`--eval-sample-every 7` the 0-60 validation set has 70 images instead of 9.

| Checkpoint | PSNR (70 images) | PSNR inside mattes | SSIM / LPIPS |
|---|---|---|---|
| Pilot | 29.63 | 18.20 | 0.9547 / 0.099 |
| C | 29.98 | 18.69 | 0.9552 / 0.097 |

- C beats the pilot inside the mattes on 67 of 70 images (mean +0.49 dB,
  +1.24 dB on test camera 48, frame 54). The mostly black background inflates
  whole-image PSNR; the performers are about 11 dB worse than the frame
  average, so `psnr_fg` is the metric to compare D against.
- Stage share: 72% of C's 7.5M initial points fall outside the mattes (a point
  counts as performer if it projects inside the matte in at least half of the
  cameras that see it). Raw RoMa clouds are 68% stage.
- Weapons (frame 54, points projecting onto the upper staffs in test camera
  48): full-image RoMa finds 1,637, but combine's per-frame budget keeps 256 in
  C's NPZ. Masked RoMa (`--use-mask --mask-before-sample --neighbors 4`) finds
  1,955 and 557k performer points per frame (0.11 px median reprojection)
  versus 348k from the full-image cloud. Matching is not the bottleneck; the
  per-frame budget is.
- `--use-mask` alone filters after sampling, so it lowers the number of
  performer points. `--mask-before-sample` zeroes certainty outside the mattes
  before sampling. RoMa's balanced sampler builds a pairwise distance matrix
  over 4x `--num-matches`, so values much above 20,000 run out of GPU memory.

Script: `/workspace/logs/check_stage.py` (results in `/workspace/results/check_stage.json`).

### Option D result: worse than C

`scripts/pku_dymvhumans/combine_stage_performer.py` splits the existing
full-image clouds by matte vote: 2.0M stage points from frames 0, 15, 30, 45
and 60 (voxel-deduplicated from 3.4M, velocity 0, time 0.5, sigma 1.0 =
whole clip) plus 5.5M performer points from all 61 frames (19.4M available;
same smart-sampling weights as C, density voxel 0.21 m, 2-frame sigma). That
is about 90k performer points per frame versus about 42k in C.

```bash
.venv/bin/python scripts/pku_dymvhumans/combine_stage_performer.py \
    --data-dir $L --output-path $L/stage_performer_0to60.npz \
    --frame-start 0 --frame-end 60 --stage-budget 2000000 --performer-budget 5500000
.venv/bin/python src/simple_trainer_freetime_4d_pure_relocation.py default_keyframe \
    --data-dir $L --init-npz-path $L/stage_performer_0to60.npz \
    --result-dir /root/results/D_stage_perf_0to60 --start-frame 0 --end-frame 61 \
    --min-duration-frames 1.0 --lambda-duration-reg 0 --velocity-lr-start 1e-2 \
    --max-steps 7000 --eval-steps 7000 --save-steps 7000 \
    --eval-sample-every 7 --eval-matte-dir $L/masks
```

| Run | PSNR (70 images) | PSNR inside mattes | SSIM / LPIPS |
|---|---|---|---|
| Pilot | 29.63 | 18.20 | 0.9547 / 0.099 |
| C | 29.98 | 18.69 | 0.9552 / 0.097 |
| D | 29.75 | 18.49 | 0.9530 / 0.108 |

- D is below C inside the mattes on 53 of 70 images (mean -0.21 dB) and worst
  on test camera 48, frame 54 (15.89 versus 17.46 dB). Renders are speckled,
  with holes in the second performer's face.
- The static stage did not survive training: of 2.0M whole-clip Gaussians,
  278k still have sigma above 15 frames at step 7k (median opacity 0.008,
  19% dead). The rest faded and were relocated as copies of short-lived
  Gaussians or narrowed, so the stage was rebuilt per frame with fewer
  Gaussians than C spends on it.
- The upper staffs are missing in both C and D on the held-out test camera,
  so more initial performer points alone do not recover the weapons.

Outputs: `/workspace/results/D_stage_perf_0to60` (with checkpoint), NPZ in
`/workspace/data/stage_performer_0to60.npz`, local review copy `D_review/`.

### Missing weapons: lost in training, not a held-out-view problem

C's checkpoint rendered at frame 54 from test camera 48 and its training
neighbours 46, 47, 49 and 50: the raised staffs are missing in all five. In the
top 20% of the matte rows (the staffs) the render's mean brightness is
0.01-0.11 against 0.28-0.52 in the ground truth, in training and held-out
cameras alike.

Gaussians projecting onto the upper staffs in camera 48 at frame 54:

| | Combined opacity > 0.05 | > 0.2 | Footprint p50 | Colour (mean RGB) p50 / p90 |
|---|---|---|---|---|
| Initialization | 9,361 | 6,666 | 4.3 px | 0.10 / 0.53 |
| C after 7k steps | 2,474 | 2,125 | 0.28 px | 0.00 / 0.00 |

40% of the slots that started there were relocated and the rest faded (base
opacity p50 0.057). The survivors are sub-pixel and black. With a black
background, turning a misplaced Gaussian black costs nothing in any frame, so
staff Gaussians that are right at one frame but wrong at neighbouring frames
(thin, fast object; nearest-neighbour velocity) are darkened instead of
corrected. More initial points (D) or a longer run do not address this.

Scripts: `/workspace/logs/render_f54.py`, `/workspace/logs/count_f54.py`,
`/workspace/logs/count_f54b.py`; renders in `/workspace/results/render_f54_C`.

Infrastructure: the `/workspace` geesefs mount disconnected three times and
lost its directory symlinks across a pod restart. Data now lives in
`/root/data_local` (real copies of images, masks, `sparse/0`, point clouds
and NPZs); results are written to `/root/results` and copied to `/workspace`
afterwards.

### Reference review: papers and GitHub (Oct 21, night)

FreeTimeGS (Wang et al., CVPR 2025, arXiv 2506.05348) against our runs:

| Topic | Paper | Our runs |
|---|---|---|
| 4D opacity regularization (`lambda_4d_reg`) | 1e-2 | 1e-4 in the `default_keyframe` presets |
| Iterations | 30k for 300 frames | 7k for 61 frames, 100k for 250 |
| Gaussians | about 1M best, 2M slightly worse | 3M-15M |
| Initialization | RoMa every frame, nearest-neighbour velocity | same |
| Fast-motion data | SelfCap, 60 fps | 25 fps, about 2.4x the motion per frame |

- The regularization is the paper's fix for fast motion: with only rendering
  losses, near-opaque Gaussians block gradients and the optimization "gets
  trapped in local minima on fast-moving regions". Ablation on the fastest 10
  frames: 29.09 dB at 0, 29.79 at 1e-3, 30.75 at 1e-2, 27.33 at 1e-1. Nothing
  in this repo shows the reduction to 1e-4 was tested.
- The paper's dynamic-region PSNR crops the matte bounding box and fills the
  rest with black, so it is not comparable to `psnr_fg`.

FreeTimeGS++ (Lee et al., 2026, arXiv 2605.03337) reimplements FreeTimeGS and
tests its heuristics:

- Relocation that copies the target whole (opacity included) scores slightly
  higher PSNR but leaves haze-like wobble in static background; recomputing
  opacity and scale MCMC-style is more stable. Ours copies at 0.8x opacity and
  leaves the source unchanged, which is the haze-prone variant.
- Long-lived Gaussians take over static background on their own, but nothing
  protects them; their fix is a learnable per-Gaussian persistence gate
  (opacity x (g + (1 - g) x temporal opacity)). This is the principled form of
  Option D.
- Flow-based (UFM) velocity initialization instead of nearest-neighbour:
  -0.18 dB alone on SelfCap, so better velocities are not a big win by
  themselves.
- Per-camera affine colour correction, used only in training, gave the
  largest gain (+0.7 dB on SelfCap) and cut run-to-run standard deviation from
  0.36 to 0.09 dB.
- Identical runs on SelfCap vary by 0.36 dB (standard deviation). Our A
  (-0.06 dB) and D (-0.21 dB) differences are within that; C's +0.49 dB over
  the pilot (67 of 70 images) is likely real but unconfirmed by a second seed.
- Their SelfCap baseline config: 500k Gaussians for 60 frames, initial scale
  0.1 (ours 0.03), time-centre learning rate 1.6e-4 (ours 1e-3), velocity
  1.6e-3 (ours 1e-2), 700k RoMa points per frame, frame stride 1, relocation
  from step 500 to 25k.

GitHub: upstream issue #1 is empty; issue #3 (initialization) has one reply,
from a FreeTimeGS++ author, saying RoMa plus nearest-neighbour velocity worked;
PR #2 is a closed experiment at about 15 dB with no write-up. Nobody reports
black floaters or missing thin parts. The FreeTimeGS++ authors' repo now
returns 404; the mirror `zhixiongzuo/FreeTimeGSPlusPlus` has their configs
(non-commercial licence). `samaust/openftgs` is a pre-alpha Apache-2.0
clean-room reimplementation.

Test order on frames 0-60 (about 10 minutes each): opacity regularization at
1e-2; transparency supervision (matte plus floor as alpha, random background);
the FreeTimeGS++ baseline settings (1-2M Gaussians, scale 0.1, their learning
rates); per-camera colour correction; C with a second seed. Then MCMC-style
relocation and the persistence gate.

### Test E: opacity regularization at the paper's 1e-2

C's command with `--lambda-4d-reg 1e-2` (everything else identical, same NPZ):

```bash
.venv/bin/python src/simple_trainer_freetime_4d_pure_relocation.py default_keyframe \
    --data-dir $L --init-npz-path $L/keyframes_61frames_step1.npz \
    --result-dir /root/results/E_reg1e-2_0to60 --start-frame 0 --end-frame 61 \
    --no-auto-init-duration --init-duration 0.0333 --min-duration-frames 1.0 \
    --velocity-lr-start 1e-2 --lambda-4d-reg 1e-2 \
    --max-steps 7000 --eval-steps 7000 --save-steps 7000 \
    --eval-sample-every 7 --eval-matte-dir $L/masks
```

| Run | PSNR (70 images) | PSNR inside mattes | SSIM / LPIPS | Camera 48, frame 54 (inside mattes) |
|---|---|---|---|---|
| C (1e-4) | 29.98 | 18.69 | 0.9552 / 0.097 | 17.46 |
| E (1e-2) | 30.08 | 18.87 | 0.9560 / 0.095 | 17.02 |

- E beats C inside the mattes on 55 of 70 images (mean +0.17 dB, median
  +0.16, range -0.43 to +0.75). Consistent, but smaller than the 0.36 dB
  run-to-run spread FreeTimeGS++ reports, so it needs a second seed.
- It does not bring the weapons back. Upper-staff brightness at frame 54 is
  0.005-0.08 against 0.28-0.52 in ground truth (training and held-out cameras),
  and only 22 Gaussians over the staffs in camera 48 reach opacity 0.2 (C:
  2,125). Arms and the second performer's head are slightly more smeared.
- Training time 420 s, same as C.

Outputs: `/workspace/results/E_reg1e-2_0to60` (with checkpoint),
`/workspace/results/render_f54_E`, local review copy `E_review/`.

### Why the staffs starve (initial points at frame 54)

`/root/staff_vel.py` takes the 657 initial Gaussians centred on frame 54 that
project onto the upper staffs in camera 48, and the 49k on the rest of the
performers as a control, and checks whether they land inside the mattes in
all 60 cameras:

| | Frame 53 | Frame 54 | Frame 55 |
|---|---|---|---|
| Staffs, with kNN velocity | 24% | 46% | 15% |
| Staffs, held still | 21% | | 13% |
| Rest of performers, with kNN velocity | 64% | 71% | 64% |

- Geometry: at their own frame the staff points miss the mattes in half the
  views. Overlaid on the images (`E_review/staff_points_f54.png`), many sit in
  empty space beside the shorter staff, on camera 48's rays but at the wrong
  depth; the longer staff is mostly right. Two parallel, identical white rods
  are an easy mismatch for RoMa.
- Velocity: kNN speed is 2.8 mm/frame (median), barely better than no motion.
  The staffs move more than their own width per frame (matte overlap 0.11-0.15
  at one frame, 0 at two), so the nearest point in the next frame is not the
  same part of the staff.
- Weight: the upper staffs are 0.12% of the pixels of a frame (the performers
  6.8%), and relocation only clones Gaussians that are already opaque, so once
  the staff Gaussians fade nothing is copied back.
- With a 2-frame sigma each Gaussian is still at 61% opacity one frame away,
  where it is over black background; darkening or fading it is the cheapest
  fix for the optimizer.

### Velocities from same-camera matches (F1, F2)

`roma_pointclouds.py --temporal-velocity` also matches every camera's frame t
against its own frame t+1 with RoMa, moves both source pixels of each
triangulated point and triangulates the moved pair again; the difference is
the velocity. `combine_frames_fast_keyframes.py --flow-velocities` uses it
(nearest-neighbour velocity fills the rest), and `--speed-duration-size S`
sets each point's sigma to S / speed frames, clipped to
[`--min-duration-frames`, `--duration-frames`].

Which check to trust was settled on the upper-staff points of frame 54 in the
pairs that include camera 48 (`/workspace/logs/staff_flow_diag*.py`):

| Staff points, frame 54 to 55 | Kept | Inside mattes at 54 | Moved, at 55 | Held still, at 55 |
|---|---|---|---|---|
| RoMa 864x864, no filter | 100% | 52% | 44% | 15% |
| RoMa, moved point reprojects within 3 px in both views | 53% | 52% | 48% | 13% |
| RAFT (torchvision, full resolution), no filter | 100% | 52% | 34% | 15% |

- RoMa's certainty (median 0.04 on the staff) and forward-backward error
  (median 15 px; 0.26 px elsewhere) reject nearly every staff point although
  the forward match is right, so validity is only "the two cameras' moves
  agree within 3 px" (`--max-flow-reproj 3`). Certainty and forward-backward
  limits are optional; all three scores are saved per point
  (`velocity_quality_frame*.npy`).
- The staffs move about 20 px per frame in camera 48, 5.7 cm per frame in 3D.
  RoMa at 864x1536 did not help; RAFT tracks them less well than RoMa.
- Cost: 74 s per frame on one H100 (45 s without), two workers in parallel.

Pilots (frames 0-60, 7k steps, E's settings: `--lambda-4d-reg 1e-2
--velocity-lr-start 1e-2`, script `/root/run_F.sh`):

- F1: same-camera velocities, uniform 2-frame sigma (isolates velocity). The
  combine step's velocity weighting now sees real speeds, so fast parts also
  get more of the 7.5M budget.
- F2: F1 plus `--speed-duration-size 0.01 --min-duration-frames 0.5` (sigma
  between 0.5 and 2 frames; the staffs get 0.5), trained with
  `--min-duration-frames 0.5` and per-point durations.

Initialization check (`staff_vel.py` on F1's NPZ): staff Gaussians centred on
frame 54 now land inside the mattes in 40% of camera views at frame 55 (C's
nearest-neighbour velocities: 15%) and 29% at frame 53 (24%). There are more
of them (865 versus 657; 9,836 versus 6,666 Gaussians on the upper staffs at
frame 54) because the budget's velocity weighting now sees real speeds.

| Run | PSNR (70 images) | PSNR inside mattes | SSIM / LPIPS | Camera 48, frame 54 | Upper-staff brightness, cameras 46-50 (truth 0.28-0.52) | Staff Gaussians, opacity > 0.2 |
|---|---|---|---|---|---|---|
| C | 29.98 | 18.69 | 0.9552 / 0.097 | 17.46 | 0.01-0.11 | 2,125 |
| E | 30.08 | 18.87 | 0.9560 / 0.095 | 17.02 | 0.005-0.08 | 22 |
| F1 | 30.07 | 18.83 | 0.9559 / 0.096 | 16.67 | 0.016-0.09 | 37 |
| F2 | 30.06 | 18.79 | 0.9555 / 0.096 | 17.37 | 0.028-0.10 | 37 |

- Against E inside the mattes: F1 -0.03 dB (wins 32 of 70), F2 -0.08 dB (30
  of 70), both within run-to-run noise. F2 is +0.14 dB on frames 45-60, where
  the weapons swing fastest. Both remain above C (+0.14 and +0.10 dB).
- The staffs are still missing. F2 shows a faint sliver of the upper staff and
  slightly sharper arms in camera 48; F1 looks like E.
- So a better start is not enough: the staff Gaussians start better placed and
  more numerous and are still darkened or relocated during training. The
  remaining suspects are on the training side: black background (darkening is
  free), opacity regularization (E has 22 opaque staff Gaussians, C 2,125) and
  relocation.

Outputs: `/workspace/results/F1_flowvel_0to60`,
`/workspace/results/F2_flowvel_speeddur_0to60` (with checkpoints); point
clouds with velocities in `/root/data_local/pointclouds_flow`; NPZs
`/root/data_local/flow_61frames_step1.npz` and
`flow_speeddur_61frames_step1.npz`; logs `/workspace/logs/*F*`; local review
copy `F_review/`.
