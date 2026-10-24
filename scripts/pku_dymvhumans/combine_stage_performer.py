"""Build an init NPZ with a static stage set and per-frame performer points.

Each point of a per-frame RoMa cloud is classified by projecting it into every
camera's matte (``masks/<camera>/<frame>.png``) at its own frame: it is a
performer point if it lands inside the matte in at least ``--performer-vote``
of the cameras that see it, otherwise a stage point.

- Stage: stage points from a few frames, voxel-deduplicated, with zero
  velocity, time at the clip centre and a duration spanning the clip, so the
  stage is represented once instead of at every frame.
- Performers: performer points from every frame, velocity from the nearest
  performer point at frame t+1, short duration, then smart-sampled to the
  performer budget with the same weights as combine_frames_fast_keyframes.py.

The output uses the same keys as combine_frames_fast_keyframes.py, with
per-point durations, so the trainer keeps them when auto init duration is on.

    python scripts/pku_dymvhumans/combine_stage_performer.py \\
        --data-dir /root/data_local --output-path /root/data_local/stage_performer_0to60.npz \\
        --frame-start 0 --frame-end 60 --stage-budget 2000000 --performer-budget 5500000
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from combine_frames_fast_keyframes import (  # noqa: E402
    compute_velocity_knn, estimate_scene_scale, load_frame_data, smart_density_velocity_sampling,
)


def _load_read_write_model():
    spec = importlib.util.spec_from_file_location(
        "read_write_model", REPO_ROOT / "datasets" / "read_write_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_cameras(data_dir: Path):
    rwm = _load_read_write_model()
    cameras = rwm.read_cameras_binary(str(data_dir / "sparse" / "0" / "cameras.bin"))
    images = rwm.read_images_binary(str(data_dir / "sparse" / "0" / "images.bin"))
    cams = []
    for image in sorted(images.values(), key=lambda im: im.name):
        cam = cameras[image.camera_id]
        fx, fy, cx, cy = cam.params[:4]
        cams.append({
            "name": Path(image.name).stem,
            "K": np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32),
            "R": rwm.qvec2rotmat(image.qvec).astype(np.float32),
            "t": np.asarray(image.tvec, dtype=np.float32),
            "width": cam.width, "height": cam.height,
        })
    return cams


def matte_fraction(xyz, cams, mask_dir: Path, frame: int, threshold: int = 128):
    """Fraction of the cameras seeing each point whose matte covers it, and the camera count."""
    inside = np.zeros(len(xyz), np.int32)
    seen = np.zeros(len(xyz), np.int32)
    for cam in cams:
        matte = np.asarray(Image.open(mask_dir / cam["name"] / f"{frame:06d}.png").convert("L")) >= threshold
        pc = xyz @ cam["R"].T + cam["t"]
        uv = pc @ cam["K"].T
        z = pc[:, 2]
        x = np.round(uv[:, 0] / np.maximum(z, 1e-6)).astype(np.int64)
        y = np.round(uv[:, 1] / np.maximum(z, 1e-6)).astype(np.int64)
        ok = (z > 0.1) & (x >= 0) & (x < cam["width"]) & (y >= 0) & (y < cam["height"])
        seen += ok
        hit = np.zeros(len(xyz), bool)
        hit[ok] = matte[y[ok], x[ok]]
        inside += hit
    return inside / np.maximum(seen, 1), seen


def voxel_dedupe(xyz, voxel: float):
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return np.sort(first)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Folder with sparse/0, masks/ and pointclouds/")
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="Per-frame clouds (default: <data-dir>/pointclouds)")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, required=True, help="Last frame (inclusive)")
    parser.add_argument("--performer-vote", type=float, default=0.5,
                        help="Min fraction of seeing cameras whose matte covers a performer point")
    parser.add_argument("--min-cameras", type=int, default=2,
                        help="Drop points seen by fewer cameras")

    stage = parser.add_argument_group("Stage")
    stage.add_argument("--stage-frames", type=int, default=5,
                       help="Number of evenly spaced frames the stage is taken from")
    stage.add_argument("--stage-budget", type=int, default=2_000_000)
    stage.add_argument("--stage-voxel", type=float, default=0.004,
                       help="Dedupe voxel in world units across stage frames")
    stage.add_argument("--stage-duration", type=float, default=1.0,
                       help="Temporal sigma in normalized time (1.0 = the whole clip)")

    perf = parser.add_argument_group("Performers")
    perf.add_argument("--performer-budget", type=int, default=5_500_000)
    perf.add_argument("--performer-duration-frames", type=float, default=2.0,
                      help="Temporal sigma in frames")
    perf.add_argument("--k-neighbors", type=int, default=1)
    perf.add_argument("--max-velocity-distance", type=float, default=0.5)
    perf.add_argument("--velocity-weight", type=float, default=5.0)
    perf.add_argument("--center-weight", type=float, default=1.0)
    perf.add_argument("--voxel-size", type=float, default=None,
                      help="Smart-sampling density voxel (default: estimated from the first frame)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = args.input_dir or args.data_dir / "pointclouds"
    mask_dir = args.data_dir / "masks"
    cams = load_cameras(args.data_dir)
    rng = np.random.default_rng(args.seed)
    span = max(args.frame_end - args.frame_start, 1)
    frames = list(range(args.frame_start, args.frame_end + 1))
    stage_frames = set(np.linspace(args.frame_start, args.frame_end, args.stage_frames).round().astype(int))

    def split(frame):
        xyz, rgb = load_frame_data(input_dir, frame)
        if xyz is None:
            return None
        frac, seen = matte_fraction(xyz, cams, mask_dir, frame)
        usable = seen >= args.min_cameras
        is_perf = usable & (frac >= args.performer_vote)
        return xyz, rgb, is_perf, usable & ~is_perf

    stage_xyz, stage_rgb = [], []
    perf_pos, perf_vel, perf_rgb, perf_time, perf_hasvel = [], [], [], [], []
    stats = []
    current = split(frames[0])
    for frame in frames:
        nxt = split(frame + 1)
        if current is None:
            print(f"  frame {frame}: missing, skipped", flush=True)
            current = nxt
            continue
        xyz, rgb, is_perf, is_stage = current
        if frame in stage_frames:
            stage_xyz.append(xyz[is_stage])
            stage_rgb.append(rgb[is_stage])
        p = xyz[is_perf]
        if nxt is not None:
            vel, valid = compute_velocity_knn(p, nxt[0][nxt[2]], max_distance=args.max_velocity_distance,
                                              k=args.k_neighbors)
        else:
            vel, valid = np.zeros_like(p), np.zeros(len(p), bool)
        perf_pos.append(p)
        perf_vel.append(vel)
        perf_rgb.append(rgb[is_perf])
        perf_time.append(np.full(len(p), (frame - args.frame_start) / span, np.float32))
        perf_hasvel.append(valid)
        stats.append({"frame": frame, "points": int(len(xyz)), "performer": int(is_perf.sum()),
                      "stage": int(is_stage.sum()), "velocity_valid": float(valid.mean()) if len(p) else 0.0})
        print(f"  frame {frame}: {len(xyz):,} points, {is_perf.sum():,} performer, "
              f"{is_stage.sum():,} stage, {100 * valid.mean():.1f}% velocity", flush=True)
        current = nxt

    # Stage: dedupe across frames, then subsample uniformly to the budget
    s_xyz, s_rgb = np.concatenate(stage_xyz), np.concatenate(stage_rgb)
    n_stage_raw = len(s_xyz)
    keep = voxel_dedupe(s_xyz, args.stage_voxel)
    if len(keep) > args.stage_budget:
        keep = np.sort(rng.choice(keep, args.stage_budget, replace=False))
    s_xyz, s_rgb = s_xyz[keep], s_rgb[keep]
    print(f"\nStage: {n_stage_raw:,} points from frames {sorted(int(f) for f in stage_frames)} "
          f"-> {len(s_xyz):,} after dedupe/budget")

    # Performers: same smart sampling as the keyframe combiner
    p_pos, p_vel, p_rgb = np.concatenate(perf_pos), np.concatenate(perf_vel), np.concatenate(perf_rgb)
    p_time, p_hasvel = np.concatenate(perf_time), np.concatenate(perf_hasvel)
    n_perf_raw = len(p_pos)
    voxel = args.voxel_size or estimate_scene_scale(perf_pos[0], seed=args.seed)["suggested_voxel_size"]
    p_pos, p_vel, p_rgb, idx = smart_density_velocity_sampling(
        p_pos, p_vel, p_rgb, target_count=args.performer_budget, voxel_size=voxel,
        velocity_weight=args.velocity_weight, center_weight=args.center_weight, seed=args.seed,
    )
    p_time, p_hasvel = p_time[idx], p_hasvel[idx]
    print(f"Performers: {n_perf_raw:,} points -> {len(p_pos):,} (voxel {voxel:.4f}), "
          f"{100 * p_hasvel.mean():.1f}% with velocity")

    n_s, n_p = len(s_xyz), len(p_pos)
    positions = np.concatenate([s_xyz, p_pos]).astype(np.float32)
    velocities = np.concatenate([np.zeros_like(s_xyz), p_vel]).astype(np.float32)
    colors = np.concatenate([s_rgb, p_rgb]).astype(np.float32)
    if colors.max() > 1.0:
        colors /= 255.0
    times = np.concatenate([np.full(n_s, 0.5, np.float32), p_time])[:, None]
    durations = np.concatenate([
        np.full(n_s, args.stage_duration, np.float32),
        np.full(n_p, args.performer_duration_frames / span, np.float32),
    ])[:, None]
    has_velocity = np.concatenate([np.ones(n_s, bool), p_hasvel])

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_path,
        positions=positions, velocities=velocities, colors=colors, times=times,
        durations=durations, has_velocity=has_velocity,
        frame_start=args.frame_start, frame_end=args.frame_end + 1, time_denominator=span,
        keyframe_step=1, n_stage=n_s, mode="stage_performer_split",
    )
    summary = {"stage": n_s, "performer": n_p, "stage_raw": n_stage_raw, "performer_raw": n_perf_raw,
               "voxel": float(voxel), "frames": stats}
    with open(args.output_path.with_suffix(".json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"Saved {len(positions):,} points ({n_s:,} stage + {n_p:,} performer) to {args.output_path}")


if __name__ == "__main__":
    main()
