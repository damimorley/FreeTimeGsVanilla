#!/usr/bin/env python3
"""
Dense per-frame point clouds for FreeTimeGS from RoMa matches.

For each needed frame, neighbouring camera pairs are densely matched with RoMa,
the matches are triangulated with the calibrated poses in <data-dir>/sparse/0,
filtered, and saved in the format src/combine_frames_fast_keyframes.py reads:

    <data-dir>/pointclouds/points3d_frame000000.npy   [M, 3] float32, world units
    <data-dir>/pointclouds/colors_frame000000.npy     [M, 3] float32, RGB 0-255
    <data-dir>/pointclouds/stats.jsonl                one line of filter stats per frame

With --temporal-velocity each point also gets a velocity from matching every
camera's frame t against its frame t+1 (velocities_frame*.npy and friends);
combine with --flow-velocities to use them instead of nearest-neighbour ones.

Only the frames the keyframe pipeline uses are processed: every keyframe
(start, start+step, ..., plus end-1) and the frame after each one, which the
combine step uses for velocity. With --combine, the init NPZ is built too:

    <data-dir>/keyframes_<N>frames_step<S>.npz

Install RoMa in the environment that runs this script (GPU required in practice):
    pip install romatch

Usage:
    python scripts/pku_dymvhumans/roma_pointclouds.py \\
        --data-dir /data/zxyun/PKU-DyMVHumans/part1/1080_Kungfu_Weapon_Pair_m12m13/data_FreeTimeGS \\
        --start 0 --end 250 --keyframe-step 5 --combine
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_read_write_model():
    spec = importlib.util.spec_from_file_location(
        "read_write_model", REPO_ROOT / "datasets" / "read_write_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rwm = _load_read_write_model()


# ----------------------------------------------------------------------------
# Cameras and pairs
# ----------------------------------------------------------------------------

def load_cameras(data_dir: Path):
    cameras = rwm.read_cameras_binary(str(data_dir / "sparse" / "0" / "cameras.bin"))
    images = rwm.read_images_binary(str(data_dir / "sparse" / "0" / "images.bin"))
    cams = []
    for image in sorted(images.values(), key=lambda im: im.name):
        cam = cameras[image.camera_id]
        if cam.model != "PINHOLE":
            raise SystemExit(f"Expected undistorted PINHOLE cameras, got {cam.model}")
        fx, fy, cx, cy = cam.params
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        R = rwm.qvec2rotmat(image.qvec)
        t = np.asarray(image.tvec, dtype=np.float64)
        cams.append({
            "name": Path(image.name).stem, "K": K, "R": R, "t": t,
            "P": K @ np.hstack([R, t[:, None]]),
            "center": -R.T @ t, "axis": R[2],
            "width": cam.width, "height": cam.height,
        })
    return cams


def select_pairs(cams, neighbors: int, max_angle_deg: float):
    centers = np.stack([c["center"] for c in cams])
    axes = np.stack([c["axis"] for c in cams])
    cos_max = np.cos(np.deg2rad(max_angle_deg))
    pairs = set()
    for i in range(len(cams)):
        dist = np.linalg.norm(centers - centers[i], axis=1)
        dist[i] = np.inf
        dist[axes @ axes[i] < cos_max] = np.inf
        for j in np.argsort(dist)[:neighbors]:
            if np.isfinite(dist[j]):
                pairs.add((min(i, int(j)), max(i, int(j))))
    return sorted(pairs)


def frames_to_process(start: int, end: int, step: int):
    keyframes = list(range(start, end, step))
    if end - 1 not in keyframes:
        keyframes.append(end - 1)
    frames = set(keyframes)
    frames.update(k + 1 for k in keyframes if k + 1 < end)
    return sorted(frames)


# ----------------------------------------------------------------------------
# Matching and triangulation
# ----------------------------------------------------------------------------

def load_roma(args):
    import torch
    import romatch

    try:
        import local_corr  # noqa: F401  (optional kernel from romatch[fused-local-corr])
        use_custom_corr = True
    except ImportError:
        use_custom_corr = False
    print(f"[roma] Local correlation: {'fused CUDA kernel' if use_custom_corr else 'native PyTorch'}")

    builder = getattr(romatch, f"roma_{args.model}")
    upsample = tuple(int(x) for x in args.upsample_res.split(","))
    model = builder(device=args.device, coarse_res=args.coarse_res,
                    upsample_res=upsample if len(upsample) == 2 else upsample[0],
                    use_custom_corr=use_custom_corr)
    model.eval()
    torch.set_grad_enabled(False)
    return model


def sample_rgb(image: np.ndarray, uv: np.ndarray):
    h, w = image.shape[:2]
    x = np.clip(np.round(uv[:, 0]).astype(np.int64), 0, w - 1)
    y = np.clip(np.round(uv[:, 1]).astype(np.int64), 0, h - 1)
    return image[y, x]


def reprojection(cam, X):
    cam_pts = X @ cam["R"].T + cam["t"]
    uv = cam_pts @ cam["K"].T
    return uv[:, :2] / np.clip(uv[:, 2:3], 1e-9, None), cam_pts[:, 2]


def match_pair(model, img_a, img_b, cam_a, cam_b, rgb_a, mask_a, mask_b, args):
    warp, certainty = model.match(img_a, img_b, device=args.device)
    num = args.num_matches
    if args.mask_before_sample and mask_a is not None:
        import torch
        grid_a, grid_b = model.to_pixel_coordinates(
            warp, cam_a["height"], cam_a["width"], cam_b["height"], cam_b["width"])
        inside = ((sample_rgb(mask_a, grid_a.reshape(-1, 2).float().cpu().numpy()) >= args.mask_threshold)
                  & (sample_rgb(mask_b, grid_b.reshape(-1, 2).float().cpu().numpy()) >= args.mask_threshold))
        inside = torch.from_numpy(inside).to(certainty.device).reshape(certainty.shape)
        certainty = certainty * inside
        # Sampling draws 4x num candidates without replacement from nonzero entries
        num = min(num, int(inside.sum().item()) // 4)
        if num == 0:
            return None
    matches, cert = model.sample(warp, certainty, num=num)
    kpts_a, kpts_b = model.to_pixel_coordinates(
        matches, cam_a["height"], cam_a["width"], cam_b["height"], cam_b["width"])
    kpts_a = kpts_a.float().cpu().numpy().astype(np.float64)
    kpts_b = kpts_b.float().cpu().numpy().astype(np.float64)
    cert = cert.float().cpu().numpy()

    keep = cert >= args.min_certainty
    if mask_a is not None:
        keep &= sample_rgb(mask_a, kpts_a) >= args.mask_threshold
        keep &= sample_rgb(mask_b, kpts_b) >= args.mask_threshold
    kpts_a, kpts_b, cert = kpts_a[keep], kpts_b[keep], cert[keep]
    if len(kpts_a) == 0:
        return None

    X_h = cv2.triangulatePoints(cam_a["P"], cam_b["P"], kpts_a.T, kpts_b.T)
    X = (X_h[:3] / X_h[3]).T

    uv_a, depth_a = reprojection(cam_a, X)
    uv_b, depth_b = reprojection(cam_b, X)
    err = np.maximum(np.linalg.norm(uv_a - kpts_a, axis=1), np.linalg.norm(uv_b - kpts_b, axis=1))
    ok = ((err <= args.max_reproj)
          & (depth_a > args.min_depth) & (depth_b > args.min_depth)
          & (depth_a < args.max_depth) & (depth_b < args.max_depth))
    if not ok.any():
        return None
    return {
        "xyz": X[ok].astype(np.float32),
        "rgb": sample_rgb(rgb_a, kpts_a[ok]).astype(np.float32),
        "err": err[ok].astype(np.float32),
        "kpts_a": kpts_a[ok],
        "kpts_b": kpts_b[ok],
        "sampled": int(len(keep)),
        "after_certainty": int(keep.sum()),
    }


def temporal_flow(model, img_t, img_next, args):
    """Dense RoMa match of one camera's frame t against its frame t+1.

    Returns the forward field (t -> t+1), the backward field (t+1 -> t) and
    their certainties, all [1, C, h, w] in RoMa's normalized coordinates.
    """
    warp, certainty = model.match(img_t, img_next, device=args.device)
    w = warp.shape[2] // 2
    forward = warp[:, :, :w, 2:].permute(0, 3, 1, 2).float()
    backward = warp[:, :, w:, :2].permute(0, 3, 1, 2).float()
    return (forward, certainty[:, None, :, :w].float(),
            backward, certainty[:, None, :, w:].float())


def sample_field(field, uv, cam):
    """Bilinearly sample a [1, C, h, w] field at pixel coordinates uv [N, 2]."""
    import torch
    import torch.nn.functional as F

    grid = torch.as_tensor(uv, dtype=torch.float32, device=field.device).clone()
    grid[:, 0] = grid[:, 0] * 2 / cam["width"] - 1
    grid[:, 1] = grid[:, 1] * 2 / cam["height"] - 1
    out = F.grid_sample(field, grid[None, :, None, :], mode="bilinear", align_corners=False)
    return out[0, :, :, 0].T.cpu().numpy().astype(np.float64)


def follow_flow(flow, uv, cam):
    """Move pixels uv to frame t+1; return new pixels, certainty and forward-backward error."""
    forward, cert_f, backward, cert_b = flow
    norm_next = sample_field(forward, uv, cam)
    uv_next = np.stack([(norm_next[:, 0] + 1) * cam["width"] / 2,
                        (norm_next[:, 1] + 1) * cam["height"] / 2], axis=1)
    norm_back = sample_field(backward, uv_next, cam)
    uv_back = np.stack([(norm_back[:, 0] + 1) * cam["width"] / 2,
                        (norm_back[:, 1] + 1) * cam["height"] / 2], axis=1)
    cert = np.minimum(sample_field(cert_f, uv, cam)[:, 0], sample_field(cert_b, uv_next, cam)[:, 0])
    return uv_next, cert, np.linalg.norm(uv_back - uv, axis=1)


def temporal_velocities(chunks, pairs, cams, model, pils, frame, args):
    """Per-point velocity (world units per frame) from same-camera matches t -> t+1.

    Both source pixels of each triangulated point are moved to frame t+1 with
    RoMa's dense match of that camera against its next frame, and the moved
    pair is triangulated again. A velocity is valid if the moved point
    reprojects within --max-flow-reproj in both views (the two cameras' moves
    agree). On thin, fast parts RoMa's certainty and forward-backward error are
    poor even where the forward match is right, so those checks are optional.

    Returns velocities, validity and per-point (certainty, forward-backward px,
    reprojection px) so the thresholds can be changed without rematching.
    """
    image_dir = args.data_dir / "images"
    next_paths = {i: image_dir / cams[i]["name"] / f"{frame + 1:0{args.frame_digits}d}.png" for i in pils}
    vel = [np.zeros((len(c["xyz"]), 3), np.float32) for c in chunks]
    valid = [np.zeros(len(c["xyz"]), bool) for c in chunks]
    quality = [np.full((len(c["xyz"]), 3), np.inf, np.float32) for c in chunks]
    if not all(p.exists() for p in next_paths.values()):
        return vel, valid, quality

    moved = [{} for _ in chunks]
    for i in sorted(pils):
        flow = temporal_flow(model, pils[i], Image.open(next_paths[i]).convert("RGB"), args)
        for k, (c, (a, b)) in enumerate(zip(chunks, pairs)):
            for side, cam_index in (("a", a), ("b", b)):
                if cam_index == i:
                    moved[k][side] = follow_flow(flow, c[f"kpts_{side}"], cams[i])
        del flow

    for k, (c, (a, b)) in enumerate(zip(chunks, pairs)):
        (uv_a, cert_a, fb_a), (uv_b, cert_b, fb_b) = moved[k]["a"], moved[k]["b"]
        X_h = cv2.triangulatePoints(cams[a]["P"], cams[b]["P"], uv_a.T, uv_b.T)
        X = (X_h[:3] / X_h[3]).T
        rep_a, depth_a = reprojection(cams[a], X)
        rep_b, depth_b = reprojection(cams[b], X)
        err = np.maximum(np.linalg.norm(rep_a - uv_a, axis=1), np.linalg.norm(rep_b - uv_b, axis=1))
        v = X - c["xyz"].astype(np.float64)
        cert, cycle = np.minimum(cert_a, cert_b), np.maximum(fb_a, fb_b)
        ok = ((cert >= args.min_flow_certainty)
              & ((cycle <= args.max_flow_cycle) if args.max_flow_cycle > 0 else True)
              & (err <= args.max_flow_reproj)
              & (depth_a > args.min_depth) & (depth_b > args.min_depth)
              & (np.linalg.norm(v, axis=1) <= args.max_speed))
        vel[k][ok] = v[ok]
        valid[k] = ok
        quality[k] = np.stack([cert, cycle, err], axis=1).astype(np.float32)
    return vel, valid, quality


def statistical_outlier_filter(xyz, k: int, std_ratio: float):
    if k <= 0 or len(xyz) <= k:
        return np.ones(len(xyz), dtype=bool)
    dist, _ = cKDTree(xyz).query(xyz, k=k + 1, workers=-1)
    mean_dist = dist[:, 1:].mean(axis=1)
    return mean_dist <= mean_dist.mean() + std_ratio * mean_dist.std()


def voxel_dedupe(xyz, err, voxel: float):
    if voxel <= 0:
        return np.arange(len(xyz))
    order = np.argsort(err)
    keys = np.floor(xyz[order] / voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return order[first]


def save_npy_atomic(path: Path, array: np.ndarray):
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        np.save(f, array)
    os.replace(tmp, path)


def process_frame(frame, cams, pairs, model, args, out_dir: Path, rng):
    image_dir = args.data_dir / "images"
    mask_dir = args.data_dir / "masks"
    pils, rgbs, masks = {}, {}, {}
    used = sorted({i for pair in pairs for i in pair})
    for i in used:
        path = image_dir / cams[i]["name"] / f"{frame:0{args.frame_digits}d}.png"
        pils[i] = Image.open(path).convert("RGB")
        rgbs[i] = np.asarray(pils[i])
        if args.use_mask:
            mask_path = mask_dir / cams[i]["name"] / path.name
            masks[i] = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0

    chunks, chunk_pairs, sampled, after_cert = [], [], 0, 0
    for a, b in pairs:
        result = match_pair(model, pils[a], pils[b], cams[a], cams[b], rgbs[a],
                            masks.get(a), masks.get(b), args)
        if result is None:
            continue
        chunks.append(result)
        chunk_pairs.append((a, b))
        sampled += result["sampled"]
        after_cert += result["after_certainty"]

    if not chunks:
        raise RuntimeError(f"Frame {frame}: no points survived filtering")
    xyz = np.concatenate([c["xyz"] for c in chunks])
    rgb = np.concatenate([c["rgb"] for c in chunks])
    err = np.concatenate([c["err"] for c in chunks])
    n_triangulated = len(xyz)
    if args.temporal_velocity:
        vel, valid, quality = temporal_velocities(chunks, chunk_pairs, cams, model, pils, frame, args)
        vel, valid, quality = np.concatenate(vel), np.concatenate(valid), np.concatenate(quality)

    keep = statistical_outlier_filter(xyz, args.sor_k, args.sor_std)
    idx = voxel_dedupe(xyz[keep], err[keep], args.voxel)
    idx = np.flatnonzero(keep)[idx]
    n_after_sor = int(keep.sum())
    if args.max_points and len(idx) > args.max_points:
        idx = idx[rng.choice(len(idx), args.max_points, replace=False)]
    xyz, rgb, err = xyz[idx], rgb[idx], err[idx]

    save_npy_atomic(out_dir / f"colors_frame{frame:06d}.npy", rgb.astype(np.float32))
    save_npy_atomic(out_dir / f"points3d_frame{frame:06d}.npy", xyz.astype(np.float32))
    stats = {
        "frame": frame, "pairs": len(pairs), "sampled": sampled,
        "after_certainty": after_cert, "triangulated": n_triangulated,
        "after_outlier_filter": n_after_sor, "saved": int(len(xyz)),
        "median_reproj_px": float(np.median(err)),
    }
    if args.temporal_velocity:
        vel, valid, quality = vel[idx], valid[idx], quality[idx]
        save_npy_atomic(out_dir / f"velocities_frame{frame:06d}.npy", vel.astype(np.float32))
        save_npy_atomic(out_dir / f"velocity_valid_frame{frame:06d}.npy", valid)
        save_npy_atomic(out_dir / f"velocity_quality_frame{frame:06d}.npy", quality)
        speed = np.linalg.norm(vel[valid], axis=1)
        stats["velocity_valid"] = float(valid.mean())
        stats["speed_p50_p90_p99"] = ([float(np.percentile(speed, p)) for p in (50, 90, 99)]
                                      if len(speed) else None)
    return stats


def run_combine(args, out_dir: Path):
    total = args.end - args.start
    npz_path = args.data_dir / f"keyframes_{total}frames_step{args.keyframe_step}.npz"
    cmd = [
        args.combine_python or sys.executable,
        str(REPO_ROOT / "src" / "combine_frames_fast_keyframes.py"),
        "--input-dir", str(out_dir),
        "--output-path", str(npz_path),
        "--frame-start", str(args.start),
        "--frame-end", str(args.end - 1),
        "--keyframe-step", str(args.keyframe_step),
        "--max-velocity-distance", str(args.max_velocity_distance),
    ]
    if args.total_budget:
        cmd += ["--use-smart-sampling", "--total-budget", str(args.total_budget)]
    print("\n[combine] " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return npz_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="data_FreeTimeGS directory written by extract_and_convert.py")
    parser.add_argument("--start", type=int, default=0, help="First frame (inclusive)")
    parser.add_argument("--end", type=int, default=250, help="Last frame (exclusive)")
    parser.add_argument("--keyframe-step", type=int, default=5)
    parser.add_argument("--frame-digits", type=int, default=6)

    roma = parser.add_argument_group("RoMa")
    roma.add_argument("--model", choices=["outdoor", "indoor"], default="outdoor")
    roma.add_argument("--device", default="cuda")
    roma.add_argument("--coarse-res", type=int, default=560)
    roma.add_argument("--upsample-res", default="864",
                      help="Fine resolution, 'S' or 'H,W' (e.g. 864,1536 for 16:9)")
    roma.add_argument("--num-matches", type=int, default=20000, help="Matches sampled per pair")
    roma.add_argument("--neighbors", type=int, default=3, help="Nearest cameras matched per camera")
    roma.add_argument("--max-angle", type=float, default=60.0,
                      help="Max angle (deg) between optical axes for a pair")

    filt = parser.add_argument_group("Filtering")
    filt.add_argument("--min-certainty", type=float, default=0.3)
    filt.add_argument("--max-reproj", type=float, default=1.5, help="Pixels, in both views")
    filt.add_argument("--min-depth", type=float, default=0.1)
    filt.add_argument("--max-depth", type=float, default=50.0)
    filt.add_argument("--use-mask", action="store_true",
                      help="Keep only matches inside the masks/ alpha mattes (performers only)")
    filt.add_argument("--mask-threshold", type=float, default=0.5)
    filt.add_argument("--mask-before-sample", action="store_true",
                      help="With --use-mask, zero certainty outside the mattes before sampling so "
                           "all --num-matches land on performers")
    filt.add_argument("--sor-k", type=int, default=8, help="Outlier filter neighbours (0 disables)")
    filt.add_argument("--sor-std", type=float, default=2.5)
    filt.add_argument("--voxel", type=float, default=0.004,
                      help="Dedupe voxel size in world units (0 disables)")
    filt.add_argument("--max-points", type=int, default=1_000_000, help="Per-frame cap (0 disables)")

    motion = parser.add_argument_group("Velocity from same-camera matches")
    motion.add_argument("--temporal-velocity", action="store_true",
                        help="Also match each camera's frame t against t+1 and save per-point "
                             "velocities (velocities_frame*.npy, velocity_valid_frame*.npy, and "
                             "velocity_quality_frame*.npy = certainty, forward-backward px, reprojection px)")
    motion.add_argument("--max-flow-reproj", type=float, default=3.0,
                        help="Pixels; the moved point must reproject this close in both views")
    motion.add_argument("--min-flow-certainty", type=float, default=0.0)
    motion.add_argument("--max-flow-cycle", type=float, default=0.0,
                        help="Pixels; forward-backward error allowed in each camera (0 disables)")
    motion.add_argument("--max-speed", type=float, default=0.5, help="World units per frame")

    out = parser.add_argument_group("Output")
    out.add_argument("--output-dir", type=Path, default=None,
                     help="Point cloud folder (default: <data-dir>/pointclouds)")
    out.add_argument("--overwrite", action="store_true", help="Recompute frames that already exist")
    out.add_argument("--combine", action="store_true",
                     help="Run combine_frames_fast_keyframes.py afterwards to build the init NPZ")
    out.add_argument("--combine-python", default=None,
                     help="Python for the combine step (default: this interpreter)")
    out.add_argument("--max-velocity-distance", type=float, default=0.5)
    out.add_argument("--total-budget", type=int, default=None,
                     help="Smart-sampling point budget for the NPZ (e.g. 4000000)")
    out.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = args.output_dir or args.data_dir / "pointclouds"
    out_dir.mkdir(parents=True, exist_ok=True)
    cams = load_cameras(args.data_dir)
    pairs = select_pairs(cams, args.neighbors, args.max_angle)
    frames = frames_to_process(args.start, args.end, args.keyframe_step)
    todo = [f for f in frames if args.overwrite
            or not (out_dir / f"points3d_frame{f:06d}.npy").exists()]
    print(f"[roma] {len(cams)} cameras, {len(pairs)} pairs per frame, "
          f"{len(frames)} frames needed ({len(todo)} to compute)")

    if todo:
        model = load_roma(args)
        rng = np.random.default_rng(args.seed)
        with open(out_dir / "stats.jsonl", "a") as stats_file:
            for n, frame in enumerate(todo, 1):
                started = time.time()
                stats = process_frame(frame, cams, pairs, model, args, out_dir, rng)
                stats["seconds"] = round(time.time() - started, 1)
                stats_file.write(json.dumps(stats) + "\n")
                stats_file.flush()
                print(f"[roma] {n}/{len(todo)} frame {frame:06d}: {stats['saved']:,} points "
                      f"(median reproj {stats['median_reproj_px']:.2f}px) in {stats['seconds']}s")

    if args.combine:
        npz_path = run_combine(args, out_dir)
        print("\nTrain with:")
        print(f"  python src/simple_trainer_freetime_4d_pure_relocation.py default_keyframe_small \\\n"
              f"      --data-dir {args.data_dir} \\\n"
              f"      --init-npz-path {npz_path} \\\n"
              f"      --result-dir results/{args.data_dir.parent.name} \\\n"
              f"      --start-frame {args.start} --end-frame {args.end}")


if __name__ == "__main__":
    main()
