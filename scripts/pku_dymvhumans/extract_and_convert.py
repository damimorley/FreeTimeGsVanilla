#!/usr/bin/env python3
"""
Extract a PKU-DyMVHumans scene tar and convert it into the FreeTimeGS layout.

Default scene: part1/1080_Kungfu_Weapon_Pair_m12m13 (60 cams, 1920x1080, 250 frames).

The tar is streamed once (from R2 or a local file) and only the members needed
for FreeTimeGS are written, keeping the R2 object path and the tar's own layout:

    <out-root>/<bucket-key without .tar>/        e.g. zxyun/PKU-DyMVHumans/part1/1080_Kungfu_Weapon_Pair_m12m13/
    ├── cams/000000NN_cam.txt                     source calibration (MVSNet format)
    ├── per_view/cam_N/images/000000.png ...      full RGB frames
    ├── per_view/cam_N/pha/000000.png ...         alpha mattes (optional RoMa match mask)
    ├── per_frame/000000/sparse/0/*.bin           reference SfM, used only for seed points
    └── data_FreeTimeGS/                          <- trainer --data-dir
        ├── images/cam_NN -> ../../per_view/cam_N/images
        ├── masks/cam_NN  -> ../../per_view/cam_N/pha
        ├── sparse/0/{cameras,images,points3D}.bin
        └── manifest.json

Camera names are zero-padded (cam_00 ... cam_59) so that the lexicographic sort
used by FreeTimeParser matches the numeric order of cams/000000NN_cam.txt.

Usage:
    # Stream from R2 using R2_* variables from .env (needs boto3)
    python scripts/pku_dymvhumans/extract_and_convert.py --out-root /data

    # Stream from a local tar or stdin
    python scripts/pku_dymvhumans/extract_and_convert.py --out-root /data --tar scene.tar
    rclone cat r2:research-noncommercial/zxyun/PKU-DyMVHumans/part1/1080_Kungfu_Weapon_Pair_m12m13.tar \\
        | python scripts/pku_dymvhumans/extract_and_convert.py --out-root /data --tar -

    # Re-run only the conversion on an existing extraction
    python scripts/pku_dymvhumans/extract_and_convert.py --out-root /data --convert-only
"""

import argparse
import fnmatch
import importlib.util
import json
import os
import re
import shutil
import struct
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KEY = "zxyun/PKU-DyMVHumans/part1/1080_Kungfu_Weapon_Pair_m12m13.tar"
REFERENCE_SFM = "per_frame/000000/sparse/0"
DATA_DIR_NAME = "data_FreeTimeGS"


def _load_read_write_model():
    # Imported by path: `import datasets` would pull in torch via datasets/__init__.py.
    spec = importlib.util.spec_from_file_location(
        "read_write_model", REPO_ROOT / "datasets" / "read_write_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rwm = _load_read_write_model()


# ----------------------------------------------------------------------------
# Source streaming
# ----------------------------------------------------------------------------

def load_env_file():
    for directory in (Path.cwd(), *Path(__file__).resolve().parents):
        env_path = directory / ".env"
        if env_path.is_file():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))
            return env_path
    return None


def open_r2_stream(key: str, bucket: str = None):
    import boto3
    from botocore.config import Config

    load_env_file()
    missing = [k for k in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT")
               if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"Missing R2 settings: {', '.join(missing)} (set them or add a .env)")
    bucket = bucket or os.environ.get("R2_BUCKET")
    if not bucket:
        raise SystemExit("No bucket given: pass --bucket or set R2_BUCKET")

    client = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(read_timeout=300, retries={"max_attempts": 5}),
    )
    response = client.get_object(Bucket=bucket, Key=key)
    print(f"[extract] Streaming s3://{bucket}/{key} "
          f"({response['ContentLength'] / 2**30:.2f} GiB)")
    return response["Body"]


def wanted_patterns(include_pha: bool):
    patterns = [
        "cams/*_cam.txt",
        "per_view/cam_*/images/*.png",
        f"{REFERENCE_SFM}/*.bin",
    ]
    if include_pha:
        patterns.append("per_view/cam_*/pha/*.png")
    return patterns


def extract_stream(fileobj, scene_dir: Path, patterns, overwrite: bool):
    written = skipped = 0
    started = time.time()
    with tarfile.open(fileobj=fileobj, mode="r|", bufsize=16 * 2**20) as tf:
        for member in tf:
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            # Members are stored under "<scene_name>/..."; drop that prefix.
            rel = Path(*parts[1:]) if len(parts) > 1 else Path(parts[0])
            rel_str = rel.as_posix()
            if rel.is_absolute() or ".." in rel.parts or rel.name.startswith("."):
                continue
            if not any(fnmatch.fnmatch(rel_str, p) for p in patterns):
                continue

            dst = scene_dir / rel
            if not overwrite and dst.exists() and dst.stat().st_size == member.size:
                skipped += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".part")
            with tf.extractfile(member) as src, open(tmp, "wb") as out:
                shutil.copyfileobj(src, out, 4 * 2**20)
            os.replace(tmp, dst)
            written += 1
            if (written + skipped) % 2000 == 0:
                rate = (written + skipped) / max(time.time() - started, 1e-6)
                print(f"[extract] {written + skipped} files ({written} new) - {rate:.0f} files/s")
    print(f"[extract] Done: {written} written, {skipped} already present "
          f"in {time.time() - started:.0f}s")


# ----------------------------------------------------------------------------
# Conversion
# ----------------------------------------------------------------------------

def read_mvsnet_cam(path: Path):
    tokens = path.read_text().split()
    e = tokens.index("extrinsic")
    i = tokens.index("intrinsic")
    w2c = np.array(tokens[e + 1:e + 17], dtype=np.float64).reshape(4, 4)
    K = np.array(tokens[i + 1:i + 10], dtype=np.float64).reshape(3, 3)
    depth = np.array(tokens[i + 10:], dtype=np.float64)
    R = w2c[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-4) or np.linalg.det(R) < 0:
        raise ValueError(f"{path}: extrinsic rotation is not a proper rotation")
    return K, R, w2c[:3, 3], depth


def png_size(path: Path):
    with open(path, "rb") as f:
        header = f.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def project(K, R, t, X):
    cam = X @ R.T + t
    uv = cam @ K.T
    return uv[:, :2] / uv[:, 2:3], cam[:, 2]


def triangulate_dlt(Ps, uvs):
    rows = []
    for P, (u, v) in zip(Ps, uvs):
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    X = vt[-1]
    return X[:3] / X[3]


def seed_points_from_reference(scene_dir: Path, cams, max_reproj: float):
    """Re-triangulate the reference SfM tracks with the cams/ poses.

    The per-frame SfM is in nearly, but not exactly, the cams/ world frame, so
    its xyz values are not reused directly; its 2D tracks are.
    """
    ref = scene_dir / REFERENCE_SFM
    if not (ref / "points3D.bin").exists():
        print(f"[convert] WARNING: {ref} missing - writing sparse/0 without seed points")
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8), np.zeros(0)

    images = rwm.read_images_binary(str(ref / "images.bin"))
    points = rwm.read_points3D_binary(str(ref / "points3D.bin"))
    image_to_cam = {}
    for image in images.values():
        match = re.search(r"c_(\d+)", image.name)
        if match and int(match.group(1)) < len(cams):
            image_to_cam[image.id] = int(match.group(1))

    xyz, rgb, err = [], [], []
    for pt in points.values():
        obs = [(image_to_cam[i], images[i].xys[j])
               for i, j in zip(pt.image_ids, pt.point2D_idxs) if i in image_to_cam]
        if len(obs) < 2:
            continue
        Ps = [cams[c]["K"] @ np.hstack([cams[c]["R"], cams[c]["t"][:, None]]) for c, _ in obs]
        X = triangulate_dlt(Ps, [uv for _, uv in obs])
        errors, depths = [], []
        for c, uv in obs:
            proj, depth = project(cams[c]["K"], cams[c]["R"], cams[c]["t"], X[None])
            errors.append(np.linalg.norm(proj[0] - uv))
            depths.append(depth[0])
        if min(depths) <= 0 or np.median(errors) > max_reproj:
            continue
        xyz.append(X)
        rgb.append(pt.rgb)
        err.append(float(np.mean(errors)))

    print(f"[convert] Seed points: kept {len(xyz)}/{len(points)} reference SfM points "
          f"re-triangulated with cams/ poses (median track error <= {max_reproj}px)")
    if err:
        e = np.asarray(err)
        print(f"[convert]   reprojection error: median {np.median(e):.2f}px, "
              f"p90 {np.percentile(e, 90):.2f}px")
    if len(xyz) < 100:
        print("[convert] WARNING: very few seed points survived - check that "
              "cams/000000NN_cam.txt corresponds to per_view/cam_N")
    return np.asarray(xyz).reshape(-1, 3), np.asarray(rgb, np.uint8).reshape(-1, 3), np.asarray(err)


def link_or_copy(src: Path, dst: Path, copy: bool):
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.is_dir():
        shutil.rmtree(dst)
    if copy:
        shutil.copytree(src, dst)
    else:
        dst.symlink_to(os.path.relpath(src, dst.parent), target_is_directory=True)


def convert(scene_dir: Path, args):
    cam_files = sorted((scene_dir / "cams").glob("*_cam.txt"))
    if not cam_files:
        raise SystemExit(f"No cams/*_cam.txt under {scene_dir}")
    n_cams = len(cam_files)
    digits = max(2, len(str(n_cams - 1)))

    data_dir = scene_dir / DATA_DIR_NAME
    sparse_dir = data_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "images").mkdir(exist_ok=True)

    cams, cameras, images = [], {}, {}
    frame_counts = {}
    for idx, cam_file in enumerate(cam_files):
        if int(cam_file.name.split("_")[0]) != idx:
            raise SystemExit(f"Unexpected camera numbering: {cam_file.name} at index {idx}")
        K, R, t, depth = read_mvsnet_cam(cam_file)
        src_view = scene_dir / "per_view" / f"cam_{idx}"
        frames = sorted((src_view / "images").glob("*.png"))
        if not frames:
            raise SystemExit(f"No images in {src_view / 'images'}")
        frame_counts[idx] = len(frames)
        width, height = png_size(frames[0])

        name = f"cam_{idx:0{digits}d}"
        cams.append({"K": K, "R": R, "t": t, "depth": depth.tolist(), "name": name,
                     "source": f"per_view/cam_{idx}", "width": width, "height": height})

        cam_id = idx + 1
        cameras[cam_id] = rwm.Camera(
            id=cam_id, model="PINHOLE", width=width, height=height,
            params=np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]))
        images[cam_id] = rwm.Image(
            id=cam_id, qvec=rwm.rotmat2qvec(R), tvec=t, camera_id=cam_id,
            name=f"{name}.png", xys=np.zeros((0, 2)), point3D_ids=np.zeros(0, dtype=np.int64))

        link_or_copy(src_view / "images", data_dir / "images" / name, args.copy_images)
        if (src_view / "pha").is_dir():
            (data_dir / "masks").mkdir(exist_ok=True)
            link_or_copy(src_view / "pha", data_dir / "masks" / name, args.copy_images)

    counts = sorted(set(frame_counts.values()))
    if len(counts) > 1:
        short = {f"cam_{i}": n for i, n in frame_counts.items() if n != max(counts)}
        print(f"[convert] WARNING: uneven frame counts per camera: {short}")

    xyz, rgb, err = seed_points_from_reference(scene_dir, cams, args.seed_max_reproj)
    points3D = {
        i + 1: rwm.Point3D(id=i + 1, xyz=xyz[i], rgb=rgb[i], error=float(err[i]),
                           image_ids=np.zeros(0, dtype=np.int32),
                           point2D_idxs=np.zeros(0, dtype=np.int32))
        for i in range(len(xyz))
    }
    rwm.write_model(cameras, images, points3D, str(sparse_dir), ext=".bin")

    centers = np.stack([-c["R"].T @ c["t"] for c in cams])
    manifest = {
        "scene": scene_dir.name,
        "num_cameras": n_cams,
        "num_frames": min(counts),
        "image_size": [cams[0]["width"], cams[0]["height"]],
        "camera_ring_radius": float(np.median(np.linalg.norm(centers - centers.mean(0), axis=1))),
        "seed_points": len(points3D),
        "cameras": [{k: c[k] for k in ("name", "source", "depth")} for c in cams],
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[convert] Wrote {sparse_dir} ({n_cams} cameras, {len(points3D)} points)")
    print(f"[convert] Frames per camera: {min(counts)}; camera ring radius "
          f"{manifest['camera_ring_radius']:.2f} world units")
    verify_with_pycolmap(sparse_dir)
    print(f"\nTrainer --data-dir: {data_dir}")
    print(f"Next: python scripts/pku_dymvhumans/roma_pointclouds.py --data-dir {data_dir}")


def verify_with_pycolmap(sparse_dir: Path):
    try:
        import pycolmap
    except ImportError:
        print("[convert] pycolmap not installed here - skipped reader check")
        return
    if hasattr(pycolmap, "SceneManager"):
        manager = pycolmap.SceneManager(str(sparse_dir))
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()
        n_images, n_points = len(manager.images), len(manager.points3D)
    else:
        rec = pycolmap.Reconstruction(str(sparse_dir))
        n_images, n_points = len(rec.images), len(rec.points3D)
    print(f"[convert] pycolmap reads sparse/0: {n_images} images, {n_points} points")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-root", type=Path, required=True,
                        help="Local root that mirrors the bucket (the object key path is created under it)")
    parser.add_argument("--key", default=DEFAULT_KEY, help="Object key of the scene tar")
    parser.add_argument("--bucket", default=None, help="Bucket (default: R2_BUCKET from .env)")
    parser.add_argument("--tar", default=None,
                        help="Read a local tar instead of R2 ('-' for stdin)")
    parser.add_argument("--no-pha", action="store_true",
                        help="Skip per_view/cam_*/pha alpha mattes")
    parser.add_argument("--convert-only", action="store_true",
                        help="Skip extraction and only rebuild data_FreeTimeGS")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rewrite files that already exist with the same size")
    parser.add_argument("--copy-images", action="store_true",
                        help="Copy image folders into data_FreeTimeGS instead of symlinking")
    parser.add_argument("--seed-max-reproj", type=float, default=4.0,
                        help="Max median reprojection error (px) for re-triangulated seed points")
    args = parser.parse_args()

    scene_dir = args.out_root / re.sub(r"\.tar$", "", args.key)
    scene_dir.mkdir(parents=True, exist_ok=True)
    print(f"[extract] Scene directory: {scene_dir}")

    if not args.convert_only:
        patterns = wanted_patterns(include_pha=not args.no_pha)
        if args.tar == "-":
            extract_stream(sys.stdin.buffer, scene_dir, patterns, args.overwrite)
        elif args.tar:
            with open(args.tar, "rb") as f:
                extract_stream(f, scene_dir, patterns, args.overwrite)
        else:
            extract_stream(open_r2_stream(args.key, args.bucket), scene_dir, patterns, args.overwrite)

    convert(scene_dir, args)


if __name__ == "__main__":
    main()
