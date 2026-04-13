#!/usr/bin/env python3
"""RaCFormer inference daemon for ROS bag playback.

Watches /workspace/ros_input/ for .npz frames written by the ROS collector node,
runs RaCFormer inference, and writes JSON results to /workspace/ros_output/.

Run inside Docker:
    conda activate racformer
    python /workspace/ros_inference_daemon.py \
        --config /workspace/configs/racformer_r50_nuimg_704x256_f8.py \
        --weights /workspace/checkpoints/racformer_r50_f8.pth
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import pathlib
import time

import numpy as np
import torch
import traceback

INPUT_DIR  = pathlib.Path('/workspace/ros_input')
OUTPUT_DIR = pathlib.Path('/workspace/ros_output')

TARGET_H, TARGET_W = 256, 704
NUM_CAMS   = 6
NUM_FRAMES = 8
SCORE_THRESH = 0.3

NUSCENES_CLASSES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]


# ---------------------------------------------------------------------------
# Radar → image projection
# ---------------------------------------------------------------------------
def _project_radar(
    radar_pts: np.ndarray,   # (N, 7) [x,y,z,rcs,vx,vy,time]
    lidar2img: np.ndarray,   # (NT, 4, 4)
    H: int, W: int,
) -> tuple[np.ndarray, np.ndarray]:
    NT = lidar2img.shape[0]
    depth_maps = np.zeros((NT, H, W), dtype=np.float32)
    rcs_maps   = np.zeros((NT, H, W), dtype=np.float32)

    if len(radar_pts) == 0:
        return depth_maps, rcs_maps

    pts_h = np.hstack([radar_pts[:, :3],
                       np.ones((len(radar_pts), 1))]).T  # (4, N)

    for i, l2i in enumerate(lidar2img):
        p = l2i @ pts_h                           # (4, N)
        valid = p[2] > 0.1
        if not valid.any():
            continue

        p_v = p[:, valid]
        u = p_v[0] / p_v[2]
        v = p_v[1] / p_v[2]
        in_fov = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not in_fov.any():
            continue

        u_i = u[in_fov].astype(np.int32)
        v_i = v[in_fov].astype(np.int32)
        depth_maps[i, v_i, u_i] = np.linalg.norm(
            radar_pts[valid][in_fov, :3], axis=1,
        ).astype(np.float32)
        rcs_maps[i, v_i, u_i] = radar_pts[valid][in_fov, 3]

    return depth_maps, rcs_maps


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(config_path: str, weights_path: str):
    from mmcv import Config
    from mmdet3d.models import build_model
    from mmcv.runner import load_checkpoint

    cfgs = Config.fromfile(config_path)
    model = build_model(cfgs.model)
    model.cuda().eval()
    load_checkpoint(
        model, weights_path, map_location='cuda', strict=True,
        logger=logging.Logger(__name__, logging.ERROR),
    )
    logging.info('Model loaded from %s', weights_path)
    return model


# ---------------------------------------------------------------------------
# Single-frame inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_inference(model, npz_path: pathlib.Path) -> dict:
    data = np.load(str(npz_path), allow_pickle=True)

    imgs           = data['imgs']            # (8, 6, 3, 256, 704)
    lidar2img_np   = data['lidar2img']       # (8, 6, 4, 4)
    img_timestamps = data['img_timestamps']  # (8, 6)
    radar_pts      = data['radar_points']    # (N, 7)
    frame_ts       = float(data['frame_ts'])

    NT = NUM_FRAMES * NUM_CAMS  # 48

    img_tensor = torch.from_numpy(
        imgs.reshape(1, NT, 3, TARGET_H, TARGET_W)
    ).float().cuda()

    # img_metas: frame 0 = current (newest), frame 7 = oldest
    lidar2img_list = [
        lidar2img_np[t, c]
        for t in range(NUM_FRAMES)
        for c in range(NUM_CAMS)
    ]
    ts_list = [
        float(img_timestamps[t, c])
        for t in range(NUM_FRAMES)
        for c in range(NUM_CAMS)
    ]
    img_metas = [{
        'lidar2img':     lidar2img_list,
        'img_shape':     [(TARGET_H, TARGET_W, 3)] * NT,
        'img_timestamp': ts_list,
    }]

    # Radar depth / RCS maps
    # _project_radar returns (NT, H, W); add batch dim → (1, NT, H, W).
    # extract_feat will .view(B, N=6, T=8, H, W) internally.
    lidar2img_flat = lidar2img_np.reshape(NT, 4, 4)
    depth_np, rcs_np = _project_radar(radar_pts, lidar2img_flat, TARGET_H, TARGET_W)
    radar_depth = torch.from_numpy(depth_np).float().unsqueeze(0).cuda()  # (1, 48, H, W)
    radar_rcs   = torch.from_numpy(rcs_np  ).float().unsqueeze(0).cuda()  # (1, 48, H, W)

    # radar_points structure:
    # simple_test does radar_points[0] → extract_feat receives a list of T=8 items.
    # extract_feat does radar_points[i] → extract_pts_feat receives [tensor] (batch list).
    # extract_pts_feat iterates over the batch list: each element must be (Ni, 7).
    radar_pts_t = torch.from_numpy(radar_pts).float().cuda()
    radar_pts_per_frame = [radar_pts_t]          # batch list (B=1) for one frame
    radar_points_all = [radar_pts_per_frame] * NUM_FRAMES  # T=8 frames

    results = model.simple_test(
        img_metas=img_metas,
        img=img_tensor,
        radar_points=[radar_points_all],   # [0] → radar_points_all
        radar_depth=radar_depth,
        radar_rcs=radar_rcs,
    )

    detections = []
    pts_bbox = results[0].get('pts_bbox', {})
    boxes  = pts_bbox.get('boxes_3d')
    scores = pts_bbox.get('scores_3d')
    labels = pts_bbox.get('labels_3d')

    if boxes is not None and len(boxes) > 0:
        box_np    = boxes.tensor.cpu().numpy()   # (M, 9): x,y,z,dx,dy,dz,yaw,vx,vy
        scores_np = scores.cpu().numpy()
        labels_np = labels.cpu().numpy()

        logging.info('  raw boxes=%d  score_max=%.3f  score_min=%.3f  radar_pts=%d',
                     len(box_np), scores_np.max(), scores_np.min(), len(radar_pts))

        for j in range(len(box_np)):
            if scores_np[j] < SCORE_THRESH:
                continue
            b = box_np[j]
            detections.append({
                'x': float(b[0]), 'y': float(b[1]), 'z': float(b[2]),
                'dx': float(b[3]), 'dy': float(b[4]), 'dz': float(b[5]),
                'yaw': float(b[6]),
                'vx': float(b[7]), 'vy': float(b[8]),
                'score': float(scores_np[j]),
                'label': int(labels_np[j]),
                'class': NUSCENES_CLASSES[int(labels_np[j])],
            })

    return {'timestamp': frame_ts, 'detections': detections}


# ---------------------------------------------------------------------------
# Main watch loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
        default='/workspace/configs/racformer_r50_nuimg_704x256_f8.py')
    parser.add_argument('--weights',
        default='/workspace/checkpoints/racformer_r50_f8.pth')
    parser.add_argument('--poll_interval', type=float, default=0.05)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    import os, sys
    os.chdir('/workspace')
    sys.path.insert(0, '/workspace')
    importlib.import_module('models')
    importlib.import_module('loaders')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.info('Loading model...')
    model = load_model(args.config, args.weights)
    logging.info('Watching %s', INPUT_DIR)

    processed: set[str] = set()

    while True:
        pending = sorted(INPUT_DIR.glob('frame_*.npz'))
        if pending:
            latest_npz = pending[-1]
            stale_npzs = [
                npz_path for npz_path in pending[:-1]
                if npz_path.name not in processed
            ]

            if stale_npzs:
                logging.info('Dropping %d stale queued frame(s); keeping %s',
                             len(stale_npzs), latest_npz.name)
            for stale_npz in stale_npzs:
                processed.add(stale_npz.name)
                try:
                    stale_npz.unlink()
                except OSError:
                    pass

            if latest_npz.name not in processed:
                try:
                    result = run_inference(model, latest_npz)
                    out = OUTPUT_DIR / latest_npz.name.replace('.npz', '.json')
                    out.write_text(json.dumps(result))
                    logging.info('frame %s → %d detections',
                                 latest_npz.stem, len(result['detections']))
                except Exception:
                    logging.error('Error on %s:\n%s', latest_npz.name, traceback.format_exc())
                finally:
                    processed.add(latest_npz.name)
                    try:
                        latest_npz.unlink()
                    except OSError:
                        pass

        if len(processed) > 500:
            processed = set(list(processed)[-200:])

        time.sleep(args.poll_interval)


if __name__ == '__main__':
    main()
