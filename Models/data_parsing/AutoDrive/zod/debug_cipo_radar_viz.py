#!/usr/bin/env python3
"""
Debug viz: image + AutoSpeed bboxes + radar BEV with CIPO association highlighted.
Run after step1 (timestamp association). Outputs every N frames for visual verification.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import cv2

_REPO_ROOT = Path(__file__).resolve().parents[4]
_ZOD_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_ZOD_SCRIPT_DIR))

from PIL import Image

try:
    from Models.inference.auto_speed_infer import AutoSpeedNetworkInfer
except ImportError:
    from inference.auto_speed_infer import AutoSpeedNetworkInfer

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    DBSCAN = None


_LAT_BUFFER_M = 0.5
_LAT_BUFFER_PATH_M = 1.0  # when no CIPO: curvature path only, ±1.0m (no FOV/azimuth)


def radar_spherical_to_cartesian(pts):
    az = pts["azimuth_angle"].astype(np.float64)
    el = pts["elevation_angle"].astype(np.float64)
    rg = pts["radar_range"].astype(np.float64)
    x = rg * np.cos(el) * np.cos(az)
    y = rg * np.cos(el) * np.sin(az)
    z = rg * np.sin(el)
    return x, y, z


def pixel_to_h_angle_deg(u: float, W: float, H: float, hfov_deg: float) -> float:
    """Horizontal angle (deg) from optical axis for full-frame pixel u."""
    return ((u - W / 2) / (W / 2)) * (hfov_deg / 2)


def cam_dir_to_radar_azimuth(h_angle_deg, cam_ext, radar_ext):
    h_rad = np.deg2rad(h_angle_deg)
    dir_cam = np.array([np.sin(h_rad), 0.0, np.cos(h_rad)])
    R_cam = np.array(cam_ext)[:3, :3]
    R_radar = np.array(radar_ext)[:3, :3]
    dir_world = R_cam @ dir_cam
    dir_radar = R_radar.T @ dir_world
    return float(np.arctan2(dir_radar[1], dir_radar[0]))


def _polar_vel_dist(a, b, range_scale=4.0, lat_buffer=0.5, vel_scale=1.5):
    """Polar+velocity distance: range, lateral, velocity."""
    dr = abs(a[0] - b[0])
    r_avg = (a[0] + b[0]) / 2
    daz = abs(np.angle(np.exp(1j * (a[1] - b[1]))))
    d_lateral = r_avg * abs(np.sin(daz)) if r_avg > 0 else 0.0
    dv = abs(a[2] - b[2])
    return np.sqrt((dr / range_scale) ** 2 + (d_lateral / lat_buffer) ** 2 + (dv / vel_scale) ** 2)


def get_radar_xy_and_clusters(radar_data, ts_ns, z_min=-0.5, z_max=1.0, range_scale=4.0, lat_buffer=0.5, vel_scale=1.5, min_samples=2):
    pts = radar_data[radar_data["timestamp"] == ts_ns]
    if len(pts) == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=int), pts, []
    x, y, z = radar_spherical_to_cartesian(pts)
    mask = (z >= z_min) & (z <= z_max)
    x_f, y_f = x[mask], y[mask]
    xy = np.column_stack([x_f, y_f])
    pts_f = pts[mask]
    if len(xy) == 0 or DBSCAN is None:
        return xy, np.zeros(len(xy), dtype=int), pts_f, []

    rg = pts_f["radar_range"].astype(np.float64)
    az = pts_f["azimuth_angle"].astype(np.float64)
    rr = pts_f["range_rate"].astype(np.float64)
    polar_vel = np.column_stack([rg, az, rr])
    metric = lambda a, b: _polar_vel_dist(a, b, range_scale, lat_buffer, vel_scale)
    labels = DBSCAN(eps=1.0, min_samples=min_samples, metric=metric).fit(polar_vel).labels_
    clusters = []
    for lbl in set(labels):
        if lbl < 0:
            continue
        m = labels == lbl
        clusters.append({
            "azimuth": float(np.mean(pts_f["azimuth_angle"][m])),
            "range": float(np.mean(pts_f["radar_range"][m])),
            "range_rate": float(np.mean(pts_f["range_rate"][m])),
            "indices": np.where(m)[0],
        })
    return xy, labels, pts_f, clusters


def path_points_from_curvature(curvature_inv_m: float, max_dist: float = 100, n_pts: int = 100):
    """Ackermann bicycle model: circular arc. Returns list of (x,y) in radar frame (x forward, y left)."""
    k = curvature_inv_m
    if abs(k) < 1e-6:
        return [(s, 0.0) for s in np.linspace(0, max_dist, n_pts)]
    R = 1.0 / k
    pts = []
    for s in np.linspace(0, max_dist, n_pts):
        x = R * np.sin(k * s)
        y = R * (1 - np.cos(k * s))
        pts.append((x, y))
    return pts


def _path_azimuth_at_range(curvature_inv_m: float, range_m: float) -> float:
    """
    Azimuth (rad) of the curvature path at given range from ego.
    Circular arc: at range r, az = atan2(y,x) where (x,y) on path. Small-angle: az ≈ κ*r/2 (NOT κ*r).
    """
    k = curvature_inv_m
    if abs(k) < 1e-9:
        return 0.0
    R = 1.0 / k
    r = min(range_m, 2 * R - 1e-6)
    theta = 2 * np.arcsin(r / (2 * R))
    x = R * np.sin(theta)
    y = R * (1 - np.cos(theta))
    return float(np.arctan2(y, x))


def find_nearest_cluster_on_path(clusters, curvature_inv_m: float, lat_buffer_m: float = 0.5):
    """Cluster most on path (min lateral deviation), not nearest by range."""
    if not clusters:
        return None, -1
    in_path = []
    for i, c in enumerate(clusters):
        r, az = c["range"], c["azimuth"]
        az_path = _path_azimuth_at_range(curvature_inv_m, r)
        daz = abs(np.angle(np.exp(1j * (az - az_path))))
        d_lateral = r * abs(np.sin(daz))
        if d_lateral <= lat_buffer_m:
            in_path.append((i, c, d_lateral))
    if not in_path:
        return None, -1
    best_idx = min(in_path, key=lambda x: (x[2], x[1]["range"]))[0]
    return clusters[best_idx], best_idx


def find_cluster_on_path_direct(radar_data, ts_ns, curvature_inv_m, pts_f_ref, lat_buffer_m=1.0,
                                 z_min=-0.5, z_max=1.0, range_gap_m=4.0, vel_gap_ms=3.0, min_pts=2):
    """
    No-CIPO path search on raw radar points (no DBSCAN, no FOV constraint).
    Returns (cluster_dict, indices_into_pts_f) where pts_f_ref is the already z-filtered pts_f.
    min_pts=2: reject isolated noise/footpath points when no prior context.
    """
    if len(pts_f_ref) == 0:
        return None, None
    rg = pts_f_ref["radar_range"].astype(np.float64)
    az = pts_f_ref["azimuth_angle"].astype(np.float64)
    rr = pts_f_ref["range_rate"].astype(np.float64)

    on_path = []
    for i in range(len(pts_f_ref)):
        az_path = _path_azimuth_at_range(curvature_inv_m, rg[i])
        daz = abs(np.angle(np.exp(1j * (az[i] - az_path))))
        d_lateral = rg[i] * abs(np.sin(daz))
        if d_lateral <= lat_buffer_m:
            on_path.append((i, float(rg[i]), float(az[i]), float(rr[i]), float(d_lateral)))

    if not on_path:
        return None, None

    on_path.sort(key=lambda p: p[1])
    groups = [[on_path[0]]]
    for pt in on_path[1:]:
        last = groups[-1][-1]
        if abs(pt[1] - last[1]) <= range_gap_m and abs(pt[3] - last[3]) <= vel_gap_ms:
            groups[-1].append(pt)
        else:
            groups.append([pt])

    best, best_indices, best_score = None, None, (float("inf"), float("inf"))
    for group in groups:
        if len(group) < min_pts:
            continue
        indices = [p[0] for p in group]
        mean_dlat = float(np.mean([p[4] for p in group]))
        mean_range = float(np.mean([p[1] for p in group]))
        score = (mean_dlat, mean_range)
        if score < best_score:
            best_score = score
            best_indices = indices
            best = {
                "range": mean_range,
                "azimuth": float(np.mean([p[2] for p in group])),
                "range_rate": float(np.mean([p[3] for p in group])),
            }
    return best, best_indices


def find_nearest_cluster_lateral(clusters, azimuth_radar, lat_buffer_m=0.5):
    """Cluster within ±lat_buffer_m lateral distance from CIPO ray."""
    if not clusters:
        return None, -1
    in_cone = []
    for i, c in enumerate(clusters):
        daz = abs(np.angle(np.exp(1j * (c["azimuth"] - azimuth_radar))))
        d_lateral = c["range"] * abs(np.sin(daz))
        if d_lateral <= lat_buffer_m:
            in_cone.append((i, c))
    if not in_cone:
        return None, -1
    best_idx = min(in_cone, key=lambda x: x[1]["range"])[0]
    return clusters[best_idx], best_idx


_BEV_X_RANGE = (0, 150)   # forward range in meters (radar ~150m)
_BEV_Y_RANGE = (-40, 40)  # lateral range in meters


def draw_bev(xy, labels, pts_f, matched_indices, az_radar_deg, scale=6, x_range=None, y_range=None, lat_buffer_m=0.5, curvature_inv_m=None, path_only=False):
    """
    path_only: when True (no CIPO), draw buffer around curvature path only, not FOV/azimuth ray.
    x_range: (min, max) forward range in m. Default (0, 150) for full radar range.
    """
    x_range = x_range or _BEV_X_RANGE
    y_range = y_range or _BEV_Y_RANGE
    bev_h = int((x_range[1] - x_range[0]) * scale)
    bev_w = int((y_range[1] - y_range[0]) * scale)
    bev = np.ones((bev_h, bev_w, 3), dtype=np.uint8) * 28

    def to_pixel(x, y):
        row = int((x_range[1] - x) * scale)
        col = int((y_range[1] - y) * scale)
        return np.clip(row, 0, bev_h - 1), np.clip(col, 0, bev_w - 1)

    # Grid
    for x in range(0, int(x_range[1]) + 1, 25):
        r, c = to_pixel(x, y_range[0])
        cv2.line(bev, (c, r), (bev_w - 1, r), (55, 55, 55), 1)
        cv2.putText(bev, f"{x}m", (5, r + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 130, 130), 1)
    for y in range(y_range[0], y_range[1] + 1, 20):
        r, c = to_pixel(x_range[0], y)
        cv2.line(bev, (c, 0), (c, bev_h - 1), (55, 55, 55), 1)

    # All radar points (gray)
    for i in range(len(xy)):
        r, c = to_pixel(xy[i, 0], xy[i, 1])
        cv2.circle(bev, (c, r), 2, (85, 85, 85), -1)

    # Predicted path from curvature (Ackermann) - draw first
    path_pts = None
    if curvature_inv_m is not None:
        path_pts = path_points_from_curvature(curvature_inv_m)
        for i in range(len(path_pts) - 1):
            r1, c1 = to_pixel(path_pts[i][0], path_pts[i][1])
            r2, c2 = to_pixel(path_pts[i + 1][0], path_pts[i + 1][1])
            cv2.line(bev, (c1, r1), (c2, r2), (0, 255, 0), 2)
        cv2.putText(bev, f"path k={curvature_inv_m:.3f}", (5, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    if path_only and path_pts is not None:
        # No CIPO: buffer zone around curvature path only (no FOV/azimuth)
        pts_left, pts_right = [], []
        for i in range(len(path_pts)):
            x, y = path_pts[i][0], path_pts[i][1]
            if i + 1 < len(path_pts):
                dx = path_pts[i + 1][0] - x
                dy = path_pts[i + 1][1] - y
            else:
                dx = path_pts[i][0] - path_pts[i - 1][0]
                dy = path_pts[i][1] - path_pts[i - 1][1]
            n = np.sqrt(dx * dx + dy * dy) + 1e-9
            perp_x, perp_y = -dy / n, dx / n
            left_pt = (x - lat_buffer_m * perp_x, y - lat_buffer_m * perp_y)
            right_pt = (x + lat_buffer_m * perp_x, y + lat_buffer_m * perp_y)
            pts_left.append(to_pixel(left_pt[0], left_pt[1]))
            pts_right.append(to_pixel(right_pt[0], right_pt[1]))
        poly = np.array([[c, r] for r, c in pts_left] + [[c, r] for r, c in reversed(pts_right)], dtype=np.int32)
        overlay = bev.copy()
        cv2.fillPoly(overlay, [poly], (40, 60, 60))
        cv2.addWeighted(overlay, 0.4, bev, 0.6, 0, bev)
        cv2.putText(bev, f"path d=+-{lat_buffer_m}m", (5, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 150, 150), 1)
    else:
        # CIPO: buffer zone around CIPO ray (FOV)
        az_rad = np.deg2rad(az_radar_deg)
        perp = np.array([-np.sin(az_rad), np.cos(az_rad)])
        pts_left, pts_right = [], []
        for d in np.linspace(0, 80, 40):
            cx, cy = d * np.cos(az_rad), d * np.sin(az_rad)
            left_pt = (cx - lat_buffer_m * perp[0], cy - lat_buffer_m * perp[1])
            right_pt = (cx + lat_buffer_m * perp[0], cy + lat_buffer_m * perp[1])
            pts_left.append(to_pixel(left_pt[0], left_pt[1]))
            pts_right.append(to_pixel(right_pt[0], right_pt[1]))
        poly = np.array([[c, r] for r, c in pts_left] + [[c, r] for r, c in reversed(pts_right)], dtype=np.int32)
        overlay = bev.copy()
        cv2.fillPoly(overlay, [poly], (40, 60, 60))
        cv2.addWeighted(overlay, 0.4, bev, 0.6, 0, bev)
        for i in range(len(pts_left) - 1):
            r1, c1 = pts_left[i]
            r2, c2 = pts_left[i + 1]
            cv2.line(bev, (c1, r1), (c2, r2), (80, 120, 120), 1)
            r1, c1 = pts_right[i]
            r2, c2 = pts_right[i + 1]
            cv2.line(bev, (c1, r1), (c2, r2), (80, 120, 120), 1)
        r_mid, c_mid = to_pixel(40 * np.cos(az_rad), 40 * np.sin(az_rad))
        cv2.putText(bev, f"d=+-{lat_buffer_m}m", (max(0, c_mid - 30), max(20, r_mid - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 150, 150), 1)
        # CIPO azimuth ray (yellow)
        for d in np.linspace(0, 80, 50):
            xr = d * np.cos(az_rad)
            yr = d * np.sin(az_rad)
            r, c = to_pixel(xr, yr)
            cv2.circle(bev, (c, r), 2, (0, 255, 255), -1)

    # Matched cluster (green)
    if matched_indices is not None:
        for i in matched_indices:
            r, c = to_pixel(xy[i, 0], xy[i, 1])
            cv2.circle(bev, (c, r), 6, (0, 255, 0), -1)
            cv2.circle(bev, (c, r), 8, (255, 255, 255), 2)

    # Ego
    r0, c0 = to_pixel(0, 0)
    cv2.circle(bev, (c0, r0), 8, (0, 255, 255), -1)
    cv2.circle(bev, (c0, r0), 10, (255, 255, 255), 2)

    # Labels
    cv2.putText(bev, f"Radar BEV 0-{x_range[1]}m (z:-0.5~1m)", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    if path_only:
        cv2.putText(bev, "path only (no CIPO)", (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 200), 1)
    else:
        cv2.putText(bev, f"CIPO az: {az_radar_deg:.1f} deg", (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    leg = "gray=all  green=matched  cyan=buffer"
    if path_only:
        leg += "  path=green"
    else:
        leg += "  yellow=ray  path=green"
    cv2.putText(bev, leg, (5, bev_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    return bev


def draw_text_panel(img, lines, x=20, y0=50, font_scale=0.7, color=(0, 255, 0)):
    """Draw text with dark background for readability."""
    for i, line in enumerate(lines):
        y = y0 + i * 32
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 4, y + 4), (0, 0, 0), -1)
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)


def main():
    parser = argparse.ArgumentParser(description="Debug CIPO-radar association viz")
    parser.add_argument("--sequence", type=str, default="000479")
    parser.add_argument("--zod-root", type=str, default=None, required=True, help="Path to the ZOD dataset root")
    parser.add_argument("--every", type=int, default=10, help="Output every N frames")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to AutoSpeed weights (default: {zod_root}/models/autodrive.pt)",
    )
    args = parser.parse_args()

    seq = args.sequence
    zod = Path(args.zod_root)
    from zod_utils import default_autospeed_checkpoint, get_images_blur_dir, get_calibration_path, sequence_output_dir

    model_path = Path(args.model_path) if args.model_path else default_autospeed_checkpoint(zod)

    out_dir = Path(args.output_dir) if args.output_dir else sequence_output_dir(zod, seq) / "debug_viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = get_images_blur_dir(zod, seq)
    calib_path = get_calibration_path(zod, seq)
    assoc_path = zod / "associations" / f"{seq}_associations.json"

    if not assoc_path.exists():
        print(f"Run step1 first: python step1_timestamp_association.py --sequence {seq} --zod-root {zod}")
        return
    if not calib_path.exists():
        print(f"Calibration not found: {calib_path}")
        return

    with open(assoc_path) as f:
        assoc = json.load(f)
    with open(calib_path) as f:
        calib = json.load(f)["FC"]
    W, H = calib["image_dimensions"][0], calib["image_dimensions"][1]
    hfov_deg = calib["field_of_view"][0]
    cam_ext = np.array(calib["extrinsics"])
    radar_ext = np.array(calib["radar_extrinsics"])

    radar_data = np.load(assoc["radar_npy_path"], allow_pickle=True)
    model = AutoSpeedNetworkInfer(str(model_path))

    color_map = {1: (0, 0, 255), 2: (0, 255, 255), 3: (255, 255, 0)}

    n_saved = 0
    for idx, rec in enumerate(assoc["associations"]):
        if idx % args.every != 0:
            continue

        img_path = img_dir / rec["image"]
        if not img_path.exists():
            continue

        img_pil = Image.open(img_path).convert("RGB")
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        preds = model.inference(img_pil)

        # Level 1, 2 only (in-path). Exclude Level 3 (cyan, off-path).
        CIPO_CLASSES = (1, 2)

        # Draw AutoSpeed preds
        for p in preds:
            x1, y1, x2, y2, conf, cls = p
            c = color_map.get(int(cls), (255, 255, 255))
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
            if int(cls) in CIPO_CLASSES:
                cv2.putText(img, f"L{int(cls)} {conf:.2f}", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)

        # Title bar
        cv2.rectangle(img, (0, 0), (img.shape[1], 28), (40, 40, 40), -1)
        cv2.putText(img, f"seq {seq}  frame {idx}  L1=red L2=yellow L3=cyan", (15, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # CIPO case uses vel_scale=1.5 (velocity matters for tracking)
        # No-CIPO case uses vel_scale=50 (velocity ignored: car front/back have different range_rates)
        xy, labels, pts_f, clusters = get_radar_xy_and_clusters(
            radar_data, rec["radar_timestamp_ns"], lat_buffer=_LAT_BUFFER_M
        )
        cipo = [p for p in preds if int(p[5]) in CIPO_CLASSES]

        if not cipo:
            curvature = rec.get("curvature_inv_m", 0.0)
            cluster_path, matched_path = find_cluster_on_path_direct(
                radar_data, rec["radar_timestamp_ns"], curvature, pts_f,
                lat_buffer_m=_LAT_BUFFER_PATH_M,
            )
            bev = draw_bev(xy, labels, pts_f, matched_path, 0, x_range=_BEV_X_RANGE, y_range=_BEV_Y_RANGE,
                          lat_buffer_m=_LAT_BUFFER_PATH_M, curvature_inv_m=curvature, path_only=True)
            lines = ["No CIPO (L1/L2)", f"path k={curvature:.4f}"]
            if cluster_path:
                lines.extend([f"path fallback: {cluster_path['range']:.1f}m", f"speed: {cluster_path['range_rate']:.1f} m/s"])
            draw_text_panel(img, lines, y0=35)
        else:
            cipo.sort(key=lambda p: (p[1] + p[3]) / 2, reverse=True)
            x1, y1, x2, y2, conf, cls = cipo[0]
            u = (x1 + x2) / 2
            cv2.circle(img, (int(u), int(y2)), 10, (0, 255, 0), 2)
            cv2.putText(img, "CIPO", (int(u) - 20, int(y2) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            h_angle_deg = pixel_to_h_angle_deg(u, W, H, hfov_deg)
            az_radar = cam_dir_to_radar_azimuth(h_angle_deg, cam_ext, radar_ext)
            az_radar_deg = np.rad2deg(az_radar)

            cluster, _ = find_nearest_cluster_lateral(clusters, az_radar, lat_buffer_m=_LAT_BUFFER_M)
            matched_indices = cluster["indices"].tolist() if cluster else None

            lines = [
                f"H_ang (cam): {h_angle_deg:.2f} deg",
                f"az (radar):  {az_radar_deg:.2f} deg",
            ]
            if cluster:
                lines.extend([f"dist: {cluster['range']:.1f} m", f"speed: {cluster['range_rate']:.1f} m/s"])
            else:
                lines.append("no cluster in az cone")
            draw_text_panel(img, lines, y0=35)

            curvature = rec.get("curvature_inv_m")
            bev = draw_bev(xy, labels, pts_f, matched_indices, az_radar_deg, x_range=_BEV_X_RANGE, y_range=_BEV_Y_RANGE,
                          lat_buffer_m=_LAT_BUFFER_M, curvature_inv_m=curvature)

        # Resize and stack
        h, w = img.shape[:2]
        target_h = 640
        img_s = cv2.resize(img, (int(w * target_h / h), target_h), interpolation=cv2.INTER_AREA)
        bev_s = cv2.resize(bev, (img_s.shape[1] // 2, target_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.hstack([img_s, bev_s])

        out_path = out_dir / f"frame_{idx:04d}_{Path(rec['image']).stem}.png"
        cv2.imwrite(str(out_path), canvas)
        n_saved += 1
        print(f"Saved {out_path.name}")

    print(f"Done: {n_saved} frames -> {out_dir}")


if __name__ == "__main__":
    main()
