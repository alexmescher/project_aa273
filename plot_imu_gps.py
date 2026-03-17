"""
Plot IMU-integrated trajectory vs GPS ground truth.

Reads the trajectory CSV produced by imu_baseline_optimal.py (which contains
absolute epoch timestamps) and overlays it on GPS track from a GPX file.

Usage:
    python plot_imu_vs_gps.py
    python plot_imu_vs_gps.py --traj my_traj.csv --gpx my_track.gpx --show
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

# Defaults (flight 1 / test2)
DEFAULT_TRAJ = ROOT / "imu_trajectory.csv"
DEFAULT_GPX  = ROOT / "data" / "test2" / "04-Mar-2026-1405.gpx"


def parse_gpx(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse GPX file using stdlib XML (no external deps).

    Returns (epoch_s, lat_deg, lon_deg, ele_m) arrays.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(path)
    root = tree.getroot()
    ns = root.tag.split("}", 1)[0] + "}" if "}" in root.tag else ""

    epochs, lats, lons, eles = [], [], [], []
    for trkpt in root.iter(f"{ns}trkpt"):
        lat = trkpt.attrib.get("lat")
        lon = trkpt.attrib.get("lon")
        if lat is None or lon is None:
            continue

        time_el = trkpt.find(f"{ns}time")
        if time_el is None or not (time_el.text or "").strip():
            continue

        t_txt = time_el.text.strip()
        if t_txt.endswith("Z"):
            t_txt = t_txt.replace("Z", "+00:00")
        dt = datetime.fromisoformat(t_txt)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        ele_el = trkpt.find(f"{ns}ele")
        ele = float(ele_el.text) if ele_el is not None and ele_el.text else float("nan")

        epochs.append(dt.timestamp())
        lats.append(float(lat))
        lons.append(float(lon))
        eles.append(ele)

    return (np.array(epochs), np.array(lats), np.array(lons), np.array(eles))


def latlon_to_local_m(lat, lon, lat0, lon0):
    """Equirectangular projection to local East/North meters."""
    R = 6378137.0
    x = (np.radians(lon) - np.radians(lon0)) * np.cos(np.radians(lat0)) * R
    y = (np.radians(lat) - np.radians(lat0)) * R
    return x, y


def best_2d_rotation(x_imu, y_imu, x_gps, y_gps, t_imu, t_gps):
    """Find best yaw rotation to align IMU frame to GPS East-North frame."""
    t_min = max(t_imu.min(), t_gps.min())
    t_max = min(t_imu.max(), t_gps.max())
    if t_max <= t_min:
        return x_imu, y_imu

    t_common = t_gps[(t_gps >= t_min) & (t_gps <= t_max)]
    if len(t_common) < 3:
        return x_imu, y_imu

    xi = np.interp(t_common, t_imu, x_imu)
    yi = np.interp(t_common, t_imu, y_imu)
    xg = np.interp(t_common, t_gps, x_gps)
    yg = np.interp(t_common, t_gps, y_gps)

    S_xx = np.sum(xi * xg + yi * yg)
    S_xy = np.sum(xi * yg - yi * xg)
    theta = math.atan2(S_xy, S_xx)
    c, s = math.cos(theta), math.sin(theta)
    return c * x_imu - s * y_imu, s * x_imu + c * y_imu


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Plot IMU trajectory vs GPS.")
    parser.add_argument("--traj", type=Path, default=DEFAULT_TRAJ,
                        help="Trajectory CSV from imu_baseline_optimal.py")
    parser.add_argument("--gpx", type=Path, default=DEFAULT_GPX,
                        help="GPX file with GPS ground truth")
    parser.add_argument("--show", action="store_true",
                        help="Open interactive plot window")
    parser.add_argument("-o", "--output", default="imu_vs_gps.png",
                        help="Output image filename")
    args = parser.parse_args()

    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Load trajectory ──
    traj = pd.read_csv(args.traj)
    assert "timestamp_s" in traj.columns, "CSV must have timestamp_s (absolute epoch)"
    assert "x" in traj.columns and "y" in traj.columns

    imu_t = traj["timestamp_s"].values
    imu_elapsed = traj["elapsed_s"].values if "elapsed_s" in traj.columns else imu_t - imu_t[0]
    x_imu = traj["x"].values
    y_imu = traj["y"].values
    z_imu = traj["z"].values if "z" in traj.columns else np.zeros(len(traj))

    # ── Load GPS ──
    gps_t, gps_lat, gps_lon, gps_ele = parse_gpx(args.gpx)

    # Find flight-start GPS point (first consecutive point, not isolated fix)
    flight_gps_idx = 0
    for i in range(1, len(gps_t)):
        if gps_t[i] - gps_t[i-1] < 5.0:
            flight_gps_idx = i
            break

    # Filter GPS to the IMU time window
    imu_t0, imu_t1 = imu_t[0], imu_t[-1]
    mask = (gps_t >= imu_t0) & (gps_t <= imu_t1)

    if mask.sum() < 2:
        # Try broader: all GPS from flight start onward, clipped to 100s
        mask = (gps_t >= gps_t[flight_gps_idx]) & (gps_t <= gps_t[flight_gps_idx] + 100)
        print(f"  Limited GPS overlap — using {mask.sum()} points from flight start")

    gps_t_f = gps_t[mask]
    gps_lat_f = gps_lat[mask]
    gps_lon_f = gps_lon[mask]
    gps_ele_f = gps_ele[mask]

    # Convert GPS to local meters (origin = first filtered point)
    x_gps, y_gps = latlon_to_local_m(gps_lat_f, gps_lon_f,
                                      gps_lat_f[0], gps_lon_f[0])
    z_gps = gps_ele_f - gps_ele_f[0]
    z_gps[np.isnan(z_gps)] = 0.0
    gps_elapsed = gps_t_f - gps_t_f[0]

    # IMU origin = first sample
    x_imu = x_imu - x_imu[0]
    y_imu = y_imu - y_imu[0]
    z_imu = z_imu - z_imu[0]

    # Align IMU yaw to GPS frame
    x_imu, y_imu = best_2d_rotation(x_imu, y_imu, x_gps, y_gps,
                                     imu_elapsed, gps_elapsed)

    # ── Compute drift metrics ──
    xy_drift = np.sqrt(x_imu**2 + y_imu**2)
    final_xy = xy_drift[-1]
    final_z = abs(z_imu[-1])
    duration = imu_elapsed[-1]

    # ── Plot ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    utc_start = datetime.fromtimestamp(imu_t[0], tz=timezone.utc).strftime("%H:%M:%S")
    utc_end = datetime.fromtimestamp(imu_t[-1], tz=timezone.utc).strftime("%H:%M:%S")
    fig.suptitle(f"IMU Dead-Reckoning vs GPS Ground Truth\n"
                 f"UTC {utc_start} → {utc_end}  ({duration:.0f}s)",
                 fontsize=14, fontweight="bold")

    # 1. XY top-down (first 30s)
    ax = axes[0, 0]
    n30 = min(30 * 200, len(x_imu))
    gps_mask_30 = gps_elapsed <= 30
    ax.plot(x_gps[gps_mask_30], y_gps[gps_mask_30], "k-o", markersize=5,
            linewidth=2.5, label="GPS (30s)", zorder=5)
    ax.plot(x_imu[:n30], y_imu[:n30], "b-", linewidth=1.5, alpha=0.8,
            label="IMU (30s)")
    ax.plot(0, 0, "r*", markersize=15, zorder=10, label="Start")
    ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
    ax.set_title("First 30 seconds"); ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2); ax.set_aspect("equal")

    # 2. Drift over time
    ax = axes[0, 1]
    ax.plot(imu_elapsed, xy_drift, "b-", linewidth=1.5, label="Horizontal |xy|")
    ax.plot(imu_elapsed, np.abs(z_imu), "r-", linewidth=1.5, alpha=0.7,
            label="Vertical |z|")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Drift (m)")
    ax.set_title("Drift over time"); ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    # Annotate
    for t_ann in [10, 30, 50, 100]:
        idx = min(int(t_ann * 200), len(xy_drift) - 1)
        if idx < len(xy_drift):
            ax.annotate(f"{xy_drift[idx]:.0f}m", xy=(imu_elapsed[idx], xy_drift[idx]),
                        fontsize=8, color="blue", fontweight="bold", ha="center",
                        va="bottom", xytext=(0, 8), textcoords="offset points")

    # 3. Drift bar chart
    ax = axes[1, 0]
    times = [1, 5, 10, 20, 30, 50, 70, 100]
    xy_vals, z_vals = [], []
    for t in times:
        idx = min(int(t * 200), len(x_imu) - 1)
        xy_vals.append(math.sqrt(x_imu[idx]**2 + y_imu[idx]**2))
        z_vals.append(abs(z_imu[idx]))
    x_pos = np.arange(len(times)); w = 0.35
    bars1 = ax.bar(x_pos - w/2, xy_vals, w, label="Horizontal", color="#2563eb", alpha=0.8)
    bars2 = ax.bar(x_pos + w/2, z_vals, w, label="Vertical", color="#ea580c", alpha=0.8)
    ax.set_xticks(x_pos); ax.set_xticklabels([f"{t}s" for t in times])
    ax.set_ylabel("Drift (m)"); ax.set_title("Drift at key timepoints")
    ax.legend(fontsize=9); ax.set_yscale("log"); ax.set_ylim(0.01, max(max(xy_vals), max(z_vals)) * 3)
    ax.grid(True, alpha=0.2, axis="y")
    for bar, val in zip(bars1, xy_vals):
        if val > 0.05:
            ax.text(bar.get_x() + bar.get_width()/2, val * 1.3, f"{val:.0f}m",
                    ha="center", fontsize=7, color="#2563eb", fontweight="bold")
    for bar, val in zip(bars2, z_vals):
        if val > 0.05:
            ax.text(bar.get_x() + bar.get_width()/2, val * 1.3, f"{val:.0f}m",
                    ha="center", fontsize=7, color="#ea580c", fontweight="bold")

    # 4. Full trajectory
    ax = axes[1, 1]
    gps_mask_full = gps_elapsed <= duration
    ax.plot(x_gps[gps_mask_full], y_gps[gps_mask_full], "k-o", markersize=3,
            linewidth=2, label="GPS", zorder=5)
    ax.plot(x_imu, y_imu, "b-", linewidth=0.8, alpha=0.6,
            label=f"IMU ({duration:.0f}s)")
    ax.plot(0, 0, "r*", markersize=15, zorder=10)
    ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
    ax.set_title(f"Full {duration:.0f}s trajectory")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2); ax.set_aspect("equal")

    plt.tight_layout()
    out = ROOT / args.output
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")

    # ── 3D plot (separate figure) ──
    fig3d = plt.figure(figsize=(12, 9))
    ax3d = fig3d.add_subplot(projection="3d")

    # GPS
    gps_mask_full = gps_elapsed <= duration
    ax3d.plot(x_gps[gps_mask_full], y_gps[gps_mask_full], z_gps[gps_mask_full],
              "k-o", markersize=4, linewidth=2.5, label="GPS ground truth", zorder=5)

    # IMU — clip z for readability if it explodes
    z_cap = min(max(abs(z_imu).max(), 100), 5000)
    z_plot = np.clip(z_imu, -z_cap, z_cap)
    ax3d.plot(x_imu, y_imu, z_plot, "b-", linewidth=1.2, alpha=0.7,
              label=f"IMU dead-reckoning ({duration:.0f}s)")

    ax3d.plot([0], [0], [0], "r*", markersize=15, zorder=10, label="Start")

    ax3d.set_xlabel("East (m)")
    ax3d.set_ylabel("North (m)")
    ax3d.set_zlabel("Up (m)")
    ax3d.set_title(f"3D Trajectory: IMU vs GPS\n"
                   f"UTC {utc_start} → {utc_end}  ({duration:.0f}s)\n"
                   f"Final drift: |xy|={final_xy:.0f}m, |z|={final_z:.0f}m",
                   fontsize=12, fontweight="bold")
    ax3d.legend(fontsize=10)

    out_3d = ROOT / args.output.replace(".png", "_3d.png")
    fig3d.tight_layout()
    fig3d.savefig(out_3d, dpi=150)
    print(f"Saved: {out_3d}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()