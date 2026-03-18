#!/usr/bin/env python3
"""
Fuse vision odometry (dead-reckoning) with noisy GPS using an EKF.

State:  x = [px, py, vx, vy]
Process: vision delta_xy drives velocity; position integrated at each step.
Update:  noisy GPS absolute position.
"""
import argparse
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def latlon_to_local_xy(lat_deg, lon_deg, lat0_deg, lon0_deg):
    R_earth = 6378137.0
    lat  = np.deg2rad(lat_deg);  lon  = np.deg2rad(lon_deg)
    lat0 = np.deg2rad(lat0_deg); lon0 = np.deg2rad(lon0_deg)
    x = R_earth * (lon - lon0) * np.cos(lat0)
    y = R_earth * (lat - lat0)
    return x, y


def load_vision_csv(path):
    df = pd.read_csv(path)
    required = {"t_epoch", "x_m", "y_m", "Pxx", "Pxy", "Pyx", "Pyy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Vision CSV missing columns: {sorted(missing)}")
    df = df.rename(columns={"t_epoch": "t", "x_m": "x", "y_m": "y"})
    return df[["t", "x", "y", "Pxx", "Pxy", "Pyx", "Pyy"]].dropna().sort_values("t").reset_index(drop=True)


def load_gpx(path):
    ns = {"gpx": "http://www.topografix.com/GPX/1/1"}
    tree = ET.parse(path)
    root = tree.getroot()
    pts = []
    for trkpt in root.findall(".//gpx:trkpt", ns):
        lat = float(trkpt.attrib["lat"])
        lon = float(trkpt.attrib["lon"])
        time_elem = trkpt.find("gpx:time", ns)
        if time_elem is None:
            continue
        t = pd.to_datetime(time_elem.text, utc=True).timestamp()
        pts.append((t, lat, lon))
    if not pts:
        raise ValueError("No GPX track points found.")
    df = pd.DataFrame(pts, columns=["t", "lat", "lon"]).sort_values("t").reset_index(drop=True)
    lat0, lon0 = df.loc[0, "lat"], df.loc[0, "lon"]
    x, y = latlon_to_local_xy(df["lat"].values, df["lon"].values, lat0, lon0)
    df["x"] = x
    df["y"] = y
    return df[["t", "x", "y"]]


def interp_columns(t_query, t_src, values_src):
    out = np.zeros((len(t_query), values_src.shape[1]), dtype=float)
    for j in range(values_src.shape[1]):
        out[:, j] = np.interp(t_query, t_src, values_src[:, j])
    return out


# ---------------------------------------------------------------------------
# EKF
# ---------------------------------------------------------------------------

class VisionGpsEKF:
    """
    State: [px, py, vx, vy]

    Predict:
        px += vx * dt
        py += vy * dt
        vx  = vision_delta_x / dt   (vision dead-reckoning velocity)
        vy  = vision_delta_y / dt

    Update:
        z = [px, py] from GPS + Gaussian noise
    """

    def __init__(self, x0, P0, q_pos=1.0, q_vel=1.0, maha_thresh=9.21):
        self.x = np.asarray(x0, dtype=float).reshape(4)
        self.P = np.asarray(P0, dtype=float).reshape(4, 4)
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.maha_thresh = maha_thresh

    def predict(self, dt, vision_dx, vision_dy):
        F = np.array([
            [1., 0., dt, 0.],
            [0., 1., 0., dt],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ])
        self.x = F @ self.x
        # Override velocity with vision-derived estimate
        self.x[2] = vision_dx / dt
        self.x[3] = vision_dy / dt

        Q = np.diag([
            self.q_pos * dt**2,
            self.q_pos * dt**2,
            self.q_vel,
            self.q_vel,
        ])
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    def update_gps(self, z_xy, R_gps):
        H = np.array([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
        ])
        z = np.asarray(z_xy, dtype=float).reshape(2)
        R = np.asarray(R_gps, dtype=float).reshape(2, 2)
        R = 0.5 * (R + R.T)

        innov = z - H @ self.x
        S = H @ self.P @ H.T + R
        S = 0.5 * (S + S.T)

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False, np.nan

        maha = float(innov.T @ S_inv @ innov)
        if maha > self.maha_thresh:
            return False, maha

        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ innov
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return True, maha


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rigid_align_2d(src_xy, dst_xy):
    src_mean = np.mean(src_xy, axis=0)
    dst_mean = np.mean(dst_xy, axis=0)
    src_c = src_xy - src_mean
    dst_c = dst_xy - dst_mean
    H = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = dst_mean - R @ src_mean
    return R, t


def apply_rigid_2d(xy, R, t):
    return (R @ xy.T).T + t


def compute_ate_rmse(est_xy, gt_xy):
    err = est_xy - gt_xy
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fuse vision odometry with noisy GPS.")
    parser.add_argument("--vision_csv", default="results/vision_traj_xy.csv")
    parser.add_argument("--gpx_file",   default="data/04-Mar-2026-1323.gpx")
    parser.add_argument("--gps_sigma",  type=float, default=5.0,
                        help="Std dev of Gaussian noise added to GPS [m]")
    parser.add_argument("--gps_rate",   type=float, default=1.0,
                        help="Simulated GPS update rate [Hz]")
    parser.add_argument("--q_pos",      type=float, default=0.1)
    parser.add_argument("--q_vel",      type=float, default=1.0)
    parser.add_argument("--maha_thresh",type=float, default=9.21)
    parser.add_argument("--save_csv",   default="results/vision_gps_fused.csv")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    vision_df = load_vision_csv(args.vision_csv)
    gps_df    = load_gpx(args.gpx_file)

    t_vis = vision_df["t"].values

    # Align GPS origin to vision frame: find heading from GPS and rotate
    # GPS local ENU starts at (0,0); vision starts at (0,0) too.
    # Compute initial GPS heading and rotate vision to match.
    gps_xy_raw = gps_df[["x", "y"]].values
    for i in range(1, min(10, len(gps_xy_raw))):
        dx, dy = gps_xy_raw[i] - gps_xy_raw[0]
        if np.hypot(dx, dy) > 1e-6:
            theta = np.arctan2(dy, dx)
            break
    else:
        theta = 0.0

    # Interpolate GPS onto vision time grid
    gps_interp = interp_columns(t_vis, gps_df["t"].values, gps_xy_raw)

    # Rotate vision trajectory to align with GPS initial heading
    c, s = np.cos(theta), np.sin(theta)
    R_align = np.array([[c, -s], [s, c]])
    vis_xy_raw = vision_df[["x", "y"]].values
    vis_xy_aligned = (R_align @ vis_xy_raw.T).T + gps_interp[0]

    # Build noisy GPS: subsample to gps_rate and add noise
    vis_dt = float(np.median(np.diff(t_vis)))
    gps_period = 1.0 / args.gps_rate
    gps_step = max(1, int(round(gps_period / vis_dt)))
    gps_indices = np.arange(0, len(t_vis), gps_step)
    gps_noise = rng.normal(0.0, args.gps_sigma, size=(len(gps_indices), 2))
    noisy_gps = gps_interp[gps_indices] + gps_noise
    R_gps = np.eye(2) * args.gps_sigma**2

    # Vision covariance
    vis_cov = vision_df[["Pxx", "Pxy", "Pyx", "Pyy"]].values.reshape(-1, 2, 2)

    # EKF initialised at GPS start, zero velocity
    x0 = np.array([gps_interp[0, 0], gps_interp[0, 1], 0.0, 0.0])
    P0 = np.diag([args.gps_sigma**2, args.gps_sigma**2, 1.0, 1.0])
    ekf = VisionGpsEKF(x0, P0, q_pos=args.q_pos, q_vel=args.q_vel, maha_thresh=args.maha_thresh)

    fused_states = [ekf.x.copy()]
    gps_update_idx = set(gps_indices.tolist())
    gps_ptr = 0
    accepted = [True]
    maha_vals = [0.0]

    for k in range(1, len(t_vis)):
        dt = float(t_vis[k] - t_vis[k - 1])
        if dt <= 0:
            dt = 1e-3

        # Vision delta in GPS-aligned frame
        dxy = vis_xy_aligned[k] - vis_xy_aligned[k - 1]
        ekf.predict(dt, dxy[0], dxy[1])

        ok, maha = False, np.nan
        if k in gps_update_idx:
            z = noisy_gps[gps_ptr]
            gps_ptr += 1
            ok, maha = ekf.update_gps(z, R_gps)

        fused_states.append(ekf.x.copy())
        accepted.append(ok)
        maha_vals.append(maha)

    fused_states = np.asarray(fused_states)
    fused_xy = fused_states[:, :2]

    # ATE (align each to clean GPS)
    R_v, t_v = rigid_align_2d(vis_xy_aligned, gps_interp)
    vis_aligned = apply_rigid_2d(vis_xy_aligned, R_v, t_v)

    R_f, t_f = rigid_align_2d(fused_xy, gps_interp)
    fused_aligned = apply_rigid_2d(fused_xy, R_f, t_f)

    print("\nATE RMSE [m]")
    print(f"  Vision only: {compute_ate_rmse(vis_aligned,    gps_interp):.3f}")
    print(f"  Fused:       {compute_ate_rmse(fused_aligned,  gps_interp):.3f}")
    print(f"\nGPS updates accepted: {int(np.sum(accepted))} / {len(gps_indices)}")

    # Save
    out = pd.DataFrame({
        "t": t_vis,
        "vision_x": vis_xy_aligned[:, 0],
        "vision_y": vis_xy_aligned[:, 1],
        "fused_x":  fused_xy[:, 0],
        "fused_y":  fused_xy[:, 1],
        "gps_x":    gps_interp[:, 0],
        "gps_y":    gps_interp[:, 1],
    })
    out.to_csv(args.save_csv, index=False)
    print(f"Saved: {args.save_csv}")

    # Plot
    pad = 20.0
    x_min = gps_interp[:, 0].min() - pad;  x_max = gps_interp[:, 0].max() + pad
    y_min = gps_interp[:, 1].min() - pad;  y_max = gps_interp[:, 1].max() + pad

    _, ax = plt.subplots(figsize=(9, 7))
    ax.plot(gps_interp[:, 0], gps_interp[:, 1], color="black",      linewidth=2,   label="GPS (clean)")
    ax.scatter(noisy_gps[:, 0], noisy_gps[:, 1], color="gray",      s=10, alpha=0.5, label=f"GPS noisy (σ={args.gps_sigma}m)")
    ax.plot(vis_xy_aligned[:, 0], vis_xy_aligned[:, 1], color="tab:orange", linestyle="--", label="Vision only")
    ax.plot(fused_xy[:, 0], fused_xy[:, 1], color="tab:blue",       linewidth=2,   label="Fused")
    ax.set_xlim(x_min, x_max);  ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]");  ax.set_ylabel("y [m]")
    ax.set_title(f"Vision + GPS Fusion  (GPS σ={args.gps_sigma} m, {args.gps_rate} Hz)")
    ax.grid(True);  ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
