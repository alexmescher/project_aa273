#!/usr/bin/env python3
import argparse
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def latlon_to_local_xy(lat_deg, lon_deg, lat0_deg, lon0_deg):
    """Convert lat/lon to local ENU-like x/y using equirectangular approximation."""
    R_earth = 6378137.0
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    lat0 = np.deg2rad(lat0_deg)
    lon0 = np.deg2rad(lon0_deg)

    x = R_earth * (lon - lon0) * np.cos(lat0)
    y = R_earth * (lat - lat0)
    return x, y


def interp_columns(t_query, t_src, values_src):
    out = np.zeros((len(t_query), values_src.shape[1]), dtype=float)
    for j in range(values_src.shape[1]):
        out[:, j] = np.interp(t_query, t_src, values_src[:, j])
    return out


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

    t = dst_mean - (R @ src_mean)
    return R, t


def apply_rigid_2d(xy, R, t):
    return (R @ xy.T).T + t


def compute_ate_rmse(est_xy, gt_xy):
    err = est_xy - gt_xy
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))


def load_imu_csv(path):
    """
    Expected sample columns:
    timestamp_s,elapsed_s,x,y,z,vx,vy,vz,qw,qx,qy,qz
    """
    df = pd.read_csv(path)

    required = {"timestamp_s", "x", "y", "vx", "vy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"IMU CSV missing columns: {sorted(missing)}")

    out = df[["timestamp_s", "x", "y", "vx", "vy"]].copy()
    out = out.rename(columns={"timestamp_s": "t"})
    return out.dropna().sort_values("t").reset_index(drop=True)


def load_vision_csv(path):
    """
    Expected sample columns:
    t_epoch,x_m,y_m,Pxx,Pxy,Pyx,Pyy
    """
    df = pd.read_csv(path)

    required = {"t_epoch", "x_m", "y_m", "Pxx", "Pxy", "Pyx", "Pyy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Vision CSV missing columns: {sorted(missing)}")

    out = df[["t_epoch", "x_m", "y_m", "Pxx", "Pxy", "Pyx", "Pyy"]].copy()
    out = out.rename(columns={"t_epoch": "t", "x_m": "x", "y_m": "y"})
    return out.dropna().sort_values("t").reset_index(drop=True)


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
    lat0 = df.loc[0, "lat"]
    lon0 = df.loc[0, "lon"]

    x, y = latlon_to_local_xy(df["lat"].values, df["lon"].values, lat0, lon0)
    df["x"] = x
    df["y"] = y
    return df[["t", "x", "y"]]


class XYVelocityEKF:
    """
    State:
        x = [px, py, vx, vy]

    Prediction:
        p_k+1 = p_k + v_k * dt
        v_k+1 = [vx_imu, vy_imu]   (IMU estimated velocity used as input)

    Update:
        z = [x_vision, y_vision]
    """

    def __init__(self, x0, P0, q_pos=0.05, q_vel=0.5, maha_thresh=9.21):
        self.x = np.asarray(x0, dtype=float).reshape(4)
        self.P = np.asarray(P0, dtype=float).reshape(4, 4)
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.maha_thresh = maha_thresh

        self.history = []

    def predict(self, dt, imu_vx, imu_vy):
        F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)

        self.x = F @ self.x
        self.x[2] = imu_vx
        self.x[3] = imu_vy

        Q = np.diag([
            self.q_pos * dt * dt,
            self.q_pos * dt * dt,
            self.q_vel,
            self.q_vel,
        ])
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    def update_vision(self, z_xy, R_xy):
        H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=float)

        z = np.asarray(z_xy, dtype=float).reshape(2)
        R = np.asarray(R_xy, dtype=float).reshape(2, 2)
        R = 0.5 * (R + R.T)

        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        S = 0.5 * (S + S.T)

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False, np.nan

        maha = float(y.T @ S_inv @ y)
        if maha > self.maha_thresh:
            return False, maha

        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ y

        I = np.eye(4)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        return True, maha


def main():
    parser = argparse.ArgumentParser(description="Fuse IMU and vision trajectories and compare to GPX.")
    parser.add_argument("--imu_csv", default="results/imu_trajectory.csv")
    parser.add_argument("--vision_csv", default="results/vision_traj_xy.csv")
    parser.add_argument("--gpx_file", default="data/04-Mar-2026-1405.gpx")
    parser.add_argument("--vision_time_offset", type=float, default=0.0,
                        help="Seconds added to vision timestamps to align with IMU")
    parser.add_argument("--q_pos", type=float, default=0.05)
    parser.add_argument("--q_vel", type=float, default=0.5)
    parser.add_argument("--maha_thresh", type=float, default=9.21)
    parser.add_argument("--save_csv", default="results/fused_traj.csv")
    args = parser.parse_args()

    imu_df = load_imu_csv(args.imu_csv)
    vision_df = load_vision_csv(args.vision_csv)
    gps_df = load_gpx(args.gpx_file)

    vision_df["t"] = vision_df["t"] + args.vision_time_offset

    imu_t0 = imu_df["t"].iloc[0]
    imu_t1 = imu_df["t"].iloc[-1]
    vis_t0 = vision_df["t"].iloc[0]
    vis_t1 = vision_df["t"].iloc[-1]

    overlap_start = max(imu_t0, vis_t0)
    overlap_end = min(imu_t1, vis_t1)

    if overlap_end <= overlap_start:
        raise ValueError(
            "IMU and vision timestamps do not overlap.\n"
            f"IMU range:    [{imu_t0:.3f}, {imu_t1:.3f}]\n"
            f"Vision range: [{vis_t0:.3f}, {vis_t1:.3f}]\n"
            "Use matching files or set --vision_time_offset."
        )

    imu_df = imu_df[(imu_df["t"] >= overlap_start) & (imu_df["t"] <= overlap_end)].reset_index(drop=True)
    t_common = imu_df["t"].values

    vision_xy = interp_columns(
        t_common,
        vision_df["t"].values,
        vision_df[["x", "y"]].values,
    )

    vision_cov = interp_columns(
        t_common,
        vision_df["t"].values,
        vision_df[["Pxx", "Pxy", "Pyx", "Pyy"]].values,
    ).reshape(-1, 2, 2)

    gps_xy = interp_columns(
        t_common,
        gps_df["t"].values,
        gps_df[["x", "y"]].values,
    )

    imu_xy = imu_df[["x", "y"]].values
    imu_v = imu_df[["vx", "vy"]].values

    x0 = np.array([imu_xy[0, 0], imu_xy[0, 1], imu_v[0, 0], imu_v[0, 1]], dtype=float)
    P0 = np.diag([1.0, 1.0, 1.0, 1.0])

    ekf = XYVelocityEKF(
        x0=x0,
        P0=P0,
        q_pos=args.q_pos,
        q_vel=args.q_vel,
        maha_thresh=args.maha_thresh,
    )

    fused_states = [ekf.x.copy()]
    accepted = [True]
    maha_vals = [0.0]

    for k in range(1, len(t_common)):
        dt = float(t_common[k] - t_common[k - 1])
        ekf.predict(dt, imu_v[k, 0], imu_v[k, 1])
        ok, maha = ekf.update_vision(vision_xy[k], vision_cov[k])

        fused_states.append(ekf.x.copy())
        accepted.append(ok)
        maha_vals.append(maha)

    fused_states = np.asarray(fused_states)
    fused_xy = fused_states[:, :2]

    # Align trajectories to GPS for comparison
    R_imu, t_imu = rigid_align_2d(imu_xy, gps_xy)
    imu_aligned = apply_rigid_2d(imu_xy, R_imu, t_imu)

    R_vis, t_vis = rigid_align_2d(vision_xy, gps_xy)
    vision_aligned = apply_rigid_2d(vision_xy, R_vis, t_vis)

    R_fused, t_fused = rigid_align_2d(fused_xy, gps_xy)
    fused_aligned = apply_rigid_2d(fused_xy, R_fused, t_fused)

    print("\nATE RMSE [m]")
    print(f"IMU only:    {compute_ate_rmse(imu_aligned, gps_xy):.3f}")
    print(f"Vision only: {compute_ate_rmse(vision_aligned, gps_xy):.3f}")
    print(f"Fused:       {compute_ate_rmse(fused_aligned, gps_xy):.3f}")
    print(f"\nAccepted vision updates: {np.sum(accepted)} / {len(accepted)}")

    out_df = pd.DataFrame({
        "t_epoch": t_common,
        "imu_x": imu_xy[:, 0],
        "imu_y": imu_xy[:, 1],
        "imu_vx": imu_v[:, 0],
        "imu_vy": imu_v[:, 1],
        "vision_x": vision_xy[:, 0],
        "vision_y": vision_xy[:, 1],
        "fused_x": fused_xy[:, 0],
        "fused_y": fused_xy[:, 1],
        "fused_vx": fused_states[:, 2],
        "fused_vy": fused_states[:, 3],
        "gps_x": gps_xy[:, 0],
        "gps_y": gps_xy[:, 1],
        "vision_update_accepted": np.asarray(accepted, dtype=int),
        "mahalanobis": np.asarray(maha_vals),
    })
    out_df.to_csv(args.save_csv, index=False)
    print(f"Saved: {args.save_csv}")

    plt.figure(figsize=(9, 7))
    plt.plot(gps_xy[:, 0], gps_xy[:, 1], label="GPS ground truth")
    plt.plot(imu_aligned[:, 0], imu_aligned[:, 1], label="IMU aligned")
    plt.plot(vision_aligned[:, 0], vision_aligned[:, 1], label="Vision aligned")
    plt.plot(fused_aligned[:, 0], fused_aligned[:, 1], label="Fused aligned", linewidth=2)

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Trajectory Comparison")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()