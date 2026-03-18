#!/usr/bin/env python3
"""
eskf_vio.py  —  Error-State Extended Kalman Filter for Visual-Inertial Odometry
               following the formulation in Section III-C of the paper.

Nominal state  x̂  = [p(3), v(3), q(4), ba(3), bg(3)]   (13 values)
Error  state   δx  = [δp(3), δv(3), δθ(3), δba(3), δbg(3)]  ∈ R^15

Prediction  (Eqs. 3–7):   raw IMU at 200 Hz propagates the nominal state.
Update      (Eqs. 15–17): vision horizontal velocity [vx_cam, vy_cam].
                           (Paper Eq. 8 also includes z_baro and ψ̇_cam;
                           those are omitted here — data not available.)
Injection   (Eq. 18):     error state injected into nominal state then reset.
"""

import argparse
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from imu_preprocessing import (
    load_imu_csv, detect_accel_scale_bug, correct_accel_scale,
    correct_gyro_scale, lowpass_imu, estimate_static_biases,
    clip_to_gps_window,
)
from imu_integrator import (
    G_MPS2,
    quat_mult, quat_conj, quat_normalize,
    quat_rotate_vec, quat_from_gyro,
)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def skew(v: np.ndarray) -> np.ndarray:
    """3×3 skew-symmetric (cross-product) matrix."""
    return np.array([
        [ 0.0,   -v[2],  v[1]],
        [ v[2],   0.0,  -v[0]],
        [-v[1],   v[0],  0.0 ],
    ], dtype=float)


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Quaternion [w, x, y, z] → 3×3 rotation matrix (body → world)."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=float)


# ---------------------------------------------------------------------------
# GPS / evaluation helpers
# ---------------------------------------------------------------------------

def latlon_to_local_xy(lat_deg, lon_deg, lat0_deg, lon0_deg):
    R_earth = 6_378_137.0
    lat  = np.deg2rad(lat_deg);  lon  = np.deg2rad(lon_deg)
    lat0 = np.deg2rad(lat0_deg); lon0 = np.deg2rad(lon0_deg)
    return (R_earth * (lon - lon0) * np.cos(lat0),
            R_earth * (lat - lat0))


def load_gpx(path):
    ns = {"gpx": "http://www.topografix.com/GPX/1/1"}
    root = ET.parse(path).getroot()
    pts = []
    for tp in root.findall(".//gpx:trkpt", ns):
        te = tp.find("gpx:time", ns)
        if te is None:
            continue
        t = pd.to_datetime(te.text, utc=True).timestamp()
        pts.append((t, float(tp.attrib["lat"]), float(tp.attrib["lon"])))
    df = pd.DataFrame(pts, columns=["t", "lat", "lon"]).sort_values("t").reset_index(drop=True)
    x, y = latlon_to_local_xy(df["lat"].values, df["lon"].values,
                               df.loc[0, "lat"], df.loc[0, "lon"])
    df["x"] = x;  df["y"] = y
    return df[["t", "x", "y"]]


def rigid_align_2d(src_xy, dst_xy):
    sm = src_xy.mean(0);  dm = dst_xy.mean(0)
    H  = (src_xy - sm).T @ (dst_xy - dm)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1;  R = Vt.T @ U.T
    return R, dm - R @ sm


def apply_rigid_2d(xy, R, t):
    return (R @ xy.T).T + t


def ate_rmse(est, gt):
    return float(np.sqrt(np.mean(np.sum((est - gt)**2, axis=1))))


# ---------------------------------------------------------------------------
# ESKF
# ---------------------------------------------------------------------------

class ESKF:
    """
    Error-State EKF following Sola [12] and Section III-C of the paper.

    Nominal state x̂ = [p(3), v(3), q(4), ba(3), bg(3)]
    Error  state δx = [δp(3), δv(3), δθ(3), δba(3), δbg(3)] ∈ R^15

    Index layout of the 15-dim error state:
        0:3   δp   position error
        3:6   δv   velocity error
        6:9   δθ   small-angle orientation error
        9:12  δba  accelerometer bias error
        12:15 δbg  gyroscope bias error
    """

    def __init__(self, p0, v0, q0, ba0, bg0, P0,
                 sigma_a, sigma_g, sigma_ba, sigma_bg,
                 maha_thresh=9.21):
        self.p  = np.asarray(p0,  float).copy()
        self.v  = np.asarray(v0,  float).copy()
        self.q  = quat_normalize(np.asarray(q0, float).copy())
        self.ba = np.asarray(ba0, float).copy()
        self.bg = np.asarray(bg0, float).copy()
        self.P  = np.asarray(P0,  float).copy()

        self.sigma_a   = sigma_a    # accel measurement noise  (m/s²/√Hz)
        self.sigma_g   = sigma_g    # gyro  measurement noise  (rad/s/√Hz)
        self.sigma_ba  = sigma_ba   # accel bias random walk   (m/s²/√Hz)
        self.sigma_bg  = sigma_bg   # gyro  bias random walk   (rad/s/√Hz)
        self.maha_thresh = maha_thresh
        self.gravity   = np.array([0.0, 0.0, G_MPS2])

    # ------------------------------------------------------------------
    # Prediction  (Eqs. 3–7, 12–13)
    # ------------------------------------------------------------------

    def predict(self, accel_body: np.ndarray, omega_body: np.ndarray, dt: float):
        """
        Propagate the nominal state forward one IMU step and grow the
        error-state covariance.

        Parameters
        ----------
        accel_body : (3,)  raw accelerometer reading in body frame (m/s²)
        omega_body : (3,)  raw gyroscope reading in body frame (rad/s)
        dt         : float timestep (s)
        """
        # ---- Nominal state propagation (Eqs. 3–7) ----

        # Eq. 3: bias-corrected angular velocity
        omega = omega_body - self.bg

        # Eq. 4: integrate orientation
        dq    = quat_from_gyro(omega, dt)
        q_new = quat_normalize(quat_mult(self.q, dq))

        # Eq. 5: rotate bias-corrected accel to world frame, remove gravity
        R      = quat_to_rot(self.q)          # rotation at current step
        a_body = accel_body - self.ba
        a_world = R @ a_body - self.gravity

        # Eqs. 6–7: Euler-integrate velocity and position
        v_new = self.v + a_world * dt
        p_new = self.p + self.v * dt + 0.5 * a_world * dt**2

        # ---- Error-state covariance propagation (Eqs. 12–13) ----
        #
        # Continuous-time Jacobian F of the error dynamics (15×15):
        #
        #   δṗ  =  δv
        #   δv̇  = -R [ã×] δθ - R δba           (ã = accel after bias removal)
        #   δθ̇  = -[ω×] δθ  - δbg              (ω  = gyro  after bias removal)
        #   δḃa =  0
        #   δḃg =  0
        F = np.zeros((15, 15))
        F[0:3,  3:6]  =  np.eye(3)          # δṗ = δv
        F[3:6,  6:9]  = -R @ skew(a_body)   # δv̇ ← orientation error
        F[3:6,  9:12] = -R                   # δv̇ ← accel bias error
        F[6:9,  6:9]  = -skew(omega)         # δθ̇ ← orientation self-coupling
        F[6:9, 12:15] = -np.eye(3)           # δθ̇ ← gyro bias error

        # First-order Euler discretisation: Φ = I + F Δt  (Eq. 13)
        Phi = np.eye(15) + F * dt

        # Discrete process noise Q (15×15)
        # Noise drives δv (accel noise), δθ (gyro noise), δba, δbg.
        # δp is not directly driven (it integrates δv).
        Q = np.zeros((15, 15))
        Q[3:6,   3:6]   = self.sigma_a  ** 2 * np.eye(3) * dt * 10
        Q[6:9,   6:9]   = self.sigma_g  ** 2 * np.eye(3) * dt * 10
        Q[9:12,  9:12]  = self.sigma_ba ** 2 * np.eye(3) * dt * 10
        Q[12:15, 12:15] = self.sigma_bg ** 2 * np.eye(3) * dt * 10

        # Eq. 12: propagate covariance
        self.P = Phi @ self.P @ Phi.T + Q
        self.P = 0.5 * (self.P + self.P.T)

        # Commit nominal state
        self.p = p_new
        self.v = v_new
        self.q = q_new

    # ------------------------------------------------------------------
    # Update  (Eqs. 15–17)
    # ------------------------------------------------------------------

    def update_vision_velocity(self, vx_cam: float, vy_cam: float,
                               R_vis: np.ndarray):
        """
        Vision horizontal velocity update (paper Eq. 8, rows 1–2).

        Measurement:       z  = [vx_cam, vy_cam]
        Predicted meas.:   h  = [v̂x, v̂y]
        Observation matrix H  (2×15): selects δvx and δvy.

        Assumes the camera x,y axes are aligned with the world x,y axes
        (nadir camera, yaw ≈ 0).  For a full implementation the estimated
        yaw from the nominal quaternion should transform the measurement
        into the world frame.
        """
        z = np.array([vx_cam, vy_cam])
        h = self.v[:2]          # predicted horizontal velocity

        H = np.zeros((2, 15))
        H[0, 3] = 1.0           # selects δvx
        H[1, 4] = 1.0           # selects δvy

        return self._update(z, h, H, R_vis)

    def update_altitude(self, altitude_m: float, sigma_alt: float):
        """
        Altitude constraint from barometer (paper Eq. 8, row 3).
        z = z_baro,  h(x̂) = p̂z,  H selects δpz (index 2).
        Prevents unconstrained z drift that contaminates x/y via R.
        """
        z = np.array([altitude_m])
        h = np.array([self.p[2]])
        H = np.zeros((1, 15));  H[0, 2] = 1.0
        R = np.array([[sigma_alt**2]])
        return self._update(z, h, H, R)

    def _update(self, z, h, H, R_meas):
        """Generic ESKF update (Eqs. 15–17) with Mahalanobis gating."""
        innov = z - h
        S = H @ self.P @ H.T + R_meas
        S = 0.5 * (S + S.T)

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False, np.nan

        maha = float(innov.T @ S_inv @ innov)
        if maha > self.maha_thresh:
            return False, maha

        K  = self.P @ H.T @ S_inv          # Eq. 15: Kalman gain
        dx = K @ innov                      # Eq. 16: error-state estimate

        # Eq. 17: Joseph-form covariance update (numerically stable)
        IKH = np.eye(15) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_meas @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        self._inject_and_reset(dx)          # Eq. 18
        return True, maha

    def _inject_and_reset(self, dx: np.ndarray):
        """
        Inject the error-state estimate into the nominal state (Eq. 18)
        then reset δx ← 0 (handled implicitly — we never store δx).

        Position, velocity, biases: vector addition.
        Orientation: quaternion perturbation q ← q ⊗ [1, δθ/2].
        """
        self.p  += dx[0:3]
        self.v  += dx[3:6]

        # Small-angle quaternion perturbation (body frame)
        dtheta = dx[6:9]
        dq = np.array([1.0,
                        dtheta[0] / 2.0,
                        dtheta[1] / 2.0,
                        dtheta[2] / 2.0])
        self.q = quat_normalize(quat_mult(self.q, dq))

        self.ba += dx[9:12]
        self.bg += dx[12:15]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ESKF Visual-Inertial Odometry")
    parser.add_argument("--imu_csv",    default="data/imu_20260304_132152.csv")
    parser.add_argument("--vision_csv", default="results/vision_traj_xy.csv")
    parser.add_argument("--gpx_file",   default="data/04-Mar-2026-1323.gpx")
    parser.add_argument("--duration_s", type=float, default=300.0)

    # Process noise spectral densities (tuning knobs).
    # sigma_a must be large enough to cover gravity leakage from gyro bias
    # drift (~3 deg/s bias → ~0.5 m/s² spurious horizontal acceleration).
    parser.add_argument("--sigma_a",   type=float, default=2.0,
                        help="Accel measurement noise (m/s²/√Hz)")
    parser.add_argument("--sigma_g",   type=float, default=0.05,
                        help="Gyro measurement noise (rad/s/√Hz)")
    parser.add_argument("--sigma_ba",  type=float, default=0.01,
                        help="Accel bias random-walk noise (m/s²/√Hz)")
    parser.add_argument("--sigma_bg",  type=float, default=0.001,
                        help="Gyro bias random-walk noise (rad/s/√Hz)")
    parser.add_argument("--sigma_vis", type=float, default=0.5,
                        help="Vision velocity measurement noise (m/s)")
    parser.add_argument("--altitude_m", type=float, default=50.0,
                        help="Known flight altitude for z constraint (m). Set 0 to disable.")
    parser.add_argument("--sigma_alt", type=float, default=20.0,
                        help="Altitude measurement noise (m)")
    parser.add_argument("--maha_thresh", type=float, default=50.0)
    parser.add_argument("--save_csv",  default="results/eskf_traj.csv")
    args = parser.parse_args()

    # ── Load and preprocess IMU ──────────────────────────────────────────
    print("Loading IMU ...")
    imu_df = load_imu_csv(args.imu_csv)
    print(f"  {len(imu_df)} samples, {imu_df['elapsed_s'].iloc[-1]:.1f} s")

    print("Correcting scale bugs ...")
    accel_idx, _ = detect_accel_scale_bug(imu_df)
    if accel_idx >= 0:
        correct_accel_scale(imu_df)
        correct_gyro_scale(imu_df, accel_idx)

    print("Low-pass filtering ...")
    imu_df = lowpass_imu(imu_df)

    # Calibrate biases from first 5 s (static, motors off)
    print("Calibrating biases ...")
    accel_bias, gyro_bias, q_init = estimate_static_biases(imu_df, n_samples=1000)

    # Clip to GPS time window
    print("Clipping to GPS window ...")
    imu_df, _ = clip_to_gps_window(imu_df, args.gpx_file,
                                   duration_s=args.duration_s)
    print(f"  Running ESKF over {len(imu_df)} IMU samples "
          f"({imu_df['elapsed_s'].iloc[-1]:.1f} s)")

    t_imu   = imu_df["timestamp_s"].values
    ax_arr  = imu_df["ax"].values;  ay_arr = imu_df["ay"].values;  az_arr = imu_df["az"].values
    gx_arr  = imu_df["gx"].values;  gy_arr = imu_df["gy"].values;  gz_arr = imu_df["gz"].values
    dt_arr  = imu_df["dt"].values

    # ── Load vision ──────────────────────────────────────────────────────
    print("Loading vision ...")
    vis_df = pd.read_csv(args.vision_csv)
    vis_df = vis_df.rename(columns={"t_epoch": "t", "x_m": "x", "y_m": "y"})
    vis_df = vis_df.dropna().sort_values("t").reset_index(drop=True)

    # Finite-difference accumulated positions → velocity (m/s)
    t_vis  = vis_df["t"].values
    dt_vis = np.diff(t_vis)
    vx_vis = np.diff(vis_df["x"].values) / dt_vis
    vy_vis = np.diff(vis_df["y"].values) / dt_vis
    t_vis_vel = t_vis[1:]   # timestamps for each velocity sample

    R_vis = np.eye(2) * args.sigma_vis**2

    # ── Load GPS (ground truth for evaluation only) ──────────────────────
    print("Loading GPS ...")
    gps_df = load_gpx(args.gpx_file)

    # ── Initialise ESKF ─────────────────────────────────────────────────
    # Initial covariance: modest uncertainty on everything
    P0 = np.diag([
        1.0, 1.0, 1.0,           # position    (m²)
        0.1, 0.1, 0.1,           # velocity    (m/s)²
        0.01, 0.01, 0.01,        # orientation (rad²)
        0.01, 0.01, 0.01,        # accel bias  (m/s²)²
        0.001, 0.001, 0.001,     # gyro  bias  (rad/s)²
    ])

    ekf = ESKF(
        p0=np.zeros(3),
        v0=np.zeros(3),
        q0=q_init,
        ba0=accel_bias,
        bg0=gyro_bias,
        P0=P0,
        sigma_a=args.sigma_a,
        sigma_g=args.sigma_g,
        sigma_ba=args.sigma_ba,
        sigma_bg=args.sigma_bg,
        maha_thresh=args.maha_thresh,
    )

    # ── Predict-update loop ──────────────────────────────────────────────
    print("Running ESKF ...")
    positions  = [ekf.p.copy()]
    timestamps = [t_imu[0]]
    n_accepted = 0
    vis_ptr    = 0

    for k in range(1, len(t_imu)):
        t_now = t_imu[k]
        dt    = float(dt_arr[k])

        accel_body = np.array([ax_arr[k], ay_arr[k], az_arr[k]])
        omega_body = np.array([gx_arr[k], gy_arr[k], gz_arr[k]])

        # Prediction step at IMU rate
        ekf.predict(accel_body, omega_body, dt)

        # Altitude constraint every step (prevents z drift contaminating x/y)
        if args.altitude_m > 0:
            ekf.update_altitude(args.altitude_m, args.sigma_alt)

        # Apply all vision velocity measurements up to t_now
        while vis_ptr < len(t_vis_vel) and t_vis_vel[vis_ptr] <= t_now:
            ok, _ = ekf.update_vision_velocity(
                vx_vis[vis_ptr], vy_vis[vis_ptr], R_vis
            )
            if ok:
                n_accepted += 1
            vis_ptr += 1

        positions.append(ekf.p.copy())
        timestamps.append(t_now)

    positions  = np.asarray(positions)
    timestamps = np.asarray(timestamps)
    eskf_xy    = positions[:, :2]

    n_vis_total = len(t_vis_vel)
    print(f"Vision updates accepted: {n_accepted} / {n_vis_total}")
    print(f"ESKF position range: x=[{eskf_xy[:,0].min():.1f}, {eskf_xy[:,0].max():.1f}]"
          f"  y=[{eskf_xy[:,1].min():.1f}, {eskf_xy[:,1].max():.1f}]")

    # ── Evaluate against GPS ─────────────────────────────────────────────
    gps_x = np.interp(timestamps, gps_df["t"].values, gps_df["x"].values)
    gps_y = np.interp(timestamps, gps_df["t"].values, gps_df["y"].values)
    gps_xy = np.column_stack([gps_x, gps_y])

    R_al, t_al = rigid_align_2d(eskf_xy, gps_xy)
    eskf_aligned = apply_rigid_2d(eskf_xy, R_al, t_al)

    print(f"\nATE RMSE (after rigid alignment): {ate_rmse(eskf_aligned, gps_xy):.2f} m")

    # ── Save ─────────────────────────────────────────────────────────────
    pd.DataFrame({
        "t": timestamps,
        "x": positions[:, 0],
        "y": positions[:, 1],
        "z": positions[:, 2],
    }).to_csv(args.save_csv, index=False)
    print(f"Saved: {args.save_csv}")

    # ── Plot ─────────────────────────────────────────────────────────────
    _, ax = plt.subplots(figsize=(9, 7))
    ax.plot(gps_df["x"].values, gps_df["y"].values,
            color="black", linewidth=2, label="GPS (ground truth)")
    ax.plot(eskf_aligned[:, 0], eskf_aligned[:, 1],
            color="tab:blue", linewidth=1.5, label="ESKF (aligned)")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]");  ax.set_ylabel("y [m]")
    ax.set_title("ESKF VIO Trajectory vs GPS")
    ax.grid(True);  ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
