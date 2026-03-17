"""
Optimal IMU dead-reckoning baseline.

Pipeline:
  1. Load IMU CSV
  2. Accel + gyro scale bug correction
  3. Low-pass filter
  4. GPS-based flight windowing
  5. Calibration at flight start
  6. Dead-reckoning integration
  7. Export trajectory CSV
"""

import math
import argparse

import numpy as np

from imu_integrator import integrate, G_MPS2
from imu_preprocessing import (
    load_imu_csv, detect_accel_scale_bug, correct_accel_scale,
    correct_gyro_scale, lowpass_imu, load_gpx_timestamps,
    quat_from_accel, G2MPS2, DEG2RAD,
)


def main():
    parser = argparse.ArgumentParser(description="Optimal IMU dead-reckoning baseline")
    parser.add_argument("imu_csv", nargs="?",
                        default="data/test2/imu_20260304_140431.csv")
    parser.add_argument("--gpx", default="data/test2/04-Mar-2026-1405.gpx")
    parser.add_argument("-o", "--output", default="imu_trajectory.csv")
    parser.add_argument("--duration", type=float, default=100.0)
    parser.add_argument("--lpf", type=float, default=15.0,
                        help="Low-pass cutoff Hz (0=disable)")
    args = parser.parse_args()

    # 1. Load
    print(f"Loading {args.imu_csv} ...")
    df = load_imu_csv(args.imu_csv)
    print(f"  {len(df)} samples, {df['elapsed_s'].iloc[-1]:.1f}s")

    # 2. Fix accel scale bug
    print("Fixing accel scale bug ...")
    accel_trans_idx, accel_scale = detect_accel_scale_bug(df)
    if accel_trans_idx >= 0:
        correct_accel_scale(df)
    else:
        print("  Not detected")

    # 3. Fix gyro scale bug
    print("Fixing gyro scale bug ...")
    if accel_trans_idx >= 0:
        correct_gyro_scale(df, accel_trans_idx)
    else:
        print("  Skipped (no accel transition to anchor)")

    # 4. Recompute SI columns after corrections
    df["ax"] = df["ax_g"] * G2MPS2
    df["ay"] = df["ay_g"] * G2MPS2
    df["az"] = df["az_g"] * G2MPS2
    df["gx"] = df["gx_dps"] * DEG2RAD
    df["gy"] = df["gy_dps"] * DEG2RAD
    df["gz"] = df["gz_dps"] * DEG2RAD

    # 5. Low-pass filter
    if args.lpf > 0:
        print("Applying low-pass filter ...")
        lowpass_imu(df, cutoff_hz=args.lpf)

    # 6. GPS windowing
    print(f"Finding flight start from GPS ...")
    gps_epochs = load_gpx_timestamps(args.gpx)

    flight_gps_idx = 0
    for i in range(1, len(gps_epochs)):
        dt_gps = gps_epochs[i] - gps_epochs[i-1]
        if dt_gps < 5.0:
            flight_gps_idx = i
            break

    gps_start = gps_epochs[flight_gps_idx]
    sr = 200
    gps_start_idx = int(np.searchsorted(df["timestamp_s"].values, gps_start))
    end_idx = min(len(df), gps_start_idx + int(args.duration * sr))

    gap = gps_start - gps_epochs[0]
    print(f"  GPS first fix: {gps_epochs[0]:.0f} (ground)")
    print(f"  Flight start:  {gps_start:.0f} (GPS point {flight_gps_idx}, {gap:.0f}s later)")
    print(f"    -> IMU sample {gps_start_idx} (t={gps_start_idx/sr:.1f}s into recording)")

    # 7. Calibrate at flight start
    print("Calibrating at flight start ...")
    cal = df.iloc[gps_start_idx:gps_start_idx + 400]  # 2s
    accel_mean = np.array([cal["ax"].mean(), cal["ay"].mean(), cal["az"].mean()])
    measured_g = np.linalg.norm(accel_mean)
    q_init = quat_from_accel(accel_mean)
    accel_bias = np.zeros(3)
    gyro_bias = np.array([cal["gx"].mean(), cal["gy"].mean(), cal["gz"].mean()])

    print(f"  Measured |g| = {measured_g:.2f} m/s² ({measured_g/G_MPS2*100:.1f}% of true g)")
    print(f"  Gyro bias = [{np.degrees(gyro_bias[0]):.2f}, "
          f"{np.degrees(gyro_bias[1]):.2f}, {np.degrees(gyro_bias[2]):.2f}] deg/s")

    # 8. Clip to flight window
    flight = df.iloc[gps_start_idx:end_idx].copy().reset_index(drop=True)
    flight["elapsed_s"] = flight["timestamp_s"] - flight["timestamp_s"].iloc[0]
    print(f"  Flight window: {len(flight)} samples ({args.duration:.0f}s)")

    # 9. Integrate
    print("Integrating ...")
    traj = integrate(flight, q_init, accel_bias, gyro_bias, gravity_mag=measured_g)

    drift_xy = math.sqrt(traj["x"].iloc[-1]**2 + traj["y"].iloc[-1]**2)
    drift_z = abs(traj["z"].iloc[-1])
    print(f"  Final: x={traj['x'].iloc[-1]:.1f}m  y={traj['y'].iloc[-1]:.1f}m  z={traj['z'].iloc[-1]:.1f}m")
    print(f"  Horizontal drift: {drift_xy:.1f}m")
    print(f"  Vertical drift:   {drift_z:.1f}m")

    traj.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")

    return traj, flight


if __name__ == "__main__":
    traj, flight = main()
