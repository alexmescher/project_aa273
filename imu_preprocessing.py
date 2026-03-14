"""
imu_preprocessing.py — IMU data loading, calibration, and correction.

Handles everything between "raw CSV on disk" and "clean DataFrame ready
for integration": unit conversion, scale-bug correction, filtering,
bias estimation, GPS windowing, and takeoff detection.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from imu_integrator import (
    G_MPS2, quat_rotate_vec, quat_conj,
    quat_normalize, quat_mult, quat_from_gyro,
)


# ── Constants ────────────────────────────────
DEG2RAD = math.pi / 180.0
G2MPS2 = G_MPS2  # 1 g = 9.80665 m/s²


# ── Data loading ─────────────────────────────

def _parse_header_float(path: Path, key: str) -> Optional[float]:
    """Extract a float value from '# key=value' header comments."""
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("#"):
                break
            if key in line:
                try:
                    return float(line.split("=")[1].strip())
                except (ValueError, IndexError):
                    return None
    return None


def load_imu_csv(path: str | Path) -> pd.DataFrame:
    """
    Load BerryIMU CSV and convert to SI units.

    Input columns:  epoch_s, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps
    Output adds:    ax/ay/az (m/s²), gx/gy/gz (rad/s), dt (seconds)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"IMU CSV not found: {path}")

    df = pd.read_csv(path, comment="#")
    df = df.dropna(subset=["epoch_s"]).reset_index(drop=True)

    # Accel: g → m/s²
    df["ax"] = df["ax_g"] * G2MPS2
    df["ay"] = df["ay_g"] * G2MPS2
    df["az"] = df["az_g"] * G2MPS2

    # Gyro: deg/s → rad/s
    df["gx"] = df["gx_dps"] * DEG2RAD
    df["gy"] = df["gy_dps"] * DEG2RAD
    df["gz"] = df["gz_dps"] * DEG2RAD

    # Timestamps
    df["timestamp_s"] = df["epoch_s"].astype(float)
    start_time = _parse_header_float(path, "epoch_start_s")
    sample_rate = _parse_header_float(path, "sample_rate_hz")

    if start_time is not None and sample_rate is not None:
        df["timestamp_s"] = start_time + np.arange(len(df)) / sample_rate
        df["dt"] = 1.0 / sample_rate
    else:
        df["dt"] = 1.0 / 200.0

    df["elapsed_s"] = df["timestamp_s"] - df["timestamp_s"].iloc[0]
    return df


# ── Accel scale-bug correction ───────────────

def detect_accel_scale_bug(df: pd.DataFrame, min_ratio: float = 3.0) -> tuple[int, float]:
    """
    Detect MPU-6050 accel scale bug (external process reconfigures register,
    causing ~10x magnitude drop). Returns (transition_idx, scale_factor).
    Returns (-1, 1.0) if not detected.
    """
    amag = np.sqrt(df["ax_g"]**2 + df["ay_g"]**2 + df["az_g"]**2)

    window = 200
    if len(amag) < window * 4:
        return -1, 1.0

    rolling = amag.rolling(window, center=True).mean()

    transition_idx = -1
    for i in range(window, len(rolling) - window):
        if pd.isna(rolling.iloc[i]):
            continue
        if rolling.iloc[i - window] > 0.7 and rolling.iloc[i] < 0.3:
            future = amag.iloc[i:i + 400]
            if len(future) >= 400 and future.mean() < 0.3:
                transition_idx = i
                break

    if transition_idx < 0:
        return -1, 1.0

    pre_start = max(0, transition_idx - 2000)
    pre_end = transition_idx - 100
    pre_mag = amag.iloc[pre_start:pre_end].median()

    best_post_mag = None
    best_post_std = 999.0
    search_end = min(len(amag), transition_idx + 12000)
    win = 200
    for scan in range(transition_idx + 200, search_end - win, 50):
        chunk = amag.iloc[scan:scan + win]
        s = float(chunk.std())
        if s < best_post_std:
            best_post_std = s
            best_post_mag = float(chunk.median())

    if best_post_mag is None or best_post_mag < 0.005:
        return -1, 1.0

    scale_factor = float(pre_mag / best_post_mag)
    if scale_factor < min_ratio:
        return -1, 1.0

    return transition_idx, scale_factor


def correct_accel_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and correct accel scale bug in-place."""
    idx, scale = detect_accel_scale_bug(df)
    if idx < 0:
        print("  Accel scale bug: NOT detected (data looks clean)")
        return df

    print(f"  Accel scale bug: DETECTED at sample {idx} (t={idx/200:.1f}s)")
    print(f"    Correction factor: {scale:.2f}x applied to samples {idx}+")

    df.loc[idx:, "ax_g"] *= scale
    df.loc[idx:, "ay_g"] *= scale
    df.loc[idx:, "az_g"] *= scale
    if "ax" in df.columns:
        df.loc[idx:, "ax"] *= scale
        df.loc[idx:, "ay"] *= scale
        df.loc[idx:, "az"] *= scale
    return df


# ── Gyro scale-bug correction ────────────────

def correct_gyro_scale(df: pd.DataFrame, accel_transition_idx: int) -> pd.DataFrame:
    """
    Correct gyro scale bug. Same external process that reconfigures the
    accel also changes the gyro range register, causing a ~10x jump.
    """
    gmag = np.sqrt(df["gx_dps"]**2 + df["gy_dps"]**2 + df["gz_dps"]**2)

    if accel_transition_idx < 500:
        print("  Gyro scale: skipped (transition too early)")
        return df

    pre = gmag.iloc[max(0, accel_transition_idx-1000):accel_transition_idx]
    pre_mag = pre.median()

    best_post_mag = None
    best_std = 999.0
    search_end = min(len(gmag), accel_transition_idx + 12000)
    for scan in range(accel_transition_idx + 200, search_end - 200, 50):
        chunk = gmag.iloc[scan:scan+200]
        s = float(chunk.std())
        if s < best_std:
            best_std = s
            best_post_mag = float(chunk.median())

    if best_post_mag is None or best_post_mag < 0.1 or pre_mag < 0.1:
        print("  Gyro scale: could not compute (no quiet periods)")
        return df

    scale = pre_mag / best_post_mag

    if scale < 0.3 or scale > 0.95:
        if best_post_mag / pre_mag > 3.0:
            scale = pre_mag / best_post_mag
        else:
            print(f"  Gyro scale: ratio {1/scale:.2f}x too small, skipping")
            return df

    print(f"  Gyro scale bug: DETECTED")
    print(f"    Pre magnitude: {pre_mag:.2f}deg/s, Post: {best_post_mag:.2f}deg/s")
    print(f"    Correction factor: {1/scale:.2f}x -> applying {scale:.4f}")

    df.loc[accel_transition_idx:, "gx_dps"] *= scale
    df.loc[accel_transition_idx:, "gy_dps"] *= scale
    df.loc[accel_transition_idx:, "gz_dps"] *= scale
    if "gx" in df.columns:
        df.loc[accel_transition_idx:, "gx"] *= scale
        df.loc[accel_transition_idx:, "gy"] *= scale
        df.loc[accel_transition_idx:, "gz"] *= scale
    return df


# ── Low-pass filter ──────────────────────────

def lowpass_imu(df: pd.DataFrame, cutoff_hz: float = 15.0,
                sample_rate: float = 200.0, order: int = 2) -> pd.DataFrame:
    """Butterworth low-pass filter on accel and gyro columns."""
    nyq = sample_rate / 2.0
    b, a = butter(order, cutoff_hz / nyq, btype='low')
    for col in ["ax", "ay", "az", "gx", "gy", "gz"]:
        if col in df.columns:
            df[col] = filtfilt(b, a, df[col].values)
    for col in ["ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"]:
        if col in df.columns:
            df[col] = filtfilt(b, a, df[col].values)
    print(f"  Low-pass filtered at {cutoff_hz} Hz (Butterworth order {order})")
    return df


# ── GPS loading ──────────────────────────────

def load_gpx_timestamps(gpx_path: str | Path) -> np.ndarray:
    """Load GPS track point epoch timestamps from a GPX file."""
    import xml.etree.ElementTree as ET
    from datetime import datetime, timezone

    gpx_path = Path(gpx_path)
    if not gpx_path.exists():
        raise FileNotFoundError(f"GPX file not found: {gpx_path}")

    tree = ET.parse(gpx_path)
    root = tree.getroot()
    ns = ""
    if root.tag.startswith("{") and "}" in root.tag:
        ns = root.tag.split("}", 1)[0] + "}"

    epochs = []
    for trkpt in root.iter(f"{ns}trkpt"):
        time_el = trkpt.find(f"{ns}time")
        if time_el is None or not (time_el.text or "").strip():
            continue
        t_txt = time_el.text.strip()
        if t_txt.endswith("Z"):
            dt = datetime.fromisoformat(t_txt.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(t_txt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        epochs.append(dt.timestamp())

    if not epochs:
        raise ValueError(f"No timestamped track points in {gpx_path}")
    return np.array(sorted(epochs))


# ── Orientation from gravity ─────────────────

def quat_from_accel(accel: np.ndarray) -> np.ndarray:
    """
    Estimate orientation from a static accelerometer reading (gravity alignment).
    Recovers roll and pitch; yaw is unobservable → set to zero.

    Parameters
    ----------
    accel : (3,) array in m/s²

    Returns
    -------
    q : (4,) unit quaternion [w, x, y, z] (body → world)
    """
    g_body = accel / np.linalg.norm(accel)
    roll  = math.atan2(g_body[1], g_body[2])
    pitch = math.atan2(-g_body[0], math.sqrt(g_body[1]**2 + g_body[2]**2))
    yaw   = 0.0

    cr, sr = math.cos(roll/2),  math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2),   math.sin(yaw/2)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ])


# ── Calibration ──────────────────────────────

def find_quiet_calibration_window(df: pd.DataFrame, target_idx: int,
                                   search_radius_s: float = 30.0,
                                   window_s: float = 2.0,
                                   sr: int = 200) -> tuple[int, int]:
    """Find the quietest period near target_idx for calibration."""
    amag = np.sqrt(df["ax_g"]**2 + df["ay_g"]**2 + df["az_g"]**2)
    win = int(window_s * sr)
    search_start = max(0, target_idx - int(search_radius_s * sr))
    search_end = min(len(df) - win, target_idx + int(search_radius_s * sr))

    best_score = 999.0
    best_start = target_idx
    for i in range(search_start, search_end, 50):
        chunk = amag.iloc[i:i+win]
        mean_mag = float(chunk.mean())
        std_mag = float(chunk.std())
        score = abs(mean_mag - 1.0) + std_mag * 10
        if score < best_score:
            best_score = score
            best_start = i
    return best_start, best_start + win


def calibrate_at_window(df: pd.DataFrame, start_idx: int,
                         end_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate orientation, accel bias, and gyro bias from a quiet window."""
    window = df.iloc[start_idx:end_idx]
    accel_mean = np.array([
        window["ax"].mean(), window["ay"].mean(), window["az"].mean()
    ])
    gyro_mean = np.array([
        window["gx"].mean(), window["gy"].mean(), window["gz"].mean()
    ])

    q = quat_from_accel(accel_mean)
    expected = quat_rotate_vec(quat_conj(q), np.array([0.0, 0.0, G_MPS2]))
    accel_bias = accel_mean - expected

    amag = np.linalg.norm(accel_mean) / G_MPS2
    print(f"  Calibration window: samples {start_idx}-{end_idx} "
          f"(t={start_idx/200:.1f}-{end_idx/200:.1f}s)")
    print(f"    |accel| = {amag:.3f}g")
    print(f"    accel_bias (m/s²) = [{accel_bias[0]:.4f}, {accel_bias[1]:.4f}, {accel_bias[2]:.4f}]")
    print(f"    gyro_bias  (deg/s)  = [{np.degrees(gyro_mean[0]):.2f}, "
          f"{np.degrees(gyro_mean[1]):.2f}, {np.degrees(gyro_mean[2]):.2f}]")
    return q, accel_bias, gyro_mean


def estimate_static_biases(
    df: pd.DataFrame, n_samples: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate IMU biases and initial orientation from a static period
    (first n_samples of the recording, motors off).

    Returns (accel_bias, gyro_bias, q_init).
    """
    init = df.iloc[:n_samples]
    gyro_bias = np.array([init["gx"].mean(), init["gy"].mean(), init["gz"].mean()])
    accel_static = np.array([init["ax"].mean(), init["ay"].mean(), init["az"].mean()])

    q_init = quat_from_accel(accel_static)
    expected_body = quat_rotate_vec(quat_conj(q_init), np.array([0.0, 0.0, G_MPS2]))
    accel_bias = accel_static - expected_body

    print(f"  Static calibration ({n_samples} samples, {n_samples/200:.1f}s):")
    print(f"    Accel bias (m/s²): [{accel_bias[0]:.4f}, {accel_bias[1]:.4f}, {accel_bias[2]:.4f}]")
    print(f"    Gyro bias  (deg/s):  [{np.degrees(gyro_bias[0]):.2f}, "
          f"{np.degrees(gyro_bias[1]):.2f}, {np.degrees(gyro_bias[2]):.2f}]")
    return accel_bias, gyro_bias, q_init


# ── Takeoff detection & flight clipping ──────

def detect_takeoff(df: pd.DataFrame, accel_threshold_g: float = 1.15,
                   window: int = 50) -> int:
    """
    Detect takeoff from accel magnitude exceeding 1g threshold.
    Returns sample index of takeoff.
    """
    mag = np.sqrt(df["ax_g"]**2 + df["ay_g"]**2 + df["az_g"]**2)
    count = 0
    for i in range(len(mag)):
        if mag.iloc[i] > accel_threshold_g:
            count += 1
            if count >= window:
                return i - window + 1
        else:
            count = 0
    print("  WARNING: No clear takeoff detected, using start of data")
    return 0


def clip_to_gps_window(
    df: pd.DataFrame, gpx_path: str | Path, duration_s: float = 100.0,
) -> tuple[pd.DataFrame, float]:
    """Clip IMU data to start at the first GPS timestamp."""
    gps_epochs = load_gpx_timestamps(gpx_path)
    gps_start = gps_epochs[0]
    imu_timestamps = df["timestamp_s"].values
    gps_start_idx = int(np.searchsorted(imu_timestamps, gps_start))

    if gps_start_idx >= len(df):
        raise ValueError(
            f"First GPS point ({gps_start:.0f}) is after all IMU data. "
            f"IMU range: {imu_timestamps[0]:.0f} – {imu_timestamps[-1]:.0f}"
        )

    sample_rate = 1.0 / df["dt"].iloc[0]
    end_idx = min(len(df), gps_start_idx + int(duration_s * sample_rate))
    clipped = df.iloc[gps_start_idx:end_idx].copy().reset_index(drop=True)
    clipped["elapsed_s"] = clipped["timestamp_s"] - clipped["timestamp_s"].iloc[0]

    imu_offset = gps_start_idx / sample_rate
    print(f"  First GPS point: {gps_start:.3f} epoch")
    print(f"    -> IMU sample {gps_start_idx} ({imu_offset:.1f}s into recording)")
    print(f"  Clipped to {len(clipped)} samples ({duration_s:.0f}s)")
    return clipped, gps_start


def clip_to_flight(
    df: pd.DataFrame, duration_s: float = 100.0, pre_takeoff_s: float = 1.0,
) -> pd.DataFrame:
    """Clip IMU data to [takeoff - pre_takeoff_s, takeoff + duration_s]."""
    takeoff_idx = detect_takeoff(df)
    sample_rate = 1.0 / df["dt"].iloc[0]
    pre_samples = int(pre_takeoff_s * sample_rate)
    post_samples = int(duration_s * sample_rate)

    start = max(0, takeoff_idx - pre_samples)
    end = min(len(df), takeoff_idx + post_samples)
    clipped = df.iloc[start:end].copy().reset_index(drop=True)
    clipped["timestamp_s"] = clipped["timestamp_s"] - clipped["timestamp_s"].iloc[0]
    clipped["elapsed_s"] = clipped["timestamp_s"]

    takeoff_time = pre_samples / sample_rate
    print(f"  Takeoff at sample {takeoff_idx} ({takeoff_idx/sample_rate:.1f}s into recording)")
    print(f"  Clipped to {len(clipped)} samples: {pre_takeoff_s:.1f}s pre + {duration_s:.0f}s flight")
    return clipped


def propagate_orientation_to_takeoff(
    df: pd.DataFrame, q_init: np.ndarray, gyro_bias: np.ndarray,
) -> np.ndarray:
    """
    Integrate gyro from recording start to takeoff to get correct
    orientation at takeoff (the drone may be moved between recording
    start and takeoff).
    """
    takeoff_idx = detect_takeoff(df)
    q = q_init.copy()
    gx = df["gx"].values
    gy = df["gy"].values
    gz = df["gz"].values
    dt_arr = df["dt"].values

    for i in range(1, takeoff_idx):
        omega = np.array([
            gx[i] - gyro_bias[0],
            gy[i] - gyro_bias[1],
            gz[i] - gyro_bias[2],
        ])
        q = quat_normalize(quat_mult(q, quat_from_gyro(omega, dt_arr[i])))

    angle = 2 * math.acos(min(1.0, abs(quat_mult(quat_conj(q_init), q)[0])))
    print(f"  Orientation change (start -> takeoff): {math.degrees(angle):.1f} deg")
    return q
