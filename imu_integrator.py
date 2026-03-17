"""
imu_integrator.py — Quaternion math and IMU state propagation.

Provides the numerical core for dead-reckoning integration and ESKF
prediction. No data loading, no I/O — just math.

Units:
    - Accel: m/s²
    - Gyro: rad/s
    - Position: meters
    - Orientation: unit quaternion [w, x, y, z] (Hamilton, scalar-first)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd


# ── Constants ────────────────────────────────
G_MPS2 = 9.80665  # standard gravity (m/s²)


# ── Quaternion helpers ───────────────────────

def quat_mult(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Hamilton product q ⊗ r."""
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = r
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate (inverse for unit quaternions)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize to unit length."""
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q:  v' = q ⊗ [0,v] ⊗ q*."""
    v_quat = np.array([0.0, v[0], v[1], v[2]])
    rotated = quat_mult(quat_mult(q, v_quat), quat_conj(q))
    return rotated[1:]


def quat_from_gyro(omega: np.ndarray, dt: float) -> np.ndarray:
    """
    Incremental rotation quaternion from angular velocity and timestep.

    Uses exact axis-angle formula (not small-angle approximation).
    Returns Δq such that q_new = q_old ⊗ Δq.
    """
    angle = np.linalg.norm(omega) * dt
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = omega / np.linalg.norm(omega)
    half = angle / 2.0
    return np.array([
        math.cos(half),
        axis[0] * math.sin(half),
        axis[1] * math.sin(half),
        axis[2] * math.sin(half),
    ])


# ── State propagation ───────────────────────

def propagate_state(
    q: np.ndarray,
    vel: np.ndarray,
    pos: np.ndarray,
    accel_body: np.ndarray,
    omega: np.ndarray,
    dt: float,
    accel_bias: np.ndarray,
    gyro_bias: np.ndarray,
    gravity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Single-step IMU state propagation (one sample).

    Parameters
    ----------
    q : (4,) quaternion [w,x,y,z] — current orientation (body→world)
    vel : (3,) — current velocity in world frame (m/s)
    pos : (3,) — current position in world frame (m)
    accel_body : (3,) — raw accelerometer reading (m/s²)
    omega : (3,) — raw gyroscope reading (rad/s)
    dt : float — timestep (s)
    accel_bias : (3,) — accelerometer bias (m/s²)
    gyro_bias : (3,) — gyroscope bias (rad/s)
    gravity : (3,) — gravity vector in world frame, e.g. [0, 0, 9.81]

    Returns
    -------
    (new_q, new_vel, new_pos)
    """
    # 1. Gyro: subtract bias, update orientation
    omega_corrected = omega - gyro_bias
    q_new = quat_normalize(quat_mult(q, quat_from_gyro(omega_corrected, dt)))

    # 2. Accel: subtract bias, rotate to world frame, remove gravity
    a_body = accel_body - accel_bias
    a_world = quat_rotate_vec(q_new, a_body)
    a_linear = a_world - gravity

    # 3. Euler integration
    vel_new = vel + a_linear * dt
    pos_new = pos + vel_new * dt

    return q_new, vel_new, pos_new


# ── Convenience wrapper ──────────────────────

def integrate(
    df: pd.DataFrame,
    q_init: np.ndarray,
    accel_bias: np.ndarray,
    gyro_bias: np.ndarray,
    gravity_mag: Optional[float] = None,
) -> pd.DataFrame:
    """
    Dead-reckoning integration over a full IMU DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ax, ay, az (m/s²), gx, gy, gz (rad/s), dt (s).
    q_init : (4,) quaternion — initial orientation.
    accel_bias : (3,) — accelerometer bias (m/s²).
    gyro_bias : (3,) — gyroscope bias (rad/s).
    gravity_mag : float, optional
        Gravity magnitude to use. If None, uses true g = 9.80665.
        Using the measured |accel| at rest absorbs sensor scale drift.

    Returns
    -------
    pd.DataFrame with columns:
        timestamp_s, elapsed_s, x, y, z, vx, vy, vz, qw, qx, qy, qz
    """
    n = len(df)
    g_mag = gravity_mag if gravity_mag is not None else G_MPS2
    gravity = np.array([0.0, 0.0, g_mag])

    positions = np.zeros((n, 3))
    velocities = np.zeros((n, 3))
    orientations = np.zeros((n, 4))
    orientations[0] = q_init

    q = q_init.copy()
    vel = np.zeros(3)
    pos = np.zeros(3)

    ax = df["ax"].values; ay = df["ay"].values; az = df["az"].values
    gx = df["gx"].values; gy = df["gy"].values; gz = df["gz"].values
    dt_arr = df["dt"].values

    for i in range(1, n):
        accel_body = np.array([ax[i], ay[i], az[i]])
        omega = np.array([gx[i], gy[i], gz[i]])

        q, vel, pos = propagate_state(
            q, vel, pos, accel_body, omega, dt_arr[i],
            accel_bias, gyro_bias, gravity,
        )
        positions[i] = pos
        velocities[i] = vel
        orientations[i] = q

    return pd.DataFrame({
        "timestamp_s": df["timestamp_s"].values,
        "elapsed_s": df["elapsed_s"].values if "elapsed_s" in df.columns
                     else df["timestamp_s"].values - df["timestamp_s"].values[0],
        "x": positions[:, 0], "y": positions[:, 1], "z": positions[:, 2],
        "vx": velocities[:, 0], "vy": velocities[:, 1], "vz": velocities[:, 2],
        "qw": orientations[:, 0], "qx": orientations[:, 1],
        "qy": orientations[:, 2], "qz": orientations[:, 3],
    })
