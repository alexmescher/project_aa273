# project_aa273
Visual-Inertial Odometry for UAV Trajectory Estimation

## IMU Dead-Reckoning Baseline

Open-loop trajectory estimation from a BerryIMU v3 (MPU-6050) mounted on a quadrotor, sampled at 200 Hz. This serves as the prediction-only baseline before adding filter corrections (ESKF).

### Code Structure

| File | Description |
|---|---|
| `imu_integrator.py` | Quaternion math and state propagation. Exposes `propagate_state()` for single-step updates (used by the ESKF) and `integrate()` for full dead-reckoning over a DataFrame. |
| `imu_preprocessing.py` | Data loading, sensor correction, and calibration. Handles BerryIMU CSV parsing, accel/gyro scale-bug detection and correction (~10x register misconfiguration), Butterworth low-pass filtering, GPS-based flight windowing, bias estimation from static periods, and takeoff detection. |
| `imu_baseline_optimal.py` | Pipeline script that wires preprocessing and integration together. Run this to produce a trajectory CSV. |
| `plot_imu_gps.py` | Plots IMU-integrated trajectory against GPS ground truth (2D and 3D), with automatic yaw alignment. |

### Data

Flight data lives in `data/test1/` and `data/test2/`, each containing:
- `imu_*.csv` — raw IMU recording (accel in g, gyro in deg/s, with epoch timestamps)
- `*.gpx` — GPS ground truth track from a phone app
- `SWP_*.MP4` — onboard video

### Usage

```bash
# Run the dead-reckoning baseline (defaults to test2 data)
python imu_baseline_optimal.py

# Custom inputs
python imu_baseline_optimal.py data/test1/imu_20260304_132152.csv --gpx data/test1/04-Mar-2026-1323.gpx

# Plot against GPS
python plot_imu_gps.py
```

### Dependencies

```
numpy, pandas, scipy, matplotlib
```