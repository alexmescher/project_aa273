import csv
import cv2
import numpy as np

from vision import CameraModel, VisionPipeline, VisionPipelineConfig


def load_scalar_csv(path: str) -> dict[float, dict[str, float]]:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row["timestamp_s"])
            out[t] = {
                "altitude_m": float(row["altitude_m"]),
                "yaw_rad": float(row["yaw_rad"]),
            }
    return out


def nearest_measurement(meas_map: dict[float, dict[str, float]], t: float) -> dict[str, float]:
    keys = np.array(list(meas_map.keys()), dtype=float)
    idx = int(np.argmin(np.abs(keys - t)))
    return meas_map[float(keys[idx])]


def main():
    cam = CameraModel.from_npz("data/camera_calib.npz")
    cfg = VisionPipelineConfig()
    pipeline = VisionPipeline(cam, cfg)

    # aux = load_scalar_csv("alts_and_yaw.csv")
    cap = cv2.VideoCapture("data/SWP_0002.MP4")
    # cap = cv2.VideoCapture("data/short_clip.mp4")

    fps = cap.get(cv2.CAP_PROP_FPS)
    dt = 1.0 / fps

    t0_epoch = 1772659571.0 # Unix timestamp of first frame
    frame_idx = 0

    traj = [np.zeros(2)]
    times = []
    covs = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t_epoch = t0_epoch + frame_idx * dt
        frame_idx += 1
        # t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        # t_s = t_ms / 1000.0

        # meas_aux = nearest_measurement(aux, t_s)
        # altitude_m = meas_aux["altitude_m"]
        # yaw_rad = meas_aux["yaw_rad"]
        altitude_m = 50
        yaw_rad = 0

        z = pipeline.process_frame(
            frame_bgr=frame,
            altitude_m=altitude_m,
            yaw_rad=yaw_rad,
            timestamp_s=t_epoch
        )

        if z is not None:
            traj.append(traj[-1] + z.delta_xy_m_nav)
            covs.append(z.covariance)
            times.append(t_epoch)
            print(
                f"t={t_epoch:8.3f}  "
                f"dxy_nav={z.delta_xy_m_nav}  "
                f"inliers={z.num_inliers}  "
                f"ratio={z.inlier_ratio:.2f}"
            )

    cap.release()

    traj = np.array(traj)
    covs = np.array(covs)
    times = np.array(times)

    cov_flat = covs.reshape(len(covs), 4)

    out = np.column_stack([
        times,
        traj,
        cov_flat
    ])

    np.savetxt(
        "results/vision_traj_xy.csv",
        out,
        delimiter=",",
        header="t_epoch,x_m,y_m,Pxx,Pxy,Pyx,Pyy",
        comments=""
    )


if __name__ == "__main__":
    main()