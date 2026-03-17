from __future__ import annotations
import numpy as np
from .camera_model import CameraModel


def pixel_translation_to_metric(
    translation_px: np.ndarray,
    altitude_m: float,
    camera: CameraModel
) -> np.ndarray:
    du, dv = float(translation_px[0]), float(translation_px[1])
    dx = du * altitude_m / camera.fx
    dy = dv * altitude_m / camera.fy
    return np.array([dx, dy], dtype=np.float64)


def rotate_cam_to_nav(delta_xy_cam: np.ndarray, yaw_rad: float | None) -> np.ndarray:
    if yaw_rad is None:
        return delta_xy_cam.copy()

    c = np.cos(yaw_rad)
    s = np.sin(yaw_rad)
    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float64)
    return R @ delta_xy_cam