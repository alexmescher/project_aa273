from __future__ import annotations
import cv2
import numpy as np

from .camera_model import CameraModel
from .config import VisionPipelineConfig
from .feature_tracker import FeatureTracker
from .motion_estimator import MotionEstimator
from .scale_conversion import pixel_translation_to_metric, rotate_cam_to_nav
from .vision_types import VisionMeasurement, TrackResult


class VisionPipeline:
    def __init__(self, camera: CameraModel, config: VisionPipelineConfig | None = None):
        self.camera = camera
        self.config = config or VisionPipelineConfig()
        self.tracker = FeatureTracker(self.config.feature, self.config.lk)
        self.motion_estimator = MotionEstimator(self.config.ransac)

        self.prev_gray: np.ndarray | None = None
        self.prev_bgr: np.ndarray | None = None
        self.prev_timestamp_s: float | None = None

    def reset(self) -> None:
        self.prev_gray = None
        self.prev_bgr = None
        self.prev_timestamp_s = None

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        altitude_m: float,
        yaw_rad: float | None = None,
        timestamp_s: float | None = None
    ) -> VisionMeasurement | None:
        if not np.isfinite(altitude_m):
            self._store_current(frame_bgr, timestamp_s)
            return None

        if altitude_m < self.config.measurement.min_altitude_m or altitude_m > self.config.measurement.max_altitude_m:
            self._store_current(frame_bgr, timestamp_s)
            return None

        curr_bgr, curr_gray = self._preprocess(frame_bgr)

        if self.prev_gray is None:
            self.prev_gray = curr_gray
            self.prev_bgr = curr_bgr
            self.prev_timestamp_s = timestamp_s
            return None

        prev_pts = self.tracker.detect(self.prev_gray)
        track_result = self.tracker.track(self.prev_gray, curr_gray, prev_pts)

        valid_prev = track_result.prev_pts[track_result.valid_mask]
        valid_curr = track_result.curr_pts[track_result.valid_mask]

        motion = self.motion_estimator.estimate(valid_prev, valid_curr)

        if not motion.success:
            self.prev_gray = curr_gray
            self.prev_bgr = curr_bgr
            self.prev_timestamp_s = timestamp_s
            return None

        dt = None
        if timestamp_s is not None and self.prev_timestamp_s is not None:
            dt = max(1e-6, timestamp_s - self.prev_timestamp_s)

        delta_xy_cam = pixel_translation_to_metric(
            motion.translation_px, altitude_m, self.camera
        )
        delta_xy_nav = rotate_cam_to_nav(delta_xy_cam, yaw_rad)

        speed_ok = True
        if dt is not None:
            speed = float(np.linalg.norm(delta_xy_nav) / dt)
            speed_ok = speed <= self.config.measurement.max_visual_speed_mps
            if not speed_ok:
                self.prev_gray = curr_gray
                self.prev_bgr = curr_bgr
                self.prev_timestamp_s = timestamp_s
                return None

        covariance = self._estimate_covariance(
            altitude_m=altitude_m,
            fx=self.camera.fx,
            fy=self.camera.fy,
            residual_px_rms=motion.residual_px_rms,
            num_inliers=motion.num_inliers,
        )

        meas = VisionMeasurement(
            timestamp_s=timestamp_s,
            delta_xy_m_cam=delta_xy_cam,
            delta_xy_m_nav=delta_xy_nav,
            covariance=covariance,
            translation_px=motion.translation_px,
            altitude_m=float(altitude_m),
            num_tracks=int(track_result.valid_mask.sum()),
            num_inliers=motion.num_inliers,
            inlier_ratio=motion.inlier_ratio,
            quality={
                "residual_px_rms": motion.residual_px_rms,
                "model_name": motion.model_name,
                "speed_gate_passed": speed_ok,
            }
        )

        self.prev_gray = curr_gray
        self.prev_bgr = curr_bgr
        self.prev_timestamp_s = timestamp_s
        return meas

    def _preprocess(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        undist = self.camera.undistort(frame_bgr)

        border = self.config.preprocess.border_crop_px
        if border > 0:
            undist = undist[border:-border, border:-border]

        if self.config.preprocess.resize_width is not None:
            h, w = undist.shape[:2]
            new_w = self.config.preprocess.resize_width
            new_h = int(round(h * (new_w / w)))
            undist = cv2.resize(undist, (new_w, new_h), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

        if self.config.preprocess.use_clahe:
            clahe = cv2.createCLAHE(
                clipLimit=self.config.preprocess.clahe_clip_limit,
                tileGridSize=self.config.preprocess.clahe_tile_grid_size
            )
            gray = clahe.apply(gray)

        return undist, gray

    def _estimate_covariance(
        self,
        altitude_m: float,
        fx: float,
        fy: float,
        residual_px_rms: float,
        num_inliers: int
    ) -> np.ndarray:
        n = max(1, num_inliers)
        sigma_px = max(self.config.measurement.nominal_sigma_px, residual_px_rms)

        sigma_x = (sigma_px * altitude_m / fx) / np.sqrt(n)
        sigma_y = (sigma_px * altitude_m / fy) / np.sqrt(n)

        return np.array([
            [sigma_x ** 2, 0.0],
            [0.0, sigma_y ** 2]
        ], dtype=np.float64)

    def _store_current(self, frame_bgr: np.ndarray, timestamp_s: float | None) -> None:
        curr_bgr, curr_gray = self._preprocess(frame_bgr)
        self.prev_gray = curr_gray
        self.prev_bgr = curr_bgr
        self.prev_timestamp_s = timestamp_s