from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class TrackResult:
    prev_pts: np.ndarray          # shape (N, 2)
    curr_pts: np.ndarray          # shape (N, 2)
    fb_error: np.ndarray          # shape (N,)
    lk_error: np.ndarray          # shape (N,)
    valid_mask: np.ndarray        # shape (N,), bool


@dataclass
class MotionEstimate:
    model_name: str
    transform: np.ndarray | None
    translation_px: np.ndarray    # shape (2,)
    inlier_mask: np.ndarray       # shape (N,), bool
    residual_px_rms: float
    num_inliers: int
    inlier_ratio: float
    success: bool
    reason: str = ""


@dataclass
class VisionMeasurement:
    timestamp_s: float | None
    delta_xy_m_cam: np.ndarray    # shape (2,)
    delta_xy_m_nav: np.ndarray    # shape (2,)
    covariance: np.ndarray        # shape (2,2)
    translation_px: np.ndarray    # shape (2,)
    altitude_m: float
    num_tracks: int
    num_inliers: int
    inlier_ratio: float
    quality: dict[str, Any] = field(default_factory=dict)