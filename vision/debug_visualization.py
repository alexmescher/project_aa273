from __future__ import annotations
import cv2
import numpy as np
from .vision_types import TrackResult, MotionEstimate


def draw_tracks(
    image_bgr: np.ndarray,
    track_result: TrackResult,
    motion: MotionEstimate | None = None
) -> np.ndarray:
    out = image_bgr.copy()

    prev_pts = track_result.prev_pts
    curr_pts = track_result.curr_pts
    valid = track_result.valid_mask

    inlier_mask = None
    if motion is not None and motion.inlier_mask.shape[0] == curr_pts[valid].shape[0]:
        inlier_mask = motion.inlier_mask

    valid_prev = prev_pts[valid]
    valid_curr = curr_pts[valid]

    for i, (p0, p1) in enumerate(zip(valid_prev, valid_curr)):
        x0, y0 = int(round(p0[0])), int(round(p0[1]))
        x1, y1 = int(round(p1[0])), int(round(p1[1]))

        color = (0, 255, 0)
        if inlier_mask is not None and not inlier_mask[i]:
            color = (0, 0, 255)

        cv2.line(out, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
        cv2.circle(out, (x1, y1), 2, color, -1, cv2.LINE_AA)

    return out