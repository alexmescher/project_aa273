from __future__ import annotations
import cv2
import numpy as np
from .config import RansacConfig
from .vision_types import MotionEstimate


class MotionEstimator:
    def __init__(self, cfg: RansacConfig):
        self.cfg = cfg

    def estimate(self, prev_pts: np.ndarray, curr_pts: np.ndarray) -> MotionEstimate:
        n = prev_pts.shape[0]
        if n < 3:
            return MotionEstimate(
                model_name=self.cfg.method,
                transform=None,
                translation_px=np.zeros(2, dtype=np.float64),
                inlier_mask=np.zeros((n,), dtype=bool),
                residual_px_rms=np.inf,
                num_inliers=0,
                inlier_ratio=0.0,
                success=False,
                reason="too_few_points"
            )

        if self.cfg.method == "translation":
            return self._estimate_translation(prev_pts, curr_pts)

        return self._estimate_partial_affine(prev_pts, curr_pts)

    def _estimate_translation(self, prev_pts: np.ndarray, curr_pts: np.ndarray) -> MotionEstimate:
        flow = curr_pts - prev_pts
        t = np.median(flow, axis=0)

        residuals = np.linalg.norm(flow - t[None, :], axis=1)
        inlier_mask = residuals <= self.cfg.reproj_threshold_px
        num_inliers = int(inlier_mask.sum())
        inlier_ratio = float(num_inliers / len(flow))

        success = (
            num_inliers >= self.cfg.min_inliers and
            inlier_ratio >= self.cfg.min_inlier_ratio
        )

        T = np.array([[1.0, 0.0, t[0]],
                      [0.0, 1.0, t[1]]], dtype=np.float64)

        return MotionEstimate(
            model_name="translation",
            transform=T,
            translation_px=t.astype(np.float64),
            inlier_mask=inlier_mask,
            residual_px_rms=float(np.sqrt(np.mean(residuals[inlier_mask] ** 2))) if num_inliers > 0 else np.inf,
            num_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
            success=success,
            reason="" if success else "insufficient_inliers"
        )

    def _estimate_partial_affine(self, prev_pts: np.ndarray, curr_pts: np.ndarray) -> MotionEstimate:
        M, inliers = cv2.estimateAffinePartial2D(
            prev_pts, curr_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.cfg.reproj_threshold_px,
            maxIters=self.cfg.max_iters,
            confidence=self.cfg.confidence,
            refineIters=10
        )

        if M is None or inliers is None:
            return MotionEstimate(
                model_name="partial_affine",
                transform=None,
                translation_px=np.zeros(2, dtype=np.float64),
                inlier_mask=np.zeros((len(prev_pts),), dtype=bool),
                residual_px_rms=np.inf,
                num_inliers=0,
                inlier_ratio=0.0,
                success=False,
                reason="ransac_failed"
            )

        inlier_mask = inliers.reshape(-1).astype(bool)
        num_inliers = int(inlier_mask.sum())
        inlier_ratio = float(num_inliers / len(prev_pts))

        prev_h = np.hstack([prev_pts, np.ones((len(prev_pts), 1), dtype=np.float32)])
        pred = (M @ prev_h.T).T
        residuals = np.linalg.norm(curr_pts - pred, axis=1)

        translation_px = M[:, 2].astype(np.float64)
        success = (
            num_inliers >= self.cfg.min_inliers and
            inlier_ratio >= self.cfg.min_inlier_ratio
        )

        return MotionEstimate(
            model_name="partial_affine",
            transform=M.astype(np.float64),
            translation_px=translation_px,
            inlier_mask=inlier_mask,
            residual_px_rms=float(np.sqrt(np.mean(residuals[inlier_mask] ** 2))) if num_inliers > 0 else np.inf,
            num_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
            success=success,
            reason="" if success else "insufficient_inliers"
        )