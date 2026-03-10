from __future__ import annotations
import cv2
import numpy as np
from .config import FeatureConfig, LKConfig
from .vision_types import TrackResult


class FeatureTracker:
    def __init__(self, feature_cfg: FeatureConfig, lk_cfg: LKConfig):
        self.feature_cfg = feature_cfg
        self.lk_cfg = lk_cfg
        self.fast = cv2.FastFeatureDetector_create(
            threshold=feature_cfg.fast_threshold,
            nonmaxSuppression=feature_cfg.nonmax_suppression
        )

    def detect(self, gray: np.ndarray) -> np.ndarray:
        keypoints = self.fast.detect(gray, None)
        if not keypoints:
            return np.empty((0, 2), dtype=np.float32)

        pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        responses = np.array([kp.response for kp in keypoints], dtype=np.float32)

        h, w = gray.shape[:2]
        m = self.feature_cfg.edge_margin_px
        keep = (
            (pts[:, 0] >= m) & (pts[:, 0] < w - m) &
            (pts[:, 1] >= m) & (pts[:, 1] < h - m)
        )
        pts = pts[keep]
        responses = responses[keep]

        if pts.shape[0] == 0:
            return pts

        return self._grid_select(pts, responses, w, h)

    def _grid_select(
        self,
        pts: np.ndarray,
        responses: np.ndarray,
        width: int,
        height: int
    ) -> np.ndarray:
        rows = self.feature_cfg.grid_rows
        cols = self.feature_cfg.grid_cols
        max_features = self.feature_cfg.max_features
        per_cell = max(1, max_features // (rows * cols))

        selected = []
        cell_w = width / cols
        cell_h = height / rows

        for r in range(rows):
            for c in range(cols):
                x0, x1 = c * cell_w, (c + 1) * cell_w
                y0, y1 = r * cell_h, (r + 1) * cell_h
                mask = (
                    (pts[:, 0] >= x0) & (pts[:, 0] < x1) &
                    (pts[:, 1] >= y0) & (pts[:, 1] < y1)
                )
                cell_pts = pts[mask]
                cell_resp = responses[mask]

                if cell_pts.shape[0] == 0:
                    continue

                order = np.argsort(-cell_resp)
                cell_pts = cell_pts[order[:per_cell]]
                selected.append(cell_pts)

        if not selected:
            return np.empty((0, 2), dtype=np.float32)

        selected = np.vstack(selected).astype(np.float32)

        if selected.shape[0] > max_features:
            selected = selected[:max_features]

        return selected

    def track(self, prev_gray: np.ndarray, curr_gray: np.ndarray, prev_pts: np.ndarray) -> TrackResult:
        if prev_pts.size == 0:
            empty_n = 0
            return TrackResult(
                prev_pts=np.empty((0, 2), dtype=np.float32),
                curr_pts=np.empty((0, 2), dtype=np.float32),
                fb_error=np.empty((0,), dtype=np.float32),
                lk_error=np.empty((0,), dtype=np.float32),
                valid_mask=np.zeros((empty_n,), dtype=bool),
            )

        p0 = prev_pts.reshape(-1, 1, 2).astype(np.float32)
        term_crit = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.lk_cfg.max_iter,
            self.lk_cfg.eps,
        )

        p1, st_fwd, err_fwd = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, p0, None,
            winSize=self.lk_cfg.win_size,
            maxLevel=self.lk_cfg.max_level,
            criteria=term_crit
        )

        p0_back, st_back, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, prev_gray, p1, None,
            winSize=self.lk_cfg.win_size,
            maxLevel=self.lk_cfg.max_level,
            criteria=term_crit
        )

        prev_pts_2 = p0.reshape(-1, 2)
        curr_pts_2 = p1.reshape(-1, 2)
        back_pts_2 = p0_back.reshape(-1, 2)
        fb_error = np.linalg.norm(prev_pts_2 - back_pts_2, axis=1)
        lk_error = err_fwd.reshape(-1)

        valid = (
            st_fwd.reshape(-1).astype(bool) &
            st_back.reshape(-1).astype(bool) &
            np.isfinite(curr_pts_2).all(axis=1) &
            (fb_error <= self.lk_cfg.fb_max_error_px) &
            (lk_error <= self.lk_cfg.lk_max_error)
        )

        return TrackResult(
            prev_pts=prev_pts_2,
            curr_pts=curr_pts_2,
            fb_error=fb_error.astype(np.float32),
            lk_error=lk_error.astype(np.float32),
            valid_mask=valid
        )