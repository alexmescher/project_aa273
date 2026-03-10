from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class PreprocessConfig:
    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    border_crop_px: int = 0
    resize_width: int | None = None


@dataclass
class FeatureConfig:
    max_features: int = 300
    fast_threshold: int = 20
    nonmax_suppression: bool = True
    grid_rows: int = 4
    grid_cols: int = 4
    edge_margin_px: int = 12


@dataclass
class LKConfig:
    win_size: Tuple[int, int] = (21, 21)
    max_level: int = 3
    max_iter: int = 30
    eps: float = 0.01
    fb_max_error_px: float = 1.5
    lk_max_error: float = 30.0


@dataclass
class RansacConfig:
    method: str = "partial_affine"   # "translation" or "partial_affine"
    reproj_threshold_px: float = 2.0
    max_iters: int = 2000
    confidence: float = 0.99
    min_inliers: int = 25
    min_inlier_ratio: float = 0.35


@dataclass
class MeasurementConfig:
    min_altitude_m: float = 0.5
    max_altitude_m: float = 120.0
    nominal_sigma_px: float = 1.0
    max_visual_speed_mps: float = 20.0


@dataclass
class VisionPipelineConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    lk: LKConfig = field(default_factory=LKConfig)
    ransac: RansacConfig = field(default_factory=RansacConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)