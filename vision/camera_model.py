from __future__ import annotations
import json
from dataclasses import dataclass
import cv2
import numpy as np
import yaml


@dataclass
class CameraModel:
    K: np.ndarray
    dist: np.ndarray

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    @classmethod
    def from_npz(cls, path: str) -> "CameraModel":
        data = np.load(path)
        return cls(K=data["K"].astype(np.float64),
                   dist=data["dist"].astype(np.float64).reshape(-1, 1))

    @classmethod
    def from_yaml(cls, path: str) -> "CameraModel":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        K = np.array(data["K"], dtype=np.float64)
        dist = np.array(data["dist"], dtype=np.float64).reshape(-1, 1)
        return cls(K=K, dist=dist)

    @classmethod
    def from_json(cls, path: str) -> "CameraModel":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        K = np.array(data["K"], dtype=np.float64)
        dist = np.array(data["dist"], dtype=np.float64).reshape(-1, 1)
        return cls(K=K, dist=dist)

    def undistort(self, image: np.ndarray) -> np.ndarray:
        return cv2.undistort(image, self.K, self.dist)