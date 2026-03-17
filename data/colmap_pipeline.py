#!/usr/bin/env python3
from pathlib import Path
import pycolmap
import numpy as np

IMAGE_DIR = "cal_images"      # frames used by COLMAP (1200x675)
DB_PATH = "colmap.db"
SPARSE_DIR = "sparse"

COLMAP_SIZE = (1200, 675)
RAW_SIZE = (2704, 1520)


def scale_K(K, old_size, new_size):
    old_w, old_h = old_size
    new_w, new_h = new_size

    sx = new_w / old_w
    sy = new_h / old_h

    K_scaled = K.copy()
    K_scaled[0,0] *= sx
    K_scaled[1,1] *= sy
    K_scaled[0,2] *= sx
    K_scaled[1,2] *= sy

    return K_scaled


def main():

    Path(SPARSE_DIR).mkdir(parents=True, exist_ok=True)

    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = "SIMPLE_RADIAL"

    extraction_options = pycolmap.FeatureExtractionOptions()

    pycolmap.extract_features(
        database_path=DB_PATH,
        image_path=IMAGE_DIR,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_options,
        extraction_options=extraction_options,
    )

    # -----------------------------
    # Sequential matching
    # -----------------------------
    pycolmap.match_sequential(
        database_path=DB_PATH
    )

    # -----------------------------
    # Sparse reconstruction
    # -----------------------------
    maps = pycolmap.incremental_mapping(
        database_path=DB_PATH,
        image_path=IMAGE_DIR,
        output_path=SPARSE_DIR
    )

    if len(maps) == 0:
        raise RuntimeError("COLMAP failed to reconstruct.")

    recon = pycolmap.Reconstruction(f"{SPARSE_DIR}/0")

    camera = list(recon.cameras.values())[0]

    print("Camera model:", camera.model)
    print("Image size used in COLMAP:", camera.width, camera.height)

    # OPENCV params
    f, cx, cy, k1 = camera.params

    K = np.array([
        [f, 0.0, cx],
        [0.0, f, cy],
        [0.0, 0.0, 1.0]
    ])

    dist = np.array([k1, 0.0, 0.0, 0.0, 0.0])

    # -----------------------------
    # Scale K to raw image size
    # -----------------------------
    K_raw = scale_K(K, COLMAP_SIZE, RAW_SIZE)

    print("\nIntrinsic matrix for 2704x1520:")
    print(K_raw)

    print("\nDistortion coefficients:")
    print(dist)


if __name__ == "__main__":
    main()