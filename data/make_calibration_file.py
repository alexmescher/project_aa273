import numpy as np


def main():
    # COLMAP Simple Radial Parameters, 2704x1520
    # K = np.array([
    # [1223.12, 0.0, 1352.0],
    # [0.0, 1222.32, 760.0],
    # [0.0, 0.0, 1.0]
    # ], dtype=float)

    # dist = np.array([
    # -0.0237434806,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0
    # ], dtype=float)

    # Initial Estimated Params
    K = np.array([
        [1500.2, 0, 960.1],
        [0, 1498.7, 540.3],
        [0, 0, 1]
    ], dtype=float)

    dist = np.array([-0.12, 0.03, 0.001, -0.0005, 0.0], dtype=float)

    np.savez("camera_calib.npz", K=K, dist=dist)
    print("Saved camera_calib.npz")
    print("K =\n", K)
    print("dist =", dist)


if __name__ == "__main__":
    main()