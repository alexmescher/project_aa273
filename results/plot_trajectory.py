import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_vision_traj(path: str):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 7:
        raise ValueError("CSV must contain: t,x,y,Pxx,Pxy,Pyx,Pyy")
    t = data[:, 0]
    x = data[:, 1]
    y = data[:, 2]
    cov = data[:, 3:7].reshape(-1, 2, 2)
    return t, x, y, cov


def load_gps_traj(path: str):
    """Parse a GPX file and return local ENU (east, north) coords in metres."""
    import xml.etree.ElementTree as ET
    ns = {"gpx": "http://www.topografix.com/GPX/1/1"}
    tree = ET.parse(path)
    root = tree.getroot()
    pts = root.findall(".//gpx:trkpt", ns)
    lats = np.array([float(p.attrib["lat"]) for p in pts])
    lons = np.array([float(p.attrib["lon"]) for p in pts])

    # Flat-earth projection centred on first point
    R = 6_371_000.0
    lat0, lon0 = np.deg2rad(lats[0]), np.deg2rad(lons[0])
    east  = R * np.cos(lat0) * (np.deg2rad(lons) - lon0)
    north = R * (np.deg2rad(lats) - lat0)
    return east, north


def initial_heading(x: np.ndarray, y: np.ndarray, n_pts: int = 10) -> float:
    """Estimate initial heading from the first nonzero displacement in the GPS track."""
    for i in range(1, min(n_pts, len(x))):
        dx, dy = x[i] - x[0], y[i] - y[0]
        if np.hypot(dx, dy) > 1e-6:
            return np.arctan2(dy, dx)
    return 0.0


def rotate_2d(x: np.ndarray, y: np.ndarray, theta: float):
    c, s = np.cos(theta), np.sin(theta)
    xr = c * x - s * y
    yr = s * x + c * y
    return xr, yr


def main():
    parser = argparse.ArgumentParser(description="Plot vision trajectory overlaid on GPS")
    parser.add_argument(
        "csv_file",
        nargs="?",
        default="vision_traj_xy.csv",
        help="Path to vision trajectory CSV",
    )
    parser.add_argument(
        "--gps",
        default="../data/04-Mar-2026-1323.gpx",
        help="Path to GPX file",
    )
    parser.add_argument("--show-points", action="store_true")
    parser.add_argument("--equal-axis", action="store_true")

    args = parser.parse_args()

    t, vx, vy, _ = load_vision_traj(args.csv_file)
    gps_x, gps_y = load_gps_traj(args.gps)

    # Align vision trajectory: rotate to GPS initial heading, translate to GPS start
    theta = initial_heading(gps_x, gps_y)
    vx_rot, vy_rot = rotate_2d(vx, vy, theta)
    vx_rot += gps_x[0]
    vy_rot += gps_y[0]

    gps_len = float(np.sum(np.hypot(np.diff(gps_x), np.diff(gps_y))))
    vis_len = float(np.sum(np.hypot(np.diff(vx_rot), np.diff(vy_rot))))
    print(f"GPS trajectory length:    {gps_len:.1f} m")
    print(f"Vision trajectory length: {vis_len:.1f} m")

    marker = "o" if args.show_points else None

    _, ax = plt.subplots(figsize=(8, 6))

    ax.plot(gps_x, gps_y, color="tab:blue", marker=marker, label="GPS")
    ax.plot(gps_x[0], gps_y[0], "o", color="tab:blue")
    ax.plot(gps_x[-1], gps_y[-1], "x", color="tab:blue")

    ax.plot(vx_rot, vy_rot, color="tab:orange", marker=marker, label="Vision")
    ax.plot(vx_rot[0], vy_rot[0], "o", color="tab:orange")
    ax.plot(vx_rot[-1], vy_rot[-1], "x", color="tab:orange")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Vision vs GPS Trajectory")
    ax.grid(True)
    ax.legend()

    if args.equal_axis:
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()