import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_traj_csv(path: str):
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


def main():
    parser = argparse.ArgumentParser(description="Plot x-y trajectory from CSV")
    parser.add_argument(
        "csv_file",
        nargs="?",
        default="vision_traj_xy.csv",
        help="Path to trajectory CSV file",
    )
    parser.add_argument(
        "--show-points",
        action="store_true",
        help="Show markers at each trajectory sample",
    )
    parser.add_argument(
        "--equal-axis",
        action="store_true",
        help="Use equal scaling on x and y axes",
    )

    args = parser.parse_args()

    t, x, y, cov = load_traj_csv(args.csv_file)

    plt.figure(figsize=(8, 6))

    if args.show_points:
        plt.plot(x, y, marker="o")
    else:
        plt.plot(x, y)

    plt.plot(x[0], y[0], marker="o", linestyle="None", label="start")
    plt.plot(x[-1], y[-1], marker="x", linestyle="None", label="end")

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Estimated X-Y Trajectory")
    plt.grid(True)
    plt.legend()

    if args.equal_axis:
        plt.axis("equal")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()