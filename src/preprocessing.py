from __future__ import annotations

from pathlib import Path
from typing import Optional

import math
import pandas as pd
import gpxpy


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute great-circle distance between two lat/lon points in meters.
    """
    r_earth = 6371000.0  # meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r_earth * c


def load_gpx(gpx_file: str | Path) -> pd.DataFrame:
    """
    Load a GPX file into a pandas DataFrame.

    Parameters
    ----------
    gpx_file : str or Path
        Path to the .gpx file.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per GPX track point and columns:
        ['track_id', 'segment_id', 'point_id',
         'latitude', 'longitude', 'elevation_m', 'time',
         'elapsed_time_s', 'step_distance_m', 'cumulative_distance_m']
    """
    gpx_file = Path(gpx_file)

    if not gpx_file.exists():
        raise FileNotFoundError(f"GPX file not found: {gpx_file}")

    with gpx_file.open("r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)

    rows = []

    for track_id, track in enumerate(gpx.tracks):
        for segment_id, segment in enumerate(track.segments):
            for point_id, point in enumerate(segment.points):
                rows.append(
                    {
                        "track_id": track_id,
                        "segment_id": segment_id,
                        "point_id": point_id,
                        "latitude": point.latitude,
                        "longitude": point.longitude,
                        "elevation_m": point.elevation,
                        "time": point.time,
                    }
                )

    if not rows:
        raise ValueError(f"No track points found in GPX file: {gpx_file}")

    df = pd.DataFrame(rows)

    # Ensure time is pandas datetime when present
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")

    # Compute elapsed time from first valid timestamp
    if df["time"].notna().any():
        t0 = df["time"].dropna().iloc[0]
        df["elapsed_time_s"] = (df["time"] - t0).dt.total_seconds()
    else:
        df["elapsed_time_s"] = pd.NA

    # Compute step and cumulative distances
    step_distances = [0.0]

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        d = haversine_distance_m(
            prev["latitude"],
            prev["longitude"],
            curr["latitude"],
            curr["longitude"],
        )
        step_distances.append(d)

    df["step_distance_m"] = step_distances
    df["cumulative_distance_m"] = df["step_distance_m"].cumsum()

    return df


def summarize_gpx(df: pd.DataFrame) -> dict:
    """
    Return a simple summary dictionary for a GPX dataframe.
    """
    summary = {
        "n_points": len(df),
        "start_latitude": df.iloc[0]["latitude"],
        "start_longitude": df.iloc[0]["longitude"],
        "end_latitude": df.iloc[-1]["latitude"],
        "end_longitude": df.iloc[-1]["longitude"],
        "total_distance_m": float(df["cumulative_distance_m"].iloc[-1]),
    }

    if "elapsed_time_s" in df.columns and df["elapsed_time_s"].notna().any():
        summary["total_time_s"] = float(df["elapsed_time_s"].dropna().iloc[-1])
    else:
        summary["total_time_s"] = None

    return summary


if __name__ == "__main__":
    # Example standalone usage
    gpx_path = "data/example.gpx"

    df = load_gpx(gpx_path)
    print(df.head())

    summary = summarize_gpx(df)
    print("\nSummary:")
    for key, value in summary.items():
        print(f"{key}: {value}")