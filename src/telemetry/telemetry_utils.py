"""Shared telemetry helpers: time conversion, overlap regions, and G-force physics.

G-forces are computed from real data only:
- With X/Y position channels (FastF1 ``pos_data``), lateral and longitudinal
  acceleration come from the trajectory (velocity/acceleration vectors derived
  from smoothed positions).
- Without positions, only longitudinal G can be derived (from speed over time);
  lateral G is reported as 0.0 with ``lateral_g_available=False``.

Nothing in this module fabricates data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

G_ACCELERATION = 9.81


def timedelta_to_seconds(td: Any) -> float:
    """Convert timedelta-like values (pandas/numpy/python) to float seconds."""
    if td is None:
        return 0.0
    if hasattr(td, "total_seconds"):
        return float(td.total_seconds())
    if hasattr(td, "item"):
        return float(td.item()) / 1e9
    return float(td)


def ensure_time_seconds(df: pd.DataFrame, source_col: str = "Time") -> pd.DataFrame:
    """Add a float 'TimeSeconds' column derived from a timedelta column, in place."""
    if "TimeSeconds" in df.columns or source_col not in df.columns:
        return df
    df["TimeSeconds"] = df[source_col].apply(timedelta_to_seconds)
    return df


def find_overlap_region(
    car_a_df: pd.DataFrame, car_b_df: pd.DataFrame, column: str = "DistanceOffset"
) -> tuple[float, float] | None:
    """Find the distance region where both cars have telemetry overlap."""
    a_start, a_end = car_a_df[column].min(), car_a_df[column].max()
    b_start, b_end = car_b_df[column].min(), car_b_df[column].max()

    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)

    if overlap_start >= overlap_end:
        return None
    return (overlap_start, overlap_end)


def compute_g_forces(
    df: pd.DataFrame,
    smoothing_window: int = 5,
) -> pd.DataFrame:
    """Compute lateral and longitudinal G-forces from telemetry.

    Requires a 'Speed' column (km/h) plus either:
    - 'X'/'Y' position columns (metres) and a time column -> full lateral +
      longitudinal physics, or
    - only a time column -> longitudinal only, lateral reported as unavailable.

    Positions are smoothed with a centred rolling mean before numerical
    differentiation; FastF1 position samples are low-rate (~4 Hz) and noisy.
    """
    if df.empty or "Speed" not in df.columns:
        df["lateral_g"] = 0.0
        df["longitudinal_g"] = 0.0
        df["lateral_g_available"] = False
        return df

    time_col = "TimeSeconds" if "TimeSeconds" in df.columns else "Time"
    has_time = time_col in df.columns and len(df) > 1
    has_position = {"X", "Y"}.issubset(df.columns) and has_time

    if not has_time:
        df["lateral_g"] = 0.0
        df["longitudinal_g"] = 0.0
        df["lateral_g_available"] = False
        return df

    ensure_time_seconds(df, "Time" if time_col == "Time" else "Time")
    t = df["TimeSeconds"].to_numpy(dtype=float)

    if has_position:
        window = min(smoothing_window, max(1, len(df) // 4)) if len(df) >= 4 else 1
        x = (
            df["X"].interpolate(limit_direction="both")
            .rolling(window, center=True, min_periods=1)
            .mean()
            .to_numpy(float)
        )
        y = (
            df["Y"].interpolate(limit_direction="both")
            .rolling(window, center=True, min_periods=1)
            .mean()
            .to_numpy(float)
        )

        # Hybrid method: speed magnitude comes from the (reliable) Speed
        # channel; direction comes from the position track. Double-
        # differentiating ~4 Hz positions directly amplifies quantization
        # noise into 10x velocity errors, but the heading (direction) of the
        # smoothed gradient is stable. Lateral accel = v * yaw_rate.
        speed_ms = df["Speed"].to_numpy(dtype=float) / 3.6
        heading = np.unwrap(
            np.arctan2(np.gradient(y, t), np.gradient(x, t))
        )
        heading = (
            pd.Series(heading)
            .rolling(window, center=True, min_periods=1)
            .mean()
            .to_numpy(float)
        )
        yaw_rate = np.gradient(heading, t)

        df["lateral_g"] = np.clip(np.abs(speed_ms * yaw_rate) / G_ACCELERATION, 0.0, 12.0)
        df["longitudinal_g"] = np.clip(
            np.abs(np.gradient(speed_ms, t)) / G_ACCELERATION, 0.0, 12.0
        )
        df["lateral_g_available"] = True
        return df

    speed_ms = df["Speed"].to_numpy(dtype=float) / 3.6
    accel_longitudinal = np.gradient(speed_ms, t)
    df["lateral_g"] = 0.0
    df["longitudinal_g"] = np.clip(np.abs(accel_longitudinal) / G_ACCELERATION, 0.0, 12.0)
    df["lateral_g_available"] = False
    return df


def merge_position_channels(
    df: pd.DataFrame, pos_df: pd.DataFrame, on: str = "SessionTime"
) -> pd.DataFrame:
    """Merge X/Y (and Position if present) channels from pos_data onto telemetry.

    Both frames must share a time column; a nearest-time as-of merge is used
    because car telemetry and position data sample at different rates.
    """
    if pos_df is None or pos_df.empty:
        return df

    if "X" not in pos_df.columns or "Y" not in pos_df.columns:
        return df

    left = df.copy().reset_index(drop=True)
    if on not in left.columns or on not in pos_df.columns:
        return left

    left["_t"] = left[on].apply(timedelta_to_seconds)
    channels = ["X", "Y"] + (["Position"] if "Position" in pos_df.columns else [])
    right = pos_df[[on] + channels].copy()
    right["_t"] = right[on].apply(timedelta_to_seconds)

    left = pd.merge_asof(
        left.sort_values("_t"),
        right[["_t"] + channels].sort_values("_t"),
        on="_t",
        direction="nearest",
    )
    return left.drop(columns=["_t"]).sort_values(on)
