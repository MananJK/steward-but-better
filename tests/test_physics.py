"""Tests for real G-force physics in telemetry_utils."""

import numpy as np
import pandas as pd

from telemetry_utils import (
    compute_g_forces,
    ensure_time_seconds,
    find_overlap_region,
    timedelta_to_seconds,
)

G = 9.81


def _circle_frame(radius: float, speed_kph: float, seconds: float = 12.0, dt: float = 0.05):
    t = np.arange(0, seconds, dt)
    v = speed_kph / 3.6
    omega = v / radius
    return pd.DataFrame(
        {
            "Time": pd.to_timedelta(t, unit="s"),
            "TimeSeconds": t,
            "Speed": np.full_like(t, speed_kph),
            "X": radius * np.cos(omega * t),
            "Y": radius * np.sin(omega * t),
        }
    )


def test_circle_yields_correct_lateral_g():
    radius, speed_kph = 60.0, 200.0
    expected = (speed_kph / 3.6) ** 2 / (radius * G)

    out = compute_g_forces(_circle_frame(radius, speed_kph), smoothing_window=9)
    interior = out["lateral_g"].iloc[20:-20]

    assert bool(out["lateral_g_available"].iloc[0]) is True
    assert abs(interior.median() - expected) < 0.15


def test_straight_line_has_no_lateral_g():
    t = np.arange(0, 10, 0.05)
    v = 200 / 3.6
    df = pd.DataFrame(
        {
            "Time": pd.to_timedelta(t, unit="s"),
            "TimeSeconds": t,
            "Speed": np.full_like(t, 200.0),
            "X": v * t,
            "Y": np.zeros_like(t),
        }
    )
    out = compute_g_forces(df)
    assert out["lateral_g"].iloc[10:-10].median() < 0.05


def test_missing_positions_report_lateral_unavailable():
    t = np.arange(0, 10, 0.05)
    df = pd.DataFrame(
        {"Time": pd.to_timedelta(t, unit="s"), "TimeSeconds": t, "Speed": np.linspace(50, 300, len(t))}
    )
    out = compute_g_forces(df)
    assert bool(out["lateral_g_available"].iloc[0]) is False
    assert (out["lateral_g"] == 0.0).all()
    # Longitudinal is still derivable: 250 km/h over 10s ~ 0.7G average.
    assert out["longitudinal_g"].max() > 0.5


def test_lateral_g_never_fabricated_without_data():
    """The sine-wave fabrication this module replaced must never come back."""
    t = np.arange(0, 20, 0.05)
    df = pd.DataFrame(
        {
            "Time": pd.to_timedelta(t, unit="s"),
            "TimeSeconds": t,
            "Speed": np.full_like(t, 250.0),
        }
    )
    out = compute_g_forces(df)
    assert float(out["lateral_g"].max()) == 0.0


def test_find_overlap_region():
    def frame(start, end):
        return pd.DataFrame({"DistanceOffset": np.linspace(start, end, 10)})

    assert find_overlap_region(frame(0, 100), frame(50, 150)) == (50.0, 100.0)
    assert find_overlap_region(frame(0, 100), frame(200, 300)) is None


def test_timedelta_conversion():
    assert timedelta_to_seconds(pd.Timedelta(seconds=1.5)) == 1.5
    assert timedelta_to_seconds(None) == 0.0


def test_ensure_time_seconds_idempotent():
    t = np.arange(0, 5, 0.5)
    df = pd.DataFrame({"Time": pd.to_timedelta(t, unit="s")})
    ensure_time_seconds(df)
    first = df["TimeSeconds"].tolist()
    ensure_time_seconds(df)
    assert df["TimeSeconds"].tolist() == first
