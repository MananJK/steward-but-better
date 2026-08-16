"""Tests for the driver-agnostic detector's trigger semantics."""

import numpy as np
import pandas as pd

from driver_agnostic_detector import DriverAgnosticDetector


def _frames(seconds_apart: float, speed: float = 250.0):
    """Two cars on the same trajectory; car B passes every point on track
    `seconds_apart` after car A (i.e. B is behind by that time gap)."""
    t = np.arange(0, 10, 0.2)
    dist = np.cumsum(np.full_like(t, speed / 3.6 * 0.2))
    car_a = pd.DataFrame(
        {
            "Time": pd.to_timedelta(t, unit="s"),
            "Speed": np.full_like(t, speed),
            "Brake": np.zeros_like(t, dtype=bool),
            "DistanceOffset": dist,
            "X": dist,
            "Y": np.zeros_like(t),
        }
    )
    car_b = pd.DataFrame(
        {
            "Time": pd.to_timedelta(t + seconds_apart, unit="s"),
            "Speed": np.full_like(t, speed),
            "Brake": np.zeros_like(t, dtype=bool),
            "DistanceOffset": dist,
            "X": dist,
            "Y": np.full_like(t, 3.5),
        }
    )
    return car_a, car_b


def test_constant_large_gap_does_not_trigger():
    car_a, car_b = _frames(seconds_apart=2.0)
    detector = DriverAgnosticDetector()
    result = detector._check_proximity_trigger(car_a, car_b)
    assert result["triggered"] is False


def test_close_in_time_at_same_point_triggers():
    """Cars within 0.5s of each other at the same point on track trigger."""
    car_a, car_b = _frames(seconds_apart=0.3)
    detector = DriverAgnosticDetector()
    result = detector._check_proximity_trigger(car_a, car_b)
    assert result["triggered"] is True
    assert result["time_proximity_triggered"] is True


def test_five_seconds_apart_never_counts_as_proximity():
    """Any-two-samples matching would have fired here; matching must be
    distance-aligned (same point on track), not any-pair."""
    car_a, car_b = _frames(seconds_apart=5.0)
    detector = DriverAgnosticDetector()
    result = detector._check_proximity_trigger(car_a, car_b)
    assert result["triggered"] is False
    assert result["time_proximity_triggered"] is False


def test_lateral_g_is_zero_without_positions():
    car_a, car_b = _frames(seconds_apart=0.4)
    detector = DriverAgnosticDetector()
    out_a = detector._calculate_g_forces(car_a.copy())
    assert float(out_a["lateral_g"].max()) == 0.0  # honest zero, no fabrication
