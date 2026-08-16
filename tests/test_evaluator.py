"""Tests for the overtake evaluator: geometry, dive-bomb direction, honesty."""

import numpy as np
import pandas as pd

from incident_evaluator import evaluate_overtake_legality


def _car_frame(t, speed, lateral_offset, brake_start):
    dist = np.cumsum(speed / 3.6 * np.diff(t, prepend=t[0]))
    return pd.DataFrame(
        {
            "Time": pd.to_timedelta(t, unit="s"),
            "Speed": speed,
            "Brake": (t >= brake_start) & (t < brake_start + 1.0),
            "DistanceOffset": dist - dist[0],
            "X": dist,
            "Y": np.full_like(t, lateral_offset, dtype=float),
        }
    )


def _corner_speed(t):
    return np.where(t < 2.0, 300 - 40 * (t**2), np.maximum(120, 120 + 30 * (t - 2)))


def test_lateral_gap_measured_from_positions():
    t = np.arange(0, 6, 0.05)
    car_a = _car_frame(t, _corner_speed(t), lateral_offset=0.0, brake_start=1.6)
    car_b = _car_frame(t, _corner_speed(t), lateral_offset=3.5, brake_start=1.6)

    result = evaluate_overtake_legality(car_a, car_b, "A", "B")
    apex = result["apex_analysis"]
    assert apex["lateral_distance_m"] is not None
    assert apex["lateral_distance_m"] >= 3.0  # ~3.5m apart laterally
    assert apex["violation"] is False


def test_tight_gap_flags_violation():
    t = np.arange(0, 6, 0.05)
    car_a = _car_frame(t, _corner_speed(t), lateral_offset=0.0, brake_start=1.6)
    car_b = _car_frame(t, _corner_speed(t), lateral_offset=1.0, brake_start=1.6)

    result = evaluate_overtake_legality(car_a, car_b, "A", "B")
    assert result["apex_analysis"]["violation"] is True
    assert result["verdict"]["verdict"] == "PENALTY"


def test_dive_bomb_labels_the_late_braker():
    """The car braking LATER is the dive-bomber, not the earlier one."""
    t = np.arange(0, 6, 0.05)
    car_a = _car_frame(t, _corner_speed(t), lateral_offset=0.0, brake_start=1.6)
    car_b = _car_frame(t, _corner_speed(t), lateral_offset=3.5, brake_start=2.1)

    result = evaluate_overtake_legality(car_a, car_b, "A", "B")
    braking = result["braking_analysis"]
    assert braking["dive_bomb_detected"] is True
    assert braking["late_braker"] == "B"  # B braked 0.5s later


def test_simultaneous_braking_is_not_a_dive_bomb():
    t = np.arange(0, 6, 0.05)
    car_a = _car_frame(t, _corner_speed(t), lateral_offset=0.0, brake_start=1.8)
    car_b = _car_frame(t, _corner_speed(t), lateral_offset=3.5, brake_start=1.8)

    result = evaluate_overtake_legality(car_a, car_b, "A", "B")
    assert result["braking_analysis"]["dive_bomb_detected"] is False


def test_missing_positions_is_inconclusive_not_violation():
    """Along-track offsets must never be misread as lateral distance."""
    t = np.arange(0, 6, 0.05)
    car_a = _car_frame(t, _corner_speed(t), lateral_offset=0.0, brake_start=1.6).drop(
        columns=["X", "Y"]
    )
    car_b = _car_frame(t, _corner_speed(t), lateral_offset=0.0, brake_start=1.6).drop(
        columns=["X", "Y"]
    )

    result = evaluate_overtake_legality(car_a, car_b, "A", "B")
    apex = result["apex_analysis"]
    assert apex["lateral_distance_m"] is None
    assert apex["violation"] is None
    assert result["verdict"]["verdict"] == "INCONCLUSIVE"


def test_no_overlap_returns_no_data():
    t = np.arange(0, 6, 0.05)
    car_a = _car_frame(t, _corner_speed(t), 0.0, 1.6)
    car_b = _car_frame(t, _corner_speed(t), 0.0, 1.6)
    car_b["DistanceOffset"] = car_b["DistanceOffset"] + 5000  # far away

    result = evaluate_overtake_legality(car_a, car_b, "A", "B")
    assert result["verdict"]["verdict"] == "NO_DATA"
