"""F1 Overtake Incident Evaluator for StewardButBetter.

Evaluates the legality of overtaking moves from two-car telemetry.

Geometry notes:
- Lateral separation requires X/Y position channels (FastF1 pos_data). It is
  measured as the Euclidean car-to-car distance at the apex instant (the two
  cars are nearly abreast there, so this approximates the lateral gap).
  Without X/Y the apex analysis reports `unknown` instead of mislabelling
  along-track offsets as lateral distance.
- Dive-bombing: the car that reaches full braking LATER (and still makes the
  corner alongside) braked later than its rival — that is the aggressive
  move. The earlier braker is the one reacting defensively.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from telemetry_utils import find_overlap_region, timedelta_to_seconds


class IncidentEvaluator:
    """Evaluates F1 overtake incidents for legality."""

    F1_CAR_WIDTH = 2.0
    DIVE_BOMB_THRESHOLD_MS = 200.0

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    def evaluate_overtake_legality(
        self,
        car_a_df: pd.DataFrame,
        car_b_df: pd.DataFrame,
        car_a_name: str = "Car A",
        car_b_name: str = "Car B",
    ) -> dict[str, Any]:
        """Evaluate the legality of an overtaking move between two cars."""
        self._logger.info(f"Evaluating overtake between {car_a_name} and {car_b_name}")

        car_a_df = car_a_df.copy().sort_values("DistanceOffset").reset_index(drop=True)
        car_b_df = car_b_df.copy().sort_values("DistanceOffset").reset_index(drop=True)

        overlap_region = find_overlap_region(car_a_df, car_b_df)

        if overlap_region is None:
            return self._create_no_overlap_result(car_a_name, car_b_name)

        car_a_segment = car_a_df[
            (car_a_df["DistanceOffset"] >= overlap_region[0])
            & (car_a_df["DistanceOffset"] <= overlap_region[1])
        ]
        car_b_segment = car_b_df[
            (car_b_df["DistanceOffset"] >= overlap_region[0])
            & (car_b_df["DistanceOffset"] <= overlap_region[1])
        ]

        # Heuristic: the slower car at the apex tends to be the one on the
        # inside line. Real inside/outside requires position data.
        inside_car, outside_car, inside_name, outside_name = self._determine_positions(
            car_a_segment, car_b_segment, car_a_name, car_b_name
        )

        apex_analysis = self._analyze_apex(
            inside_car, outside_car, inside_name, outside_name
        )
        brake_analysis = self._analyze_braking_points(
            car_a_segment, car_b_segment, car_a_name, car_b_name
        )

        return {
            "incident_summary": {
                "overtaking_car": outside_name,
                "defending_car": inside_name,
                "overlap_detected": True,
                "corner_region_m": {
                    "start": round(overlap_region[0], 2),
                    "end": round(overlap_region[1], 2),
                },
            },
            "apex_analysis": apex_analysis,
            "braking_analysis": brake_analysis,
            "verdict": self._determine_verdict(apex_analysis, brake_analysis),
        }

    def _find_apex(self, segment_df: pd.DataFrame) -> dict[str, Any]:
        """Locate the apex (point of minimum velocity) in a corner."""
        if segment_df.empty or "Speed" not in segment_df.columns:
            return {"distance_offset": None, "speed_kmh": None, "time": None, "index": None}

        min_speed_idx = segment_df["Speed"].idxmin()
        apex_row = segment_df.loc[min_speed_idx]

        time_val = None
        if "Time" in segment_df.columns and pd.notna(apex_row.get("Time")):
            time_val = timedelta_to_seconds(apex_row["Time"])
        elif "TimeSeconds" in segment_df.columns:
            time_val = float(apex_row["TimeSeconds"])

        return {
            "distance_offset": float(apex_row["DistanceOffset"]),
            "speed_kmh": float(apex_row["Speed"]),
            "time": time_val,
            "index": int(min_speed_idx),
        }

    def _determine_positions(
        self,
        car_a_segment: pd.DataFrame,
        car_b_segment: pd.DataFrame,
        car_a_name: str,
        car_b_name: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
        """Determine which car is likely inside vs outside at the corner apex."""
        apex_a_speed = self._find_apex(car_a_segment)["speed_kmh"]
        apex_b_speed = self._find_apex(car_b_segment)["speed_kmh"]

        if apex_a_speed is None or apex_b_speed is None:
            apex_a_speed = car_a_segment["Speed"].mean() if not car_a_segment.empty else 0
            apex_b_speed = car_b_segment["Speed"].mean() if not car_b_segment.empty else 0

        if apex_a_speed <= apex_b_speed:
            return car_a_segment, car_b_segment, car_a_name, car_b_name
        return car_b_segment, car_a_segment, car_b_name, car_a_name

    def _position_at_time(self, df: pd.DataFrame, time_val: float) -> tuple[float, float] | None:
        """(X, Y) of a car at a given time, or None when positions are missing."""
        if not {"X", "Y", "Time"}.issubset(df.columns):
            return None
        times = df["Time"].apply(timedelta_to_seconds)
        row = df.loc[(times - time_val).abs().idxmin()]
        x, y = row.get("X"), row.get("Y")
        if pd.isna(x) or pd.isna(y):
            return None
        return float(x), float(y)

    def _analyze_apex(
        self,
        inside_car: pd.DataFrame,
        outside_car: pd.DataFrame,
        inside_name: str,
        outside_name: str,
    ) -> dict[str, Any]:
        """Analyze the car-to-car spatial gap at the apex from X/Y positions."""
        apex_inside = self._find_apex(inside_car)
        apex_outside = self._find_apex(outside_car)

        if (
            apex_inside["distance_offset"] is None
            or apex_outside["distance_offset"] is None
        ):
            return {
                "inside_car": inside_name,
                "outside_car": outside_name,
                "lateral_distance_m": None,
                "sufficient_space": None,
                "violation": None,
                "note": "insufficient telemetry",
            }

        inside_pos = (
            self._position_at_time(inside_car, apex_inside["time"])
            if apex_inside["time"] is not None
            else None
        )
        outside_pos = (
            self._position_at_time(outside_car, apex_inside["time"])
            if apex_inside["time"] is not None
            else None
        )

        if inside_pos is None or outside_pos is None:
            return {
                "inside_car": inside_name,
                "outside_car": outside_name,
                "inside_apex_speed_kmh": round(apex_inside["speed_kmh"], 1),
                "outside_apex_speed_kmh": round(apex_outside["speed_kmh"], 1),
                "lateral_distance_m": None,
                "required_clearance_m": self.F1_CAR_WIDTH,
                "sufficient_space": None,
                "violation": None,
                "note": "no X/Y position channels; lateral gap unknown",
            }

        gap_m = float(np.hypot(inside_pos[0] - outside_pos[0], inside_pos[1] - outside_pos[1]))
        sufficient_space = gap_m >= self.F1_CAR_WIDTH

        return {
            "inside_car": inside_name,
            "outside_car": outside_name,
            "inside_apex_speed_kmh": round(apex_inside["speed_kmh"], 1),
            "outside_apex_speed_kmh": round(apex_outside["speed_kmh"], 1),
            "lateral_distance_m": round(gap_m, 2),
            "required_clearance_m": self.F1_CAR_WIDTH,
            "sufficient_space": sufficient_space,
            "violation": not sufficient_space,
        }

    def _analyze_braking_points(
        self,
        car_a_segment: pd.DataFrame,
        car_b_segment: pd.DataFrame,
        car_a_name: str,
        car_b_name: str,
    ) -> dict[str, Any]:
        """Compare full-braking onset to detect dive-bombing (later braking)."""
        brake_a = self._find_first_100_brake(car_a_segment)
        brake_b = self._find_first_100_brake(car_b_segment)

        if brake_a is None or brake_b is None:
            return {
                "car_a_brake_time": brake_a["time"] if brake_a else None,
                "car_b_brake_time": brake_b["time"] if brake_b else None,
                "dive_bomb_detected": False,
                "late_braker": None,
                "time_difference_ms": None,
            }

        time_diff_ms = abs(brake_a["time"] - brake_b["time"]) * 1000.0
        dive_bomb_detected = time_diff_ms > self.DIVE_BOMB_THRESHOLD_MS

        # The dive-bomber brakes LATER than the rival, not earlier.
        late_braker = None
        if dive_bomb_detected:
            late_braker = car_a_name if brake_a["time"] > brake_b["time"] else car_b_name

        return {
            "car_a_name": car_a_name,
            "car_b_name": car_b_name,
            "car_a_brake_time": round(brake_a["time"], 3),
            "car_b_brake_time": round(brake_b["time"], 3),
            "time_difference_ms": round(time_diff_ms, 1),
            "dive_bomb_threshold_ms": self.DIVE_BOMB_THRESHOLD_MS,
            "dive_bomb_detected": dive_bomb_detected,
            "late_braker": late_braker,
        }

    def _find_first_100_brake(self, df: pd.DataFrame) -> dict[str, Any] | None:
        """Find the first instance of 100% brake application."""
        if df is None or df.empty or "Brake" not in df.columns:
            return None

        brake_100_mask = df["Brake"] >= 1.0
        if not brake_100_mask.any():
            return None

        brake_row = df[brake_100_mask].iloc[0]

        time_val = None
        if "Time" in df.columns and pd.notna(brake_row.get("Time")):
            time_val = timedelta_to_seconds(brake_row["Time"])
        elif "TimeSeconds" in df.columns:
            time_val = float(brake_row["TimeSeconds"])

        if time_val is None:
            return None

        return {
            "time": time_val,
            "distance_offset": float(brake_row["DistanceOffset"]),
            "speed_kmh": float(brake_row["Speed"]),
        }

    def _determine_verdict(
        self, apex_analysis: dict[str, Any], brake_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Determine the overall verdict based on apex and braking analysis."""
        violations = []
        unknowns = []

        if apex_analysis.get("violation") is None:
            unknowns.append("Lateral gap at apex could not be measured (no position data)")
        elif apex_analysis.get("violation", False):
            violations.append("Insufficient lateral clearance at apex")

        if brake_analysis.get("dive_bomb_detected", False):
            violations.append(
                f"Dive-bomb detected - {brake_analysis['late_braker']} braked late into the corner"
            )

        if violations:
            verdict = "PENALTY"
            summary = f"Violation(s) detected: {'; '.join(violations)}"
        elif unknowns:
            verdict = "INCONCLUSIVE"
            summary = "; ".join(unknowns)
        else:
            verdict = "NO_INVESTIGATION"
            summary = "No rule violations detected. Overtake appears legal."

        return {"verdict": verdict, "violations": violations, "summary": summary}

    def _create_no_overlap_result(
        self, car_a_name: str, car_b_name: str
    ) -> dict[str, Any]:
        """Create result when no telemetry overlap is found."""
        return {
            "incident_summary": {
                "car_a": car_a_name,
                "car_b": car_b_name,
                "overlap_detected": False,
            },
            "apex_analysis": {"error": "No telemetry overlap between cars"},
            "braking_analysis": {"error": "No telemetry overlap between cars"},
            "verdict": {
                "verdict": "NO_DATA",
                "violations": [],
                "summary": (
                    f"Insufficient telemetry overlap between {car_a_name} and "
                    f"{car_b_name} to make determination."
                ),
            },
        }

    def save_incident_report(self, incident_facts: dict[str, Any], filename: str) -> None:
        """Save incident facts to JSON file."""
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(incident_facts, f, indent=2)
        self._logger.info(f"Saved incident report to {output_path}")


def evaluate_overtake_legality(
    car_a_df: pd.DataFrame,
    car_b_df: pd.DataFrame,
    car_a_name: str = "Car A",
    car_b_name: str = "Car B",
) -> dict[str, Any]:
    """Evaluate the legality of an overtaking move between two F1 cars."""
    evaluator = IncidentEvaluator()
    return evaluator.evaluate_overtake_legality(car_a_df, car_b_df, car_a_name, car_b_name)


if __name__ == "__main__":
    # Synthetic smoke test: two fabricated cars through a corner. Car B brakes
    # 0.4s later than Car A (dive-bomb shape) and runs 1.2m from Car A at the
    # apex (below the 2.0m car-width threshold).
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    t = np.arange(0, 6, 0.05)

    def car_frame(speed_profile, lateral_offset, brake_start):
        speed = speed_profile(t)
        dist = np.cumsum(speed / 3.6 * 0.05)
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

    corner = lambda tt: np.where(tt < 2.0, 300 - 40 * (tt ** 2), np.maximum(120, 120 + 30 * (tt - 2)))
    car_a = car_frame(corner, lateral_offset=0.0, brake_start=1.6)
    car_b = car_frame(corner, lateral_offset=1.2, brake_start=2.0)

    result = evaluate_overtake_legality(car_a, car_b, "Car A", "Car B")
    print(json.dumps(result, indent=2))
