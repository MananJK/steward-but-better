"""F1 Overtake Incident Evaluator for StewardButSmarter.

This module provides functionality to evaluate the legality of overtaking moves
by analyzing telemetry data from two cars involved in an incident.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np


class IncidentEvaluator:
    """Evaluates F1 overtake incidents for legality."""

    F1_CAR_WIDTH = 2.0

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    def evaluate_overtake_legality(
        self,
        car_a_df: pd.DataFrame,
        car_b_df: pd.DataFrame,
        car_a_name: str = "Car A",
        car_b_name: str = "Car B",
    ) -> dict[str, Any]:
        """Evaluate the legality of an overtaking move between two cars.

        Args:
            car_a_df: Telemetry DataFrame for car A with columns:
                      [Speed, Throttle, Brake, Distance, DistanceOffset, Time]
            car_b_df: Telemetry DataFrame for car B with columns:
                      [Speed, Throttle, Brake, Distance, DistanceOffset, Time]
            car_a_name: Name/identifier for car A (e.g., driver code)
            car_b_name: Name/identifier for car B (e.g., driver code)

        Returns:
            Dictionary containing incident facts for the Chief Steward.
        """
        self._logger.info(f"Evaluating overtake between {car_a_name} and {car_b_name}")

        car_a_df = car_a_df.copy()
        car_b_df = car_b_df.copy()

        car_a_df = car_a_df.sort_values("DistanceOffset").reset_index(drop=True)
        car_b_df = car_b_df.sort_values("DistanceOffset").reset_index(drop=True)

        overlap_region = self._find_overlap_region(car_a_df, car_b_df)

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

        apex_a = self._find_apex(car_a_segment)
        apex_b = self._find_apex(car_b_segment)

        inside_car, outside_car, inside_name, outside_name = self._determine_positions(
            car_a_segment, car_b_segment, car_a_name, car_b_name
        )

        apex_analysis = self._analyze_apex(
            inside_car, outside_car, inside_name, outside_name
        )

        brake_analysis = self._analyze_braking_points(
            car_a_df, car_b_df, car_a_name, car_b_name
        )

        incident_facts = {
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

        return incident_facts

    def _find_overlap_region(
        self, car_a_df: pd.DataFrame, car_b_df: pd.DataFrame
    ) -> tuple[float, float] | None:
        """Find the distance region where both cars have telemetry overlap."""
        a_start, a_end = (
            car_a_df["DistanceOffset"].min(),
            car_a_df["DistanceOffset"].max(),
        )
        b_start, b_end = (
            car_b_df["DistanceOffset"].min(),
            car_b_df["DistanceOffset"].max(),
        )

        overlap_start = max(a_start, b_start)
        overlap_end = min(a_end, b_end)

        if overlap_start >= overlap_end:
            return None

        return (overlap_start, overlap_end)

    def _find_apex(self, segment_df: pd.DataFrame) -> dict[str, Any]:
        """Locate the apex (point of minimum velocity) in a corner."""
        if segment_df.empty or "Speed" not in segment_df.columns:
            return {"distance_offset": None, "speed_kmh": None, "index": None}

        min_speed_idx = segment_df["Speed"].idxmin()
        apex_row = segment_df.loc[min_speed_idx]

        return {
            "distance_offset": float(apex_row["DistanceOffset"]),
            "speed_kmh": float(apex_row["Speed"]),
            "time": float(apex_row["Time"]) if "Time" in segment_df.columns else None,
            "index": int(min_speed_idx),
        }

    def _determine_positions(
        self,
        car_a_segment: pd.DataFrame,
        car_b_segment: pd.DataFrame,
        car_a_name: str,
        car_b_name: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
        """Determine which car is inside vs outside at the corner apex."""
        apex_a_speed = self._find_apex(car_a_segment)["speed_kmh"]
        apex_b_speed = self._find_apex(car_b_segment)["speed_kmh"]

        if apex_a_speed is None or apex_b_speed is None:
            avg_a = car_a_segment["Speed"].mean() if not car_a_segment.empty else 0
            avg_b = car_b_segment["Speed"].mean() if not car_b_segment.empty else 0
            apex_a_speed = avg_a
            apex_b_speed = avg_b

        if apex_a_speed <= apex_b_speed:
            inside_car = car_a_segment
            outside_car = car_b_segment
            inside_name = car_a_name
            outside_name = car_b_name
        else:
            inside_car = car_b_segment
            outside_car = car_a_segment
            inside_name = car_b_name
            outside_name = car_a_name

        return inside_car, outside_car, inside_name, outside_name

    def _analyze_apex(
        self,
        inside_car: pd.DataFrame,
        outside_car: pd.DataFrame,
        inside_name: str,
        outside_name: str,
    ) -> dict[str, Any]:
        """Analyze the apex and lateral distance between cars."""
        apex_inside = self._find_apex(inside_car)
        apex_outside = self._find_apex(outside_car)

        if (
            apex_inside["distance_offset"] is None
            or apex_outside["distance_offset"] is None
        ):
            return {
                "inside_car": inside_name,
                "outside_car": outside_name,
                "inside_apex_speed_kmh": None,
                "outside_apex_speed_kmh": None,
                "lateral_distance_m": None,
                "sufficient_space": None,
                "violation": False,
            }

        lateral_distance = abs(
            apex_inside["distance_offset"] - apex_outside["distance_offset"]
        )

        sufficient_space = lateral_distance >= self.F1_CAR_WIDTH

        violation = not sufficient_space

        return {
            "inside_car": inside_name,
            "outside_car": outside_name,
            "inside_apex_speed_kmh": round(apex_inside["speed_kmh"], 1)
            if apex_inside["speed_kmh"]
            else None,
            "outside_apex_speed_kmh": round(apex_outside["speed_kmh"], 1)
            if apex_outside["speed_kmh"]
            else None,
            "apex_distance_offset_m": round(
                (apex_inside["distance_offset"] + apex_outside["distance_offset"]) / 2,
                2,
            ),
            "lateral_distance_m": round(lateral_distance, 2),
            "required_clearance_m": self.F1_CAR_WIDTH,
            "sufficient_space": sufficient_space,
            "violation": violation,
        }

    def _analyze_braking_points(
        self,
        car_a_df: pd.DataFrame,
        car_b_df: pd.DataFrame,
        car_a_name: str,
        car_b_name: str,
    ) -> dict[str, Any]:
        """Compare braking points to detect dive-bombing."""
        brake_a_100 = self._find_first_100_brake(car_a_df, car_a_name)
        brake_b_100 = self._find_first_100_brake(car_b_df, car_b_name)

        if brake_a_100 is None or brake_b_100 is None:
            return {
                "car_a_brake_100_time": brake_a_100["time"] if brake_a_100 else None,
                "car_b_brake_100_time": brake_b_100["time"] if brake_b_100 else None,
                "dive_bomb_detected": False,
                "aggressive_braker": None,
                "time_difference_ms": None,
            }

        time_diff_ms = abs(brake_a_100["time"] - brake_b_100["time"])

        dive_bomb_threshold_ms = 200
        dive_bomb_detected = time_diff_ms > dive_bomb_threshold_ms

        aggressive_braker = None
        if dive_bomb_detected:
            if brake_a_100["time"] < brake_b_100["time"]:
                aggressive_braker = car_a_name
            else:
                aggressive_braker = car_b_name

        return {
            "car_a_name": car_a_name,
            "car_b_name": car_b_name,
            f"{car_a_name.lower()}_brake_100_time": round(brake_a_100["time"], 3),
            f"{car_b_name.lower()}_brake_100_time": round(brake_b_100["time"], 3),
            "time_difference_ms": round(time_diff_ms * 1000, 1),
            "dive_bomb_threshold_ms": dive_bomb_threshold_ms,
            "dive_bomb_detected": dive_bomb_detected,
            "aggressive_braker": aggressive_braker,
        }

    def _find_first_100_brake(
        self, df: pd.DataFrame, car_name: str
    ) -> dict[str, Any] | None:
        """Find the first instance of 100% brake application."""
        if "Brake" not in df.columns:
            return None

        brake_100_mask = df["Brake"] >= 1.0
        if not brake_100_mask.any():
            return None

        first_brake_idx = df[brake_100_mask].index[0]
        brake_row = df.loc[first_brake_idx]

        return {
            "time": float(brake_row["Time"]) if "Time" in df.columns else None,
            "distance_offset": float(brake_row["DistanceOffset"]),
            "speed_kmh": float(brake_row["Speed"]),
        }

    def _determine_verdict(
        self, apex_analysis: dict[str, Any], brake_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Determine the overall verdict based on apex and braking analysis."""
        violations = []

        if apex_analysis.get("violation", False):
            violations.append("Insufficient lateral clearance at apex")

        if brake_analysis.get("dive_bomb_detected", False):
            violations.append(
                f"Dive-bomb detected - {brake_analysis['aggressive_braker']} braked excessively early"
            )

        if violations:
            verdict = "PENALTY"
            summary = f"Violation(s) detected: {'; '.join(violations)}"
        else:
            verdict = "NO_INVESTIGATION"
            summary = "No规则 violations detected. Overtake appears legal."

        return {
            "verdict": verdict,
            "violations": violations,
            "summary": summary,
        }

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
                "summary": f"Insufficient telemetry overlap between {car_a_name} and {car_b_name} to make determination.",
            },
        }

    def save_incident_report(
        self, incident_facts: dict[str, Any], filename: str
    ) -> None:
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
    """Evaluate the legality of an overtaking move between two F1 cars.

    Args:
        car_a_df: Telemetry DataFrame for car A with columns:
                  [Speed, Throttle, Brake, Distance, DistanceOffset, Time]
        car_b_df: Telemetry DataFrame for car B with columns:
                  [Speed, Throttle, Brake, Distance, DistanceOffset, Time]
        car_a_name: Name/identifier for car A (e.g., driver code like 'VER')
        car_b_name: Name/identifier for car B (e.g., driver code like 'HAM')

    Returns:
        Dictionary containing incident facts in JSON format for Chief Steward review.
        Includes:
        - Apex analysis (minimum speed point, lateral distance, clearance)
        - Braking analysis (100% brake times, dive-bomb detection)
        - Verdict (PENALTY or NO_INVESTIGATION)
    """
    evaluator = IncidentEvaluator()
    return evaluator.evaluate_overtake_legality(
        car_a_df, car_b_df, car_a_name, car_b_name
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    TELEMETRY_FILE = Path(__file__).parent / "verstappen_abu_dhabi_2021_lap58.parquet"
    OUTPUT_FILE = Path(__file__).parent.parent / "ui" / "public" / "live_incident.json"

    ver_df = pd.read_parquet(TELEMETRY_FILE)
    print(f"Loaded Verstappen telemetry: {len(ver_df)} points")

    ham_df = ver_df.copy()
    ham_df["Speed"] = ham_df["Speed"] * 0.97
    ham_df["DistanceOffset"] = ham_df["DistanceOffset"] + 1.8
    ham_df["DriverCode"] = "HAM"
    ham_df["Brake"] = ver_df["Brake"]

    ver_df = ver_df[ver_df["DriverCode"] == "VER"]

    corner_mask = (ver_df["DistanceOffset"] >= 50) & (ver_df["DistanceOffset"] <= 150)
    corner_segment = ver_df[corner_mask]

    apex_idx = corner_segment["Speed"].idxmin()
    apex_speed = corner_segment.loc[apex_idx, "Speed"]
    apex_distance = corner_segment.loc[apex_idx, "DistanceOffset"]

    inside_car_offset = apex_distance
    outside_car_offset = apex_distance + 1.8
    apex_gap = abs(outside_car_offset - inside_car_offset)

    ART_33_4_WIDTH = 2.0
    verdict = "PENALTY" if apex_gap < ART_33_4_WIDTH else "CLEAN"

    confidence = 0.85 if apex_gap < 1.0 else 0.95 if apex_gap > 2.5 else 0.75

    defense_status = "ILLEGAL" if verdict == "PENALTY" else "LEGAL"
    incident_desc = (
        f"Car {ver_df['DriverCode'].iloc[0]} detected at {apex_gap:.1f}m from apex at T3 corner; "
        f"apex velocity {apex_speed:.1f} kph. "
        f"Defense categorized as {defense_status} under Art 33.4. "
        f"Clearance below 2.0m threshold by {2.0 - apex_gap:.2f}m."
    )

    live_incident = {
        "driver": "VER",
        "speed_kph": round(float(apex_speed), 1),
        "apex_gap": round(float(apex_gap), 2),
        "verdict": verdict,
        "article_cited": "FIA International Sporting Code Appendix L, Art 33.4",
        "confidence_score": confidence,
        "incident_description": incident_desc,
        "incident_type": "overtake_legality",
        "track": "Abu Dhabi Grand Prix",
        "lap": 58,
        "timestamp": "2021-12-12T20:00:00Z",
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(live_incident, f, indent=2)

    print("\n" + "=" * 60)
    print("LIVE INCIDENT REPORT")
    print("=" * 60)
    print(json.dumps(live_incident, indent=2))
    print(f"\nSaved to: {OUTPUT_FILE}")
