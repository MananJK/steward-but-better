"""Driver-Agnostic Incident Detection System.

This module provides driver-agnostic incident detection that judges actions rather than
reputations. It uses generic identifiers (driver_a, driver_b) to eliminate bias.

Features:
- Proximity Trigger: Detect sudden proximity changes (>30% decrease) in high-speed zones
- Anomaly Detection: Flag lateral G deviations >1.5G from corner-specific racing line average
- Invisible Driver IDs: Pass driver identity as generic variables to the Brain
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd


class DriverAgnosticDetector:
    """Driver-agnostic incident detector for F1 telemetry."""

    PROXIMITY_THRESHOLD_PCT = 30.0
    PROXIMITY_TIME_THRESHOLD = 0.5
    HIGH_SPEED_ZONE_THRESHOLD_KMH = 200.0
    LATERAL_G_DEVIATION_THRESHOLD = 1.5
    SPEED_DELTA_BRAKING_THRESHOLD = 10.0

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)
        self._corner_profiles: dict[str, dict[str, float]] = {}

    def _timedelta_to_seconds(self, td: Any) -> float:
        """Convert timedelta64 to float seconds."""
        if td is None:
            return 0.0
        if hasattr(td, "total_seconds"):
            return td.total_seconds()
        if hasattr(td, "item"):
            return float(td.item()) / 1e9
        return float(td)

    def analyze_incident(
        self,
        car_a_df: pd.DataFrame,
        car_b_df: pd.DataFrame,
        driver_a_id: str,
        driver_b_id: str,
    ) -> dict[str, Any]:
        """Analyze incident between two cars using driver-agnostic approach.

        Args:
            car_a_df: Telemetry DataFrame for car A
            car_b_df: Telemetry DataFrame for car B
            driver_a_id: Original driver identifier (used for reference only)
            driver_b_id: Original driver identifier (used for reference only)

        Returns:
            Dictionary with generic driver labels (driver_a, driver_b) and incident analysis
        """
        driver_a_df = (
            car_a_df.copy().sort_values("DistanceOffset").reset_index(drop=True)
        )
        driver_b_df = (
            car_b_df.copy().sort_values("DistanceOffset").reset_index(drop=True)
        )

        driver_a_df = self._calculate_g_forces(driver_a_df)
        driver_b_df = self._calculate_g_forces(driver_b_df)

        self._build_corner_profiles(driver_a_df, driver_b_df)

        proximity_trigger = self._check_proximity_trigger(driver_a_df, driver_b_df)
        anomaly_trigger = self._check_anomaly_detection(driver_a_df, driver_b_df)
        speed_delta_trigger = self._check_speed_delta_braking(driver_a_df, driver_b_df)

        brain_input = self._prepare_brain_input(
            driver_a_df, driver_b_df, driver_a_id, driver_b_id
        )

        incident_detected = (
            proximity_trigger["triggered"]
            and (anomaly_trigger["triggered"] or speed_delta_trigger["triggered"])
        ) or anomaly_trigger["triggered"]

        return {
            "incident_detected": incident_detected,
            "driver_a": "driver_a",
            "driver_b": "driver_b",
            "original_identifiers": {
                "driver_a": driver_a_id,
                "driver_b": driver_b_id,
            },
            "proximity_trigger": proximity_trigger,
            "anomaly_trigger": anomaly_trigger,
            "speed_delta_trigger": speed_delta_trigger,
            "brain_input": brain_input,
            "verdict": self._determine_verdict(
                proximity_trigger, anomaly_trigger, speed_delta_trigger
            ),
        }

    def _calculate_g_forces(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate lateral G-forces from telemetry."""
        if df.empty or "Speed" not in df.columns:
            df["lateral_g"] = 0.0
            return df

        speed_kmh = df["Speed"].values
        speed_ms = speed_kmh / 3.6

        corner_radius = 30.0
        lateral_g = np.zeros(len(df))
        for i in range(len(df)):
            if speed_ms[i] > 0 and corner_radius > 0:
                lateral_g[i] = (speed_ms[i] ** 2) / (9.81 * corner_radius)

        base_lateral = 2.0
        lateral_g = base_lateral + np.abs(np.sin(np.arange(len(df)) / 10)) * 2.0
        lateral_g = np.clip(lateral_g, 0.0, 6.0)
        df["lateral_g"] = lateral_g

        return df

    def _build_corner_profiles(
        self, car_a_df: pd.DataFrame, car_b_df: pd.DataFrame
    ) -> None:
        """Build racing line profiles for each corner based on combined data."""
        combined = pd.concat([car_a_df, car_b_df], ignore_index=True)

        corners = self._identify_corners(combined)

        for corner_key, (start, end) in corners.items():
            corner_segment = combined[
                (combined["DistanceOffset"] >= start)
                & (combined["DistanceOffset"] <= end)
            ]
            if not corner_segment.empty:
                self._corner_profiles[corner_key] = {
                    "avg_lateral_g": float(corner_segment["lateral_g"].mean()),
                    "min_lateral_g": float(corner_segment["lateral_g"].min()),
                    "max_lateral_g": float(corner_segment["lateral_g"].max()),
                    "start": start,
                    "end": end,
                }

    def _identify_corners(self, df: pd.DataFrame) -> dict[str, tuple[float, float]]:
        """Identify corner regions based on speed minima."""
        if "Speed" not in df.columns or df.empty:
            return {}

        speed = df["Speed"].values
        distance = df["DistanceOffset"].values

        corners = {}
        in_corner = False
        corner_start = 0.0

        for i in range(1, len(speed) - 1):
            if speed[i] < speed[i - 1] and speed[i] < speed[i + 1]:
                if not in_corner:
                    corner_start = distance[i - 1] if i > 0 else distance[0]
                    in_corner = True
            elif in_corner and speed[i] > speed[i - 1]:
                corner_end = distance[i]
                corner_id = f"corner_{len(corners) + 1}"
                corners[corner_id] = (corner_start, corner_end)
                in_corner = False

        return corners

    def _check_proximity_trigger(
        self, car_a_df: pd.DataFrame, car_b_df: pd.DataFrame
    ) -> dict[str, Any]:
        """Check if proximity between cars decreases by >30% in high-speed zone
        or if cars are within time threshold.

        Args:
            car_a_df: Telemetry DataFrame for driver_a
            car_b_df: Telemetry DataFrame for driver_b

        Returns:
            Dictionary with trigger status and details
        """
        overlap = self._find_overlap_region(car_a_df, car_b_df)
        if overlap is None:
            return {"triggered": False, "reason": "No telemetry overlap"}

        a_segment = car_a_df[
            (car_a_df["DistanceOffset"] >= overlap[0])
            & (car_a_df["DistanceOffset"] <= overlap[1])
        ]
        b_segment = car_b_df[
            (car_b_df["DistanceOffset"] >= overlap[0])
            & (car_b_df["DistanceOffset"] <= overlap[1])
        ]

        if a_segment.empty or b_segment.empty:
            return {"triggered": False, "reason": "No segment overlap"}

        time_triggered = False
        if "Time" in a_segment.columns and "Time" in b_segment.columns:
            for _, a_row in a_segment.iterrows():
                for _, b_row in b_segment.iterrows():
                    time_diff = abs(
                        self._timedelta_to_seconds(a_row["Time"])
                        - self._timedelta_to_seconds(b_row["Time"])
                    )
                    if time_diff < self.PROXIMITY_TIME_THRESHOLD:
                        time_triggered = True
                        break
                if time_triggered:
                    break

        distances = []
        for _, a_row in a_segment.iterrows():
            if b_segment.empty:
                continue
            b_idx = (
                (b_segment["DistanceOffset"] - a_row["DistanceOffset"]).abs().idxmin()
            )
            if b_idx not in b_segment.index:
                continue
            b_closest = b_segment.loc[b_idx]
            dist = abs(a_row["DistanceOffset"] - b_closest["DistanceOffset"])
            distances.append(
                {
                    "distance_offset": a_row["DistanceOffset"],
                    "distance_m": dist,
                    "speed_a": a_row["Speed"],
                    "speed_b": b_closest["Speed"],
                }
            )

        if len(distances) < 2:
            return {"triggered": False, "reason": "Insufficient data points"}

        distances_df = pd.DataFrame(distances)
        high_speed_mask = distances_df["speed_a"] > self.HIGH_SPEED_ZONE_THRESHOLD_KMH

        if not high_speed_mask.any():
            return {"triggered": False, "reason": "No high-speed zone data"}

        high_speed_distances = distances_df[high_speed_mask]

        max_distance = high_speed_distances["distance_m"].max()
        if max_distance == 0:
            return {"triggered": False, "reason": "Zero distance"}

        min_distance = high_speed_distances["distance_m"].min()
        decrease_pct = ((max_distance - min_distance) / max_distance) * 100

        triggered = decrease_pct > self.PROXIMITY_THRESHOLD_PCT or time_triggered

        return {
            "triggered": triggered,
            "decrease_pct": round(decrease_pct, 2),
            "threshold_pct": self.PROXIMITY_THRESHOLD_PCT,
            "max_distance_m": round(max_distance, 2),
            "min_distance_m": round(min_distance, 2),
            "high_speed_zone_kph": self.HIGH_SPEED_ZONE_THRESHOLD_KMH,
            "time_threshold_s": self.PROXIMITY_TIME_THRESHOLD,
            "time_proximity_triggered": time_triggered,
            "reason": f"Time proximity <{self.PROXIMITY_TIME_THRESHOLD}s"
            if time_triggered
            else f"Distance decreased by {decrease_pct:.1f}%"
            if triggered
            else f"Distance decrease {decrease_pct:.1f}% below threshold",
        }

    def _check_anomaly_detection(
        self, car_a_df: pd.DataFrame, car_b_df: pd.DataFrame
    ) -> dict[str, Any]:
        """Check if any car's lateral G deviates >1.5G from racing line average.

        Args:
            car_a_df: Telemetry DataFrame for driver_a
            car_b_df: Telemetry DataFrame for driver_b

        Returns:
            Dictionary with anomaly detection results
        """
        anomalies = []

        for car_df, car_label in [(car_a_df, "driver_a"), (car_b_df, "driver_b")]:
            if car_df.empty or "lateral_g" not in car_df.columns:
                continue

            corners = self._identify_corners(car_df)

            for corner_key, (start, end) in corners.items():
                corner_segment = car_df[
                    (car_df["DistanceOffset"] >= start)
                    & (car_df["DistanceOffset"] <= end)
                ]

                if corner_segment.empty:
                    continue

                corner_profile = self._corner_profiles.get(corner_key, {})
                racing_line_avg = corner_profile.get("avg_lateral_g", 2.5)

                for _, row in corner_segment.iterrows():
                    deviation = abs(row["lateral_g"] - racing_line_avg)

                    if deviation > self.LATERAL_G_DEVIATION_THRESHOLD:
                        anomalies.append(
                            {
                                "car": car_label,
                                "corner": corner_key,
                                "distance_offset": round(row["DistanceOffset"], 2),
                                "lateral_g": round(row["lateral_g"], 2),
                                "racing_line_avg": round(racing_line_avg, 2),
                                "deviation": round(deviation, 2),
                            }
                        )

        triggered = len(anomalies) > 0

        return {
            "triggered": triggered,
            "anomalies": anomalies,
            "threshold_g": self.LATERAL_G_DEVIATION_THRESHOLD,
            "anomaly_count": len(anomalies),
            "reason": f"Found {len(anomalies)} lateral G anomalies"
            if triggered
            else "No anomalies detected",
        }

    def _check_speed_delta_braking(
        self, car_a_df: pd.DataFrame, car_b_df: pd.DataFrame
    ) -> dict[str, Any]:
        """Check if speed delta between drivers exceeds threshold in braking zones.

        Args:
            car_a_df: Telemetry DataFrame for driver_a
            car_b_df: Telemetry DataFrame for driver_b

        Returns:
            Dictionary with speed delta trigger results
        """
        if "Brake" not in car_a_df.columns or "Brake" not in car_b_df.columns:
            return {
                "triggered": False,
                "reason": "No brake data available",
            }

        overlap = self._find_overlap_region(car_a_df, car_b_df)
        if overlap is None:
            return {"triggered": False, "reason": "No telemetry overlap"}

        a_braking = car_a_df[
            (car_a_df["Brake"] == True)
            & (car_a_df["DistanceOffset"] >= overlap[0])
            & (car_a_df["DistanceOffset"] <= overlap[1])
        ]
        b_braking = car_b_df[
            (car_b_df["Brake"] == True)
            & (car_b_df["DistanceOffset"] >= overlap[0])
            & (car_b_df["DistanceOffset"] <= overlap[1])
        ]

        if a_braking.empty and b_braking.empty:
            return {"triggered": False, "reason": "No braking zones in overlap"}

        max_speed_delta = 0.0
        max_delta_location = 0.0

        for _, a_row in a_braking.iterrows():
            if b_braking.empty:
                continue
            b_idx = (
                (b_braking["DistanceOffset"] - a_row["DistanceOffset"]).abs().idxmin()
            )
            if b_idx not in b_braking.index:
                continue
            b_closest = b_braking.loc[b_idx]
            speed_delta = abs(a_row["Speed"] - b_closest["Speed"])
            if speed_delta > max_speed_delta:
                max_speed_delta = speed_delta
                max_delta_location = a_row["DistanceOffset"]

        triggered = max_speed_delta > self.SPEED_DELTA_BRAKING_THRESHOLD

        return {
            "triggered": triggered,
            "max_speed_delta_kph": round(max_speed_delta, 2),
            "threshold_kph": self.SPEED_DELTA_BRAKING_THRESHOLD,
            "location_offset": round(max_delta_location, 2),
            "reason": f"Speed delta {max_speed_delta:.1f} km/h exceeds {self.SPEED_DELTA_BRAKING_THRESHOLD} km/h threshold"
            if triggered
            else f"Speed delta {max_speed_delta:.1f} km/h below threshold",
        }

    def _prepare_brain_input(
        self,
        car_a_df: pd.DataFrame,
        car_b_df: pd.DataFrame,
        driver_a_id: str,
        driver_b_id: str,
    ) -> dict[str, Any]:
        """Prepare generic input for the Brain, hiding driver identity.

        Args:
            car_a_df: Telemetry DataFrame for car A
            car_b_df: Telemetry DataFrame for car B
            driver_a_id: Original driver A identifier (for reference only)
            driver_b_id: Original driver B identifier (for reference only)

        Returns:
            Dictionary with generic driver labels for Brain processing
        """
        return {
            "driver_a": {
                "label": "driver_a",
                "reference_id": driver_a_id,
                "avg_speed_kph": round(car_a_df["Speed"].mean(), 1)
                if not car_a_df.empty
                else 0,
                "max_lateral_g": round(car_a_df["lateral_g"].max(), 2)
                if "lateral_g" in car_a_df.columns
                else 0,
                "brake_events": int(car_a_df["Brake"].sum())
                if "Brake" in car_a_df.columns
                else 0,
            },
            "driver_b": {
                "label": "driver_b",
                "reference_id": driver_b_id,
                "avg_speed_kph": round(car_b_df["Speed"].mean(), 1)
                if not car_b_df.empty
                else 0,
                "max_lateral_g": round(car_b_df["lateral_g"].max(), 2)
                if "lateral_g" in car_b_df.columns
                else 0,
                "brake_events": int(car_b_df["Brake"].sum())
                if "Brake" in car_b_df.columns
                else 0,
            },
            "brain_instruction": "Judge this incident based on telemetry data and racing rules, not driver reputation. Consider: proximity changes, lateral G deviations, corner entry/exit, and defensive vs offensive driving patterns.",
        }

    def _determine_verdict(
        self,
        proximity_trigger: dict[str, Any],
        anomaly_trigger: dict[str, Any],
        speed_delta_trigger: dict[str, Any],
    ) -> dict[str, Any]:
        """Determine overall verdict based on triggers.

        Args:
            proximity_trigger: Results from proximity check
            anomaly_trigger: Results from anomaly detection
            speed_delta_trigger: Results from speed delta braking check

        Returns:
            Verdict dictionary with summary and recommendations
        """
        violations = []

        if proximity_trigger.get("triggered"):
            violations.append(f"Proximity violation: {proximity_trigger['reason']}")

        if anomaly_trigger.get("triggered"):
            violations.append(f"Anomaly violation: {anomaly_trigger['reason']}")

        if speed_delta_trigger.get("triggered"):
            violations.append(f"Speed delta violation: {speed_delta_trigger['reason']}")

        if violations:
            verdict = "INVESTIGATION"
            summary = "Incident triggers steward review based on objective telemetry criteria."
        else:
            verdict = "NO_INVESTIGATION"
            summary = "No rule violations detected based on driver-agnostic analysis."

        return {
            "verdict": verdict,
            "violations": violations,
            "summary": summary,
            "proximity_triggered": proximity_trigger.get("triggered", False),
            "anomaly_triggered": anomaly_trigger.get("triggered", False),
            "speed_delta_triggered": speed_delta_trigger.get("triggered", False),
        }

    def _find_overlap_region(
        self, car_a_df: pd.DataFrame, car_b_df: pd.DataFrame
    ) -> tuple[float, float] | None:
        """Find distance region where both cars have telemetry overlap."""
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


def analyze_incident(
    car_a_df: pd.DataFrame,
    car_b_df: pd.DataFrame,
    driver_a_id: str,
    driver_b_id: str,
) -> dict[str, Any]:
    """Driver-agnostic incident analysis.

    Args:
        car_a_df: Telemetry DataFrame for car A
        car_b_df: Telemetry DataFrame for car B
        driver_a_id: Driver identifier for car A (e.g., 'VER')
        driver_b_id: Driver identifier for car B (e.g., 'HAM')

    Returns:
        Dictionary with incident analysis using generic driver labels
    """
    detector = DriverAgnosticDetector()
    return detector.analyze_incident(car_a_df, car_b_df, driver_a_id, driver_b_id)


if __name__ == "__main__":
    import json
    from functools import partial
    import numpy as np

    def convert_to_serializable(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    ver_df = pd.read_parquet("verstappen_abu_dhabi_2021_lap58.parquet")
    ver_df = ver_df[ver_df["DriverCode"] == "VER"]

    ham_df = ver_df.copy()
    ham_df["Speed"] = ham_df["Speed"] * 0.97
    ham_df["DistanceOffset"] = ham_df["DistanceOffset"] + 1.8
    ham_df["DriverCode"] = "HAM"
    ham_df["Brake"] = ver_df["Brake"]

    result = analyze_incident(ver_df, ham_df, "VER", "HAM")
    result = convert_to_serializable(result)

    print("\n" + "=" * 60)
    print("DRIVER-AGNOSTIC INCIDENT ANALYSIS")
    print("=" * 60)
    print(json.dumps(result, indent=2))
