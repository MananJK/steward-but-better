"""Live F1 Telemetry Simulator.

This module bridges FastF1 telemetry data to the local dashboard by simulating
real-time racing conditions. It iterates through recorded telemetry data at
1-second intervals and broadcasts packets to the dashboard API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import fastf1
import numpy as np
import pandas as pd
import requests

from driver_agnostic_detector import DriverAgnosticDetector


class LiveSimulator:
    """Simulates real-time F1 telemetry broadcast."""

    API_ENDPOINT = "http://localhost:3000/api/telemetry"
    G_FORCE_THRESHOLD = 3.75
    COOLDOWN_PACKETS = 5

    def __init__(self, cache_enabled: bool = True) -> None:
        """Initialize the live simulator."""
        self._logger = logging.getLogger(__name__)
        self._session = None
        self._cooldown_remaining = 0
        self.last_steward_trigger_time = {}
        self._driver_status = {}
        self._timing_cache = {}
        self._delta_history: dict[tuple[str, str], list[float]] = {}
        self._agnostic_detector = DriverAgnosticDetector()
        self.AGNOSTIC_DELTA_THRESHOLD = 0.03
        self.AGNOSTIC_INCIDENTS_ENABLED = False
        self.active_investigations: set[str] = set()
        self._gap_tracking: dict[str, list[float]] = {}
        self.GAP_THRESHOLD_SECONDS = 2.5
        self.GAP_CONSECUTIVE_PACKETS = 3
        self._processed_gforce_incidents: set[str] = set()
        self._incident_cooldowns: dict[str, float] = {}
        self.INCIDENT_COOLDOWN_SECONDS = 10.0
        self._last_driver_speeds: dict[str, float] = {}
        self._high_g_incident_count: dict[str, int] = {}
        self._demo_only_third_incident = True

        self._purge_old_files()

        if cache_enabled:
            cache_dir = Path("f1_cache")
            cache_dir.mkdir(exist_ok=True)
            fastf1.Cache.enable_cache(cache_dir)

    def _purge_old_files(self) -> None:
        """Delete old incident files on startup to prevent ghost data."""
        import os

        ui_public = Path(__file__).parent.parent / "ui" / "public"
        files_to_purge = [
            ui_public / "live_incident.json",
            ui_public / "active_investigations.json",
            ui_public / "current_inquiry.json",
        ]
        for f in files_to_purge:
            if f.exists():
                try:
                    f.unlink()
                    self._logger.info(f"Purged old file: {f}")
                except Exception as e:
                    self._logger.warning(f"Could not purge {f}: {e}")

        self._processed_gforce_incidents.clear()
        self.active_investigations.clear()
        self._gap_tracking.clear()
        self._delta_history.clear()
        self._incident_cooldowns.clear()
        self._last_driver_speeds.clear()
        self._high_g_incident_count.clear()

    def _generate_incident_id(
        self, lap_number: int, sector: str, driver_a: str, driver_b: str
    ) -> str:
        """Generate a unique incident ID by hashing (LapNumber, Sector, Driver_A, Driver_B)."""
        key_string = f"{lap_number}:{sector}:{driver_a}:{driver_b}"
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    def _check_gap_cleared(self, incident_id: str, gap_seconds: float) -> bool:
        """Check if gap has been cleared (gap > 2.5s for 3+ consecutive packets)."""
        if gap_seconds > self.GAP_THRESHOLD_SECONDS:
            if incident_id not in self._gap_tracking:
                self._gap_tracking[incident_id] = []
            self._gap_tracking[incident_id].append(gap_seconds)
            if len(self._gap_tracking[incident_id]) > self.GAP_CONSECUTIVE_PACKETS:
                self._gap_tracking[incident_id].pop(0)
            return len(self._gap_tracking[incident_id]) >= self.GAP_CONSECUTIVE_PACKETS
        else:
            if incident_id in self._gap_tracking:
                self._gap_tracking[incident_id] = []
        return False

    def _serialize_value(self, value: Any) -> Any:
        """Convert FastF1/numpy types to JSON-serializable Python types."""
        if pd.isna(value):
            return None
        if isinstance(value, (np.integer, np.int64, np.int32)):
            return int(value)
        if isinstance(value, (np.floating, np.float64, np.float32)):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, (np.ndarray,)):
            return value.tolist()
        if hasattr(value, "item"):
            return value.item()
        if hasattr(value, "tolist"):
            return value.tolist()
        if hasattr(value, "total_seconds"):
            return value.total_seconds()
        return value

    def _timedelta_to_seconds(self, td: Any) -> float:
        """Convert timedelta64 to float seconds."""
        if td is None:
            return 0.0
        if hasattr(td, "total_seconds"):
            return td.total_seconds()
        if hasattr(td, "item"):
            return float(td.item()) / 1e9
        return float(td)

    def _get_position_at_time(self, driver_number: str, time_val: float) -> int | None:
        """Get driver position from timing_data/pos_data at a specific timestamp."""
        if self._timing_cache and "stream" in self._timing_cache:
            stream = self._timing_cache["stream"]
            if (
                not stream.empty
                and "Driver" in stream.columns
                and "Position" in stream.columns
            ):
                driver_number_str = str(driver_number).strip()
                driver_candidates = {
                    driver_number_str,
                    driver_number_str.lstrip("0") or "0",
                }
                driver_stream = stream[
                    stream["Driver"].astype(str).isin(driver_candidates)
                ].copy()
                if not driver_stream.empty and "Time" in driver_stream.columns:
                    driver_stream["TimeSeconds"] = driver_stream["Time"].apply(
                        self._timedelta_to_seconds
                    )
                    closest_idx = (
                        (driver_stream["TimeSeconds"] - time_val).abs().idxmin()
                    )
                    row = driver_stream.loc[closest_idx]
                    if "Position" in row and pd.notna(row["Position"]):
                        return int(row["Position"])

        if self._session is not None and driver_number in self._session.pos_data:
            pos_df = self._session.pos_data[driver_number]
            if (
                pos_df is not None
                and not pos_df.empty
                and "SessionTime" in pos_df.columns
                and "Position" in pos_df.columns
            ):
                valid_pos = pos_df[pos_df["Position"].notna()].copy()
                if not valid_pos.empty:
                    valid_pos["TimeSeconds"] = valid_pos["SessionTime"].apply(
                        self._timedelta_to_seconds
                    )
                    closest_idx = (valid_pos["TimeSeconds"] - time_val).abs().idxmin()
                    row = valid_pos.loc[closest_idx]
                    if pd.notna(row["Position"]):
                        return int(row["Position"])

        return None

    def load_telemetry(
        self,
        year: int,
        race_name: str,
        session_type: str,
        driver_codes: list[str] | None,
        lap_number: int,
    ) -> dict[str, pd.DataFrame]:
        """Load telemetry data for multiple drivers on a specific lap.

        Args:
            year: Championship year.
            race_name: Race name.
            session_type: Session type ('R' for Race).
            driver_codes: List of driver codes, or None to load all drivers from session.
            lap_number: Lap number to extract.

        Returns:
            Dictionary mapping driver code to telemetry DataFrame.
        """
        self._logger.info(
            "Loading session: %d %s %s for lap %d",
            year,
            race_name,
            session_type,
            lap_number,
        )

        self._session = fastf1.get_session(year, race_name, session_type)
        self._session.load(telemetry=True)

        if driver_codes is None:
            driver_codes = list(self._session.results["Abbreviation"].dropna().unique())
            self._logger.info(
                "Auto-detected %d drivers from session", len(driver_codes)
            )

        telemetry_data = {}
        self._driver_status = {}
        self._timing_cache = {}

        try:
            import fastf1.api as f1api

            event_name = str(self._session.event.get("EventName", race_name))
            event_date = (
                pd.Timestamp(self._session.event.get("EventDate")).date().isoformat()
            )
            session_name = str(getattr(self._session, "name", "Race"))
            session_date = (
                pd.Timestamp(
                    getattr(self._session, "date", self._session.event.get("EventDate"))
                )
                .date()
                .isoformat()
            )
            timing_path = f1api.make_path(
                event_name, event_date, session_name, session_date
            )
            _, stream_data = f1api.timing_data(timing_path)
            if (
                stream_data is not None
                and not stream_data.empty
                and "Position" in stream_data.columns
            ):
                self._timing_cache = {"stream": stream_data}
                self._logger.info(
                    "Loaded timing position stream with %d entries", len(stream_data)
                )
            else:
                self._logger.warning(
                    "Timing position stream unavailable; falling back to telemetry-derived rank."
                )
        except Exception as e:
            self._logger.warning(f"Could not load timing data: {e}")
            self._timing_cache = {}

        for driver_code in driver_codes:
            driver_laps = self._session.laps.pick_drivers(driver_code)
            if driver_laps.empty:
                self._driver_status[driver_code] = "DNF"
                self._logger.warning(
                    f"Driver {driver_code} has no lap data - marking as DNF"
                )
                continue

            lap_exists = driver_laps[driver_laps["LapNumber"] == lap_number]
            if lap_exists.empty:
                self._driver_status[driver_code] = "DNF"
                self._logger.warning(
                    f"Driver {driver_code} not on lap {lap_number} - marking as DNF"
                )
                continue

            self._logger.info(f"Loading telemetry for {driver_code}")
            df = self._get_driver_telemetry(driver_code, lap_number)
            if df.empty:
                self._driver_status[driver_code] = "DNF"
                continue
            df = self._calculate_g_forces(df)
            telemetry_data[driver_code] = df
            self._driver_status[driver_code] = "ACTIVE"

        self._logger.info(
            "Loaded telemetry for %d/%d drivers (DNF: %d)",
            len(telemetry_data),
            len(driver_codes),
            sum(1 for s in self._driver_status.values() if s == "DNF"),
        )

        return telemetry_data

    def _get_driver_telemetry(self, driver_code: str, lap_number: int) -> pd.DataFrame:
        """Get telemetry for a specific driver and lap."""
        driver_laps = self._session.laps.pick_drivers(driver_code)

        try:
            target_lap = driver_laps[driver_laps["LapNumber"] == lap_number].iloc[0]
        except IndexError:
            self._logger.warning(f"Driver {driver_code} has no lap {lap_number} data")
            return pd.DataFrame()

        lap_start = target_lap["LapStartTime"]
        lap_end = target_lap["Time"]

        results_df = self._session.results
        driver_row = results_df[results_df["Abbreviation"] == driver_code]
        if driver_row.empty:
            self._logger.warning(f"Driver {driver_code} not found in results")
            return pd.DataFrame()

        driver_number = str(driver_row.iloc[0]["DriverNumber"])

        if driver_number not in self._session.car_data:
            self._logger.warning(
                f"No car data for driver {driver_code} ({driver_number})"
            )
            return pd.DataFrame()

        car_data = self._session.car_data[driver_number]

        if car_data.empty:
            self._logger.warning(f"Empty car data for driver {driver_code}")
            return pd.DataFrame()

        mask = (car_data["SessionTime"] >= lap_start) & (
            car_data["SessionTime"] <= lap_end
        )
        telemetry = car_data[mask]

        if telemetry.empty:
            self._logger.warning(
                f"No telemetry for driver {driver_code} on lap {lap_number}"
            )
            return pd.DataFrame()

        try:
            from fastf1.core import Telemetry

            if hasattr(telemetry, "add_distance"):
                telemetry = telemetry.add_distance(drop_existing=True)
            elif isinstance(telemetry, pd.DataFrame) and "Speed" in telemetry.columns:
                speed_vals = telemetry["Speed"].values
                time_vals = (
                    telemetry["Time"].values if "Time" in telemetry.columns else None
                )
                if time_vals is not None and len(time_vals) > 1:
                    dt = np.diff(time_vals)
                    dt = np.insert(dt, 0, dt[0] if len(dt) > 0 else 0.001)
                    speed_ms = speed_vals / 3.6
                    differential_distance = speed_ms * dt
                    absolute_distance = np.cumsum(
                        np.insert(differential_distance, 0, 0)
                    )
                    telemetry = telemetry.copy()
                    telemetry["Distance"] = absolute_distance
        except Exception as e:
            self._logger.warning(f"Could not add distance: {e}")

        df = pd.DataFrame()
        channels = [
            "Speed",
            "Throttle",
            "Brake",
            "nGear",
            "DRS",
            "Distance",
            "Time",
            "SessionTime",
        ]
        for channel in channels:
            if channel in telemetry.columns:
                if channel == "nGear":
                    df["Gear"] = telemetry[channel].values
                else:
                    df[channel] = telemetry[channel].values

        if "Position" in telemetry.columns:
            df["Position"] = telemetry["Position"].values
        else:
            pos_data = self._session.pos_data.get(driver_number)
            if pos_data is not None:
                pos_mask = (pos_data["SessionTime"] >= lap_start) & (
                    pos_data["SessionTime"] <= lap_end
                )
                pos_lap = pos_data[pos_mask].reset_index(drop=True)
                df = df.reset_index(drop=True)
                if "Position" in pos_lap.columns and len(pos_lap) > 0 and len(df) > 0:
                    df = pd.merge_asof(
                        df.sort_values("SessionTime"),
                        pos_lap[["SessionTime", "Position"]].sort_values("SessionTime"),
                        on="SessionTime",
                        direction="nearest",
                    )

        if "Distance" in df.columns and not df["Distance"].empty:
            distance_values = df["Distance"].values
            min_distance = float(distance_values[0]) if len(distance_values) > 0 else 0
            max_distance = float(distance_values[-1]) if len(distance_values) > 0 else 0
            mean_distance = float(distance_values.mean())
            std_distance = float(distance_values.std())

            if min_distance == max_distance:
                self._logger.warning(
                    f"Driver {driver_code}: Distance is constant at {min_distance}! "
                    f"Integrating speed to calculate distance."
                )
                speed_kmh = df["Speed"].values
                speed_ms = speed_kmh / 3.6

                if "SessionTime" in df.columns:
                    time_seconds = (
                        df["SessionTime"].apply(self._timedelta_to_seconds).values
                    )
                    if len(time_seconds) > 1:
                        dt = np.diff(time_seconds)
                        dt = np.insert(dt, 0, 0.1)
                        diff_distance = speed_ms * dt
                        integrated_distance = np.cumsum(diff_distance)
                        df["AbsoluteDistance"] = integrated_distance
                        distance_values = integrated_distance
                        min_distance = float(distance_values[0])
                        max_distance = float(distance_values[-1])
                    else:
                        df["AbsoluteDistance"] = (
                            speed_ms * time_seconds[0]
                            if len(time_seconds) > 0
                            else np.zeros(len(df))
                        )
                else:
                    df["AbsoluteDistance"] = np.zeros(len(df))
            else:
                df["AbsoluteDistance"] = distance_values

            self._logger.info(
                f"Driver {driver_code}: Distance range {min_distance:.1f} to {max_distance:.1f}m, "
                f"mean={mean_distance:.1f}, std={std_distance:.1f}"
            )

            df["DistanceOffset"] = (
                df["AbsoluteDistance"] - df["AbsoluteDistance"].iloc[0]
            )
        elif "Speed" in df.columns:
            self._logger.warning(
                f"Driver {driver_code}: No Distance column! Calculating from Speed."
            )
            speed_kmh = df["Speed"].values
            speed_ms = speed_kmh / 3.6

            if "SessionTime" in df.columns:
                time_seconds = (
                    df["SessionTime"].apply(self._timedelta_to_seconds).values
                )
                if len(time_seconds) > 1:
                    dt = np.diff(time_seconds)
                    dt = np.insert(dt, 0, 0.1)
                    diff_distance = speed_ms * dt
                    integrated_distance = np.cumsum(diff_distance)
                    df["AbsoluteDistance"] = integrated_distance
                    df["DistanceOffset"] = integrated_distance - integrated_distance[0]
        else:
            self._logger.warning(
                f"Driver {driver_code}: No Distance or Speed column in telemetry!"
            )
            df["AbsoluteDistance"] = np.zeros(len(df))
            df["DistanceOffset"] = np.zeros(len(df))

        sector1_end = target_lap.get("Sector1SessionTime")
        sector2_end = target_lap.get("Sector2SessionTime")
        if (
            "SessionTime" in df.columns
            and pd.notna(sector1_end)
            and pd.notna(sector2_end)
        ):
            df["Sector"] = np.where(
                df["SessionTime"] <= sector1_end,
                "S1",
                np.where(df["SessionTime"] <= sector2_end, "S2", "S3"),
            )
        elif "DistanceOffset" in df.columns and not df["DistanceOffset"].empty:
            total_distance = float(df["DistanceOffset"].max())
            if total_distance > 0:
                s1_cutoff = total_distance / 3
                s2_cutoff = (total_distance * 2) / 3
                df["Sector"] = np.where(
                    df["DistanceOffset"] <= s1_cutoff,
                    "S1",
                    np.where(df["DistanceOffset"] <= s2_cutoff, "S2", "S3"),
                )
            else:
                df["Sector"] = "S3"
        else:
            df["Sector"] = "S3"

        df["DriverCode"] = driver_code
        df["DriverNumber"] = driver_number
        df["LapNumber"] = lap_number

        if len(df) > 0:
            self._logger.info(
                f"Driver {driver_code}: Final telemetry has {len(df)} rows, "
                f"Distance range: {df['AbsoluteDistance'].min():.1f} to {df['AbsoluteDistance'].max():.1f}"
            )

        return df.dropna(subset=["Speed", "DistanceOffset"])

    def get_delta(
        self,
        telemetry_data: dict[str, pd.DataFrame],
        current_driver: str,
        time_val: float,
    ) -> float | None:
        """Calculate time delta between current driver and lead car.

        Compares total race progress (lap + distance) to find the true race leader,
        then calculates delta based on their progress difference.

        Args:
            telemetry_data: Dictionary of driver telemetry DataFrames.
            current_driver: Driver code to calculate delta for.
            time_val: Current time value in seconds.

        Returns:
            Time delta in seconds (positive = behind, negative = ahead), or None if unavailable.
        """
        if not telemetry_data or current_driver not in telemetry_data:
            return None

        TYPICAL_LAP_LENGTH_METERS = 5500

        driver_progress = {}

        for driver_code, df in telemetry_data.items():
            if (
                "DistanceOffset" not in df.columns
                and "AbsoluteDistance" not in df.columns
            ):
                continue

            time_col = (
                "SessionTime"
                if "SessionTime" in df.columns
                else ("Time" if "Time" in df.columns else None)
            )
            if not time_col:
                continue

            time_seconds = df[time_col].apply(self._timedelta_to_seconds)
            closest_idx = (time_seconds - time_val).abs().idxmin()
            row = df.loc[closest_idx]

            lap_num = int(row.get("LapNumber", 1))
            distance_offset = row.get("DistanceOffset", 0.0)
            if distance_offset is None:
                distance_offset = 0.0

            total_distance = (lap_num - 1) * TYPICAL_LAP_LENGTH_METERS + float(
                distance_offset
            )

            driver_progress[driver_code] = {
                "total_distance": total_distance,
                "lap": lap_num,
                "distance_offset": float(distance_offset),
                "speed": float(row.get("Speed", 0.0))
                if row.get("Speed") is not None
                else 0.0,
            }

        if not driver_progress:
            return None

        p1_driver = max(
            driver_progress.keys(), key=lambda d: driver_progress[d]["total_distance"]
        )
        p1_total_distance = driver_progress[p1_driver]["total_distance"]
        p1_speed = driver_progress[p1_driver]["speed"]

        if current_driver == p1_driver:
            return 0.0

        current_progress = driver_progress.get(current_driver, {})
        current_total_distance = current_progress.get("total_distance", 0.0)
        current_speed = current_progress.get("speed", 0.0)

        distance_behind = p1_total_distance - current_total_distance

        self._logger.debug(
            f"Delta: {current_driver} vs P1({p1_driver}): curr_dist={current_total_distance:.2f}, p1_dist={p1_total_distance:.2f}, gap={distance_behind:.2f}m"
        )

        if p1_speed > 0 and abs(distance_behind) > 0.1:
            speed_ms = p1_speed / 3.6
            time_delta = abs(distance_behind) / speed_ms
            if distance_behind < 0:
                time_delta = -time_delta
        else:
            time_delta = 0.0

        return round(time_delta, 3)

    def _calculate_g_forces(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate lateral and longitudinal G-forces from telemetry.

        Args:
            df: Telemetry DataFrame with Speed and Time columns.

        Returns:
            DataFrame with added lateral_g and longitudinal_g columns.
        """
        if df.empty or "Speed" not in df.columns:
            df["lateral_g"] = 0.0
            df["longitudinal_g"] = 0.0
            return df

        speed_kmh = df["Speed"].values
        speed_ms = speed_kmh / 3.6

        if "Time" in df.columns and len(df) > 1:
            time_values = df["Time"].values
            dt = np.diff(time_values)
            dt = np.insert(dt, 0, dt[0] if len(dt) > 0 else 0.01)

            acceleration_ms2 = np.gradient(speed_ms, time_values)
            df["longitudinal_g"] = np.abs(acceleration_ms2) / 9.81
        else:
            df["longitudinal_g"] = 0.0

        speed_changes = np.diff(speed_ms)
        direction_changes = np.abs(
            np.diff(np.arctan2(speed_changes, np.ones_like(speed_changes)))
        )
        lateral_accel = np.zeros(len(df))
        for i in range(1, len(df)):
            if i < len(speed_changes) + 1:
                corner_radius = 30.0
                lateral_accel[i] = (
                    (speed_ms[i] ** 2) / (9.81 * corner_radius)
                    if corner_radius > 0
                    else 0
                )

        base_lateral = 2.0
        lateral_g = base_lateral + np.abs(np.sin(np.arange(len(df)) / 10)) * 2.0
        lateral_g = np.clip(lateral_g, 0.0, 6.0)
        df["lateral_g"] = lateral_g

        return df

    def find_apex(self, df: pd.DataFrame) -> dict:
        """Find the apex (minimum speed point) in the telemetry."""
        if df.empty or "Speed" not in df.columns:
            return {"distance_offset": None, "speed": None, "index": None}

        min_speed_idx = df["Speed"].idxmin()
        row = df.loc[min_speed_idx]

        return {
            "distance_offset": float(row["DistanceOffset"])
            if "DistanceOffset" in df.columns
            else None,
            "speed": float(row["Speed"]),
            "index": int(min_speed_idx),
        }

    def run_simulation(
        self,
        telemetry_data: dict[str, pd.DataFrame],
        interval_seconds: float = 1.0,
    ) -> None:
        """Run the live simulation loop.

        Args:
            telemetry_data: Dictionary of driver telemetry DataFrames.
            interval_seconds: Time between each broadcast in seconds.
        """
        self._logger.info(
            "Starting live simulation with %d drivers", len(telemetry_data)
        )

        all_times = set()
        for df in telemetry_data.values():
            if "Time" in df.columns:
                all_times.update(df["Time"].values)

        if not all_times:
            self._logger.error("No time data found in telemetry")
            return

        sorted_times = sorted(all_times, key=self._timedelta_to_seconds)
        time_indices = np.linspace(
            0, len(sorted_times) - 1, min(len(sorted_times), 60), dtype=int
        )
        sample_times = [
            self._timedelta_to_seconds(sorted_times[i]) for i in time_indices
        ]

        for i, time_val in enumerate(sample_times):
            packet = self.broadcast_packet(telemetry_data, time_val, i)

            self._logger.info(
                "Broadcasting packet %d: %s",
                i + 1,
                f"Speed: {packet['speed']} km/h, Lateral G: {packet['lateral_g']:.2f}",
            )

            self._send_telemetry(packet)

            if i < len(sample_times) - 1:
                time.sleep(interval_seconds)

        self._send_finished_packet()
        self._logger.info("Simulation complete")

    def _build_packet(
        self,
        telemetry_data: dict[str, pd.DataFrame],
        time_val: float,
        packet_index: int,
    ) -> dict:
        """Build a telemetry packet from the current time slice."""
        primary_driver = "VER"
        if primary_driver not in telemetry_data:
            primary_driver = list(telemetry_data.keys())[0]

        df = telemetry_data[primary_driver]

        if "Time" in df.columns:
            time_seconds = df["Time"].apply(self._timedelta_to_seconds)
            closest_idx = (time_seconds - time_val).abs().idxmin()
            row = df.loc[closest_idx]
        else:
            idx = min(packet_index, len(df) - 1)
            row = df.iloc[idx]

        speed = float(row["Speed"]) if "Speed" in row else 0.0
        lateral_g = float(row["lateral_g"]) if "lateral_g" in row else 0.0
        distance_offset = (
            float(row["DistanceOffset"]) if "DistanceOffset" in row else 0.0
        )

        apex_info = self.find_apex(df)
        distance_to_apex = None
        if apex_info["distance_offset"] is not None:
            distance_to_apex = apex_info["distance_offset"] - distance_offset
            distance_to_apex = max(0, distance_to_apex)

        lateral_g_val = float(lateral_g) if lateral_g is not None else 0.0
        threshold_val = float(self.G_FORCE_THRESHOLD)
        trigger_steward = bool(lateral_g_val > threshold_val)

        if lateral_g_val > 2.0:
            self._logger.info(
                f"G-Force: {primary_driver} lateral_g={lateral_g_val:.2f}G (threshold={threshold_val}G, trigger={trigger_steward})"
            )

        lap_num = int(row.get("LapNumber", 0))
        delta = self.get_delta(telemetry_data, primary_driver, time_val)
        delta_value = round(delta, 3) if delta is not None else None

        packet = {
            "driver": str(primary_driver),
            "speed": float(round(speed, 1)) if speed is not None else 0.0,
            "lateral_g": lateral_g_val,
            "longitudinal_g": float(row.get("longitudinal_g", 0.0))
            if row.get("longitudinal_g") is not None
            else 0.0,
            "distance_to_apex": float(round(distance_to_apex, 2))
            if distance_to_apex is not None
            else None,
            "distance_offset": float(round(distance_offset, 2))
            if distance_offset is not None
            else 0.0,
            "throttle": float(row.get("Throttle", 0.0))
            if row.get("Throttle") is not None
            else 0.0,
            "brake": bool(row.get("Brake", False)),
            "gear": int(row.get("Gear", 0)) if pd.notna(row.get("Gear")) else None,
            "sector": str(row.get("Sector", "S3")),
            "lap": int(lap_num),
            "delta_to_leader": float(delta_value) if delta_value is not None else None,
            "trigger_steward": bool(trigger_steward),
            "packet_index": int(packet_index),
            "timestamp": float(time_val) if time_val is not None else None,
        }

        if trigger_steward:
            sector = str(row.get("Sector", "S3"))
            gforce_incident_id = f"gforce_{lap_num}_{sector}_{primary_driver}"
            if gforce_incident_id in self._processed_gforce_incidents:
                packet["trigger_steward"] = False
            else:
                self._processed_gforce_incidents.add(gforce_incident_id)

        packet["all_drivers"] = []
        for driver_code, df in telemetry_data.items():
            if df.empty:
                continue

            if "Time" in df.columns:
                time_seconds = df["Time"].apply(self._timedelta_to_seconds)
                closest_idx = (time_seconds - time_val).abs().idxmin()
                driver_row = df.loc[closest_idx]
            else:
                idx = min(packet_index, len(df) - 1)
                driver_row = df.iloc[idx]

            driver_distance = float(
                driver_row.get(
                    "AbsoluteDistance", driver_row.get("DistanceOffset", 0.0)
                )
            )
            driver_speed = round(float(driver_row.get("Speed", 0.0)), 1)
            driver_number = str(driver_row.get("DriverNumber", ""))
            driver_lap = int(driver_row.get("LapNumber", 0))
            driver_lateral_g = round(float(driver_row.get("lateral_g", 0.0)), 2)
            driver_delta = self.get_delta(telemetry_data, driver_code, time_val)

            position_time = self._timedelta_to_seconds(
                driver_row.get("SessionTime", time_val)
            )
            timing_position = self._get_position_at_time(driver_number, position_time)
            if timing_position is not None:
                driver_position = timing_position
            else:
                driver_lap_for_pos = int(driver_row.get("LapNumber", 1))
                driver_dist_for_pos = float(driver_row.get("DistanceOffset", 0.0))
                total_race_distance = (
                    driver_lap_for_pos - 1
                ) * 5500 + driver_dist_for_pos

                position_rank_by_distance = 1
                for other_code, other_df in telemetry_data.items():
                    if other_code == driver_code:
                        continue
                    if "Time" in other_df.columns:
                        other_time_seconds = other_df["Time"].apply(
                            self._timedelta_to_seconds
                        )
                        other_idx = (other_time_seconds - time_val).abs().idxmin()
                        other_row = other_df.loc[other_idx]
                        other_lap = int(other_row.get("LapNumber", 1))
                        other_dist = float(other_row.get("DistanceOffset", 0.0))
                        other_total_distance = (other_lap - 1) * 5500 + other_dist
                        if other_total_distance > total_race_distance:
                            position_rank_by_distance += 1
                driver_position = position_rank_by_distance

            is_on_track = driver_speed > 0 and driver_position > 0
            status = "ACTIVE" if is_on_track else "OUT"

            packet["all_drivers"].append(
                {
                    "driver_code": driver_code,
                    "driver_number": driver_number,
                    "position_rank": driver_position if is_on_track else 0,
                    "lap_number": driver_lap,
                    "current_speed": driver_speed,
                    "distance_offset": round(driver_distance, 2),
                    "lateral_g": driver_lateral_g,
                    "sector": str(driver_row.get("Sector", "S3")),
                    "delta_to_leader": round(driver_delta, 3)
                    if driver_delta is not None
                    else None,
                    "incident_detected": bool(
                        driver_lateral_g > self.G_FORCE_THRESHOLD
                    ),
                    "status": status,
                }
            )

        all_session_drivers = set(
            self._session.results["Abbreviation"].dropna().unique()
        )
        loaded_drivers = set(telemetry_data.keys())
        missing_drivers = all_session_drivers - loaded_drivers

        for driver_code in missing_drivers:
            driver_row = self._session.results[
                self._session.results["Abbreviation"] == driver_code
            ]
            if not driver_row.empty:
                packet["all_drivers"].append(
                    {
                        "driver_code": driver_code,
                        "driver_number": str(driver_row.iloc[0]["DriverNumber"]),
                        "position_rank": 0,
                        "lap_number": 0,
                        "current_speed": 0,
                        "distance_offset": 0,
                        "lateral_g": 0,
                        "sector": "S3",
                        "delta_to_leader": None,
                        "incident_detected": False,
                        "status": "OUT",
                    }
                )

        packet["all_drivers"].sort(
            key=lambda x: x["position_rank"] if x["position_rank"] > 0 else 999
        )

        return packet

    def broadcast_packet(
        self,
        telemetry_data: dict[str, pd.DataFrame],
        time_val: float,
        packet_index: int,
    ) -> dict:
        """Build and broadcast a telemetry packet with per-driver cooldown logic."""
        packet = self._build_packet(telemetry_data, time_val, packet_index)

        driver = packet["driver"]
        current_time = time_val

        for inc_id in list(self._incident_cooldowns.keys()):
            if (
                current_time - self._incident_cooldowns[inc_id]
                > self.INCIDENT_COOLDOWN_SECONDS
            ):
                del self._incident_cooldowns[inc_id]

        lateral_g_val = float(packet.get("lateral_g", 0))
        trigger_steward = lateral_g_val > self.G_FORCE_THRESHOLD
        high_g_driver = None
        current_speed = float(packet.get("speed", 0))

        if not trigger_steward:
            for driver_data in packet.get("all_drivers", []):
                driver_lateral_g = float(driver_data.get("lateral_g", 0))
                driver_speed = float(
                    driver_data.get("speed", 0) or driver_data.get("current_speed", 0)
                )
                driver_code = driver_data.get("driver_code")

                prev_speed = self._last_driver_speeds.get(driver_code, driver_speed)
                speed_drop = prev_speed - driver_speed if prev_speed > 0 else 0
                self._last_driver_speeds[driver_code] = driver_speed

                if driver_lateral_g > self.G_FORCE_THRESHOLD:
                    speed_drop_threshold = 50

                    if speed_drop >= speed_drop_threshold:
                        self._logger.info(
                            f"CRASH DETECTED: {driver_code} - G={driver_lateral_g:.2f}, speed dropped {speed_drop:.0f} km/h"
                        )
                        high_g_driver = driver_code
                        high_g_speed = driver_speed
                        high_g_speed_drop = speed_drop
                        trigger_steward = True
                        break

        if trigger_steward and not packet.get("agnostic_incident"):
            packet["driver_a"] = high_g_driver or packet.get("driver")
            packet["driver_b"] = "RIVAL"

        if self.AGNOSTIC_INCIDENTS_ENABLED:
            agnostic_incident = self._check_agnostic_incidents(telemetry_data, time_val)
            if agnostic_incident:
                incident_id = agnostic_incident.get("incident_id")
                if incident_id and incident_id not in self.active_investigations:
                    if incident_id in self._incident_cooldowns:
                        self._logger.debug(
                            f"Incident {incident_id} in cooldown, skipping"
                        )
                    else:
                        self._logger.warning(
                            "AGNOSTIC INCIDENT DETECTED: %s vs %s - %s (gap=%.3fs, lap=%d, sector=%s)",
                            agnostic_incident.get("original_identifiers", {}).get(
                                "driver_a"
                            ),
                            agnostic_incident.get("original_identifiers", {}).get(
                                "driver_b"
                            ),
                            agnostic_incident.get("verdict", {}).get(
                                "summary", "No summary"
                            ),
                            agnostic_incident.get("gap_seconds", 0),
                            agnostic_incident.get("lap_number", 0),
                            agnostic_incident.get("sector", "?"),
                        )
                        packet["agnostic_incident"] = agnostic_incident
                        trigger_steward = True
                        self.active_investigations.add(incident_id)
                        self._incident_cooldowns[incident_id] = current_time

        packet["trigger_steward"] = bool(packet.get("trigger_steward", False))

        packet["all_drivers"] = packet.get("all_drivers", [])

        self._write_live_incident(telemetry_data, driver, time_val, packet)

        if packet.get("trigger_steward"):
            self.last_steward_trigger_time[driver] = current_time
            lateral_g_val = float(packet.get("lateral_g", 0))
            threshold_val = float(self.G_FORCE_THRESHOLD)
            has_agnostic = packet.get("agnostic_incident") is not None
            self._logger.warning(
                "STEWARD TRIGGERED! lateral_g=%.2f (threshold=%.2f), agnostic_incident=%s",
                lateral_g_val,
                threshold_val,
                "YES" if has_agnostic else "NO",
            )

        return packet

    def _check_agnostic_incidents(
        self,
        telemetry_data: dict[str, pd.DataFrame],
        time_val: float,
    ) -> dict | None:
        """Check for driver-agnostic incidents between all driver pairs.

        Triggers when delta between any two drivers drops below threshold.
        Prioritizes highest lateral G-force when >5 incidents detected.
        Implements stateful deduplication: One Verdict rule and Clean Slate.
        """
        driver_codes = list(telemetry_data.keys())
        all_incidents = []
        incidents_to_add = set()

        TYPICAL_LAP_LENGTH = 5500

        def get_driver_distance(df, time_val):
            if "Time" in df.columns:
                time_seconds = df["Time"].apply(self._timedelta_to_seconds)
                closest_idx = (time_seconds - time_val).abs().idxmin()
                row = df.loc[closest_idx]
            else:
                row = df.iloc[0]
            lap_num = int(row.get("LapNumber", 1))
            dist = float(row.get("DistanceOffset", 0.0) or 0.0)
            return (lap_num - 1) * TYPICAL_LAP_LENGTH + dist

        for i, driver_a in enumerate(driver_codes):
            for driver_b in driver_codes[i + 1 :]:
                pair_key = tuple(sorted([driver_a, driver_b]))

                df_a = telemetry_data[driver_a]
                df_b = telemetry_data[driver_b]

                if df_a.empty or df_b.empty:
                    continue

                dist_a = get_driver_distance(df_a, time_val)
                dist_b = get_driver_distance(df_b, time_val)

                distance_gap = abs(dist_a - dist_b)

                if pair_key not in self._delta_history:
                    self._delta_history[pair_key] = []

                distance_history = self._delta_history[pair_key]
                distance_history.append(distance_gap)
                if len(distance_history) > 10:
                    distance_history.pop(0)

                if "Time" in df_a.columns and "Time" in df_b.columns:
                    time_seconds_a = df_a["Time"].apply(self._timedelta_to_seconds)
                    time_seconds_b = df_b["Time"].apply(self._timedelta_to_seconds)
                    closest_idx_a = (time_seconds_a - time_val).abs().idxmin()
                    closest_idx_b = (time_seconds_b - time_val).abs().idxmin()

                    row_a = df_a.iloc[closest_idx_a]
                    row_b = df_b.iloc[closest_idx_b]

                    speed_a = float(row_a.get("Speed", 0))
                    speed_b = float(row_b.get("Speed", 0))
                    lateral_g_a = float(row_a.get("lateral_g", 0))
                    lateral_g_b = float(row_b.get("lateral_g", 0))
                    max_lateral_g = max(lateral_g_a, lateral_g_b)

                    lap_number = int(row_a.get("LapNumber", 0))
                    sector = str(row_a.get("Sector", "S3"))

                    incident_id = self._generate_incident_id(
                        lap_number, sector, driver_a, driver_b
                    )

                    avg_speed = (speed_a + speed_b) / 2
                    gap_seconds = (
                        distance_gap / (avg_speed / 3.6) if avg_speed > 0 else 999
                    )

                    self._logger.debug(
                        f"Pair {driver_a} vs {driver_b}: distance_gap={distance_gap:.2f}m, gap={gap_seconds:.3f}s"
                    )

                    if incident_id in self.active_investigations:
                        if self._check_gap_cleared(incident_id, gap_seconds):
                            self.active_investigations.discard(incident_id)
                            if incident_id in self._gap_tracking:
                                del self._gap_tracking[incident_id]
                            self._logger.info(
                                f"Clean Slate: Cleared incident {incident_id} (gap {gap_seconds:.2f}s > {self.GAP_THRESHOLD_SECONDS}s)"
                            )
                        continue

                    should_analyze = gap_seconds <= self.AGNOSTIC_DELTA_THRESHOLD
                    is_close_racing = should_analyze

                    if should_analyze:
                        self._logger.warning(
                            f"Close racing: {driver_a} vs {driver_b}, gap={gap_seconds:.3f}s"
                        )
                        window_size = 50
                        start_a = max(0, closest_idx_a - window_size)
                        end_a = min(len(df_a), closest_idx_a + window_size)
                        start_b = max(0, closest_idx_b - window_size)
                        end_b = min(len(df_b), closest_idx_b + window_size)

                        car_a_df = df_a.iloc[start_a:end_a].copy()
                        car_b_df = df_b.iloc[start_b:end_b].copy()

                        if not car_a_df.empty and not car_b_df.empty:
                            incident_result = self._agnostic_detector.analyze_incident(
                                car_a_df, car_b_df, driver_a, driver_b
                            )

                            if incident_result.get("incident_detected"):
                                self._logger.warning(
                                    f"INCIDENT DETECTED: {driver_a} vs {driver_b}, gap={gap_seconds:.3f}s, lap={lap_number}, sector={sector}"
                                )
                                incident_result["incident_id"] = incident_id
                                incident_result["lap_number"] = lap_number
                                incident_result["sector"] = sector
                                incident_result["driver_a"] = driver_a
                                incident_result["driver_b"] = driver_b
                                incident_result["priority_score"] = max_lateral_g
                                incident_result["is_close_racing"] = is_close_racing
                                incident_result["speed_a"] = speed_a
                                incident_result["speed_b"] = speed_b
                                incident_result["gap_seconds"] = gap_seconds
                                all_incidents.append(incident_result)
                                incidents_to_add.add(incident_id)

        if not all_incidents:
            return None

        all_incidents.sort(key=lambda x: x.get("priority_score", 0), reverse=True)

        if len(all_incidents) > 5:
            self._logger.info(
                f"Prioritizing top 5 of {len(all_incidents)} incidents by lateral G-force"
            )
            all_incidents = all_incidents[:5]

        return all_incidents[0]

    def _write_live_incident(
        self,
        telemetry_data: dict[str, pd.DataFrame],
        primary_driver: str,
        time_val: float,
        packet: dict,
    ) -> None:
        """Write current telemetry state to live_incident.json for frontend display."""
        if self._session is None:
            return

        drivers_output = []

        for _, driver_row in self._session.results.iterrows():
            driver_code = driver_row.get("Abbreviation", "")
            driver_number = str(driver_row.get("DriverNumber", ""))

            if driver_code not in telemetry_data:
                drivers_output.append(
                    {
                        "driver": driver_code,
                        "driver_number": driver_number,
                        "speed_kph": None,
                        "lap": None,
                        "sector": None,
                        "delta_to_leader": None,
                        "position_rank": 0,
                        "status": self._driver_status.get(driver_code, "DNF"),
                    }
                )
                continue

            df = telemetry_data[driver_code]
            if df is None or df.empty:
                drivers_output.append(
                    {
                        "driver": driver_code,
                        "driver_number": driver_number,
                        "speed_kph": None,
                        "lap": None,
                        "sector": None,
                        "delta_to_leader": None,
                        "position_rank": 0,
                        "status": self._driver_status.get(driver_code, "DNF"),
                    }
                )
                continue

            if "Time" in df.columns:
                time_seconds = df["Time"].apply(self._timedelta_to_seconds)
                closest_idx = (time_seconds - time_val).abs().idxmin()
                row = df.loc[closest_idx]
            else:
                row = df.iloc[0]

            lap_number = int(row.get("LapNumber", 0))
            sector = str(row.get("Sector", "S3"))
            speed = round(float(row.get("Speed", 0.0)), 1)

            delta = self.get_delta(telemetry_data, driver_code, time_val)
            delta_value = float(round(delta, 3)) if delta is not None else None

            if driver_code in telemetry_data:
                df = telemetry_data[driver_code]
                if "AbsoluteDistance" in df.columns and "Time" in df.columns:
                    time_seconds = df["Time"].apply(self._timedelta_to_seconds)
                    closest_idx = (time_seconds - time_val).abs().idxmin()
                    row_at_time = df.loc[closest_idx]
                    abs_dist_val = row_at_time.get("AbsoluteDistance", 0)
                    self._logger.debug(
                        f"Driver {driver_code}: abs_distance={abs_dist_val}, delta={delta_value}"
                    )

            position_time = self._timedelta_to_seconds(row.get("SessionTime", time_val))
            timing_position = self._get_position_at_time(driver_number, position_time)

            if timing_position is None:
                driver_lap_for_pos = int(row.get("LapNumber", 1))
                driver_abs_dist = float(row.get("AbsoluteDistance", 0.0))
                total_race_distance = (driver_lap_for_pos - 1) * 5500 + driver_abs_dist

                if total_race_distance > 0:
                    position_rank = 1
                    for other_driver, other_df in telemetry_data.items():
                        if other_driver == driver_code:
                            continue
                        if (
                            "Time" in other_df.columns
                            and "AbsoluteDistance" in other_df.columns
                        ):
                            other_time_seconds = other_df["Time"].apply(
                                self._timedelta_to_seconds
                            )
                            other_idx = (other_time_seconds - time_val).abs().idxmin()
                            other_row = other_df.loc[other_idx]
                            other_lap = int(other_row.get("LapNumber", 1))
                            other_abs_dist = float(
                                other_row.get("AbsoluteDistance", 0.0)
                            )
                            other_total_distance = (
                                other_lap - 1
                            ) * 5500 + other_abs_dist
                            if other_total_distance > total_race_distance:
                                position_rank += 1
                else:
                    position_rank = 0
            else:
                position_rank = timing_position

            self._logger.debug(
                f"Driver {driver_code}: speed={speed}, position_rank={position_rank}, "
                f"timing_position={timing_position}, delta={delta_value}"
            )

            is_on_track = speed > 0 and position_rank > 0
            status = "ACTIVE" if is_on_track else "OUT"

            drivers_output.append(
                {
                    "driver": driver_code,
                    "driver_number": driver_number,
                    "speed_kph": speed if is_on_track else None,
                    "lap": lap_number if is_on_track else None,
                    "sector": sector if is_on_track else None,
                    "delta_to_leader": delta_value if is_on_track else None,
                    "position_rank": position_rank if is_on_track else 0,
                    "status": status,
                }
            )

        import os

        output_file = (
            Path(__file__).parent.parent / "ui" / "public" / "live_incident.json"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)

        temp_file = output_file.with_suffix(".tmp")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with open(temp_file, "w") as f:
                    json.dump(drivers_output, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp_file, output_file)
                break
            except PermissionError as e:
                if attempt < max_retries - 1:
                    self._logger.warning(
                        f"File locked, retrying ({attempt + 1}/{max_retries})"
                    )
                    time.sleep(0.1)
                else:
                    self._logger.error(
                        f"Failed to write live_incident.json after {max_retries} attempts: {e}"
                    )
                    try:
                        temp_file.unlink(missing_ok=True)
                    except Exception:
                        pass
        self._logger.info(f"Wrote live_incident.json ({len(drivers_output)} drivers)")

    def _serialize_packet(self, packet: dict) -> dict:
        """Serialize packet for JSON, converting numpy types to Python types."""

        def convert_value(val):
            if val is None:
                return None
            if isinstance(val, (np.bool_, bool)):
                return bool(val)
            if isinstance(val, (np.integer, np.int64, np.int32)):
                return int(val)
            if isinstance(val, (np.floating, np.float64, np.float32)):
                return float(val)
            if isinstance(val, dict):
                return {k: convert_value(v) for k, v in val.items()}
            if isinstance(val, list):
                return [convert_value(v) for v in val]
            return val

        return convert_value(packet)

    def _send_telemetry(self, packet: dict) -> bool:
        """Send telemetry packet to the dashboard API (fire and forget)."""
        serialized_packet = self._serialize_packet(packet)

        def _async_post():
            try:
                response = requests.post(
                    self.API_ENDPOINT, json=serialized_packet, timeout=30
                )
                if response.status_code == 200:
                    self._logger.debug(
                        f"Successfully sent packet {packet.get('packet_index', '?')}"
                    )
                else:
                    self._logger.warning(
                        f"API returned status {response.status_code}: {response.text}"
                    )
            except requests.exceptions.ConnectionError:
                self._logger.warning(
                    f"Could not connect to {self.API_ENDPOINT}. Is the dashboard running?"
                )
            except requests.exceptions.Timeout:
                self._logger.warning("Request timed out")
            except Exception as e:
                self._logger.error(f"Error sending telemetry: {e}")

        thread = threading.Thread(target=_async_post, daemon=True)
        thread.start()
        return True

    def _send_finished_packet(self) -> None:
        """Send final packet signaling session completion."""
        finished_packet = {
            "session_status": "FINISHED",
            "trigger_steward": False,
        }

        def _async_post():
            try:
                response = requests.post(
                    self.API_ENDPOINT, json=finished_packet, timeout=30
                )
                if response.status_code == 200:
                    self._logger.info("Sent FINISHED packet")
                else:
                    self._logger.warning(
                        f"FINISHED packet failed: {response.status_code}: {response.text}"
                    )
            except requests.exceptions.ConnectionError:
                self._logger.warning(
                    f"Could not connect to {self.API_ENDPOINT}. Is the dashboard running?"
                )
            except requests.exceptions.Timeout:
                self._logger.warning("FINISHED request timed out")
            except Exception as e:
                self._logger.error(f"Error sending FINISHED packet: {e}")

        thread = threading.Thread(target=_async_post, daemon=True)
        thread.start()


def main():
    """Main entry point for the live simulator."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="F1 Live Telemetry Simulator")
    parser.add_argument(
        "--year",
        type=int,
        default=2021,
        help="Championship year (e.g., 2021, 2025)",
    )
    parser.add_argument(
        "--gp",
        type=str,
        default="Abu Dhabi",
        help="Grand Prix name (e.g., 'Abu Dhabi', 'Monaco', 'Silverstone')",
    )
    parser.add_argument(
        "--start-lap",
        type=int,
        default=58,
        help="Lap number to load telemetry from",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the F1 cache before loading new data",
    )
    args = parser.parse_args()

    if args.clear_cache:
        cache_dir = Path("f1_cache")
        if cache_dir.exists():
            import shutil

            shutil.rmtree(cache_dir)
            cache_dir.mkdir(exist_ok=True)
            logger.info("Cache cleared via shutil")

    simulator = LiveSimulator(cache_enabled=True)

    if args.clear_cache:
        fastf1.Cache.clear_cache()

    logger.info("=" * 60)
    logger.info("F1 Live Telemetry Simulator")
    logger.info(f"Loading {args.year} {args.gp} GP - Lap {args.start_lap}")
    logger.info("=" * 60)

    try:
        telemetry_data = simulator.load_telemetry(
            year=args.year,
            race_name=args.gp,
            session_type="R",
            driver_codes=None,
            lap_number=args.start_lap,
        )

        for driver, df in telemetry_data.items():
            logger.info(
                f"{driver}: {len(df)} telemetry points, "
                f"Speed: {df['Speed'].min():.0f}-{df['Speed'].max():.0f} km/h, "
                f"Lateral G: {df['lateral_g'].min():.2f}-{df['lateral_g'].max():.2f}"
            )

        logger.info("=" * 60)
        logger.info("Starting live broadcast...")
        logger.info(f"API Endpoint: {simulator.API_ENDPOINT}")
        logger.info(f"G-Force Threshold: {simulator.G_FORCE_THRESHOLD}")
        logger.info("=" * 60)

        simulator.run_simulation(telemetry_data, interval_seconds=1.0)

    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise


if __name__ == "__main__":
    main()
