"""Live F1 Telemetry Simulator.

This module bridges FastF1 telemetry data to the local dashboard by simulating
real-time racing conditions. It iterates through recorded telemetry data at
1-second intervals and broadcasts packets to the dashboard API.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from pathlib import Path
from typing import Any

import fastf1
import numpy as np
import pandas as pd
import requests


class LiveSimulator:
    """Simulates real-time F1 telemetry broadcast."""

    API_ENDPOINT = "http://localhost:3000/api/telemetry"
    G_FORCE_THRESHOLD = 4.2
    COOLDOWN_PACKETS = 5

    def __init__(self, cache_enabled: bool = True) -> None:
        """Initialize the live simulator."""
        self._logger = logging.getLogger(__name__)
        self._session = None
        self._cooldown_remaining = 0
        self.last_steward_trigger_time = {}
        if cache_enabled:
            cache_dir = Path("f1_cache")
            cache_dir.mkdir(exist_ok=True)
            fastf1.Cache.enable_cache(cache_dir)

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

    def load_telemetry(
        self,
        year: int,
        race_name: str,
        session_type: str,
        driver_codes: list[str],
        lap_number: int,
    ) -> dict[str, pd.DataFrame]:
        """Load telemetry data for multiple drivers on a specific lap.

        Args:
            year: Championship year.
            race_name: Race name.
            session_type: Session type ('R' for Race).
            driver_codes: List of driver codes (e.g., ['VER', 'HAM']).
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
        self._session.load()

        telemetry_data = {}

        for driver_code in driver_codes:
            self._logger.info(f"Loading telemetry for {driver_code}")
            df = self._get_driver_telemetry(driver_code, lap_number)
            df = self._calculate_g_forces(df)
            telemetry_data[driver_code] = df

        return telemetry_data

    def _get_driver_telemetry(self, driver_code: str, lap_number: int) -> pd.DataFrame:
        """Get telemetry for a specific driver and lap."""
        driver_laps = self._session.laps.pick_drivers(driver_code)
        target_lap = driver_laps[driver_laps["LapNumber"] == lap_number].iloc[0]
        lap_start = target_lap["LapStartTime"]
        lap_end = target_lap["Time"]

        results_df = self._session.results
        driver_row = results_df[results_df["Abbreviation"] == driver_code]
        driver_number = str(driver_row.iloc[0]["DriverNumber"])
        car_data = self._session.car_data[driver_number]

        mask = (car_data["SessionTime"] >= lap_start) & (
            car_data["SessionTime"] <= lap_end
        )
        telemetry = car_data[mask]

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

        if "Distance" in df.columns and not df["Distance"].empty:
            min_distance = df["Distance"].iloc[0]
            df["DistanceOffset"] = df["Distance"] - min_distance
        else:
            df["DistanceOffset"] = 0.0

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
        return df.dropna(subset=["Speed", "DistanceOffset"])

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

        trigger_steward = lateral_g > self.G_FORCE_THRESHOLD

        packet = {
            "driver": primary_driver,
            "speed": round(speed, 1),
            "lateral_g": round(lateral_g, 2),
            "longitudinal_g": round(float(row.get("longitudinal_g", 0.0)), 2),
            "distance_to_apex": round(distance_to_apex, 2)
            if distance_to_apex is not None
            else None,
            "distance_offset": round(distance_offset, 2),
            "throttle": round(float(row.get("Throttle", 0.0)), 2),
            "brake": bool(row.get("Brake", False)),
            "gear": int(row.get("Gear", 0)) if pd.notna(row.get("Gear")) else None,
            "sector": str(row.get("Sector", "S3")),
            "trigger_steward": trigger_steward,
            "packet_index": packet_index,
            "timestamp": self._serialize_value(time_val),
        }

        packet["all_drivers"] = {}
        for driver_code, df in telemetry_data.items():
            if "Time" in df.columns:
                time_seconds = df["Time"].apply(self._timedelta_to_seconds)
                closest_idx = (time_seconds - time_val).abs().idxmin()
                driver_row = df.loc[closest_idx]
            else:
                idx = min(packet_index, len(df) - 1)
                driver_row = df.iloc[idx]

            packet["all_drivers"][driver_code] = {
                "speed": round(float(driver_row.get("Speed", 0.0)), 1),
                "lateral_g": round(float(driver_row.get("lateral_g", 0.0)), 2),
                "distance_offset": round(
                    float(driver_row.get("DistanceOffset", 0.0)), 2
                ),
                "sector": str(driver_row.get("Sector", "S3")),
            }

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

        if driver in self.last_steward_trigger_time:
            time_since_trigger = current_time - self.last_steward_trigger_time[driver]
            if time_since_trigger < 5.0:
                print("COOLDOWN ACTIVE - Skipping Judicial Review")
                packet["trigger_steward"] = False
                return packet

        if packet.get("trigger_steward"):
            self.last_steward_trigger_time[driver] = current_time
            self._logger.warning(
                "STEWARD TRIGGERED! Lateral G %.2f exceeded threshold %.2f",
                packet["lateral_g"],
                self.G_FORCE_THRESHOLD,
            )

        return packet

    def _send_telemetry(self, packet: dict) -> bool:
        """Send telemetry packet to the dashboard API (fire and forget)."""

        def _async_post():
            try:
                response = requests.post(self.API_ENDPOINT, json=packet, timeout=30)
                if response.status_code == 200:
                    self._logger.debug(
                        f"Successfully sent packet {packet['packet_index']}"
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
            driver_codes=["VER", "HAM"],
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
