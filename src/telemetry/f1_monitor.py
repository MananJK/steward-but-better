"""F1 Telemetry Extractor for incident data analysis.

This module provides functionality to extract high-precision telemetry data from
fastf1 for specific race incidents, supporting multi-car comparisons through
distance offset calculation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import fastf1
import pandas as pd

from dotenv import load_dotenv
load_dotenv(override=True)


class TelemetryExtractor:
    """Extracts high-precision telemetry data from F1 sessions."""

    def __init__(self, cache_enabled: bool = True) -> None:
        """Initialize the telemetry extractor.

        Args:
            cache_enabled: Whether to enable fastf1's cache for session data.
        """
        self._logger = logging.getLogger(__name__)
        if cache_enabled:
            cache_dir = Path("f1_cache")
            cache_dir.mkdir(exist_ok=True)
            fastf1.Cache.enable_cache(cache_dir)
        self._session = None

    def get_incident_data(
        self,
        year: int,
        race_name: str,
        session_type: str,
        driver_code: str,
        lap_number: int,
    ) -> pd.DataFrame:
        """Extract telemetry for a specific driver on a specific lap.

        Args:
            year: Championship year (e.g., 2021).
            race_name: Race name (e.g., 'Abu Dhabi').
            session_type: Session type - 'FP1', 'FP2', 'FP3', 'Q', 'R', 'SQ'.
            driver_code: Three-letter driver code (e.g., 'VER', 'HAM').
            lap_number: Lap number to extract telemetry from.

        Returns:
            DataFrame containing high-precision telemetry with distance offset.
        """
        self._logger.info(
            "Loading session: %d %s %s for driver %s lap %d",
            year,
            race_name,
            session_type,
            driver_code,
            lap_number,
        )

        self._session = fastf1.get_session(year, race_name, session_type)
        self._session.load()

        driver_laps = self._session.laps.pick_drivers(driver_code)
        if driver_laps.empty:
            raise ValueError(f"No laps found for driver {driver_code} in session")

        target_lap = driver_laps[driver_laps["LapNumber"] == lap_number]
        if target_lap.empty:
            raise ValueError(
                f"Lap {lap_number} not found for driver {driver_code}. "
                f"Available laps: {driver_laps['LapNumber'].min()}-{driver_laps['LapNumber'].max()}"
            )

        lap_row = target_lap.iloc[0]
        lap_start = lap_row["LapStartTime"]
        lap_end = lap_row["Time"]

        results_df = self._session.results
        driver_row = results_df[results_df["Abbreviation"] == driver_code]
        if driver_row.empty:
            raise ValueError(f"Driver {driver_code} not found in session results")
        driver_number = str(driver_row.iloc[0]["DriverNumber"])
        car_data = self._session.car_data[driver_number]

        if car_data.empty:
            raise ValueError(
                f"No car data found for driver {driver_code} (number {driver_number})"
            )

        mask = (car_data["SessionTime"] >= lap_start) & (
            car_data["SessionTime"] <= lap_end
        )
        telemetry = car_data[mask]

        if telemetry.empty:
            raise ValueError(f"No telemetry data for lap {lap_number}")

        telemetry_df = self._extract_high_precision_telemetry(telemetry)
        telemetry_df = self._calculate_distance_offset(telemetry_df)

        telemetry_df["Year"] = year
        telemetry_df["RaceName"] = race_name
        telemetry_df["SessionType"] = session_type
        telemetry_df["DriverCode"] = driver_code
        telemetry_df["LapNumber"] = lap_number

        self._logger.info(
            "Extracted %d telemetry points for %s lap %d",
            len(telemetry_df),
            driver_code,
            lap_number,
        )

        return telemetry_df

    def _extract_high_precision_telemetry(
        self, telemetry: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract high-precision telemetry channels."""
        channels = ["Speed", "Throttle", "Brake", "nGear", "DRS"]
        rename_map = {"nGear": "Gear"}

        data = {}
        for channel in channels:
            if channel in telemetry.columns:
                data[channel] = telemetry[channel].values
            else:
                self._logger.warning("Channel %s not found in telemetry data", channel)
                data[channel] = [None] * len(telemetry)

        df = pd.DataFrame(data)
        df = df.rename(columns=rename_map)

        if "Time" in telemetry.columns:
            df["Time"] = telemetry["Time"].values

        if "Distance" in telemetry.columns:
            df["Distance"] = telemetry["Distance"].values
        elif "Speed" in df.columns and len(df["Speed"]) > 0:
            df["Distance"] = self._calculate_distance_from_speed(df["Speed"])

        df = df.dropna(subset=["Speed", "Distance"])

        return df

    def _calculate_distance_from_speed(self, speed: pd.Series) -> pd.Series:
        """Calculate approximate distance from speed data using cumulative sum.

        This is a fallback when direct distance data is unavailable.
        Assumes equal time intervals between samples.
        """
        speed_ms = speed / 3.6
        time_delta = 1.0 / 20.0
        distance_delta = speed_ms * time_delta
        return distance_delta.cumsum()

    def _calculate_distance_offset(self, telemetry_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate distance offset relative to first data point.

        This normalizes distance to start from 0, enabling multi-car comparisons
        across different laps and sessions.
        """
        if "Distance" in telemetry_df.columns and not telemetry_df["Distance"].empty:
            min_distance = telemetry_df["Distance"].iloc[0]
            telemetry_df["DistanceOffset"] = telemetry_df["Distance"] - min_distance
        else:
            telemetry_df["DistanceOffset"] = 0.0

        return telemetry_df

    def save_to_parquet(self, dataframe: pd.DataFrame, filename: str) -> Path:
        """Save telemetry DataFrame to parquet file.

        Args:
            dataframe: Telemetry DataFrame to save.
            filename: Output filename (with or without .parquet extension).

        Returns:
            Path to the saved file.
        """
        output_path = Path(filename)
        if output_path.suffix != ".parquet":
            output_path = output_path.with_suffix(".parquet")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        dataframe.to_parquet(output_path, index=False)

        self._logger.info("Saved telemetry data to %s", output_path)

        return output_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    extractor = TelemetryExtractor(cache_enabled=True)

    print("\n" + "=" * 60)
    print("F1 Telemetry Validation Test")
    print("=" * 60)

    try:
        telemetry_data = extractor.get_incident_data(
            year=2021,
            race_name="Abu Dhabi",
            session_type="R",
            driver_code="VER",
            lap_number=58,
        )

        print(f"\nExtracted {len(telemetry_data)} telemetry points")
        print(f"Columns: {list(telemetry_data.columns)}")
        print(
            f"\nSpeed range: {telemetry_data['Speed'].min():.1f} - {telemetry_data['Speed'].max():.1f} km/h"
        )
        print(
            f"Throttle range: {telemetry_data['Throttle'].min():.0%} - {telemetry_data['Throttle'].max():.0%}"
        )
        print(f"Brake events: {telemetry_data['Brake'].sum()}")
        print(
            f"Gear range: {telemetry_data['Gear'].min():.0f} - {telemetry_data['Gear'].max():.0f}"
        )

        output_path = extractor.save_to_parquet(
            telemetry_data, "verstappen_abu_dhabi_2021_lap58.parquet"
        )
        print(f"\nSaved to: {output_path}")

        print("\nFirst 5 rows:")
        print(telemetry_data.head().to_string())

        print("\nLast 5 rows:")
        print(telemetry_data.tail().to_string())

        print("\n" + "=" * 60)
        print("Validation PASSED")
        print("=" * 60)

    except Exception as e:
        print(f"\nValidation FAILED: {e}")
        raise
