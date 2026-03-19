"""
src/data_engineering/telemetry_processor.py
=============================================
Lap normalization and per-lap telemetry aggregation.

Engineering responsibility:
    Take a raw FastF1 Session object and produce a clean, analysis-ready
    per-lap DataFrame. This module is the quality gate between raw API data
    and every downstream model. Its job is not to model anything — it is to
    produce a trustworthy, schema-consistent dataset.

Why SC/VSC laps are flagged (not dropped):
    Safety car laps are flagged rather than deleted because they serve two
    different purposes downstream:
        1. Degradation modelling: SC laps MUST be excluded — the car is not
           pushing, tyre temperatures drop, and the lap time is meaningless
           as a degradation signal.
        2. Race simulation: SC laps MUST be included — they affect total
           race time, pit stop timing windows, and gap calculations.
    Dropping them from the raw dataset would force both modules to re-derive
    them from scratch. Flagging preserves optionality.

Why in-laps and out-laps are flagged:
    The in-lap (pit entry lap) is slow in sector 3 due to the pit entry
    manoeuvre — the driver lifts after the last corner. The out-lap is slow
    because the tyres are cold and the driver is building operating temperature.
    Both are NOT representative of steady-state tyre degradation and must be
    excluded from the degradation regression. But both contribute real time to
    total race duration and must be included in simulation.

Lap normalization pipeline:
    1. Assert column presence (fail fast)
    2. Convert LapTime timedelta → float seconds
    3. Hard-floor filter (remove physically impossible lap times)
    4. Flag: is_pit_entry_lap, is_pit_exit_lap, is_sc_lap, is_vsc_lap,
             is_anomalous_lap
    5. Derive stint_number and tyre_age (with FastF1 native fallback)
    6. Normalise compound names
    7. Compute is_representative composite flag
    8. Return clean DataFrame with consistent snake_case schema
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import numpy as np

from src.constants import (
    ALL_CANONICAL_COMPOUNDS,
    MIN_RACE_LAP_TIME_SEC,
    SLOW_LAP_MULTIPLIER,
    TYRE_LIFE_TRUST_THRESHOLD,
    MIN_TYRE_AGE_FOR_REGRESSION,
    TRACK_STATUS_SC,
    TRACK_STATUS_VSC,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Constants local to this module
# ===========================================================================

# Columns required to exist in the FastF1 laps DataFrame.
# If any are absent the session data is too incomplete to process.
REQUIRED_LAP_COLUMNS: set[str] = {
    "Driver", "DriverNumber", "LapNumber", "LapTime",
    "PitInTime", "PitOutTime", "TrackStatus",
}

# Optional columns used if present; gracefully absent if not.
OPTIONAL_LAP_COLUMNS: set[str] = {
    "Compound", "TyreLife", "FreshTyre", "Stint",
    "IsPersonalBest", "Sector1Time", "Sector2Time", "Sector3Time",
    "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
}


# ===========================================================================
# Internal helpers
# ===========================================================================

def _assert_required_columns(
    df: pd.DataFrame,
    required: set[str],
    context: str,
) -> None:
    """
    Raise a descriptive ValueError if any required columns are absent.

    Args:
        df:       DataFrame to check.
        required: Set of column names that must exist.
        context:  Calling function name for error context.

    Raises:
        ValueError: Lists all missing columns and all available columns.
    """
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"[{context}] FastF1 laps DataFrame missing required columns: "
            f"{sorted(missing)}.\n"
            f"Available columns: {sorted(df.columns.tolist())}\n"
            f"This may indicate an incomplete FastF1 load or an unsupported "
            f"session type."
        )


def _track_status_contains(series: pd.Series, code: str) -> pd.Series:
    """
    Test whether a FastF1 TrackStatus string contains a given status code.

    FastF1 encodes track status as a concatenated string of active codes per
    lap. e.g. "14" means codes "1" (green) and "4" (SC) were both active
    during that lap. We use substring search, NOT equality, because multiple
    codes can be active simultaneously.

    Args:
        series: Series of TrackStatus strings (may contain NaN).
        code:   Single-character status code to search for.

    Returns:
        Boolean Series — True where the code is present.
    """
    return series.apply(
        lambda s: code in str(s) if pd.notna(s) else False
    )


def _normalise_compound(series: pd.Series) -> pd.Series:
    """
    Normalise FastF1 Compound strings to canonical uppercase names.

    FastF1 generally returns uppercase compound names but can return NaN,
    lowercase variations, or non-standard strings for laps where compound
    data was not telemetered. Any unrecognised value becomes "UNKNOWN".

    Args:
        series: Raw Compound column from FastF1 laps DataFrame.

    Returns:
        Normalised compound Series with values from ALL_CANONICAL_COMPOUNDS
        or "UNKNOWN".
    """
    normalised = series.str.upper().str.strip()
    return normalised.apply(
        lambda c: c if c in ALL_CANONICAL_COMPOUNDS else "UNKNOWN"
    )


def _flag_lap_quality(laps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Append five boolean quality-flag columns to a laps DataFrame.

    All flag column names use snake_case — distinct from FastF1's PascalCase
    native columns, making provenance immediately visible in any DataFrame.

    Flags (True = condition is present):
        is_pit_entry_lap : Lap on which car entered pit lane.
                           FastF1 sets PitInTime on this lap.
        is_pit_exit_lap  : Lap on which car exited pit lane (first lap of
                           new stint). FastF1 sets PitOutTime on this lap.
                           Excluded from degradation: tyres are cold.
        is_sc_lap        : Lap run fully or partially under Safety Car.
                           TrackStatus contains code "4".
                           Excluded from degradation: car not pushing,
                           tyre thermal state undefined.
        is_vsc_lap       : Lap run under Virtual Safety Car.
                           TrackStatus contains code "6".
                           Excluded for same reason as SC laps.
        is_anomalous_lap : Lap time > SLOW_LAP_MULTIPLIER × session best.
                           Covers: red flags, driver errors, traffic incidents,
                           formation laps accidentally included.

    Args:
        laps_df: Must contain PitInTime, PitOutTime, TrackStatus, LapTime.

    Returns:
        New DataFrame with five additional boolean columns. Input not mutated.

    Raises:
        ValueError: If required columns are absent.
    """
    _assert_required_columns(
        laps_df,
        required={"PitInTime", "PitOutTime", "TrackStatus", "LapTime"},
        context="_flag_lap_quality",
    )

    df = laps_df.copy()

    df["is_pit_entry_lap"] = df["PitInTime"].notna()
    df["is_pit_exit_lap"]  = df["PitOutTime"].notna()
    df["is_sc_lap"]        = _track_status_contains(df["TrackStatus"], TRACK_STATUS_SC)
    df["is_vsc_lap"]       = _track_status_contains(df["TrackStatus"], TRACK_STATUS_VSC)

    lap_times_sec = df["LapTime"].dt.total_seconds()
    valid_times   = lap_times_sec.dropna()

    if valid_times.empty:
        logger.warning(
            "_flag_lap_quality: no valid LapTime values — "
            "is_anomalous_lap set to False for all rows."
        )
        df["is_anomalous_lap"] = False
    else:
        session_best_sec = valid_times.min()
        threshold_sec    = session_best_sec * SLOW_LAP_MULTIPLIER
        df["is_anomalous_lap"] = lap_times_sec > threshold_sec
        logger.debug(
            "_flag_lap_quality: slow-lap threshold=%.3fs "
            "(%.1fx session best %.3fs)  anomalous=%d laps",
            threshold_sec, SLOW_LAP_MULTIPLIER, session_best_sec,
            int(df["is_anomalous_lap"].sum()),
        )

    return df


def _derive_stint_metadata(laps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive stint_number and tyre_age columns, preferring FastF1 native data.

    FastF1 provides Stint and TyreLife columns but they can contain NaN
    for laps with missing telemetry. We use them when coverage is high
    (>= TYRE_LIFE_TRUST_THRESHOLD) and derive them ourselves otherwise.

    Derivation logic (fallback path):
        stint_number: cumulative count of pit-exit laps per driver, clipped
                      to a minimum of 1 so the opening stint is stint 1.
        tyre_age:     cumulative lap count within (driver, stint_number),
                      1-indexed so the pit-exit lap is tyre_age=1.

    Args:
        laps_df: Must contain Driver, LapNumber, is_pit_exit_lap.

    Returns:
        New DataFrame with stint_number (int) and tyre_age (int) appended.
    """
    _assert_required_columns(
        laps_df,
        required={"Driver", "LapNumber", "is_pit_exit_lap"},
        context="_derive_stint_metadata",
    )

    df = laps_df.copy().sort_values(["Driver", "LapNumber"])

    # --- stint_number ---
    native_stint_ok = (
        "Stint" in df.columns and df["Stint"].notna().all()
    )
    if native_stint_ok:
        df["stint_number"] = df["Stint"].astype(int)
        logger.debug("stint_number: using FastF1 native Stint column.")
    else:
        logger.debug(
            "stint_number: deriving from cumulative pit-exit count per driver."
        )
        df["stint_number"] = (
            df.groupby("Driver")["is_pit_exit_lap"]
            .transform("cumsum")
            .astype(int)
            .clip(lower=1)
        )

    # --- tyre_age ---
    tyre_life_coverage = (
        df["TyreLife"].notna().mean()
        if "TyreLife" in df.columns else 0.0
    )
    if tyre_life_coverage >= TYRE_LIFE_TRUST_THRESHOLD:
        df["tyre_age"] = df["TyreLife"].astype("Int64")
        logger.debug(
            "tyre_age: using FastF1 TyreLife (coverage=%.1f%%).",
            tyre_life_coverage * 100,
        )
    else:
        logger.debug(
            "tyre_age: deriving from lap count within each stint "
            "(TyreLife coverage=%.1f%% < %.0f%% threshold).",
            tyre_life_coverage * 100,
            TYRE_LIFE_TRUST_THRESHOLD * 100,
        )
        df["tyre_age"] = (
            df.groupby(["Driver", "stint_number"])
            .cumcount()
            .add(1)
            .astype(int)
        )

    return df


# ===========================================================================
# Public API
# ===========================================================================

def process_laps(
    session: "fastf1.core.Session",
) -> pd.DataFrame:
    """
    Process a FastF1 session into a clean per-lap DataFrame.

    This is the primary public function of this module. It is the single
    transform step between raw FastF1 data and everything downstream.

    Output schema (all module-produced columns are snake_case):
        driver_code        : str   — 3-letter driver abbreviation (e.g. "VER")
        driver_number      : str   — car number string (e.g. "1")
        lap_number         : int   — race lap number (1-indexed)
        lap_time_sec       : float — lap time in seconds (NaN if unrecorded)
        compound           : str   — canonical compound name or "UNKNOWN"
        tyre_age           : int   — laps completed on current set (1 = fresh)
        stint_number       : int   — stint index (1 = first stint)
        is_new_tyre        : bool  — True if tyre fitted new (not pre-used)
        track_status       : str   — raw FastF1 TrackStatus string
        is_pit_entry_lap   : bool  — see _flag_lap_quality
        is_pit_exit_lap    : bool  — see _flag_lap_quality
        is_sc_lap          : bool  — see _flag_lap_quality
        is_vsc_lap         : bool  — see _flag_lap_quality
        is_anomalous_lap   : bool  — see _flag_lap_quality
        is_representative  : bool  — True iff ALL flags above are False.
                                     Primary filter for degradation modelling.

    Args:
        session: Fully loaded FastF1 Session object from fastf1_loader.py.

    Returns:
        DataFrame sorted by (driver_code, lap_number).

    Raises:
        ValueError: If session contains no lap data or required columns absent.
        RuntimeError: If the processed dataset is empty after filtering.
    """
    import fastf1.core  # Local import to avoid circular dependency at module level

    laps = session.laps

    if laps is None or laps.empty:
        raise ValueError(
            f"Session '{session.event.get('EventName', 'unknown')}' "
            f"contains no lap data. Ensure session.load(laps=True) was called."
        )

    event_name = session.event.get("EventName", "unknown")
    logger.info(
        "process_laps: event='%s'  raw_laps=%d", event_name, len(laps)
    )

    _assert_required_columns(laps, REQUIRED_LAP_COLUMNS, "process_laps")

    # Pull all needed columns (required + available optional)
    available_optional = OPTIONAL_LAP_COLUMNS & set(laps.columns)
    working_cols = list(REQUIRED_LAP_COLUMNS | available_optional)
    df = laps[working_cols].copy()

    # Convert timedelta → seconds (NaT → NaN automatically)
    df["lap_time_sec"] = df["LapTime"].dt.total_seconds()

    # Hard-floor filter: remove physically impossible lap times
    n_before = len(df)
    df = df[
        df["lap_time_sec"].isna()
        | (df["lap_time_sec"] >= MIN_RACE_LAP_TIME_SEC)
    ].copy()
    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.warning(
            "process_laps: removed %d laps with lap_time_sec < %.1fs "
            "(likely timing artefacts from red flag or session reset).",
            n_removed, MIN_RACE_LAP_TIME_SEC,
        )

    # Quality flags
    df = _flag_lap_quality(df)

    # Stint metadata
    df = _derive_stint_metadata(df)

    # Compound normalisation
    if "Compound" in df.columns:
        df["compound"] = _normalise_compound(df["Compound"])
        n_unknown = int((df["compound"] == "UNKNOWN").sum())
        if n_unknown > 0:
            logger.warning(
                "process_laps: %d laps assigned compound='UNKNOWN' "
                "(FastF1 telemetry gap or unrecognised compound string).",
                n_unknown,
            )
    else:
        logger.warning(
            "process_laps: Compound column absent — all laps set to 'UNKNOWN'."
        )
        df["compound"] = "UNKNOWN"

    # is_new_tyre
    if "FreshTyre" in df.columns:
        df["is_new_tyre"] = df["FreshTyre"].fillna(True).astype(bool)
    else:
        df["is_new_tyre"] = True

    # Composite representative flag
    df["is_representative"] = ~(
        df["is_pit_entry_lap"]
        | df["is_pit_exit_lap"]
        | df["is_sc_lap"]
        | df["is_vsc_lap"]
        | df["is_anomalous_lap"]
    )

    # Rename FastF1 PascalCase → our snake_case convention
    df = df.rename(columns={
        "Driver":       "driver_code",
        "DriverNumber": "driver_number",
        "LapNumber":    "lap_number",
        "TrackStatus":  "track_status",
    })

    df["lap_number"] = df["lap_number"].astype(int)

    output_cols = [
        "driver_code", "driver_number", "lap_number", "lap_time_sec",
        "compound", "tyre_age", "stint_number", "is_new_tyre",
        "track_status",
        "is_pit_entry_lap", "is_pit_exit_lap",
        "is_sc_lap", "is_vsc_lap", "is_anomalous_lap", "is_representative",
    ]
    # Only include output columns that exist
    output_cols = [c for c in output_cols if c in df.columns]

    result = (
        df[output_cols]
        .sort_values(["driver_code", "lap_number"])
        .reset_index(drop=True)
    )

    if result.empty:
        raise RuntimeError(
            f"process_laps: processed dataset for '{event_name}' is empty. "
            "Check session data quality."
        )

    repr_pct = result["is_representative"].mean() * 100
    logger.info(
        "process_laps: complete | rows=%d  representative=%.1f%%  "
        "sc_laps=%d  vsc_laps=%d  anomalous=%d  unknown_compound=%d",
        len(result),
        repr_pct,
        int(result["is_sc_lap"].sum()),
        int(result["is_vsc_lap"].sum()),
        int(result["is_anomalous_lap"].sum()),
        int((result["compound"] == "UNKNOWN").sum()),
    )

    if repr_pct < 40.0:
        logger.warning(
            "process_laps: only %.1f%% of laps are representative. "
            "Check for extended SC/VSC periods or session data quality issues.",
            repr_pct,
        )

    return result


def process_pit_stops(
    session: "fastf1.core.Session",
) -> pd.DataFrame:
    """
    Reconstruct pit stop events from FastF1 lap timing data.

    FastF1 does not expose stationary time directly. We approximate total
    pit lane time as: PitOutTime(lap N+1) − PitInTime(lap N).
    This captures entry + stationary + exit — the total time delta vs
    staying out — which is what the race simulator needs.

    Unmatched inlaps (retirement in pit lane, final-lap stop) are preserved
    with NaN stop_duration_sec and logged explicitly by driver.

    Output schema:
        driver_code       : str
        pit_lap           : int        — lap on which car entered pit lane
        pit_in_time       : Timedelta  — session elapsed time at pit entry
        pit_out_time      : Timedelta  — session elapsed time at pit exit (NaT if retirement)
        stop_duration_sec : float      — total pit lane time in seconds (NaN if unmatched)

    Args:
        session: Fully loaded FastF1 Session object.

    Returns:
        DataFrame sorted by (driver_code, pit_lap).
    """
    laps = session.laps
    EMPTY = pd.DataFrame(columns=[
        "driver_code", "pit_lap", "pit_in_time",
        "pit_out_time", "stop_duration_sec",
    ])

    if "PitInTime" not in laps.columns:
        logger.warning(
            "process_pit_stops: PitInTime absent from session — "
            "returning empty DataFrame."
        )
        return EMPTY

    pit_entry = (
        laps[laps["PitInTime"].notna()][["Driver", "LapNumber", "PitInTime"]]
        .copy()
        .rename(columns={
            "Driver": "driver_code",
            "LapNumber": "pit_lap",
            "PitInTime": "pit_in_time",
        })
    )
    pit_entry["pit_lap"] = pit_entry["pit_lap"].astype(int)

    if "PitOutTime" not in laps.columns:
        logger.warning(
            "process_pit_stops: PitOutTime absent — "
            "stop_duration_sec will be NaN for all events."
        )
        pit_entry["pit_out_time"]      = pd.NaT
        pit_entry["stop_duration_sec"] = float("nan")
        return pit_entry.sort_values(["driver_code", "pit_lap"]).reset_index(drop=True)

    pit_exit = (
        laps[laps["PitOutTime"].notna()][["Driver", "LapNumber", "PitOutTime"]]
        .copy()
        .rename(columns={
            "Driver": "driver_code",
            "LapNumber": "out_lap",
            "PitOutTime": "pit_out_time",
        })
    )
    pit_exit["out_lap"] = pit_exit["out_lap"].astype(int)
    pit_entry["out_lap"] = pit_entry["pit_lap"] + 1

    merged = pit_entry.merge(pit_exit, on=["driver_code", "out_lap"], how="left")

    unmatched = merged["pit_out_time"].isna()
    if unmatched.any():
        logger.warning(
            "process_pit_stops: %d unmatched pit entries "
            "(retirement in pit lane or final-lap stop). Drivers: %s",
            int(unmatched.sum()),
            merged.loc[unmatched, "driver_code"].tolist(),
        )

    merged["stop_duration_sec"] = (
        merged["pit_out_time"] - merged["pit_in_time"]
    ).dt.total_seconds()

    result = (
        merged[["driver_code", "pit_lap", "pit_in_time",
                "pit_out_time", "stop_duration_sec"]]
        .sort_values(["driver_code", "pit_lap"])
        .reset_index(drop=True)
    )

    valid = result["stop_duration_sec"].dropna()
    logger.info(
        "process_pit_stops: %d stops | drivers=%d | "
        "mean=%.1fs  min=%.1fs  max=%.1fs",
        len(result),
        result["driver_code"].nunique(),
        valid.mean() if not valid.empty else float("nan"),
        valid.min()  if not valid.empty else float("nan"),
        valid.max()  if not valid.empty else float("nan"),
    )
    return result


def process_driver_info(
    session: "fastf1.core.Session",
) -> pd.DataFrame:
    """
    Extract driver and constructor metadata for all session drivers.

    Output schema:
        driver_code   : str — 3-letter abbreviation (e.g. "VER")
        driver_number : str — car number string (e.g. "1")
        full_name     : str — "FirstName LastName"
        team_name     : str — constructor name (e.g. "Red Bull Racing")

    Args:
        session: Fully loaded FastF1 Session object.

    Returns:
        DataFrame with one row per driver.

    Raises:
        RuntimeError: If zero drivers could be retrieved (systemic failure).
    """
    records, failed = [], []

    for number in session.drivers:
        try:
            info = session.get_driver(number)
            records.append({
                "driver_code":   info.get("Abbreviation", "UNK"),
                "driver_number": str(number),
                "full_name":     (
                    f"{info.get('FirstName', '')} "
                    f"{info.get('LastName', '')}"
                ).strip(),
                "team_name":     info.get("TeamName", "Unknown"),
            })
        except Exception as exc:
            failed.append(number)
            logger.warning(
                "process_driver_info: failed for driver #%s — %s",
                number, exc,
            )

    if not records:
        raise RuntimeError(
            "process_driver_info: could not retrieve info for any driver. "
            "The session data may be corrupt or incomplete."
        )
    if failed:
        logger.warning(
            "process_driver_info: partial failure — %d/%d drivers missing. "
            "Numbers: %s",
            len(failed), len(session.drivers), failed,
        )

    result = pd.DataFrame(records)
    logger.info(
        "process_driver_info: %d/%d drivers retrieved.",
        len(result), len(session.drivers),
    )
    return result
