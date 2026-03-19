"""
src/data_engineering/feature_builder.py
=========================================
Feature engineering for the degradation model and ML optimizer.

Engineering responsibility:
    Transform the clean per-lap DataFrame from telemetry_processor.py into
    a feature-rich dataset that captures the signals relevant to tyre
    degradation modelling, pace prediction, and strategy classification.

Feature engineering rationale:
    Raw lap times are not useful for strategy modelling — they confound
    five separate physical signals that must be decomposed:

    1. DELTA LAP TIME VS THEORETICAL BEST
       The theoretical best lap time is the fastest any driver achieved
       on that race lap (minimum across all representative laps for that
       lap number). Expressing each lap as a delta to this baseline
       removes absolute pace differences between cars and circuits,
       making data comparable across events.

    2. PACE DROP PER LAP (compound-level degradation signal)
       For each driver-stint, the per-lap pace change vs the previous lap
       on the same tyre set. Aggregated by compound using median across
       all stints. This is the raw degradation rate before fuel correction.

    3. FUEL-CORRECTED LAP TIME
       Each lap time is reduced by the fuel load penalty:
           penalty(lap_n) = (total_laps - n) × burn_rate × sensitivity
       After correction, the remaining lap time variance is attributable
       to tyre state and track evolution — the signals we want to model.

    4. STINT POSITION (tyre_age as fraction of expected stint length)
       Normalised tyre age: tyre_age / expected_max_stint_age.
       This makes the degradation curve comparable across stints of
       different expected lengths (a 20-lap soft stint vs a 30-lap
       medium stint), enabling cross-compound model training.

    5. TRACK EVOLUTION COEFFICIENT
       The per-lap minimum corrected lap time across all drivers.
       This captures the circuit-wide rubber build-up signal.
       Normalised to 0 at lap 1 (no evolution at race start).
       Subtracted from corrected lap times to isolate tyre degradation.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.constants import (
    FUEL_BURN_RATE_KG_PER_LAP,
    FUEL_LAP_TIME_SENSITIVITY_SEC_PER_KG,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Constants
# ===========================================================================

# Number of early race laps used to fit the track evolution trend.
# We restrict to early laps where all drivers are on relatively fresh
# tyres and compound mix effects are minimal — cleaner evolution signal.
TRACK_EVOLUTION_FIT_LAPS: int = 20

# Maximum plausible track evolution rate (s/lap).
# A rate faster than this indicates a data artefact (SC period, red flag).
# Clamped to zero to avoid over-correction.
MAX_TRACK_EVOLUTION_RATE_SEC_PER_LAP: float = 0.20

# Expected stint lengths per compound (laps) at a typical circuit.
# Used only for stint_position_normalised calculation.
# These are first-order estimates — the degradation model refines them.
EXPECTED_STINT_LENGTH: dict[str, int] = {
    "SOFT":         18,
    "MEDIUM":       28,
    "HARD":         38,
    "INTERMEDIATE": 25,
    "WET":          20,
    "UNKNOWN":      25,
}

# Required input columns from telemetry_processor.process_laps().
REQUIRED_INPUT_COLUMNS: set[str] = {
    "driver_code", "lap_number", "lap_time_sec",
    "compound", "tyre_age", "stint_number", "is_representative",
}


# ===========================================================================
# Individual feature functions
# ===========================================================================

def add_fuel_corrected_lap_time(
    laps: pd.DataFrame,
    total_race_laps: int,
) -> pd.DataFrame:
    """
    Compute fuel-corrected lap time and add as a new column.

    Engineering rationale:
        A car carrying 105 kg of fuel at race start is ~3.7s slower than
        on empty tanks (105 kg × 0.035 s/kg). As fuel burns at 1.8 kg/lap,
        the car naturally gets faster. Without removing this trend, early-stint
        degradation is underestimated and late-stint is overestimated.

        Correction: subtract the fuel load PENALTY at each lap.
        penalty(n) = fuel_remaining(n) × sensitivity
        fuel_remaining(n) = max(0, (total_laps - n) × burn_rate)

        After correction, the lap time represents what the car would have
        done with empty tanks — enabling fair comparison across all laps.

    Args:
        laps:            per-lap DataFrame from telemetry_processor.
        total_race_laps: Scheduled race distance (for fuel burn calculation).

    Returns:
        DataFrame with new column: fuel_corrected_lap_sec.
    """
    df = laps.copy()

    fuel_remaining = np.maximum(
        0.0,
        (total_race_laps - df["lap_number"].astype(float))
        * FUEL_BURN_RATE_KG_PER_LAP,
    )
    fuel_penalty = fuel_remaining * FUEL_LAP_TIME_SENSITIVITY_SEC_PER_KG

    df["fuel_corrected_lap_sec"] = df["lap_time_sec"] - fuel_penalty

    logger.debug(
        "add_fuel_corrected_lap_time: fuel penalty range [%.3f, %.3f]s "
        "over laps [%d, %d]",
        float(fuel_penalty.min()), float(fuel_penalty.max()),
        int(df["lap_number"].min()), int(df["lap_number"].max()),
    )
    return df


def add_track_evolution_coefficient(
    laps: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate and subtract the track evolution trend from fuel-corrected laps.

    Engineering rationale:
        As 20 cars lay rubber over 57 laps, grip improves progressively.
        This benefits every compound equally — it is not a tyre degradation
        signal but a circuit-state signal. If not removed, all compounds
        appear to degrade less than they actually do.

    Method:
        Per race lap, take the minimum fuel_corrected_lap_sec across all
        drivers (representative laps only). This per-lap minimum proxies
        the cleanest available lap time that lap — track state dominated.
        Fit a linear regression through the first TRACK_EVOLUTION_FIT_LAPS
        of these minima. Extrapolate the trend to all laps.
        Subtract (trend - trend_at_lap_1) from each corrected lap time.

    Result columns:
        track_evolution_sec:     Estimated evolution improvement at each lap.
        evolution_corrected_lap_sec: Fuel + evolution corrected lap time.

    Args:
        laps: Must contain fuel_corrected_lap_sec and lap_number.

    Returns:
        DataFrame with two new columns added.
    """
    if "fuel_corrected_lap_sec" not in laps.columns:
        raise ValueError(
            "add_track_evolution_coefficient requires fuel_corrected_lap_sec. "
            "Call add_fuel_corrected_lap_time() first."
        )

    df = laps.copy()

    # Per-lap minimum across representative laps only
    repr_laps = df[df["is_representative"]].copy()
    per_lap_min = (
        repr_laps.groupby("lap_number")["fuel_corrected_lap_sec"]
        .min()
        .reset_index()
        .rename(columns={"fuel_corrected_lap_sec": "min_corrected"})
    )

    fit_data = per_lap_min[
        per_lap_min["lap_number"] <= TRACK_EVOLUTION_FIT_LAPS
    ].dropna()

    if len(fit_data) < 3:
        logger.warning(
            "add_track_evolution_coefficient: only %d laps for evolution fit "
            "— skipping correction (evolution_corrected_lap_sec = fuel_corrected_lap_sec).",
            len(fit_data),
        )
        df["track_evolution_sec"]        = 0.0
        df["evolution_corrected_lap_sec"] = df["fuel_corrected_lap_sec"]
        return df

    X = fit_data["lap_number"].values.reshape(-1, 1)
    y = fit_data["min_corrected"].values
    reg = LinearRegression().fit(X, y)
    rate = float(reg.coef_[0])

    if abs(rate) > MAX_TRACK_EVOLUTION_RATE_SEC_PER_LAP:
        logger.warning(
            "add_track_evolution_coefficient: evolution rate %.4f s/lap "
            "exceeds max %.2f s/lap — likely a data artefact. Clamping to 0.",
            rate, MAX_TRACK_EVOLUTION_RATE_SEC_PER_LAP,
        )
        df["track_evolution_sec"]        = 0.0
        df["evolution_corrected_lap_sec"] = df["fuel_corrected_lap_sec"]
        return df

    # Normalise: evolution = 0 at lap 1
    baseline    = float(reg.predict(np.array([[1]]))[0])
    predictions = reg.predict(df["lap_number"].values.reshape(-1, 1)).flatten()
    evolution   = predictions - baseline

    df["track_evolution_sec"]         = evolution
    df["evolution_corrected_lap_sec"] = df["fuel_corrected_lap_sec"] - evolution

    logger.debug(
        "add_track_evolution_coefficient: rate=%.4f s/lap  "
        "total improvement over %d laps = %.3fs",
        rate,
        int(df["lap_number"].max()),
        float(evolution.max() - evolution.min()),
    )
    return df


def add_delta_to_theoretical_best(
    laps: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute each lap's time delta vs the session theoretical best.

    The theoretical best for each race lap is the minimum evolution-corrected
    lap time achieved by any driver on that lap (representative laps only).
    This removes absolute pace differences between cars.

    Result column:
        delta_to_theoretical_best_sec: How much slower than the theoretical
                                       best this lap was. Always >= 0 for
                                       representative laps. May be negative
                                       for anomalous laps (expected).

    Args:
        laps: Must contain evolution_corrected_lap_sec.

    Returns:
        DataFrame with delta_to_theoretical_best_sec added.
    """
    if "evolution_corrected_lap_sec" not in laps.columns:
        raise ValueError(
            "add_delta_to_theoretical_best requires evolution_corrected_lap_sec. "
            "Call add_track_evolution_coefficient() first."
        )

    df = laps.copy()

    repr_only = df[df["is_representative"]]
    theoretical_best = (
        repr_only.groupby("lap_number")["evolution_corrected_lap_sec"]
        .min()
        .rename("theoretical_best_sec")
    )

    df = df.merge(
        theoretical_best.reset_index(),
        on="lap_number",
        how="left",
    )
    df["delta_to_theoretical_best_sec"] = (
        df["evolution_corrected_lap_sec"] - df["theoretical_best_sec"]
    )

    logger.debug(
        "add_delta_to_theoretical_best: mean delta=%.3fs  "
        "max delta=%.3fs (representative laps only)",
        float(df.loc[df["is_representative"], "delta_to_theoretical_best_sec"].mean()),
        float(df.loc[df["is_representative"], "delta_to_theoretical_best_sec"].max()),
    )
    return df


def add_pace_drop_per_lap(
    laps: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute the per-lap pace drop within each driver-stint.

    Engineering rationale:
        Pace drop (positive = slower than previous lap on same set) is the
        raw degradation signal at the individual stint level. Aggregated
        by compound across stints, this becomes the degradation rate.

    Method:
        Within each (driver_code, stint_number) group, compute the difference
        in evolution_corrected_lap_sec between consecutive representative laps.
        Non-representative laps get NaN (they don't contribute to the signal).

    Result column:
        pace_drop_sec: Lap time increase vs previous lap on same tyre set.
                       Positive = slower (degrading).
                       NaN on the first lap of each stint.

    Args:
        laps: Must contain evolution_corrected_lap_sec, is_representative,
              driver_code, stint_number, tyre_age.

    Returns:
        DataFrame with pace_drop_sec added.
    """
    if "evolution_corrected_lap_sec" not in laps.columns:
        raise ValueError(
            "add_pace_drop_per_lap requires evolution_corrected_lap_sec."
        )

    df = laps.copy()
    df = df.sort_values(["driver_code", "stint_number", "tyre_age"])

    def _stint_pace_drop(group: pd.DataFrame) -> pd.Series:
        corrected = group["evolution_corrected_lap_sec"].copy()
        # Only diff representative laps — non-representative laps are NaN
        corrected.loc[~group["is_representative"]] = np.nan
        return corrected.diff()

    _pace_result = (
        df.groupby(["driver_code", "stint_number"], group_keys=False)
        .apply(_stint_pace_drop)
    )
    # Newer pandas versions return flat index; older return MultiIndex
    if isinstance(_pace_result.index, pd.MultiIndex):
        _pace_result = _pace_result.reset_index(level=[0, 1], drop=True)
    df["pace_drop_sec"] = _pace_result.reindex(df.index)

    logger.debug(
        "add_pace_drop_per_lap: median pace_drop=%.4fs  "
        "std=%.4fs (representative laps)",
        float(df.loc[df["is_representative"], "pace_drop_sec"].median()),
        float(df.loc[df["is_representative"], "pace_drop_sec"].std()),
    )
    return df


def add_stint_position(
    laps: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute normalised tyre age (position within expected stint length).

    Engineering rationale:
        A tyre_age of 15 means different things for a soft (nearly at cliff)
        vs a hard (barely warmed up). Normalising by expected stint length
        makes degradation curves comparable across compounds, enabling
        multi-compound ML models that generalise across stints.

    Formula:
        stint_position = tyre_age / EXPECTED_STINT_LENGTH[compound]
        Clipped to [0, 1] to handle stints that run longer than expected.

    Result column:
        stint_position: float in [0, 1]. 0 = fresh tyre. 1 = end of
                        expected stint length.

    Args:
        laps: Must contain compound and tyre_age.

    Returns:
        DataFrame with stint_position added.
    """
    df = laps.copy()

    df["expected_stint_laps"] = df["compound"].map(EXPECTED_STINT_LENGTH).fillna(25)
    df["stint_position"] = (
        df["tyre_age"] / df["expected_stint_laps"]
    ).clip(0.0, 1.5)  # Allow 1.5 to capture over-extended stints

    df = df.drop(columns=["expected_stint_laps"])

    logger.debug(
        "add_stint_position: mean=%.3f  max=%.3f",
        float(df["stint_position"].mean()),
        float(df["stint_position"].max()),
    )
    return df


def add_lap_delta_from_stint_baseline(
    laps: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute each lap's evolution-corrected time delta from its own stint baseline.

    The baseline for each driver-stint is the median corrected time on laps
    with tyre_age == 2 (the first fully warmed-up lap). Using the driver's
    own baseline eliminates between-driver absolute pace differences — only
    the SHAPE of the degradation curve matters for modelling.

    Result column:
        lap_delta_from_baseline_sec: Positive = slower than baseline (degrading).

    Args:
        laps: Must contain evolution_corrected_lap_sec, driver_code,
              stint_number, tyre_age.

    Returns:
        DataFrame with lap_delta_from_baseline_sec added.
    """
    from src.constants import MIN_TYRE_AGE_FOR_REGRESSION

    if "evolution_corrected_lap_sec" not in laps.columns:
        raise ValueError(
            "add_lap_delta_from_stint_baseline requires evolution_corrected_lap_sec."
        )

    df = laps.copy()

    baseline_laps = df[
        (df["tyre_age"] == MIN_TYRE_AGE_FOR_REGRESSION)
        & df["is_representative"]
    ].groupby(["driver_code", "stint_number"])["evolution_corrected_lap_sec"].mean()
    baseline_laps = baseline_laps.rename("stint_baseline_sec")

    df = df.merge(
        baseline_laps.reset_index(),
        on=["driver_code", "stint_number"],
        how="left",
    )

    # Fallback: use each stint's minimum representative lap time as baseline
    # for stints that don't have a lap at MIN_TYRE_AGE_FOR_REGRESSION
    fallback = (
        df[df["is_representative"]]
        .groupby(["driver_code", "stint_number"])["evolution_corrected_lap_sec"]
        .min()
        .rename("fallback_baseline_sec")
    )
    df = df.merge(fallback.reset_index(), on=["driver_code", "stint_number"], how="left")
    df["stint_baseline_sec"] = df["stint_baseline_sec"].fillna(df["fallback_baseline_sec"])
    df = df.drop(columns=["fallback_baseline_sec"])

    missing = df["stint_baseline_sec"].isna().sum()
    if missing > 0:
        logger.warning(
            "add_lap_delta_from_stint_baseline: %d laps have no baseline — "
            "lap_delta_from_baseline_sec will be NaN for these.",
            missing,
        )

    df["lap_delta_from_baseline_sec"] = (
        df["evolution_corrected_lap_sec"] - df["stint_baseline_sec"]
    )

    logger.debug(
        "add_lap_delta_from_stint_baseline: mean_delta=%.3fs  "
        "max_delta=%.3fs (representative laps only)",
        float(df.loc[df["is_representative"], "lap_delta_from_baseline_sec"].mean()),
        float(df.loc[df["is_representative"], "lap_delta_from_baseline_sec"].max()),
    )
    return df


# ===========================================================================
# Pipeline entry point
# ===========================================================================

def build_feature_set(
    laps: pd.DataFrame,
    total_race_laps: int,
) -> pd.DataFrame:
    """
    Build the full feature set from a processed laps DataFrame.

    Chains all individual feature functions in dependency order:
        1. fuel_corrected_lap_sec       (requires: lap_time_sec, lap_number)
        2. track_evolution_sec,
           evolution_corrected_lap_sec  (requires: fuel_corrected_lap_sec)
        3. delta_to_theoretical_best_sec (requires: evolution_corrected_lap_sec)
        4. pace_drop_sec                (requires: evolution_corrected_lap_sec)
        5. stint_position               (requires: compound, tyre_age)
        6. lap_delta_from_baseline_sec  (requires: evolution_corrected_lap_sec)

    Args:
        laps:            Output of telemetry_processor.process_laps().
        total_race_laps: Scheduled race distance (for fuel correction).

    Returns:
        Feature-enriched DataFrame with all original columns plus:
            fuel_corrected_lap_sec
            track_evolution_sec
            evolution_corrected_lap_sec
            delta_to_theoretical_best_sec
            pace_drop_sec
            stint_position
            stint_baseline_sec
            lap_delta_from_baseline_sec

    Raises:
        ValueError: If required input columns are absent.
    """
    missing = REQUIRED_INPUT_COLUMNS - set(laps.columns)
    if missing:
        raise ValueError(
            f"build_feature_set: input DataFrame missing columns: "
            f"{sorted(missing)}. "
            f"Ensure input is from telemetry_processor.process_laps()."
        )

    logger.info(
        "build_feature_set: building features for %d laps "
        "(%d representative, total_race_laps=%d)",
        len(laps),
        int(laps["is_representative"].sum()),
        total_race_laps,
    )

    df = laps.copy()
    df = add_fuel_corrected_lap_time(df, total_race_laps)
    df = add_track_evolution_coefficient(df)
    df = add_delta_to_theoretical_best(df)
    df = add_pace_drop_per_lap(df)
    df = add_stint_position(df)
    df = add_lap_delta_from_stint_baseline(df)

    new_features = [
        "fuel_corrected_lap_sec", "track_evolution_sec",
        "evolution_corrected_lap_sec", "delta_to_theoretical_best_sec",
        "pace_drop_sec", "stint_position",
        "stint_baseline_sec", "lap_delta_from_baseline_sec",
    ]
    existing = [f for f in new_features if f in df.columns]
    missing_features = [f for f in new_features if f not in df.columns]
    if missing_features:
        logger.warning(
            "build_feature_set: some features not computed: %s",
            missing_features,
        )

    logger.info(
        "build_feature_set: complete — %d features added: %s",
        len(existing), existing,
    )
    return df
