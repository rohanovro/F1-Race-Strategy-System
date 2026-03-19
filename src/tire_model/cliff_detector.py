"""
src/tire_model/cliff_detector.py
==================================
Tyre degradation cliff detection via change-point analysis.

Engineering responsibility:
    Identify the tyre age at which degradation transitions from the linear
    wear phase into the cliff phase — the point of rapid, non-linear
    grip and pace loss.

Why cliff detection matters for strategy:
    The cliff is the hard constraint on stint length. Running beyond the
    cliff is not a linear cost — it is exponential. A driver 2 laps past
    the cliff may be 1.5-2.0s/lap slower than if they had pitted, meaning
    they lose a position every 2-3 laps to any car on fresh rubber.

    The pit window optimizer needs the cliff lap to set the LATEST viable
    pit lap. The race simulator uses it to trigger the post-cliff quadratic
    degradation regime. The safety car analyzer uses it to evaluate whether
    an SC deployment allows a "free" stop before the cliff.

Detection approaches implemented:
    1. SECOND DERIVATIVE SPIKE (primary):
       Compute the second derivative (acceleration) of the median degradation
       curve. A sudden spike indicates a change in degradation RATE — the
       defining characteristic of a cliff. This is fast, interpretable, and
       works well on smoothed data.

    2. PELT CHANGE-POINT DETECTION (secondary, requires ruptures library):
       The Pruned Exact Linear Time algorithm detects structural breakpoints
       in time series. More statistically rigorous but requires an external
       dependency (ruptures). Used as a cross-check when available.

    3. ROLLING RATE THRESHOLD (fallback):
       If neither method produces a clean signal, fall back to the first
       tyre age where the rolling per-lap degradation rate exceeds
       CLIFF_RATE_MULTIPLIER × mean rate. Robust but less precise.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ===========================================================================
# Constants
# ===========================================================================

# Minimum tyre age at which a cliff can be declared.
# Earlier ages indicate noise or out-lap distortion, not a real cliff.
CLIFF_MIN_TYRE_AGE: int = 5

# Window size for Savitzky-Golay smoothing before second-derivative analysis.
# Must be odd and >= 3. Larger values → smoother but may blur the cliff.
SMOOTHING_WINDOW: int = 5

# Second derivative threshold for cliff detection (standard deviations).
# A spike in the second derivative that exceeds this many std devs of the
# second derivative distribution is flagged as a cliff candidate.
SPIKE_THRESHOLD_STD: float = 2.0

# Cliff rate multiplier for rolling-rate fallback detector.
# A lap is a cliff candidate if its per-lap rate exceeds this multiple
# of the mean rate observed across the full stint.
CLIFF_RATE_MULTIPLIER: float = 2.5

# Minimum number of data points required for cliff detection.
# Below this, the signal is too sparse for reliable change-point analysis.
MIN_POINTS_FOR_DETECTION: int = 8


# ===========================================================================
# Internal helpers
# ===========================================================================

def _smooth_series(values: np.ndarray, window: int = SMOOTHING_WINDOW) -> np.ndarray:
    """
    Apply a simple moving-average smooth to reduce noise before differentiation.

    Smoothing is necessary because raw median degradation curves from
    real data have lap-to-lap noise from traffic, driver variance, and
    measurement precision. Differentiating noisy data amplifies noise,
    producing false spike detections.

    Args:
        values: 1D array of values to smooth.
        window: Window size (must be odd and >= 3).

    Returns:
        Smoothed array of the same length (edges use available data).
    """
    if len(values) < window:
        return values

    half = window // 2
    smoothed = np.convolve(values, np.ones(window) / window, mode="full")
    # Trim to original length with edge handling
    return smoothed[half: half + len(values)]


def _second_derivative_cliff(
    tyre_ages: np.ndarray,
    deltas: np.ndarray,
) -> Optional[int]:
    """
    Detect cliff using second-derivative spike detection.

    The second derivative of the degradation curve (d²Δt/dAge²) represents
    the RATE OF CHANGE OF DEGRADATION RATE. A positive spike indicates
    that degradation is accelerating — the physical signature of a cliff.

    Method:
        1. Smooth the delta curve to reduce noise
        2. Compute first derivative (per-lap degradation rate)
        3. Compute second derivative (acceleration of degradation)
        4. Flag the first tyre age where d²Δt/dAge² exceeds
           mean(d²Δt) + SPIKE_THRESHOLD_STD × std(d²Δt)
        5. Validate: flagged age must be >= CLIFF_MIN_TYRE_AGE

    Args:
        tyre_ages: Sorted array of tyre age values.
        deltas:    Corresponding median lap time delta values.

    Returns:
        Tyre age at cliff onset, or None if not detected.
    """
    if len(tyre_ages) < MIN_POINTS_FOR_DETECTION:
        return None

    smoothed  = _smooth_series(deltas)
    d1        = np.diff(smoothed)        # First derivative
    d2        = np.diff(d1)              # Second derivative
    d2_ages   = tyre_ages[2:]            # Corresponding tyre ages

    if len(d2) < 3:
        return None

    d2_mean   = np.mean(d2)
    d2_std    = np.std(d2)

    if d2_std < 1e-6:
        # Perfectly flat second derivative — no cliff
        return None

    threshold = d2_mean + SPIKE_THRESHOLD_STD * d2_std
    spike_mask = (d2 > threshold) & (d2_ages >= CLIFF_MIN_TYRE_AGE)

    if not spike_mask.any():
        return None

    cliff_age = int(d2_ages[spike_mask][0])
    logger.debug(
        "_second_derivative_cliff: cliff at tyre_age=%d "
        "(d2=%.4f > threshold=%.4f)",
        cliff_age,
        float(d2[spike_mask][0]),
        threshold,
    )
    return cliff_age


def _rolling_rate_cliff(
    tyre_ages: np.ndarray,
    deltas: np.ndarray,
) -> Optional[int]:
    """
    Detect cliff using rolling per-lap rate vs mean rate comparison.

    Fallback method when the second derivative approach fails to produce
    a clean signal (typically when data is too sparse or noisy).

    Method:
        Compute per-lap rate (finite difference of deltas).
        The cliff is the first lap where the rate exceeds
        CLIFF_RATE_MULTIPLIER × mean absolute rate.

    Args:
        tyre_ages: Sorted tyre age array.
        deltas:    Corresponding delta array.

    Returns:
        Tyre age at cliff onset, or None.
    """
    if len(tyre_ages) < 4:
        return None

    rates     = np.diff(deltas)
    rate_ages = tyre_ages[1:]
    mean_rate = np.mean(np.abs(rates))

    if mean_rate <= 1e-6:
        return None

    valid_mask = rate_ages >= CLIFF_MIN_TYRE_AGE
    valid_rates = rates[valid_mask]
    valid_ages  = rate_ages[valid_mask]

    cliff_mask = np.abs(valid_rates) > CLIFF_RATE_MULTIPLIER * mean_rate
    if not cliff_mask.any():
        return None

    cliff_age = int(valid_ages[cliff_mask][0])
    logger.debug(
        "_rolling_rate_cliff: cliff at tyre_age=%d "
        "(rate=%.4f > %.1fx mean=%.4f)",
        cliff_age,
        float(valid_rates[cliff_mask][0]),
        CLIFF_RATE_MULTIPLIER,
        mean_rate,
    )
    return cliff_age


def _pelt_cliff(
    tyre_ages: np.ndarray,
    deltas: np.ndarray,
) -> Optional[int]:
    """
    Detect cliff using PELT change-point detection (ruptures library).

    PELT (Pruned Exact Linear Time) is a statistically rigorous method
    for detecting structural breakpoints in time series. It minimises a
    cost function (here: RBF kernel) over all possible segmentations.

    This is optional — if ruptures is not installed the function returns
    None and the caller falls back to other methods.

    Args:
        tyre_ages: Sorted tyre age array.
        deltas:    Corresponding delta array.

    Returns:
        Tyre age at first breakpoint, or None.
    """
    try:
        import ruptures as rpt
    except ImportError:
        logger.debug("_pelt_cliff: ruptures not installed — skipping PELT detection.")
        return None

    if len(deltas) < MIN_POINTS_FOR_DETECTION:
        return None

    signal = deltas.reshape(-1, 1)
    algo   = rpt.Pelt(model="rbf").fit(signal)

    try:
        breakpoints = algo.predict(pen=1.0)
    except Exception as exc:
        logger.debug("_pelt_cliff: PELT failed — %s", exc)
        return None

    # breakpoints returns indices; convert to tyre age values
    # The last breakpoint is always len(signal) — ignore it
    valid_bp = [bp for bp in breakpoints[:-1] if bp < len(tyre_ages)]
    if not valid_bp:
        return None

    # Take the first breakpoint that meets the minimum age requirement
    for bp_idx in valid_bp:
        candidate_age = int(tyre_ages[bp_idx])
        if candidate_age >= CLIFF_MIN_TYRE_AGE:
            logger.debug(
                "_pelt_cliff: breakpoint at index=%d → tyre_age=%d",
                bp_idx, candidate_age,
            )
            return candidate_age

    return None


# ===========================================================================
# Public API
# ===========================================================================

def detect_cliff(
    tyre_ages: np.ndarray,
    lap_time_deltas: np.ndarray,
    compound: str = "UNKNOWN",
) -> Optional[int]:
    """
    Detect the degradation cliff tyre age for a single compound.

    Runs all detection methods in priority order and returns the first
    successful detection. Priority:
        1. PELT (most statistically rigorous, requires ruptures)
        2. Second derivative spike (fast, interpretable)
        3. Rolling rate threshold (robust fallback)

    Results from multiple methods are logged for comparison.

    Args:
        tyre_ages:        Sorted array of tyre age values (ints).
        lap_time_deltas:  Corresponding median lap time deltas from baseline.
        compound:         Compound name (for logging only).

    Returns:
        Tyre age integer at cliff onset, or None if no cliff detected.
    """
    if len(tyre_ages) < MIN_POINTS_FOR_DETECTION:
        logger.debug(
            "detect_cliff [%s]: insufficient data (%d points < %d min).",
            compound, len(tyre_ages), MIN_POINTS_FOR_DETECTION,
        )
        return None

    sort_idx  = np.argsort(tyre_ages)
    ages_s    = tyre_ages[sort_idx].astype(float)
    deltas_s  = lap_time_deltas[sort_idx].astype(float)

    pelt_cliff   = _pelt_cliff(ages_s, deltas_s)
    d2_cliff     = _second_derivative_cliff(ages_s, deltas_s)
    rate_cliff   = _rolling_rate_cliff(ages_s, deltas_s)

    logger.debug(
        "detect_cliff [%s]: PELT=%s  d2=%s  rate=%s",
        compound, pelt_cliff, d2_cliff, rate_cliff,
    )

    # Priority: PELT → d2 → rate → None
    cliff = pelt_cliff or d2_cliff or rate_cliff

    if cliff is not None:
        logger.info(
            "detect_cliff [%s]: cliff detected at tyre_age=%d",
            compound, cliff,
        )
    else:
        logger.info(
            "detect_cliff [%s]: no cliff detected within observed range "
            "(max_age=%d).",
            compound, int(ages_s.max()),
        )

    return cliff


def detect_all_compound_cliffs(
    feature_df: pd.DataFrame,
    compounds: Optional[list[str]] = None,
) -> dict[str, Optional[int]]:
    """
    Detect degradation cliffs for all compounds in a feature DataFrame.

    Uses the lap_delta_from_baseline_sec column (output of feature_builder.py)
    aggregated by tyre_age as the degradation signal.

    Args:
        feature_df: Output of feature_builder.build_feature_set().
                    Must contain: compound, tyre_age, lap_delta_from_baseline_sec,
                    is_representative.
        compounds:  Optional list of compounds to analyse.
                    Defaults to all compounds present in the DataFrame.

    Returns:
        Dict mapping compound name -> cliff tyre age (or None).
    """
    required = {"compound", "tyre_age",
                "lap_delta_from_baseline_sec", "is_representative"}
    missing = required - set(feature_df.columns)
    if missing:
        raise ValueError(
            f"detect_all_compound_cliffs: missing columns {sorted(missing)}. "
            f"Ensure input is from feature_builder.build_feature_set()."
        )

    if compounds is None:
        compounds = sorted(feature_df["compound"].unique().tolist())

    results: dict[str, Optional[int]] = {}

    for compound in compounds:
        comp_data = feature_df[
            (feature_df["compound"] == compound)
            & feature_df["is_representative"]
        ].copy()

        if comp_data.empty:
            logger.warning(
                "detect_all_compound_cliffs [%s]: no representative laps — skipping.",
                compound,
            )
            results[compound] = None
            continue

        agg = (
            comp_data.groupby("tyre_age")["lap_delta_from_baseline_sec"]
            .median()
            .reset_index()
            .sort_values("tyre_age")
        )

        cliff = detect_cliff(
            tyre_ages       = agg["tyre_age"].values,
            lap_time_deltas = agg["lap_delta_from_baseline_sec"].values,
            compound        = compound,
        )
        results[compound] = cliff

    logger.info(
        "detect_all_compound_cliffs: results = %s",
        {k: v for k, v in results.items()},
    )
    return results
