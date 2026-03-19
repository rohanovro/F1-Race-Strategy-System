"""
src/strategy_engine/race_simulator.py
=======================================
Vectorised lap-by-lap race simulator with Monte Carlo support.

Engineering responsibility:
    Given a complete pit stop strategy and a fitted DegradationModelSet,
    simulate every lap of the race and return a predicted total time.

    This is the hot-path function of the entire system — the optimizer
    calls it thousands of times per search. Every design decision here
    is made to maximise throughput while maintaining physical fidelity.

Performance contract:
    < 1 ms per simulate_strategy() call (NumPy vectorised path).
    10,000 strategy evaluations: < 10 seconds wall time.
    Memory: O(race_laps) per call — no cross-call accumulation.

Monte Carlo design:
    The deterministic simulator models the expected race time given a
    strategy. The Monte Carlo wrapper adds stochastic perturbations:
        - Safety car deployment (probability + expected duration)
        - Pit stop time variance (Gaussian around stationary time)
        - Tyre degradation uncertainty (Gaussian around model prediction)
    Running N Monte Carlo samples gives a DISTRIBUTION of race times
    for each strategy, enabling risk-adjusted strategy selection.

Lap time model (per lap):
    predicted_lap = base_lap_time
                  + fuel_delta(lap_n)         # fuel load penalty
                  + deg_delta(compound, age)  # tyre degradation penalty
                  + inlap_penalty             # if is_pit_entry_lap
                  + outlap_penalty            # if is_pit_exit_lap
                  + pit_loss                  # stationary + traverse time

Competitor interaction:
    The simulator models a SINGLE CAR in isolation. Competitor interaction
    (undercut/overcut) is handled by undercut_overcut.py which calls the
    simulator separately for each car's strategy and compares the resulting
    race times and track positions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.tire_model.degradation_model import DegradationModelSet
from src.constants import (
    FUEL_BURN_RATE_KG_PER_LAP,
    FUEL_LAP_TIME_SENSITIVITY_SEC_PER_KG,
    INLAP_TIME_PENALTY_SEC,
    OUTLAP_TIME_PENALTY_SEC,
    PIT_STATIONARY_TIME_SEC,
    PIT_LANE_DELTA_SEC_BAHRAIN,
    MIN_COMPOUNDS_PER_RACE,
    COMPOUND_ABBREV,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Constants
# ===========================================================================

MIN_RACE_LAPS: int    = 10
MAX_TYRE_AGE_WARNING: int = 45

# Monte Carlo defaults
MC_DEFAULT_SAMPLES:           int   = 500
MC_PIT_TIME_STD_SEC:          float = 0.4   # Gaussian std for pit stop time
MC_DEG_NOISE_STD_FRACTION:    float = 0.08  # Fraction of predicted deg delta
MC_SC_PROBABILITY_PER_LAP:    float = 0.018 # ~1 SC every 55 laps historically
MC_SC_DURATION_MEAN_LAPS:     int   = 5
MC_SC_LAP_TIME_DELTA_SEC:     float = 30.0  # SC lap ~30s slower than racing lap


# ===========================================================================
# Strategy data contracts
# ===========================================================================

@dataclass(frozen=True)
class StintSpec:
    """
    Specification for a single tyre stint within a race strategy.

    Attributes:
        compound:     Tyre compound (e.g. "SOFT").
        start_lap:    First race lap of this stint (1-indexed).
        end_lap:      Last race lap of this stint (inclusive).
        starting_age: Tyre age at stint start. 1 = new tyre.
    """
    compound:     str
    start_lap:    int
    end_lap:      int
    starting_age: int = 1

    @property
    def stint_length(self) -> int:
        return self.end_lap - self.start_lap + 1

    @property
    def max_tyre_age(self) -> int:
        return self.starting_age + self.stint_length - 1


@dataclass
class RaceStrategy:
    """
    Complete race strategy: ordered stints + circuit pit lane delta.

    Attributes:
        stints:             Ordered list of StintSpec (lap 1 → finish).
        pit_lane_delta_sec: Circuit pit lane traverse time (seconds).
        label:              Human-readable label (auto-generated if empty).
    """
    stints:             list[StintSpec]
    pit_lane_delta_sec: float = PIT_LANE_DELTA_SEC_BAHRAIN
    label:              str   = ""

    def __post_init__(self) -> None:
        if not self.label:
            seq = "-".join(COMPOUND_ABBREV.get(s.compound, "?") for s in self.stints)
            self.label = f"{self.n_stops}-stop [{seq}]"

    @property
    def n_stops(self) -> int:
        return max(0, len(self.stints) - 1)

    @property
    def pit_laps(self) -> list[int]:
        return [s.end_lap for s in self.stints[:-1]]

    @property
    def compounds_used(self) -> list[str]:
        return [s.compound for s in self.stints]

    def is_valid(self) -> tuple[bool, str]:
        """Validate FIA compound rule and physical constraints."""
        if len(set(self.compounds_used)) < MIN_COMPOUNDS_PER_RACE:
            return False, (
                f"FIA compound violation: only one compound "
                f"({set(self.compounds_used)}). Min {MIN_COMPOUNDS_PER_RACE} required."
            )
        if not self.stints:
            return False, "Strategy contains no stints."
        return True, ""

    def __repr__(self) -> str:
        return f"RaceStrategy[{self.label}]"


@dataclass
class LapResult:
    """Simulated data for a single race lap."""
    lap_number:        int
    compound:          str
    tyre_age:          int
    base_lap_sec:      float
    fuel_delta_sec:    float
    deg_delta_sec:     float
    pit_loss_sec:      float
    inlap_penalty_sec: float
    outlap_penalty_sec: float
    is_pit_entry_lap:  bool
    is_pit_exit_lap:   bool
    predicted_lap_sec: float


@dataclass
class SimulationResult:
    """Complete output of one strategy simulation."""
    strategy:              RaceStrategy
    total_race_time_sec:   float
    lap_results:           list[LapResult]
    is_valid:              bool  = True
    invalid_reason:        str   = ""
    extrapolation_warning: bool  = False

    @property
    def total_time_formatted(self) -> str:
        t = self.total_race_time_sec
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "lap_number":         r.lap_number,
                "compound":           r.compound,
                "tyre_age":           r.tyre_age,
                "base_lap_sec":       r.base_lap_sec,
                "fuel_delta_sec":     r.fuel_delta_sec,
                "deg_delta_sec":      r.deg_delta_sec,
                "pit_loss_sec":       r.pit_loss_sec,
                "inlap_penalty_sec":  r.inlap_penalty_sec,
                "outlap_penalty_sec": r.outlap_penalty_sec,
                "is_pit_entry_lap":   r.is_pit_entry_lap,
                "is_pit_exit_lap":    r.is_pit_exit_lap,
                "predicted_lap_sec":  r.predicted_lap_sec,
            }
            for r in self.lap_results
        ])

    def summary(self) -> str:
        warn  = " [EXTRAP]" if self.extrapolation_warning else ""
        valid = "" if self.is_valid else f" [INVALID: {self.invalid_reason}]"
        return f"{self.strategy.label} | {self.total_time_formatted}{warn}{valid}"


@dataclass
class MonteCarloResult:
    """Output of a Monte Carlo simulation run."""
    strategy:              RaceStrategy
    n_samples:             int
    mean_time_sec:         float
    std_time_sec:          float
    p10_time_sec:          float   # 10th percentile (optimistic)
    p50_time_sec:          float   # Median
    p90_time_sec:          float   # 90th percentile (pessimistic)
    sc_affected_fraction:  float   # Fraction of samples with >=1 SC
    sample_times:          np.ndarray = field(repr=False, default_factory=lambda: np.array([]))

    def summary(self) -> str:
        return (
            f"{self.strategy.label} | "
            f"mean={self.mean_time_sec:.1f}s  std={self.std_time_sec:.1f}s  "
            f"P10={self.p10_time_sec:.1f}s  P90={self.p90_time_sec:.1f}s  "
            f"SC_frac={self.sc_affected_fraction:.0%}"
        )


# ===========================================================================
# Strategy builder
# ===========================================================================

def build_strategy(
    pit_laps:           list[int],
    compounds:          list[str],
    total_race_laps:    int,
    starting_tyre_ages: Optional[list[int]] = None,
    pit_lane_delta_sec: float = PIT_LANE_DELTA_SEC_BAHRAIN,
) -> RaceStrategy:
    """
    Construct a RaceStrategy from flat pit-lap and compound lists.

    Args:
        pit_laps:           Ordered list of pit stop laps (empty = 0-stop).
        compounds:          One compound per stint (len = len(pit_laps) + 1).
        total_race_laps:    Race distance.
        starting_tyre_ages: Tyre age at start of each stint. Defaults to 1.
        pit_lane_delta_sec: Circuit pit lane time.

    Returns:
        RaceStrategy with fully assembled StintSpec list.

    Raises:
        ValueError: On inconsistent arguments or invalid pit laps.
    """
    n_stints = len(pit_laps) + 1

    if len(compounds) != n_stints:
        raise ValueError(
            f"build_strategy: {len(compounds)} compounds for {n_stints} stints. "
            f"Expected len(compounds) == len(pit_laps) + 1."
        )

    if starting_tyre_ages is None:
        starting_tyre_ages = [1] * n_stints
    elif len(starting_tyre_ages) != n_stints:
        raise ValueError(
            f"build_strategy: starting_tyre_ages has {len(starting_tyre_ages)} "
            f"entries; expected {n_stints}."
        )

    sorted_pits = sorted(pit_laps)
    for i, pl in enumerate(sorted_pits):
        if not (1 <= pl < total_race_laps):
            raise ValueError(
                f"build_strategy: pit_lap={pl} outside [1, {total_race_laps-1}]."
            )
        if i > 0 and pl <= sorted_pits[i - 1]:
            raise ValueError(
                f"build_strategy: duplicate/out-of-order pit laps: {sorted_pits}."
            )

    starts = [1] + [pl + 1 for pl in sorted_pits]
    ends   = sorted_pits + [total_race_laps]

    stints = [
        StintSpec(
            compound     = compounds[i].upper(),
            start_lap    = starts[i],
            end_lap      = ends[i],
            starting_age = starting_tyre_ages[i],
        )
        for i in range(n_stints)
    ]

    return RaceStrategy(
        stints             = stints,
        pit_lane_delta_sec = pit_lane_delta_sec,
    )


# ===========================================================================
# Vectorised per-lap array builder
# ===========================================================================

def _vectorise_strategy(
    strategy: RaceStrategy,
    total_race_laps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre-compute per-lap arrays from a strategy.

    Pre-computing these arrays once eliminates repeated Python attribute
    lookups inside the simulation hot loop.

    Returns tuple of 5 arrays of length total_race_laps:
        (compounds, tyre_ages, is_inlap, is_outlap, pit_loss)
    """
    n = total_race_laps
    compounds  = np.empty(n, dtype=object)
    tyre_ages  = np.zeros(n, dtype=np.int32)
    is_inlap   = np.zeros(n, dtype=bool)
    is_outlap  = np.zeros(n, dtype=bool)
    pit_loss   = np.zeros(n, dtype=np.float64)

    for stint in strategy.stints:
        for lap in range(stint.start_lap, stint.end_lap + 1):
            idx           = lap - 1
            compounds[idx] = stint.compound
            tyre_ages[idx] = stint.starting_age + (lap - stint.start_lap)

    for pl in strategy.pit_laps:
        is_inlap[pl - 1] = True
        pit_loss[pl - 1] = PIT_STATIONARY_TIME_SEC + strategy.pit_lane_delta_sec

    for stint in strategy.stints[1:]:
        is_outlap[stint.start_lap - 1] = True

    return compounds, tyre_ages, is_inlap, is_outlap, pit_loss


# ===========================================================================
# Core simulator
# ===========================================================================

def simulate_strategy(
    strategy:          RaceStrategy,
    model_set:         DegradationModelSet,
    base_lap_time_sec: float,
    total_race_laps:   int,
) -> SimulationResult:
    """
    Simulate a complete race strategy. Vectorised hot path.

    Args:
        strategy:          RaceStrategy to simulate.
        model_set:         Fitted DegradationModelSet from degradation_model.py.
        base_lap_time_sec: Reference pace at empty-fuel weight (seconds).
        total_race_laps:   Race distance.

    Returns:
        SimulationResult with total time and full per-lap breakdown.
    """
    valid, reason = strategy.is_valid()
    if not valid:
        return SimulationResult(
            strategy             = strategy,
            total_race_time_sec  = float("inf"),
            lap_results          = [],
            is_valid             = False,
            invalid_reason       = reason,
        )

    if total_race_laps < MIN_RACE_LAPS:
        return SimulationResult(
            strategy            = strategy,
            total_race_time_sec = float("inf"),
            lap_results         = [],
            is_valid            = False,
            invalid_reason      = f"total_race_laps={total_race_laps} < {MIN_RACE_LAPS}.",
        )

    compounds, tyre_ages, is_inlap, is_outlap, pit_loss = _vectorise_strategy(
        strategy, total_race_laps
    )

    # Fuel delta (vectorised)
    lap_ns         = np.arange(1, total_race_laps + 1, dtype=np.float64)
    fuel_remaining = np.maximum(0.0,
        (total_race_laps - lap_ns) * FUEL_BURN_RATE_KG_PER_LAP)
    fuel_delta     = fuel_remaining * FUEL_LAP_TIME_SENSITIVITY_SEC_PER_KG

    # Degradation delta (batched by compound)
    deg_delta          = np.zeros(total_race_laps, dtype=np.float64)
    extrapolation_warn = False

    for compound in set(compounds):
        model = model_set.get(str(compound))
        mask  = compounds == compound
        ages  = tyre_ages[mask].astype(float)

        if model is None:
            logger.warning(
                "simulate_strategy: no model for '%s' — zero degradation assumed.",
                compound,
            )
            continue

        if np.any(ages > MAX_TYRE_AGE_WARNING):
            extrapolation_warn = True
            logger.warning(
                "simulate_strategy: '%s' reaches tyre_age=%d > %d "
                "(extrapolating beyond fitted range).",
                compound, int(ages.max()), MAX_TYRE_AGE_WARNING,
            )

        deg_delta[mask] = model.predict(ages)

    # In/out lap time penalties
    inlap_penalty  = is_inlap.astype(float)  * INLAP_TIME_PENALTY_SEC
    outlap_penalty = is_outlap.astype(float) * OUTLAP_TIME_PENALTY_SEC

    # Total lap time array
    predicted_laps = (
        base_lap_time_sec
        + fuel_delta
        + deg_delta
        + inlap_penalty
        + outlap_penalty
        + pit_loss
    )
    total_time = float(predicted_laps.sum())

    lap_results = [
        LapResult(
            lap_number         = int(lap_ns[i]),
            compound           = str(compounds[i]),
            tyre_age           = int(tyre_ages[i]),
            base_lap_sec       = base_lap_time_sec,
            fuel_delta_sec     = float(fuel_delta[i]),
            deg_delta_sec      = float(deg_delta[i]),
            pit_loss_sec       = float(pit_loss[i]),
            inlap_penalty_sec  = float(inlap_penalty[i]),
            outlap_penalty_sec = float(outlap_penalty[i]),
            is_pit_entry_lap   = bool(is_inlap[i]),
            is_pit_exit_lap    = bool(is_outlap[i]),
            predicted_lap_sec  = float(predicted_laps[i]),
        )
        for i in range(total_race_laps)
    ]

    return SimulationResult(
        strategy             = strategy,
        total_race_time_sec  = total_time,
        lap_results          = lap_results,
        is_valid             = True,
        extrapolation_warning = extrapolation_warn,
    )


# ===========================================================================
# Monte Carlo wrapper
# ===========================================================================

def monte_carlo_simulate(
    strategy:           RaceStrategy,
    model_set:          DegradationModelSet,
    base_lap_time_sec:  float,
    total_race_laps:    int,
    n_samples:          int    = MC_DEFAULT_SAMPLES,
    sc_probability:     float  = MC_SC_PROBABILITY_PER_LAP,
    pit_time_std_sec:   float  = MC_PIT_TIME_STD_SEC,
    deg_noise_fraction: float  = MC_DEG_NOISE_STD_FRACTION,
    rng_seed:           Optional[int] = None,
) -> MonteCarloResult:
    """
    Monte Carlo simulation: run N perturbed simulations and return distribution.

    Each sample introduces:
        1. Pit stop time noise:  pit_time += N(0, pit_time_std_sec)
        2. Degradation noise:    deg_delta *= N(1, deg_noise_fraction)
        3. Safety car events:    each lap has sc_probability of triggering SC.
                                 Under SC, lap time += MC_SC_LAP_TIME_DELTA_SEC
                                 for MC_SC_DURATION_MEAN_LAPS laps.

    This gives a realistic race time DISTRIBUTION rather than a single point
    estimate — enabling risk-aware strategy decisions.

    Args:
        strategy:           Strategy to evaluate.
        model_set:          Fitted DegradationModelSet.
        base_lap_time_sec:  Reference pace.
        total_race_laps:    Race distance.
        n_samples:          Number of Monte Carlo iterations.
        sc_probability:     Per-lap SC deployment probability.
        pit_time_std_sec:   Std dev of pit stop time noise (seconds).
        deg_noise_fraction: Std dev of degradation multiplier noise.
        rng_seed:           Optional seed for reproducibility.

    Returns:
        MonteCarloResult with time distribution statistics.
    """
    rng = np.random.default_rng(rng_seed)

    # Get deterministic baseline arrays once
    compounds, tyre_ages, is_inlap, is_outlap, _ = _vectorise_strategy(
        strategy, total_race_laps
    )

    lap_ns = np.arange(1, total_race_laps + 1, dtype=np.float64)
    fuel_remaining = np.maximum(
        0.0, (total_race_laps - lap_ns) * FUEL_BURN_RATE_KG_PER_LAP
    )
    fuel_delta = fuel_remaining * FUEL_LAP_TIME_SENSITIVITY_SEC_PER_KG

    # Base degradation array (deterministic)
    base_deg = np.zeros(total_race_laps, dtype=np.float64)
    for compound in set(compounds):
        model = model_set.get(str(compound))
        if model is None:
            continue
        mask = compounds == compound
        base_deg[mask] = model.predict(tyre_ages[mask].astype(float))

    sample_times = np.zeros(n_samples)
    sc_count     = 0

    for i in range(n_samples):
        # --- Pit time noise ---
        pit_noise = rng.normal(0.0, pit_time_std_sec, size=len(strategy.pit_laps))
        pit_loss  = np.zeros(total_race_laps, dtype=np.float64)
        for j, pl in enumerate(strategy.pit_laps):
            pit_loss[pl - 1] = max(
                0.5,
                PIT_STATIONARY_TIME_SEC
                + strategy.pit_lane_delta_sec
                + pit_noise[j],
            )

        # --- Degradation noise ---
        deg_multiplier = rng.normal(1.0, deg_noise_fraction)
        deg_delta      = base_deg * max(0.5, deg_multiplier)

        # --- Safety car events ---
        sc_laps_added = np.zeros(total_race_laps, dtype=np.float64)
        sc_triggered  = rng.random(total_race_laps) < sc_probability
        lap_idx = 0
        has_sc  = False
        while lap_idx < total_race_laps:
            if sc_triggered[lap_idx]:
                has_sc = True
                duration = rng.integers(
                    MC_SC_DURATION_MEAN_LAPS - 2,
                    MC_SC_DURATION_MEAN_LAPS + 3,
                )
                end_idx = min(lap_idx + duration, total_race_laps)
                sc_laps_added[lap_idx:end_idx] = MC_SC_LAP_TIME_DELTA_SEC
                lap_idx = end_idx
            else:
                lap_idx += 1
        if has_sc:
            sc_count += 1

        # --- In/out lap penalties ---
        inlap_pen  = is_inlap.astype(float)  * INLAP_TIME_PENALTY_SEC
        outlap_pen = is_outlap.astype(float) * OUTLAP_TIME_PENALTY_SEC

        lap_times = (
            base_lap_time_sec
            + fuel_delta
            + deg_delta
            + inlap_pen
            + outlap_pen
            + pit_loss
            + sc_laps_added
        )
        sample_times[i] = lap_times.sum()

    result = MonteCarloResult(
        strategy             = strategy,
        n_samples            = n_samples,
        mean_time_sec        = float(np.mean(sample_times)),
        std_time_sec         = float(np.std(sample_times)),
        p10_time_sec         = float(np.percentile(sample_times, 10)),
        p50_time_sec         = float(np.percentile(sample_times, 50)),
        p90_time_sec         = float(np.percentile(sample_times, 90)),
        sc_affected_fraction = sc_count / n_samples,
        sample_times         = sample_times,
    )

    logger.info("monte_carlo_simulate: %s", result.summary())
    return result


# ===========================================================================
# Base lap time estimation
# ===========================================================================

def estimate_base_lap_time(
    feature_df:      pd.DataFrame,
    total_race_laps: int,
    percentile:      float = 5.0,
) -> float:
    """
    Estimate the reference base lap time from feature-engineered race data.

    Uses the specified percentile of evolution_corrected_lap_sec across
    all representative laps. Low percentile (5th) guards against anomalous
    fast laps while capturing the realistic lower bound of pace.

    Args:
        feature_df:      Output of feature_builder.build_feature_set().
        total_race_laps: Race distance (for column check only).
        percentile:      Percentile to use as base estimate.

    Returns:
        Base lap time in seconds.

    Raises:
        RuntimeError: If no representative laps are available.
        ValueError:   If required columns are absent.
    """
    required = {"evolution_corrected_lap_sec", "is_representative"}
    missing  = required - set(feature_df.columns)
    if missing:
        raise ValueError(
            f"estimate_base_lap_time: missing columns {sorted(missing)}. "
            "Run feature_builder.build_feature_set() first."
        )

    repr_laps = feature_df[feature_df["is_representative"]]["evolution_corrected_lap_sec"].dropna()

    if repr_laps.empty:
        raise RuntimeError(
            "estimate_base_lap_time: no representative laps — "
            "cannot estimate base lap time."
        )

    base = float(np.percentile(repr_laps.values, percentile))

    logger.info(
        "estimate_base_lap_time: %.3fs  "
        "(%.0fth percentile of %d repr laps)",
        base, percentile, len(repr_laps),
    )
    return base
