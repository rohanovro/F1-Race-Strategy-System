"""
src/strategy_engine/pit_window_optimizer.py
=============================================
Strategy search, pruning, and ranking engine.

Enumerates all valid pit stop strategies for a given race, simulates each
one, and returns a ranked leaderboard DataFrame plus an optimal strategy.

Search space:
    0-stop: len(compounds) strategies
    1-stop: pit_laps × compound_sequences
    2-stop: pit_lap_pairs × compound_sequences
    3-stop: (rarely optimal, search-space controlled by max_stops)

All pruning is applied before simulation — not after. This is critical
for performance: a pruned search space of ~3,000 strategies at < 1ms
each = < 3 seconds total.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.strategy_engine.race_simulator import (
    RaceStrategy,
    SimulationResult,
    MonteCarloResult,
    build_strategy,
    simulate_strategy,
    monte_carlo_simulate,
    estimate_base_lap_time,
    PIT_LANE_DELTA_SEC_BAHRAIN,
)
from src.tire_model.degradation_model import DegradationModelSet
from src.constants import MIN_COMPOUNDS_PER_RACE, COMPOUND_ABBREV

logger = logging.getLogger(__name__)


# ===========================================================================
# Constants
# ===========================================================================

EARLIEST_PIT_LAP:         int   = 3
MIN_STINT_LAPS:           int   = 6
MAX_STOPS:                int   = 3
EQUIVALENCE_THRESHOLD_SEC: float = 0.5
DEFAULT_LEADERBOARD_SIZE:  int   = 25


# ===========================================================================
# Data contract
# ===========================================================================

@dataclass
class OptimizationResult:
    """Complete output of the pit window optimisation."""
    circuit:              str
    season:               int
    total_race_laps:      int
    base_lap_time_sec:    float
    n_evaluated:          int
    n_valid:              int
    optimal:              Optional[SimulationResult]
    leaderboard:          pd.DataFrame
    all_results:          list[SimulationResult] = field(default_factory=list)

    def gap_to_optimal(self, result: SimulationResult) -> float:
        if self.optimal is None:
            return float("inf")
        return result.total_race_time_sec - self.optimal.total_race_time_sec

    def print_leaderboard(self, n: int = 10) -> None:
        print(f"\n{'='*70}")
        print(f"  Leaderboard — {self.circuit} {self.season}  "
              f"(base={self.base_lap_time_sec:.3f}s  laps={self.total_race_laps})")
        print(f"  Evaluated={self.n_evaluated}  Valid={self.n_valid}")
        print(f"{'='*70}")
        for _, row in self.leaderboard.head(n).iterrows():
            gap = f"+{row['gap_to_optimal_sec']:.3f}s" if row["rank"] > 1 else "OPTIMAL"
            eq  = "~" if row["is_equivalent"] and row["rank"] > 1 else " "
            print(f"  #{row['rank']:<3} {row['label']:<32} "
                  f"{row['total_time_formatted']}  {gap:>10} {eq}")
        print(f"{'='*70}\n")


# ===========================================================================
# Search space generation
# ===========================================================================

def _compound_sequences(
    available: list[str],
    n_stints: int,
) -> list[tuple[str, ...]]:
    """All compound sequences of length n_stints with >= 2 distinct compounds."""
    return [
        seq for seq in itertools.product(available, repeat=n_stints)
        if len(set(seq)) >= MIN_COMPOUNDS_PER_RACE
    ]


def _pit_lap_combinations(
    n_stops: int,
    total_laps: int,
    earliest: int = EARLIEST_PIT_LAP,
    min_stint: int = MIN_STINT_LAPS,
) -> list[tuple[int, ...]]:
    """All valid ordered pit lap tuples with minimum stint length constraints."""
    if n_stops == 0:
        return [()]

    latest = total_laps - min_stint
    combos = []
    for pit_combo in itertools.combinations(range(earliest, latest + 1), n_stops):
        # Check first stint length
        if pit_combo[0] - 1 < min_stint:
            continue
        # Check inter-stop stints
        if any(
            pit_combo[i] - pit_combo[i - 1] < min_stint
            for i in range(1, n_stops)
        ):
            continue
        combos.append(pit_combo)
    return combos


def enumerate_strategies(
    available_compounds: list[str],
    total_race_laps:    int,
    max_stops:          int   = MAX_STOPS,
    pit_lane_delta_sec: float = PIT_LANE_DELTA_SEC_BAHRAIN,
    min_stint_laps:     int   = MIN_STINT_LAPS,
    earliest_pit_lap:   int   = EARLIEST_PIT_LAP,
) -> list[RaceStrategy]:
    """Enumerate all valid strategies in the configured search space."""
    all_strategies: list[RaceStrategy] = []

    for n_stops in range(0, max_stops + 1):
        pit_combos     = _pit_lap_combinations(n_stops, total_race_laps,
                                               earliest_pit_lap, min_stint_laps)
        compound_seqs  = _compound_sequences(available_compounds, n_stops + 1)

        added = 0
        for pits in pit_combos:
            for comps in compound_seqs:
                try:
                    s = build_strategy(
                        pit_laps           = list(pits),
                        compounds          = list(comps),
                        total_race_laps    = total_race_laps,
                        pit_lane_delta_sec = pit_lane_delta_sec,
                    )
                    all_strategies.append(s)
                    added += 1
                except ValueError:
                    pass

        logger.info(
            "enumerate_strategies: %d-stop → %d strategies added",
            n_stops, added,
        )

    logger.info(
        "enumerate_strategies: total=%d strategies", len(all_strategies)
    )
    return all_strategies


# ===========================================================================
# Leaderboard builder
# ===========================================================================

def _build_leaderboard(
    results: list[SimulationResult],
    n: int = DEFAULT_LEADERBOARD_SIZE,
) -> pd.DataFrame:
    valid = sorted(
        [r for r in results if r.is_valid],
        key=lambda r: r.total_race_time_sec,
    )
    if not valid:
        return pd.DataFrame()

    opt_time = valid[0].total_race_time_sec
    rows = []
    for rank, res in enumerate(valid[:n], 1):
        gap = res.total_race_time_sec - opt_time
        rows.append({
            "rank":                rank,
            "label":               res.strategy.label,
            "n_stops":             res.strategy.n_stops,
            "compounds":           "-".join(
                COMPOUND_ABBREV.get(c, "?") for c in res.strategy.compounds_used
            ),
            "pit_laps":            ", ".join(str(p) for p in res.strategy.pit_laps) or "none",
            "total_time_sec":      round(res.total_race_time_sec, 3),
            "total_time_formatted": res.total_time_formatted,
            "gap_to_optimal_sec":  round(gap, 3),
            "is_equivalent":       gap <= EQUIVALENCE_THRESHOLD_SEC,
            "extrapolation_warning": res.extrapolation_warning,
        })
    return pd.DataFrame(rows)


# ===========================================================================
# Pipeline entry point
# ===========================================================================

def optimise_strategy(
    model_set:           DegradationModelSet,
    feature_df:          pd.DataFrame,
    available_compounds: list[str],
    total_race_laps:     int,
    circuit:             str   = "unknown",
    season:              int   = 2023,
    max_stops:           int   = MAX_STOPS,
    pit_lane_delta_sec:  float = PIT_LANE_DELTA_SEC_BAHRAIN,
    min_stint_laps:      int   = MIN_STINT_LAPS,
    earliest_pit_lap:    int   = EARLIEST_PIT_LAP,
    leaderboard_size:    int   = DEFAULT_LEADERBOARD_SIZE,
) -> OptimizationResult:
    """
    Full pit window optimisation: enumerate → simulate → rank.

    Args:
        model_set:           DegradationModelSet from degradation_model.py.
        feature_df:          Output of feature_builder.build_feature_set().
        available_compounds: Compounds available at this event.
        total_race_laps:     Race distance.
        circuit:             Circuit name.
        season:              Year.
        max_stops:           Maximum stops to search.
        pit_lane_delta_sec:  Circuit pit lane time.
        min_stint_laps:      Minimum stint length for pruning.
        earliest_pit_lap:    Earliest stop lap to consider.
        leaderboard_size:    Number of rows in output leaderboard.

    Returns:
        OptimizationResult.
    """
    logger.info(
        "optimise_strategy: %s %d  laps=%d  compounds=%s  max_stops=%d",
        circuit, season, total_race_laps, available_compounds, max_stops,
    )

    base_lap = estimate_base_lap_time(feature_df, total_race_laps)

    strategies = enumerate_strategies(
        available_compounds = available_compounds,
        total_race_laps     = total_race_laps,
        max_stops           = max_stops,
        pit_lane_delta_sec  = pit_lane_delta_sec,
        min_stint_laps      = min_stint_laps,
        earliest_pit_lap    = earliest_pit_lap,
    )

    logger.info("optimise_strategy: simulating %d strategies...", len(strategies))
    all_results: list[SimulationResult] = []
    n_valid = 0

    for s in strategies:
        r = simulate_strategy(s, model_set, base_lap, total_race_laps)
        all_results.append(r)
        if r.is_valid:
            n_valid += 1

    valid_results = [r for r in all_results if r.is_valid]
    optimal = (
        min(valid_results, key=lambda r: r.total_race_time_sec)
        if valid_results else None
    )
    leaderboard = _build_leaderboard(all_results, leaderboard_size)

    if optimal:
        logger.info("optimise_strategy: OPTIMAL — %s", optimal.summary())

    return OptimizationResult(
        circuit           = circuit,
        season            = season,
        total_race_laps   = total_race_laps,
        base_lap_time_sec = base_lap,
        n_evaluated       = len(all_results),
        n_valid           = n_valid,
        optimal           = optimal,
        leaderboard       = leaderboard,
        all_results       = all_results,
    )


def compute_pit_window_sensitivity(
    model_set:          DegradationModelSet,
    feature_df:         pd.DataFrame,
    start_compound:     str,
    next_compound:      str,
    total_race_laps:    int,
    pit_lap_range:      Optional[tuple[int, int]] = None,
    pit_lane_delta_sec: float = PIT_LANE_DELTA_SEC_BAHRAIN,
) -> pd.DataFrame:
    """
    Generate pit window sensitivity curve for a 1-stop strategy.

    Returns total predicted race time as a function of pit lap number,
    holding compound choice fixed. Shows how wide the optimal pit window is.

    Returns:
        DataFrame with: pit_lap, total_time_sec, gap_to_optimal_sec,
        is_optimal_window.
    """
    base_lap = estimate_base_lap_time(feature_df, total_race_laps)

    if pit_lap_range is None:
        pit_lap_range = (EARLIEST_PIT_LAP, total_race_laps - MIN_STINT_LAPS)

    rows = []
    for pl in range(pit_lap_range[0], pit_lap_range[1] + 1):
        try:
            s = build_strategy(
                pit_laps           = [pl],
                compounds          = [start_compound, next_compound],
                total_race_laps    = total_race_laps,
                pit_lane_delta_sec = pit_lane_delta_sec,
            )
        except ValueError:
            continue
        r = simulate_strategy(s, model_set, base_lap, total_race_laps)
        if r.is_valid:
            rows.append({"pit_lap": pl, "total_time_sec": r.total_race_time_sec})

    if not rows:
        return pd.DataFrame(columns=["pit_lap", "total_time_sec",
                                     "gap_to_optimal_sec", "is_optimal_window"])

    df = pd.DataFrame(rows)
    opt = df["total_time_sec"].min()
    df["gap_to_optimal_sec"] = df["total_time_sec"] - opt
    df["is_optimal_window"]  = df["gap_to_optimal_sec"] <= EQUIVALENCE_THRESHOLD_SEC

    optimal_pl = int(df.loc[df["gap_to_optimal_sec"] == 0, "pit_lap"].values[0])
    window = df[df["is_optimal_window"]]
    logger.info(
        "pit_window [%s→%s]: optimal_lap=%d  window=[%d,%d]",
        start_compound, next_compound, optimal_pl,
        int(window["pit_lap"].min()), int(window["pit_lap"].max()),
    )
    return df
