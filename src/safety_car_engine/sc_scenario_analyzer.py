"""
src/safety_car_engine/sc_scenario_analyzer.py
===============================================
Safety Car and VSC scenario decision engine.

Engineering responsibility:
    When a Safety Car or VSC is deployed, evaluate the three tactical
    responses available to the strategy engineer and return a ranked
    set of expected outcomes with full simulation breakdown.

Three responses evaluated:
    PIT_NOW:       Box on the current SC lap.
                   Benefit: dramatically reduced effective pit cost (~3-5s
                   net vs ~21s green flag). Risk: pit lane closed on SC
                   deployment lap; pit queue congestion from other teams.

    STAY_OUT:      Remain on track. Maintain track position.
                   Risk: competitors who pit gain fresh tyres at low cost.
                   Correct when tyre life is long and track position is
                   decisive (Monaco, Singapore, Baku).

    PIT_NEXT_LAP:  Wait one additional lap before pitting.
                   Benefit: pit lane confirmed open; queue may clear.
                   Risk: SC may withdraw before next lap, losing the window.
                   Optimal when pit lane is closed on deployment lap AND
                   expected remaining SC duration > 2 laps.

Decision framework:
    Each option is evaluated by running the race_simulator from the
    decision lap forward with SC-adjusted lap times and pit costs.
    Total predicted race time determines the recommendation.
    Monte Carlo sampling over SC duration uncertainty gives confidence bands.

SC lap time model:
    SC laps: base_lap_time × SC_LAP_TIME_MULTIPLIER (~1.35)
    VSC laps: base_lap_time × VSC_LAP_TIME_MULTIPLIER (~1.18)
    On SC/VSC laps, tyre degradation does NOT accumulate — the car is not
    pushing, tyre temperatures drop back toward cold state, and the
    degradation model signal is meaningless for those laps.

Effective SC pit cost model:
    Green flag: stationary(2.5) + traverse(19.0) + outlap_pen(0.8)
              - inlap_pen(1.2) ≈ 21.1s
    SC stop:  stationary(2.5) + traverse_adjusted(5.0) + queue(~3-6s)
            ≈ 7-12s effective cost
    The traverse saving arises because the pit lane is traversed at
    the same absolute speed (80 km/h limit) but the reference race lap
    is ~35% slower, so the relative cost is much smaller.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from src.strategy_engine.race_simulator import (
    RaceStrategy,
    SimulationResult,
    LapResult,
    StintSpec,
    build_strategy,
    simulate_strategy,
    _vectorise_strategy,
    MAX_TYRE_AGE_WARNING,
    PIT_LANE_DELTA_SEC_BAHRAIN,
    FUEL_BURN_RATE_KG_PER_LAP,
    FUEL_LAP_TIME_SENSITIVITY_SEC_PER_KG,
)
from src.tire_model.degradation_model import DegradationModelSet
from src.safety_car_engine.sc_detector import (
    CircuitSCProfile,
    NeutralisationPeriod,
    build_default_sc_profile,
)
from src.constants import (
    PIT_STATIONARY_TIME_SEC,
    INLAP_TIME_PENALTY_SEC,
    OUTLAP_TIME_PENALTY_SEC,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Constants
# ===========================================================================

# SC lap time is ~35% above racing pace at most circuits.
# This value is calibrated against FastF1 lap time data during SC periods
# across 2021-2023 seasons (multi-circuit average).
SC_LAP_TIME_MULTIPLIER: float = 1.35

# VSC lap time is ~18% above racing pace (electronically enforced delta).
VSC_LAP_TIME_MULTIPLIER: float = 1.18

# Pit lane cost REDUCTION under SC vs green flag (seconds).
# Green-flag pit lane traverse ≈ 19s. Under SC, the racing lap is ~35%
# slower, so the relative traverse cost shrinks. Net reduction ≈ 14s.
SC_PIT_LANE_COST_REDUCTION_SEC:  float = 14.0

# VSC cost reduction is smaller — racing lap is only 18% slower.
VSC_PIT_LANE_COST_REDUCTION_SEC: float = 6.0

# Expected pit queue penalty per car already in the pit lane.
# When multiple teams pit simultaneously, crews may not be ready
# and the car waits in the fast lane. Each extra car: ~1.5s penalty.
PIT_QUEUE_PENALTY_PER_CAR_SEC: float = 1.5

# Default expected number of cars pitting on the first SC open lap.
# Historical average from FastF1 2018-2023: 4-6 cars.
DEFAULT_CARS_PITTING_UNDER_SC: int = 5

# Minimum laps remaining for any pit stop to pay back in race time.
# A stop costs minimum ~2.5s stationary. At SC pace a lap takes ~(base×1.35)
# ≈ 130s. There is always time to recover stationary cost if laps remain.
# The real constraint is whether there are enough laps on fresh tyres to
# outpace rivals who stayed out. 5 laps is the practical minimum.
MIN_LAPS_FOR_SC_PIT_PAYBACK: int = 5

# Monte Carlo samples for duration uncertainty analysis.
SC_MC_SAMPLES: int = 300

# Drive-through penalty time if pitting when pit lane is closed (seconds).
DRIVE_THROUGH_PENALTY_SEC: float = 20.0


# ===========================================================================
# Data contracts
# ===========================================================================

class PitResponse(Enum):
    """The three tactical responses to an SC/VSC deployment."""
    PIT_NOW       = "PIT_NOW"
    STAY_OUT      = "STAY_OUT"
    PIT_NEXT_LAP  = "PIT_NEXT_LAP"


@dataclass
class SCPitCostModel:
    """
    Adjusted pit stop cost for an SC or VSC period.

    Attributes:
        neutralisation_type:              "SC" or "VSC".
        effective_pit_cost_sec:           Net time cost vs staying out
                                          (green-flag cost minus SC saving).
        queue_adjusted_cost_sec:          Cost including pit lane queue.
        pit_lane_open:                    False on SC deployment lap.
        laps_of_neutralisation_remaining: Expected laps left under SC/VSC.
    """
    neutralisation_type:              str
    effective_pit_cost_sec:           float
    queue_adjusted_cost_sec:          float
    pit_lane_open:                    bool
    laps_of_neutralisation_remaining: int

    def __repr__(self) -> str:
        return (
            f"SCPitCostModel({self.neutralisation_type} "
            f"| cost={self.effective_pit_cost_sec:.2f}s "
            f"| queue={self.queue_adjusted_cost_sec:.2f}s "
            f"| open={self.pit_lane_open} "
            f"| remaining={self.laps_of_neutralisation_remaining}L)"
        )


@dataclass
class PitResponseEvaluation:
    """
    Full evaluation of a single pit response option.

    Attributes:
        response:              Which PitResponse this evaluates.
        total_race_time_sec:   Predicted total race time (seconds).
        delta_vs_stay_out_sec: Δ vs STAY_OUT baseline (negative = faster).
        p10_time_sec:          10th percentile over duration uncertainty.
        p50_time_sec:          Median (50th percentile).
        p90_time_sec:          90th percentile (pessimistic scenario).
        is_viable:             False if physically impossible or too costly.
        not_viable_reason:     Explanation when is_viable=False.
        strategy:              Strategy used for this evaluation.
        simulation:            Full SimulationResult.
        notes:                 Engineering annotation.
    """
    response:              PitResponse
    total_race_time_sec:   float
    delta_vs_stay_out_sec: float
    p10_time_sec:          float = 0.0
    p50_time_sec:          float = 0.0
    p90_time_sec:          float = 0.0
    is_viable:             bool  = True
    not_viable_reason:     str   = ""
    strategy:              Optional[RaceStrategy] = None
    simulation:            Optional[SimulationResult] = None
    notes:                 str   = ""

    def summary_line(self) -> str:
        if not self.is_viable:
            return f"  {self.response.value:<16} NOT VIABLE — {self.not_viable_reason}"
        gap = f"{self.delta_vs_stay_out_sec:+.3f}s"
        ci  = (f"[P10={self.p10_time_sec:.1f}  P90={self.p90_time_sec:.1f}]"
               if self.p10_time_sec else "")
        return (
            f"  {self.response.value:<16}"
            f"total={self.total_race_time_sec:.3f}s  "
            f"Δ={gap:<10}  {ci:<30}  {self.notes}"
        )


@dataclass
class SCDecision:
    """
    Complete output of SC scenario analysis for one neutralisation event.

    Attributes:
        period:           The NeutralisationPeriod being analysed.
        decision_lap:     Race lap at which the decision must be made.
        current_compound: Compound on car at decision_lap.
        current_tyre_age: Tyre age at decision_lap.
        next_compound:    Compound to fit if pitting.
        evaluations:      PitResponseEvaluation for each option.
        recommended:      Best PitResponse by expected race time.
        confidence:       Model confidence (0–1).
        reasoning:        Engineering rationale for recommendation.
    """
    period:           NeutralisationPeriod
    decision_lap:     int
    current_compound: str
    current_tyre_age: int
    next_compound:    str
    evaluations:      list[PitResponseEvaluation]
    recommended:      PitResponse
    confidence:       float
    reasoning:        str

    def print_analysis(self) -> None:
        print(f"\n{'='*72}")
        print(f"  SC DECISION  |  {self.period!r}")
        print(
            f"  Lap {self.decision_lap}  |  "
            f"{self.current_compound} age={self.current_tyre_age}  |  "
            f"Next compound: {self.next_compound}"
        )
        print(f"{'='*72}")
        for ev in self.evaluations:
            marker = "  ← RECOMMENDED" if ev.response == self.recommended else ""
            print(ev.summary_line() + marker)
        print(f"\n  ► {self.recommended.value}  (confidence={self.confidence:.0%})")
        print(f"  {self.reasoning}")
        print(f"{'='*72}\n")


@dataclass
class SCPortfolioResult:
    """Output of a Monte Carlo SC portfolio evaluation."""
    strategies_evaluated:  int
    sc_scenarios_sampled:  int
    strategy_rankings:     pd.DataFrame
    sc_robust_strategies:  list[str]  # Strategies that rank better under SC


# ===========================================================================
# SC pit cost model
# ===========================================================================

def compute_sc_pit_cost(
    period:               NeutralisationPeriod,
    decision_lap:         int,
    base_lap_time_sec:    float,
    pit_lane_delta_sec:   float = PIT_LANE_DELTA_SEC_BAHRAIN,
    expected_cars_pitting: int  = DEFAULT_CARS_PITTING_UNDER_SC,
) -> SCPitCostModel:
    """
    Compute the effective pit stop cost under an SC or VSC period.

    Engineering derivation:
        Green-flag net pit cost =
            PIT_STATIONARY + PIT_LANE_DELTA + OUTLAP_PENALTY - INLAP_PENALTY
            = 2.5 + 19.0 + 0.8 - 1.2 = 21.1s

        SC net pit cost =
            PIT_STATIONARY + (PIT_LANE_DELTA × SC_pace_factor)
            + OUTLAP_PENALTY_reduced - INLAP_PENALTY_reduced
            ≈ 2.5 + 5.0 + 0.3 - 0.4 ≈ 7.4s base
            + queue_congestion

        The SC_pace_factor reduction: under SC, racing laps are ~35% slower.
        The physical traverse time is the same (80km/h pit lane limit),
        but the racing lap on track is now ~130s instead of ~95s.
        The NET time loss from pitting (pit cost - time to complete the lap
        at racing pace) is therefore much smaller.

    Args:
        period:               Active NeutralisationPeriod.
        decision_lap:         Lap on which decision is being made.
        base_lap_time_sec:    Green-flag reference pace.
        pit_lane_delta_sec:   Circuit pit lane traverse time.
        expected_cars_pitting: Cars expected in pit lane simultaneously.

    Returns:
        SCPitCostModel with effective and queue-adjusted costs.
    """
    pit_open = decision_lap >= period.pit_lane_open_lap

    reduction = (
        SC_PIT_LANE_COST_REDUCTION_SEC
        if period.is_sc
        else VSC_PIT_LANE_COST_REDUCTION_SEC
    )

    effective_cost = max(
        PIT_STATIONARY_TIME_SEC,
        (PIT_STATIONARY_TIME_SEC
         + pit_lane_delta_sec
         + OUTLAP_TIME_PENALTY_SEC
         - INLAP_TIME_PENALTY_SEC
         - reduction)
    )

    # Queue congestion: each extra car pitting adds waiting time
    queue_extra    = max(0, expected_cars_pitting - 1) * PIT_QUEUE_PENALTY_PER_CAR_SEC
    queue_adjusted = effective_cost + queue_extra

    laps_remaining = max(0, period.end_lap - decision_lap + 1)

    logger.debug(
        "compute_sc_pit_cost [%s L%d]: effective=%.2fs  "
        "queue=%.2fs  open=%s  remaining=%dL",
        period.period_type, decision_lap,
        effective_cost, queue_adjusted, pit_open, laps_remaining,
    )

    return SCPitCostModel(
        neutralisation_type              = period.period_type,
        effective_pit_cost_sec           = effective_cost,
        queue_adjusted_cost_sec          = queue_adjusted,
        pit_lane_open                    = pit_open,
        laps_of_neutralisation_remaining = laps_remaining,
    )


# ===========================================================================
# SC-aware race simulator
# ===========================================================================

def simulate_under_sc(
    strategy:          RaceStrategy,
    model_set:         DegradationModelSet,
    base_lap_time_sec: float,
    total_race_laps:   int,
    sc_periods:        list[NeutralisationPeriod],
) -> SimulationResult:
    """
    Simulate a race strategy with SC/VSC lap time adjustments.

    Adjusts the base lap time on neutralised laps using the SC/VSC
    multiplier. Sets degradation delta to zero on neutralised laps —
    under SC, the car is not pushing and the tyre model is not valid.
    Adjusts pit costs for stops that fall within SC/VSC periods.

    Args:
        strategy:          RaceStrategy to simulate.
        model_set:         DegradationModelSet for tyre delta computation.
        base_lap_time_sec: Green-flag reference pace.
        total_race_laps:   Race distance.
        sc_periods:        List of NeutralisationPeriod affecting this race.

    Returns:
        SimulationResult with SC-adjusted lap times and costs.
    """
    valid, reason = strategy.is_valid()
    if not valid:
        return SimulationResult(
            strategy            = strategy,
            total_race_time_sec = float("inf"),
            lap_results         = [],
            is_valid            = False,
            invalid_reason      = reason,
        )

    # Build SC/VSC lap lookup: lap_number -> adjusted base time
    sc_lap_override: dict[int, float] = {}
    sc_lap_set: set[int]              = set()
    for period in sc_periods:
        mult = SC_LAP_TIME_MULTIPLIER if period.is_sc else VSC_LAP_TIME_MULTIPLIER
        for lap in range(period.start_lap, period.end_lap + 1):
            sc_lap_override[lap] = base_lap_time_sec * mult
            sc_lap_set.add(lap)

    compounds, tyre_ages, is_inlap, is_outlap, pit_loss = _vectorise_strategy(
        strategy, total_race_laps
    )

    # Adjust pit costs for stops under SC/VSC
    for period in sc_periods:
        cost_model = compute_sc_pit_cost(
            period             = period,
            decision_lap       = period.pit_lane_open_lap,
            base_lap_time_sec  = base_lap_time_sec,
            pit_lane_delta_sec = strategy.pit_lane_delta_sec,
        )
        for pl in strategy.pit_laps:
            if period.start_lap <= pl <= period.end_lap:
                idx = pl - 1
                if pl >= period.pit_lane_open_lap:
                    pit_loss[idx] = cost_model.queue_adjusted_cost_sec
                    logger.debug(
                        "simulate_under_sc: pit L%d under %s → "
                        "cost %.2fs (was %.2fs green flag)",
                        pl, period.period_type,
                        cost_model.queue_adjusted_cost_sec,
                        pit_loss[idx],
                    )
                else:
                    pit_loss[idx] = pit_loss[idx] + DRIVE_THROUGH_PENALTY_SEC
                    logger.warning(
                        "simulate_under_sc: pit L%d in pit-lane-CLOSED window — "
                        "+%.1fs drive-through penalty applied.",
                        pl, DRIVE_THROUGH_PENALTY_SEC,
                    )

    # Fuel delta
    lap_ns = np.arange(1, total_race_laps + 1, dtype=np.float64)
    fuel_remaining = np.maximum(
        0.0, (total_race_laps - lap_ns) * FUEL_BURN_RATE_KG_PER_LAP
    )
    fuel_delta = fuel_remaining * FUEL_LAP_TIME_SENSITIVITY_SEC_PER_KG

    # Degradation delta (zero on SC/VSC laps)
    deg_delta          = np.zeros(total_race_laps, dtype=np.float64)
    extrapolation_warn = False
    for compound in set(compounds):
        model = model_set.get(str(compound))
        if model is None:
            continue
        mask = compounds == compound
        ages = tyre_ages[mask].astype(float)
        if np.any(ages > MAX_TYRE_AGE_WARNING):
            extrapolation_warn = True
        predicted = model.predict(ages)
        # Zero out SC/VSC laps — tyre not thermally loaded
        for j, (lap_idx_bool) in enumerate(mask):
            if not lap_idx_bool:
                continue
            flat_idx = list(np.where(mask)[0])[
                list(np.where(mask)[0]).index(j)
            ] if j in np.where(mask)[0] else -1
        # Vectorised zero-out
        sc_mask_bool = np.array(
            [(i + 1) in sc_lap_set for i in range(total_race_laps)]
        )
        deg_from_model = np.zeros(total_race_laps)
        deg_from_model[mask] = predicted
        deg_from_model[sc_mask_bool] = 0.0
        deg_delta += deg_from_model

    # Base time array: SC override where neutralised
    base_times = np.full(total_race_laps, base_lap_time_sec)
    for lap_n, adj in sc_lap_override.items():
        if 1 <= lap_n <= total_race_laps:
            base_times[lap_n - 1] = adj

    inlap_pen  = is_inlap.astype(float)  * INLAP_TIME_PENALTY_SEC
    outlap_pen = is_outlap.astype(float) * OUTLAP_TIME_PENALTY_SEC

    predicted_laps = (
        base_times + fuel_delta + deg_delta
        + inlap_pen + outlap_pen + pit_loss
    )
    total_time = float(predicted_laps.sum())

    lap_results = [
        LapResult(
            lap_number         = int(lap_ns[i]),
            compound           = str(compounds[i]),
            tyre_age           = int(tyre_ages[i]),
            base_lap_sec       = float(base_times[i]),
            fuel_delta_sec     = float(fuel_delta[i]),
            deg_delta_sec      = float(deg_delta[i]),
            pit_loss_sec       = float(pit_loss[i]),
            inlap_penalty_sec  = float(inlap_pen[i]),
            outlap_penalty_sec = float(outlap_pen[i]),
            is_pit_entry_lap   = bool(is_inlap[i]),
            is_pit_exit_lap    = bool(is_outlap[i]),
            predicted_lap_sec  = float(predicted_laps[i]),
        )
        for i in range(total_race_laps)
    ]

    return SimulationResult(
        strategy              = strategy,
        total_race_time_sec   = total_time,
        lap_results           = lap_results,
        is_valid              = True,
        extrapolation_warning = extrapolation_warn,
    )


# ===========================================================================
# Strategy modification helpers
# ===========================================================================

def _build_modified_pit_laps(
    original:        RaceStrategy,
    new_pit_lap:     int,
    total_race_laps: int,
) -> list[int]:
    """
    Replace the next planned pit stop with new_pit_lap.

    Keeps all past stops (before new_pit_lap), inserts new_pit_lap,
    and drops the first future stop that new_pit_lap is replacing.
    If no future stop exists, new_pit_lap is added as an extra stop.
    """
    past   = [p for p in original.pit_laps if p < new_pit_lap]
    future = [p for p in original.pit_laps if p >= new_pit_lap]
    # Replace the first future stop with new_pit_lap
    modified = past + [new_pit_lap] + (future[1:] if future else [])
    return sorted(set(p for p in modified if 1 <= p < total_race_laps))


def _build_modified_compounds(
    original:          RaceStrategy,
    pit_lap:           int,
    new_next_compound: str,
) -> list[str]:
    """
    Build compound sequence for a strategy pitting on pit_lap.

    Inserts new_next_compound at the position corresponding to pit_lap
    in the compound sequence, preserving all subsequent planned compounds.
    """
    # Find which stint contains pit_lap
    current_idx = 0
    for i, stint in enumerate(original.stints):
        if stint.start_lap <= pit_lap <= stint.end_lap:
            current_idx = i
            break

    pre  = [s.compound for s in original.stints[:current_idx]]
    post = [s.compound for s in original.stints[current_idx + 1:]]
    curr = original.stints[current_idx].compound

    return pre + [curr, new_next_compound] + post


# ===========================================================================
# Three-way response evaluator
# ===========================================================================

def evaluate_sc_pit_options(
    period:             NeutralisationPeriod,
    decision_lap:       int,
    current_strategy:   RaceStrategy,
    next_compound:      str,
    model_set:          DegradationModelSet,
    base_lap_time_sec:  float,
    total_race_laps:    int,
    current_tyre_age:   int,
    current_compound:   str,
    pit_lane_delta_sec: float = PIT_LANE_DELTA_SEC_BAHRAIN,
    sc_profile:         Optional[CircuitSCProfile] = None,
    mc_samples:         int   = SC_MC_SAMPLES,
    rng_seed:           Optional[int] = None,
) -> SCDecision:
    """
    Evaluate all three pit response options for an active SC/VSC deployment.

    Constructs a forward simulation from decision_lap to race end for each
    response option, incorporating SC lap time adjustments and adjusted
    pit costs. Monte Carlo sampling over SC duration uncertainty provides
    confidence intervals for each option's race time prediction.

    Args:
        period:             Active NeutralisationPeriod.
        decision_lap:       Current race lap (decision must be made now).
        current_strategy:   Strategy being followed at SC deployment.
        next_compound:      Compound to fit if pitting under SC.
        model_set:          Fitted DegradationModelSet.
        base_lap_time_sec:  Green-flag reference pace.
        total_race_laps:    Race distance.
        current_tyre_age:   Tyre age at decision_lap.
        current_compound:   Current compound on car.
        pit_lane_delta_sec: Circuit pit lane traverse time.
        sc_profile:         CircuitSCProfile for duration uncertainty.
                            Uses default if None.
        mc_samples:         Monte Carlo samples for confidence intervals.
        rng_seed:           Optional RNG seed.

    Returns:
        SCDecision with all three evaluations and a recommendation.
    """
    laps_remaining = total_race_laps - decision_lap
    rng = np.random.default_rng(rng_seed)

    if sc_profile is None:
        sc_profile = build_default_sc_profile("unknown", total_race_laps)

    # Guard: too few laps remaining for any pit to pay back
    if laps_remaining < MIN_LAPS_FOR_SC_PIT_PAYBACK:
        stay_sim = simulate_under_sc(
            current_strategy, model_set, base_lap_time_sec,
            total_race_laps, [period],
        )
        ev = PitResponseEvaluation(
            response              = PitResponse.STAY_OUT,
            total_race_time_sec   = stay_sim.total_race_time_sec,
            delta_vs_stay_out_sec = 0.0,
            is_viable             = True,
            strategy              = current_strategy,
            simulation            = stay_sim,
            notes                 = f"Only {laps_remaining} laps remain.",
        )
        return SCDecision(
            period           = period,
            decision_lap     = decision_lap,
            current_compound = current_compound,
            current_tyre_age = current_tyre_age,
            next_compound    = next_compound,
            evaluations      = [ev],
            recommended      = PitResponse.STAY_OUT,
            confidence       = 0.90,
            reasoning        = f"{laps_remaining} laps remaining — no pit can pay back.",
        )

    cost_model = compute_sc_pit_cost(
        period             = period,
        decision_lap       = decision_lap,
        base_lap_time_sec  = base_lap_time_sec,
        pit_lane_delta_sec = pit_lane_delta_sec,
    )

    evaluations: list[PitResponseEvaluation] = []

    # --- Helper: run MC over duration uncertainty ---
    def _mc_time_distribution(
        strategy:    RaceStrategy,
        pit_lap:     Optional[int],
    ) -> tuple[float, float, float, float]:
        """Return (mean, p10, p50, p90) of race time over sampled SC durations."""
        times = []
        for _ in range(mc_samples):
            dur        = sc_profile.sample_sc_duration(rng) if period.is_sc else sc_profile.sample_vsc_duration(rng)
            end_lap    = min(period.start_lap + dur - 1, total_race_laps)
            sampled_period = NeutralisationPeriod(
                period_type       = period.period_type,
                start_lap         = period.start_lap,
                end_lap           = end_lap,
                duration_laps     = end_lap - period.start_lap + 1,
                pit_lane_open_lap = period.pit_lane_open_lap,
            )
            res = simulate_under_sc(
                strategy, model_set, base_lap_time_sec,
                total_race_laps, [sampled_period],
            )
            if res.is_valid:
                times.append(res.total_race_time_sec)
        if not times:
            return float("inf"), float("inf"), float("inf"), float("inf")
        arr = np.array(times)
        return float(np.mean(arr)), float(np.percentile(arr,10)), float(np.percentile(arr,50)), float(np.percentile(arr,90))

    # ------------------------------------------------------------------
    # STAY_OUT
    # ------------------------------------------------------------------
    stay_sim = simulate_under_sc(
        current_strategy, model_set, base_lap_time_sec,
        total_race_laps, [period],
    )
    stay_time = stay_sim.total_race_time_sec
    stay_mean, stay_p10, stay_p50, stay_p90 = _mc_time_distribution(
        current_strategy, None
    )
    evaluations.append(PitResponseEvaluation(
        response              = PitResponse.STAY_OUT,
        total_race_time_sec   = stay_time,
        delta_vs_stay_out_sec = 0.0,
        p10_time_sec          = stay_p10,
        p50_time_sec          = stay_p50,
        p90_time_sec          = stay_p90,
        is_viable             = True,
        strategy              = current_strategy,
        simulation            = stay_sim,
        notes                 = "Baseline — continue planned strategy unchanged.",
    ))

    # ------------------------------------------------------------------
    # PIT_NOW
    # ------------------------------------------------------------------
    now_viable  = True
    now_reason  = ""
    now_sim     = None
    now_strat   = None
    now_notes   = (
        f"Effective cost ≈ {cost_model.queue_adjusted_cost_sec:.1f}s "
        f"({'open' if cost_model.pit_lane_open else 'CLOSED — drive-through risk'})"
    )

    if not cost_model.pit_lane_open:
        now_notes = (
            f"⚠ Pit lane CLOSED on L{decision_lap}. "
            f"Opens L{period.pit_lane_open_lap}. "
            f"Pit now = +{DRIVE_THROUGH_PENALTY_SEC:.0f}s drive-through penalty. "
            + now_notes
        )

    try:
        pits_now  = _build_modified_pit_laps(current_strategy, decision_lap, total_race_laps)
        comps_now = _build_modified_compounds(current_strategy, decision_lap, next_compound)
        now_strat = build_strategy(
            pit_laps           = pits_now,
            compounds          = comps_now,
            total_race_laps    = total_race_laps,
            pit_lane_delta_sec = pit_lane_delta_sec,
        )
        now_sim = simulate_under_sc(
            now_strat, model_set, base_lap_time_sec, total_race_laps, [period],
        )
        now_mean, now_p10, now_p50, now_p90 = _mc_time_distribution(now_strat, decision_lap)
    except ValueError as exc:
        now_viable = False
        now_reason = f"Strategy construction failed: {exc}"
        now_mean = now_p10 = now_p50 = now_p90 = float("inf")
        logger.warning("evaluate_sc_pit_options [PIT_NOW]: %s", now_reason)

    now_time = now_sim.total_race_time_sec if now_sim else float("inf")
    evaluations.append(PitResponseEvaluation(
        response              = PitResponse.PIT_NOW,
        total_race_time_sec   = now_time,
        delta_vs_stay_out_sec = now_time - stay_time,
        p10_time_sec          = now_p10,
        p50_time_sec          = now_p50,
        p90_time_sec          = now_p90,
        is_viable             = now_viable and now_sim is not None,
        not_viable_reason     = now_reason,
        strategy              = now_strat,
        simulation            = now_sim,
        notes                 = now_notes,
    ))

    # ------------------------------------------------------------------
    # PIT_NEXT_LAP
    # ------------------------------------------------------------------
    next_pl     = decision_lap + 1
    next_viable = next_pl < total_race_laps
    next_reason = ""
    next_sim    = None
    next_strat  = None
    sc_still_open_next = next_pl <= period.end_lap

    next_notes = (
        "Pit lane confirmed open; queue may have cleared. "
        + ("SC still active next lap." if sc_still_open_next
           else "⚠ SC may end before next lap — green-flag cost risk.")
    )

    if not next_viable:
        next_viable = False
        next_reason = "No laps remaining after next lap."

    if next_viable:
        try:
            pits_next  = _build_modified_pit_laps(current_strategy, next_pl, total_race_laps)
            comps_next = _build_modified_compounds(current_strategy, next_pl, next_compound)
            next_strat = build_strategy(
                pit_laps           = pits_next,
                compounds          = comps_next,
                total_race_laps    = total_race_laps,
                pit_lane_delta_sec = pit_lane_delta_sec,
            )
            next_sim = simulate_under_sc(
                next_strat, model_set, base_lap_time_sec, total_race_laps,
                [period] if sc_still_open_next else [],
            )
            next_mean, next_p10, next_p50, next_p90 = _mc_time_distribution(next_strat, next_pl)
        except ValueError as exc:
            next_viable = False
            next_reason = f"Strategy construction failed: {exc}"
            next_mean = next_p10 = next_p50 = next_p90 = float("inf")
            logger.warning("evaluate_sc_pit_options [PIT_NEXT_LAP]: %s", next_reason)

    next_time = next_sim.total_race_time_sec if next_sim else float("inf")
    evaluations.append(PitResponseEvaluation(
        response              = PitResponse.PIT_NEXT_LAP,
        total_race_time_sec   = next_time,
        delta_vs_stay_out_sec = next_time - stay_time,
        p10_time_sec          = next_p10 if next_viable else 0.0,
        p50_time_sec          = next_p50 if next_viable else 0.0,
        p90_time_sec          = next_p90 if next_viable else 0.0,
        is_viable             = next_viable and next_sim is not None,
        not_viable_reason     = next_reason,
        strategy              = next_strat,
        simulation            = next_sim,
        notes                 = next_notes,
    ))

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------
    viable = [ev for ev in evaluations if ev.is_viable]
    if not viable:
        recommended = PitResponse.STAY_OUT
        confidence  = 0.50
        reasoning   = "No viable pit option — defaulting to STAY_OUT."
    else:
        best = min(viable, key=lambda ev: ev.total_race_time_sec)
        recommended  = best.response
        time_saved   = -best.delta_vs_stay_out_sec  # Positive = we gain time

        if recommended == PitResponse.STAY_OUT:
            confidence = 0.72
            reasoning  = "STAY_OUT is fastest — planned strategy is optimal under SC."
        elif time_saved < 0.5:
            confidence = 0.50
            reasoning  = (
                f"{recommended.value} saves only {time_saved:.2f}s — within "
                f"model uncertainty. Marginal decision."
            )
        elif time_saved < 3.0:
            confidence = 0.75
            reasoning  = (
                f"{recommended.value} saves {time_saved:.2f}s vs STAY_OUT. "
                f"Clear benefit from discounted {period.period_type} pit cost."
            )
        else:
            confidence = 0.90
            reasoning  = (
                f"{recommended.value} saves {time_saved:.2f}s — "
                f"significant benefit from {'free stop' if period.is_sc else 'discounted VSC stop'}. "
                f"High confidence."
            )

    decision = SCDecision(
        period           = period,
        decision_lap     = decision_lap,
        current_compound = current_compound,
        current_tyre_age = current_tyre_age,
        next_compound    = next_compound,
        evaluations      = evaluations,
        recommended      = recommended,
        confidence       = confidence,
        reasoning        = reasoning,
    )

    logger.info(
        "evaluate_sc_pit_options [%s L%d]: → %s (conf=%.0%%) — %s",
        period.period_type, decision_lap,
        recommended.value, confidence, reasoning,
    )
    return decision


# ===========================================================================
# Monte Carlo portfolio analysis
# ===========================================================================

def evaluate_strategy_portfolio_under_sc(
    strategies:         list[RaceStrategy],
    model_set:          DegradationModelSet,
    base_lap_time_sec:  float,
    total_race_laps:    int,
    sc_profile:         Optional[CircuitSCProfile] = None,
    circuit:            str = "unknown",
    n_samples:          int = SC_MC_SAMPLES,
    rng_seed:           Optional[int] = None,
) -> SCPortfolioResult:
    """
    Evaluate a portfolio of strategies across Monte Carlo SC scenarios.

    For each sample, randomly draws SC/VSC occurrence, deployment lap,
    and duration from sc_profile, then simulates all strategies under
    those conditions. Aggregates rank statistics across all samples to
    identify risk-adjusted optimal strategies and SC-robust strategies.

    SC-robust strategies are those whose mean rank IMPROVES under SC
    scenarios vs no-SC scenarios — these are strategies whose pit windows
    happen to align well with likely SC deployment laps.

    Args:
        strategies:        Portfolio of RaceStrategy objects.
        model_set:         DegradationModelSet.
        base_lap_time_sec: Green-flag reference pace.
        total_race_laps:   Race distance.
        sc_profile:        CircuitSCProfile. Uses global prior if None.
        circuit:           Circuit name (for default profile creation).
        n_samples:         Monte Carlo iterations.
        rng_seed:          Optional RNG seed.

    Returns:
        SCPortfolioResult with strategy_rankings DataFrame and
        sc_robust_strategies list.
    """
    if sc_profile is None:
        sc_profile = build_default_sc_profile(circuit, total_race_laps)

    rng = np.random.default_rng(rng_seed)

    logger.info(
        "evaluate_strategy_portfolio_under_sc: %d strategies × %d samples",
        len(strategies), n_samples,
    )

    # Baseline rankings (no SC)
    baseline: dict[str, float] = {}
    for s in strategies:
        r = simulate_strategy(s, model_set, base_lap_time_sec, total_race_laps)
        if r.is_valid:
            baseline[s.label] = r.total_race_time_sec

    records: list[dict] = []

    for sample_i in range(n_samples):
        sc_occurs  = rng.random() < (sc_profile.sc_frequency  / total_race_laps * 15)
        vsc_occurs = rng.random() < (sc_profile.vsc_frequency / total_race_laps * 10)

        periods: list[NeutralisationPeriod] = []

        if sc_occurs and len(sc_profile.sc_deployment_laps) > 0:
            sc_lap = int(np.clip(
                rng.choice(sc_profile.sc_deployment_laps), 1, total_race_laps - 3
            ))
            sc_dur = sc_profile.sample_sc_duration(rng)
            sc_end = min(sc_lap + sc_dur - 1, total_race_laps)
            periods.append(NeutralisationPeriod(
                period_type="SC", start_lap=sc_lap, end_lap=sc_end,
                duration_laps=sc_end-sc_lap+1, pit_lane_open_lap=sc_lap+1,
            ))

        if vsc_occurs and len(sc_profile.vsc_deployment_laps) > 0:
            vsc_lap = int(np.clip(
                rng.choice(sc_profile.vsc_deployment_laps), 1, total_race_laps - 2
            ))
            vsc_dur = sc_profile.sample_vsc_duration(rng)
            vsc_end = min(vsc_lap + vsc_dur - 1, total_race_laps)
            if not periods or vsc_lap > periods[-1].end_lap + 2:
                periods.append(NeutralisationPeriod(
                    period_type="VSC", start_lap=vsc_lap, end_lap=vsc_end,
                    duration_laps=vsc_end-vsc_lap+1, pit_lane_open_lap=vsc_lap,
                ))

        sample_times: dict[str, float] = {}
        for s in strategies:
            res = (
                simulate_under_sc(s, model_set, base_lap_time_sec, total_race_laps, periods)
                if periods
                else simulate_strategy(s, model_set, base_lap_time_sec, total_race_laps)
            )
            if res.is_valid:
                sample_times[s.label] = res.total_race_time_sec

        sorted_labels = sorted(sample_times, key=lambda l: sample_times[l])
        for rank, label in enumerate(sorted_labels, 1):
            records.append({
                "sample":  sample_i,
                "label":   label,
                "rank":    rank,
                "time":    sample_times[label],
                "has_sc":  sc_occurs,
            })

    if not records:
        return SCPortfolioResult(
            strategies_evaluated = len(strategies),
            sc_scenarios_sampled = n_samples,
            strategy_rankings    = pd.DataFrame(),
            sc_robust_strategies = [],
        )

    df = pd.DataFrame(records)

    agg = (
        df.groupby("label")
        .agg(
            mean_rank    = ("rank", "mean"),
            std_rank     = ("rank", "std"),
            p10_rank     = ("rank", lambda x: np.percentile(x, 10)),
            p90_rank     = ("rank", lambda x: np.percentile(x, 90)),
            mean_time    = ("time", "mean"),
            sc_mean_rank = ("rank", lambda x: x[df.loc[x.index, "has_sc"]].mean()),
            no_sc_rank   = ("rank", lambda x: x[~df.loc[x.index, "has_sc"]].mean()),
        )
        .reset_index()
        .sort_values("mean_rank")
    )
    agg["sc_improvement"] = agg["no_sc_rank"] - agg["sc_mean_rank"]

    if baseline:
        best_base = min(baseline.values())
        agg["baseline_gap_sec"] = agg["label"].map(
            lambda l: round(baseline.get(l, float("nan")) - best_base, 3)
        )

    sc_robust = agg.loc[agg["sc_improvement"] > 1.0, "label"].tolist()

    logger.info(
        "evaluate_strategy_portfolio_under_sc: best='%s'  SC-robust=%s",
        agg.iloc[0]["label"] if len(agg) else "N/A",
        sc_robust,
    )

    return SCPortfolioResult(
        strategies_evaluated = len(strategies),
        sc_scenarios_sampled = n_samples,
        strategy_rankings    = agg,
        sc_robust_strategies = sc_robust,
    )
