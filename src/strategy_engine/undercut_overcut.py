"""
src/strategy_engine/undercut_overcut.py
=========================================
Competitor interaction: undercut, overcut, and mirror decision engine.

Engineering responsibility:
    Evaluate whether to pit before (undercut), stay out (overcut), or
    follow (mirror) a specific competitor, given the current race gap,
    tyre states, and degradation model predictions.

    This module adds the COMPETITOR DIMENSION absent from the isolated
    optimiser. A strategy optimal in isolation is often wrong when a
    competitor is within the strategy window.

Definitions:
    Undercut: Pit BEFORE competitor. Get fresh tyres and lap faster than
              their older set. Aim to emerge ahead when they eventually pit.
              Works when: fresh-tyre pace delta × analysis_laps > gap + pit_cost.

    Overcut:  Stay OUT while competitor pits. Bank track position advantage.
              Works when: pit_cost_they_incur > our pace loss from older tyres
              over the analysis window.

    Mirror:   Pit at the same time. Preserves relative position. Used when
              neither undercut nor overcut has a clear expected-value advantage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from src.tire_model.degradation_model import DegradationModelSet
from src.constants import (
    INLAP_TIME_PENALTY_SEC,
    OUTLAP_TIME_PENALTY_SEC,
    PIT_STATIONARY_TIME_SEC,
    PIT_LANE_DELTA_SEC_BAHRAIN,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Constants
# ===========================================================================

UNDERCUT_ANALYSIS_LAPS:          int   = 3
OVERCUT_ANALYSIS_LAPS:           int   = 5
MIN_GAP_FOR_UNDERCUT_SEC:        float = 0.5
RECOMMENDATION_CONFIDENCE_FLOOR: float = 0.60
GAP_OUTSIDE_STRATEGY_WINDOW_SEC: float = 25.0

# Circuit overtaking difficulty (0=impossible, 1=trivial).
# Scales the gap margin required for an undercut to yield an actual overtake.
OVERTAKING_DIFFICULTY_BAHRAIN: float = 0.75


# ===========================================================================
# Data contracts
# ===========================================================================

class Decision(Enum):
    UNDERCUT  = "UNDERCUT"
    OVERCUT   = "OVERCUT"
    MIRROR    = "MIRROR"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class GapScenario:
    """Current on-track state for competitor interaction analysis."""
    our_driver:           str
    their_driver:         str
    current_lap:          int
    gap_ahead_sec:        float   # Positive = they are ahead; negative = they are behind
    our_compound:         str
    our_tyre_age:         int
    their_compound:       str
    their_tyre_age:       int
    our_next_compound:    str
    their_next_compound:  str
    total_race_laps:      int
    pit_lane_delta_sec:   float = PIT_LANE_DELTA_SEC_BAHRAIN
    overtaking_difficulty: float = OVERTAKING_DIFFICULTY_BAHRAIN


@dataclass
class InteractionDecision:
    """Output of a competitor interaction analysis."""
    scenario:             GapScenario
    decision:             Decision
    confidence:           float
    undercut_gain_sec:    float
    overcut_gain_sec:     float
    fresh_tyre_delta:     float
    pit_total_cost_sec:   float
    laps_remaining:       int
    reasoning:            str

    def summary(self) -> str:
        return (
            f"{self.scenario.our_driver} vs {self.scenario.their_driver} | "
            f"L{self.scenario.current_lap}  gap={self.scenario.gap_ahead_sec:+.2f}s | "
            f"→ {self.decision.value} ({self.confidence:.0%}): {self.reasoning}"
        )


# ===========================================================================
# Pace helpers
# ===========================================================================

def _fresh_tyre_pace_gain(
    model_set:        DegradationModelSet,
    next_compound:    str,
    current_tyre_age: int,
) -> float:
    """
    Per-lap pace improvement from switching to a fresh tyre of next_compound.

    = degradation_at_current_age - degradation_at_age_1 (always 0)
    = degradation_at_current_age

    Returns 0.0 if no model for next_compound.
    """
    delta = model_set.predict(next_compound, current_tyre_age)
    return float(delta) if delta is not None else 0.0


# ===========================================================================
# Public decision engine
# ===========================================================================

def evaluate_interaction(
    scenario:  GapScenario,
    model_set: DegradationModelSet,
) -> InteractionDecision:
    """
    Evaluate undercut, overcut, and mirror for a given gap scenario.

    Decision logic:
        1. If |gap| > GAP_OUTSIDE_STRATEGY_WINDOW_SEC → MIRROR (not in window)
        2. Compute undercut_gain: fresh pace × laps - pit cost - overtake margin
        3. Compute overcut_gain: pit cost banked - their fresh pace × laps
        4. Select best option if confidence > RECOMMENDATION_CONFIDENCE_FLOOR
        5. UNCERTAIN if undercut and overcut are within 0.5s of each other
        6. Default: MIRROR (safe neutral option)

    Args:
        scenario:  GapScenario with current race state.
        model_set: Fitted DegradationModelSet.

    Returns:
        InteractionDecision with recommendation and analysis breakdown.
    """
    laps_remaining = scenario.total_race_laps - scenario.current_lap

    # Guard: outside strategy window
    if abs(scenario.gap_ahead_sec) > GAP_OUTSIDE_STRATEGY_WINDOW_SEC:
        return InteractionDecision(
            scenario          = scenario,
            decision          = Decision.MIRROR,
            confidence        = 0.95,
            undercut_gain_sec = 0.0,
            overcut_gain_sec  = 0.0,
            fresh_tyre_delta  = 0.0,
            pit_total_cost_sec = 0.0,
            laps_remaining    = laps_remaining,
            reasoning         = (
                f"Gap {scenario.gap_ahead_sec:+.1f}s outside strategy "
                f"window ({GAP_OUTSIDE_STRATEGY_WINDOW_SEC}s) — MIRROR."
            ),
        )

    total_pit_cost = (
        PIT_STATIONARY_TIME_SEC
        + scenario.pit_lane_delta_sec
        + OUTLAP_TIME_PENALTY_SEC
        - INLAP_TIME_PENALTY_SEC
    )

    our_fresh_delta   = _fresh_tyre_pace_gain(model_set, scenario.our_next_compound,
                                               scenario.our_tyre_age)
    their_fresh_delta = _fresh_tyre_pace_gain(model_set, scenario.their_next_compound,
                                               scenario.their_tyre_age)
    their_current_deg = _fresh_tyre_pace_gain(model_set, scenario.their_compound,
                                               scenario.their_tyre_age)

    # --- Undercut gain ---
    # Gap when we emerge from pit lane:
    gap_at_exit = scenario.gap_ahead_sec - total_pit_cost
    # Pace advantage per lap after pit:
    per_lap_gain = our_fresh_delta + their_current_deg
    # Required gap to actually make the overtake (circuit difficulty adjusted)
    overtake_margin = (1.0 - scenario.overtaking_difficulty) * 2.0
    undercut_gain = (
        gap_at_exit
        + per_lap_gain * min(UNDERCUT_ANALYSIS_LAPS, laps_remaining)
        - overtake_margin
    )

    # --- Overcut gain ---
    # While they are in the pit lane, we bank their pit cost.
    # When they rejoin, they chase with their_fresh_delta - our degradation loss.
    gap_banked     = total_pit_cost + OUTLAP_TIME_PENALTY_SEC
    closing_rate   = their_fresh_delta - their_current_deg
    overcut_gain   = (
        -scenario.gap_ahead_sec  # their gap behind us if they're behind
        + gap_banked
        - closing_rate * min(OVERCUT_ANALYSIS_LAPS, laps_remaining)
    )

    # Logistic confidence: smooth 0→1 mapping of gain → probability
    undercut_prob = float(1.0 / (1.0 + np.exp(-undercut_gain)))
    overcut_prob  = float(1.0 / (1.0 + np.exp(-overcut_gain / 2.0)))

    # Decision
    decision   = Decision.MIRROR
    confidence = 0.65
    reasoning  = "Neither undercut nor overcut is clearly superior — MIRROR."

    if laps_remaining < 5:
        decision   = Decision.MIRROR
        confidence = 0.90
        reasoning  = f"{laps_remaining} laps remaining — pit cannot pay back. MIRROR."

    elif abs(undercut_gain - overcut_gain) < 0.5:
        decision   = Decision.UNCERTAIN
        confidence = 0.45
        reasoning  = (
            f"Undercut gain ({undercut_gain:+.2f}s) ≈ overcut gain "
            f"({overcut_gain:+.2f}s) — insufficient precision to distinguish."
        )

    elif (undercut_gain > 0
          and undercut_prob > RECOMMENDATION_CONFIDENCE_FLOOR
          and scenario.gap_ahead_sec > MIN_GAP_FOR_UNDERCUT_SEC):
        decision   = Decision.UNDERCUT
        confidence = undercut_prob
        reasoning  = (
            f"Fresh {scenario.our_next_compound} pace +{our_fresh_delta:.2f}s/lap "
            f"+ their degradation {their_current_deg:.2f}s/lap → "
            f"{undercut_gain:+.2f}s projected gain. Undercut viable."
        )

    elif overcut_gain > 0 and overcut_prob > RECOMMENDATION_CONFIDENCE_FLOOR:
        decision   = Decision.OVERCUT
        confidence = overcut_prob
        reasoning  = (
            f"Gap banked during their stop: {gap_banked:.2f}s. "
            f"Closing rate {closing_rate:.2f}s/lap insufficient over "
            f"{OVERCUT_ANALYSIS_LAPS} laps. Overcut viable."
        )

    result = InteractionDecision(
        scenario          = scenario,
        decision          = decision,
        confidence        = confidence,
        undercut_gain_sec = undercut_gain,
        overcut_gain_sec  = overcut_gain,
        fresh_tyre_delta  = our_fresh_delta,
        pit_total_cost_sec = total_pit_cost,
        laps_remaining    = laps_remaining,
        reasoning         = reasoning,
    )

    logger.info("evaluate_interaction: %s", result.summary())
    return result


def batch_evaluate(
    our_driver:         str,
    current_lap:        int,
    our_compound:       str,
    our_tyre_age:       int,
    our_next_compound:  str,
    total_race_laps:    int,
    competitor_states:  list[dict],
    model_set:          DegradationModelSet,
    pit_lane_delta_sec: float = PIT_LANE_DELTA_SEC_BAHRAIN,
    overtaking_difficulty: float = OVERTAKING_DIFFICULTY_BAHRAIN,
) -> list[InteractionDecision]:
    """
    Evaluate interaction decisions against multiple competitors simultaneously.

    competitor_states: list of dicts with keys:
        driver_code, gap_sec, compound, tyre_age,
        next_compound (optional, defaults to MEDIUM)

    Returns decisions sorted by |gap_sec| ascending (closest competitor first —
    most urgent decision always at index 0).
    """
    decisions = []
    for comp in competitor_states:
        scenario = GapScenario(
            our_driver            = our_driver,
            their_driver          = comp["driver_code"],
            current_lap           = current_lap,
            gap_ahead_sec         = comp["gap_sec"],
            our_compound          = our_compound,
            our_tyre_age          = our_tyre_age,
            their_compound        = comp["compound"],
            their_tyre_age        = comp["tyre_age"],
            our_next_compound     = our_next_compound,
            their_next_compound   = comp.get("next_compound", "MEDIUM"),
            total_race_laps       = total_race_laps,
            pit_lane_delta_sec    = pit_lane_delta_sec,
            overtaking_difficulty = overtaking_difficulty,
        )
        decisions.append(evaluate_interaction(scenario, model_set))

    decisions.sort(key=lambda d: abs(d.scenario.gap_ahead_sec))
    return decisions
