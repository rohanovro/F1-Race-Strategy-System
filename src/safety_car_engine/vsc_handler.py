"""
src/safety_car_engine/vsc_handler.py
======================================
Virtual Safety Car (VSC) time delta modelling and pit cost adjustment.

Engineering responsibility:
    Model the time implications of the Virtual Safety Car separately from
    the full Safety Car. The VSC and SC share the same strategic logic
    (should I pit or stay out?) but have fundamentally different physics:

    FULL SC:
        All cars queue behind the physical safety car at ~80 km/h.
        Lap time: ~170s vs ~90s racing lap at Bahrain = +80s/lap.
        Pit cost: ~19s (pit lane traverse) + 2.5s stationary.
        The pit costs the same time as normal — but the SC compresses
        the field, so the GAP COST of pitting is essentially zero.
        A free stop under SC = pitting with no time penalty vs competitors.

    VSC:
        Each driver independently targets a target lap time ~40% slower
        than racing pace. There is NO physical car to queue behind.
        Lap time: ~130s vs ~90s racing lap at Bahrain = +40s/lap.
        Pit cost: ~15-17s delta (smaller because pit lane time is a
        SMALLER fraction of the slower VSC lap time).
        The gap compression is LESS than under a full SC — a stop under
        VSC still loses ~15s to competitors who stay out, vs ~2-3s under SC.

    This difference is critical for strategy:
        - Under full SC: pitting is almost always free. The only question
          is whether you have a better compound choice available.
        - Under VSC: pitting is discounted but NOT free. Only worth it if
          the tyre delta benefit outweighs the remaining pit cost.

    Conflating VSC and SC time models produces wrong pit recommendations:
        - Treating VSC as full SC → over-recommends pitting under VSC
        - Treating SC as VSC → under-recommends pitting under SC

Engineering note on VSC delta calculation:
    The VSC pit cost delta is circuit-specific because it depends on:
        1. VSC target delta time (usually ~40% slower than racing pace)
        2. Pit lane traverse time (fixed per circuit)
        3. The ratio of pit time to VSC lap time

    For Bahrain:
        Racing pace: ~93s/lap
        VSC target:  ~93 × 1.40 = ~130s/lap
        Pit lane time: 19s traverse + 2.5s stationary = 21.5s
        Time "saved" vs doing the VSC lap at racing pace: 130 - 93 = 37s
        VSC pit cost: 21.5s - 37s = -15.5s → pitting saves 15.5s on that lap
        BUT the gap to competitors stays: they also go 37s slower.
        NET cost of VSC stop = pit_stationary + traverse - vsc_lap_delta
                             = 21.5 - (vsc_lap_sec - racing_lap_sec) × 0
                             = 21.5s lost to cars behind who stay out

    In practice, the NET gap cost of a VSC stop is:
        = PIT_STATIONARY + PIT_LANE_TRAVERSE_TIME
        (because the VSC slows everyone equally — the lap time gain is offset
        by competitors also being slowed)
    This is approximately 21-22s at Bahrain vs ~2-3s under a full SC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.constants import (
    PIT_STATIONARY_TIME_SEC,
    PIT_LANE_DELTA_SEC_BAHRAIN,
    INLAP_TIME_PENALTY_SEC,
    OUTLAP_TIME_PENALTY_SEC,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Constants
# ===========================================================================

# VSC target lap time multiplier vs racing pace.
# Under VSC, drivers target a lap time 35-45% slower than racing pace.
# The FIA mandates a minimum lap time based on the fastest lap of the race.
# We use 1.40 as the centre estimate for Bahrain and most circuits.
VSC_LAP_TIME_MULTIPLIER: float = 1.40

# Under a full SC, the entire field compresses. The time gap between
# cars within the SC queue is effectively eliminated. We model this
# as a gap compression factor applied to the current on-track gap.
# 0.15 = only 15% of the original gap remains after compression.
# Based on analysis of historical FastF1 gap data under SC periods.
SC_GAP_COMPRESSION_FACTOR: float = 0.15

# Under VSC, gap compression is much less pronounced — cars are spread
# around the circuit, each driving to their own target time independently.
VSC_GAP_COMPRESSION_FACTOR: float = 0.70

# VSC net pit cost: the time lost relative to competitors who stay out.
# See module docstring for the derivation.
# This is circuit-agnostic to first order — stationary + traverse dominates.
VSC_NET_PIT_COST_SEC: float = PIT_STATIONARY_TIME_SEC + PIT_LANE_DELTA_SEC_BAHRAIN

# SC net pit cost: almost zero gap cost because field compression equalises
# the lap time loss. In practice ~2-4s is lost from the pit lane entry/exit
# delta within the SC queue.
SC_NET_PIT_COST_SEC: float = 3.0

# Minimum fresh tyre pace advantage to recommend a VSC stop (s/lap).
# Below this, the VSC net cost (21-22s) cannot be recovered within the
# remaining laps on the pace advantage alone.
VSC_MIN_TYRE_ADVANTAGE_FOR_STOP_SEC_PER_LAP: float = 0.25

# SC threshold: much lower because the pit cost is nearly zero.
SC_MIN_TYRE_ADVANTAGE_FOR_STOP_SEC_PER_LAP: float = 0.05

# Circuit-specific VSC lap time multipliers.
# Some circuits (Monaco, Hungary) are naturally slower, making the VSC
# delta smaller relative to racing pace.
VSC_MULTIPLIER_BY_CIRCUIT: dict[str, float] = {
    "bahrain":          1.40,
    "saudi_arabia":     1.38,
    "australia":        1.42,
    "japan":            1.38,
    "china":            1.40,
    "miami":            1.38,
    "monaco":           1.30,   # Monaco already slow; VSC delta smaller
    "canada":           1.40,
    "spain":            1.40,
    "austria":          1.42,
    "great_britain":    1.42,
    "hungary":          1.35,
    "belgium":          1.45,   # Spa: huge speed range; VSC delta bigger
    "netherlands":      1.38,
    "italy":            1.38,
    "singapore":        1.28,   # Singapore already very slow
    "united_states":    1.40,
    "mexico":           1.40,
    "brazil":           1.42,
    "las_vegas":        1.38,
    "abu_dhabi":        1.40,
}


# ===========================================================================
# Data contract
# ===========================================================================

@dataclass
class NeutralisationTimeDelta:
    """
    Time-model parameters for a neutralisation event.

    This object is the output of compute_neutralisation_delta() and is
    consumed by sc_scenario_analyzer to evaluate the pit decision.

    Attributes:
        neutralisation_type:   "SC" or "VSC".
        circuit:               Circuit name.
        base_racing_lap_sec:   Reference racing lap time (seconds).
        neutralised_lap_sec:   Expected lap time under neutralisation.
        lap_delta_sec:         neutralised - racing (how much slower each lap is).
        net_pit_cost_sec:      Time lost vs staying out if pitting now.
                               ≈ 3s under SC, ≈ 21s under VSC.
        gap_compression_factor: Fraction of current gap remaining after compression.
        vsc_multiplier:        VSC target multiplier used (for logging/validation).
    """
    neutralisation_type:    str
    circuit:                str
    base_racing_lap_sec:    float
    neutralised_lap_sec:    float
    lap_delta_sec:          float
    net_pit_cost_sec:       float
    gap_compression_factor: float
    vsc_multiplier:         float = VSC_LAP_TIME_MULTIPLIER

    def summary(self) -> str:
        return (
            f"NeutralisationDelta[{self.neutralisation_type}@{self.circuit}] | "
            f"racing={self.base_racing_lap_sec:.1f}s  "
            f"neutralised={self.neutralised_lap_sec:.1f}s  "
            f"Δlap={self.lap_delta_sec:+.1f}s  "
            f"net_pit_cost={self.net_pit_cost_sec:.1f}s  "
            f"gap_compression={self.gap_compression_factor:.0%}"
        )


# ===========================================================================
# Core time model
# ===========================================================================

def get_vsc_multiplier(circuit: str) -> float:
    """
    Return the VSC lap time multiplier for a given circuit.

    Falls back to the global default (VSC_LAP_TIME_MULTIPLIER) if the
    circuit is not in the lookup table.

    Args:
        circuit: Circuit name (fuzzy-matched against lookup keys).

    Returns:
        Float multiplier (e.g. 1.40 = 40% slower than racing pace).
    """
    normalised = circuit.lower().strip().replace(" ", "_").replace("-", "_")

    # Try exact match first
    if normalised in VSC_MULTIPLIER_BY_CIRCUIT:
        return VSC_MULTIPLIER_BY_CIRCUIT[normalised]

    # Try partial match (circuit name may contain extra words)
    for key in VSC_MULTIPLIER_BY_CIRCUIT:
        if key in normalised or normalised in key:
            logger.debug(
                "get_vsc_multiplier: partial match '%s' → '%s' (%.2f)",
                normalised, key, VSC_MULTIPLIER_BY_CIRCUIT[key],
            )
            return VSC_MULTIPLIER_BY_CIRCUIT[key]

    logger.debug(
        "get_vsc_multiplier: no match for '%s' — using default %.2f",
        circuit, VSC_LAP_TIME_MULTIPLIER,
    )
    return VSC_LAP_TIME_MULTIPLIER


def compute_neutralisation_delta(
    neutralisation_type: str,
    base_racing_lap_sec: float,
    circuit:             str = "bahrain",
    pit_lane_delta_sec:  float = PIT_LANE_DELTA_SEC_BAHRAIN,
) -> NeutralisationTimeDelta:
    """
    Compute the time-model parameters for a SC or VSC neutralisation.

    This function encapsulates the physics of each neutralisation type
    so that the scenario analyzer can call a single consistent interface
    regardless of whether an SC or VSC is active.

    Args:
        neutralisation_type: "SC" or "VSC".
        base_racing_lap_sec: Reference racing pace (s/lap).
        circuit:             Circuit name (for VSC multiplier lookup).
        pit_lane_delta_sec:  Circuit pit lane traverse time.

    Returns:
        NeutralisationTimeDelta with all parameters populated.

    Raises:
        ValueError: If neutralisation_type is not "SC" or "VSC".
    """
    nt = neutralisation_type.upper().strip()
    if nt not in ("SC", "VSC"):
        raise ValueError(
            f"compute_neutralisation_delta: type must be 'SC' or 'VSC', "
            f"got '{neutralisation_type}'."
        )

    if nt == "SC":
        # Under full SC, cars travel at ~80 km/h behind the safety car.
        # We model SC lap time as base_racing × SC_LAP_MULTIPLIER.
        # The multiplier is derived from typical SC lap times observed in
        # FastF1 data: Bahrain SC laps ~170s vs ~93s racing = ×1.83.
        # This is circuit-dependent (longer circuits → bigger multiplier),
        # but we use 1.8 as a reasonable universal estimate.
        sc_lap_multiplier       = 1.80
        neutralised_lap_sec     = base_racing_lap_sec * sc_lap_multiplier
        lap_delta_sec           = neutralised_lap_sec - base_racing_lap_sec
        net_pit_cost_sec        = SC_NET_PIT_COST_SEC
        gap_compression_factor  = SC_GAP_COMPRESSION_FACTOR
        vsc_multiplier          = sc_lap_multiplier

        logger.debug(
            "compute_neutralisation_delta [SC]: "
            "racing=%.1fs  sc_lap=%.1fs  Δ=+%.1fs  net_pit_cost=%.1fs",
            base_racing_lap_sec, neutralised_lap_sec,
            lap_delta_sec, net_pit_cost_sec,
        )

    else:  # VSC
        vsc_multiplier          = get_vsc_multiplier(circuit)
        neutralised_lap_sec     = base_racing_lap_sec * vsc_multiplier
        lap_delta_sec           = neutralised_lap_sec - base_racing_lap_sec
        # Net cost: stationary + traverse. The lap-time delta cancels out
        # because every competitor is also slowed by the VSC.
        net_pit_cost_sec        = PIT_STATIONARY_TIME_SEC + pit_lane_delta_sec
        gap_compression_factor  = VSC_GAP_COMPRESSION_FACTOR

        logger.debug(
            "compute_neutralisation_delta [VSC @%s]: "
            "racing=%.1fs  vsc_lap=%.1fs (×%.2f)  Δ=+%.1fs  net_pit_cost=%.1fs",
            circuit, base_racing_lap_sec, neutralised_lap_sec,
            vsc_multiplier, lap_delta_sec, net_pit_cost_sec,
        )

    result = NeutralisationTimeDelta(
        neutralisation_type    = nt,
        circuit                = circuit,
        base_racing_lap_sec    = base_racing_lap_sec,
        neutralised_lap_sec    = neutralised_lap_sec,
        lap_delta_sec          = lap_delta_sec,
        net_pit_cost_sec       = net_pit_cost_sec,
        gap_compression_factor = gap_compression_factor,
        vsc_multiplier         = vsc_multiplier,
    )

    logger.info("compute_neutralisation_delta: %s", result.summary())
    return result


def apply_gap_compression(
    current_gap_sec:        float,
    neutralisation_delta:   NeutralisationTimeDelta,
    n_neutralisation_laps:  int = 1,
) -> float:
    """
    Apply gap compression to an on-track gap during neutralisation.

    Under SC, the physical safety car bunches the field. Gaps shrink
    rapidly over the first 1-2 laps of SC deployment.
    Under VSC, compression is much smaller — drivers are independently
    pacing to a target time, not queuing behind a physical car.

    Gap after n laps of neutralisation:
        compressed_gap = current_gap × gap_compression_factor^n_laps

    The exponential decay model reflects empirical observation: most
    gap compression happens in the first SC lap; subsequent laps show
    diminishing returns as the queue has fully formed.

    Args:
        current_gap_sec:       Current on-track gap (seconds).
        neutralisation_delta:  NeutralisationTimeDelta for the event.
        n_neutralisation_laps: Number of laps under neutralisation.

    Returns:
        Compressed gap in seconds. Always >= 0.
    """
    factor   = neutralisation_delta.gap_compression_factor
    # Apply exponential decay: gap reduces by factor each lap
    compressed = current_gap_sec * (factor ** n_neutralisation_laps)
    result     = max(0.0, compressed)

    logger.debug(
        "apply_gap_compression [%s]: %.2fs → %.2fs (factor=%.2f^%d)",
        neutralisation_delta.neutralisation_type,
        current_gap_sec, result, factor, n_neutralisation_laps,
    )
    return result


def should_pit_under_neutralisation(
    neutralisation_delta: NeutralisationTimeDelta,
    fresh_tyre_pace_gain_per_lap: float,
    laps_remaining_after_stop:    int,
    tyre_age:                     int,
) -> tuple[bool, str]:
    """
    Simple heuristic: should we pit under the current neutralisation?

    This is a fast pre-filter for the scenario analyzer. It does NOT
    replace the full simulation — it provides a quick yes/no answer
    based on whether the pit cost can be recovered on pace alone.

    Break-even calculation:
        Pitting is worth it if:
        fresh_tyre_pace_gain_per_lap × laps_remaining > net_pit_cost

    With an SC-specific override: under full SC with >= 2 laps remaining,
    the answer is almost always YES because the net cost is only ~3s.

    Args:
        neutralisation_delta:          NeutralisationTimeDelta for the event.
        fresh_tyre_pace_gain_per_lap:  Per-lap pace improvement from pitting.
        laps_remaining_after_stop:     Laps left after the stop completes.
        tyre_age:                      Current tyre age (for context logging).

    Returns:
        Tuple of (should_pit: bool, reasoning: str).
    """
    nt              = neutralisation_delta.neutralisation_type
    net_cost        = neutralisation_delta.net_pit_cost_sec
    pace_gain_total = fresh_tyre_pace_gain_per_lap * laps_remaining_after_stop

    if laps_remaining_after_stop < 3:
        return False, (
            f"Only {laps_remaining_after_stop} laps after stop — "
            f"insufficient to recover {net_cost:.1f}s net pit cost."
        )

    if nt == "SC" and laps_remaining_after_stop >= 5:
        # Under full SC with plenty of race remaining, pitting is almost
        # always correct unless you just fitted brand-new tyres.
        if tyre_age <= 3:
            return False, (
                f"SC: tyre_age={tyre_age} — tyres essentially fresh. "
                f"No benefit to pitting."
            )
        return True, (
            f"SC: net_pit_cost={net_cost:.1f}s  "
            f"fresh_tyre_gain={pace_gain_total:.1f}s over {laps_remaining_after_stop} laps. "
            f"Pit strongly recommended."
        )

    # General break-even analysis
    if pace_gain_total > net_cost:
        threshold = (
            SC_MIN_TYRE_ADVANTAGE_FOR_STOP_SEC_PER_LAP
            if nt == "SC"
            else VSC_MIN_TYRE_ADVANTAGE_FOR_STOP_SEC_PER_LAP
        )
        if fresh_tyre_pace_gain_per_lap < threshold:
            return False, (
                f"{nt}: pace gain {fresh_tyre_pace_gain_per_lap:.3f}s/lap "
                f"< threshold {threshold:.3f}s/lap — tyres not degraded enough."
            )
        return True, (
            f"{nt}: pace_gain_total={pace_gain_total:.1f}s > "
            f"net_cost={net_cost:.1f}s — pit recommended."
        )

    return False, (
        f"{nt}: pace_gain_total={pace_gain_total:.1f}s < "
        f"net_cost={net_cost:.1f}s — staying out is better."
    )
