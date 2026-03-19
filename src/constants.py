"""
src/constants.py
=================
Shared physical constants and configuration values used across all modules.

Engineering principle:
    Every number that appears in more than one file belongs here.
    Duplicating constants across modules creates silent divergence bugs —
    if FUEL_BURN_RATE changes (e.g. after a regulation change), you update
    one file and forget the other, and the simulator and degradation model
    silently use different fuel physics.

    All constants are documented with their engineering source and rationale.
    No magic numbers anywhere in the codebase.
"""

from __future__ import annotations

from pathlib import Path


# ===========================================================================
# Project paths
# ===========================================================================

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
DATA_SIMULATED_DIR: Path = PROJECT_ROOT / "data" / "simulated"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"
FASTF1_CACHE_DIR: Path = DATA_RAW_DIR / "fastf1_cache"


# ===========================================================================
# Fuel physics
# ===========================================================================

# Average fuel burn per racing lap (kg/lap).
# FIA Sporting Regulations and published team data.
# Bahrain GP: ~105 kg start load, ~1.8 kg/lap burn rate.
# Conservative estimate — some cars fuel heavier for SC contingency.
FUEL_BURN_RATE_KG_PER_LAP: float = 1.8

# Lap time sensitivity to fuel load (seconds per kg).
# A 10 kg difference in fuel load = ~0.35s of lap time at most circuits.
# Widely cited in F1 engineering literature; consistent with FastF1 analysis.
FUEL_LAP_TIME_SENSITIVITY_SEC_PER_KG: float = 0.035


# ===========================================================================
# Tyre compounds
# ===========================================================================

# Canonical dry-weather compound names as returned by FastF1 (post-normalisation).
DRY_COMPOUNDS: frozenset[str] = frozenset({"SOFT", "MEDIUM", "HARD"})

# All compounds including wet-weather (for validation and normalisation).
ALL_CANONICAL_COMPOUNDS: frozenset[str] = frozenset(
    {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}
)

# Minimum number of distinct dry compounds required per race (FIA Art. 28.4).
MIN_COMPOUNDS_PER_RACE: int = 2

# Compound abbreviations for display (FOM broadcast standard).
COMPOUND_ABBREV: dict[str, str] = {
    "SOFT":         "S",
    "MEDIUM":       "M",
    "HARD":         "H",
    "INTERMEDIATE": "I",
    "WET":          "W",
    "UNKNOWN":      "?",
}

# FOM broadcast-standard compound colours (hex).
# These are not arbitrary — reviewers familiar with F1 broadcast
# will read plots using these colours instantly.
COMPOUND_COLOURS: dict[str, str] = {
    "SOFT":         "#E8002D",
    "MEDIUM":       "#FFF200",
    "HARD":         "#EBEBEB",
    "INTERMEDIATE": "#43B02A",
    "WET":          "#0067FF",
    "UNKNOWN":      "#888888",
}


# ===========================================================================
# FastF1 track status codes
# ===========================================================================

# FastF1 encodes track conditions as a concatenated string of single-digit
# codes active during each lap. e.g. "14" = Green flag + Safety Car active.
# Reference: FastF1 documentation, TrackStatus field.
TRACK_STATUS_GREEN:  str = "1"
TRACK_STATUS_YELLOW: str = "2"
TRACK_STATUS_SC:     str = "4"   # Full Safety Car
TRACK_STATUS_RED:    str = "7"   # Red flag
TRACK_STATUS_VSC:    str = "6"   # Virtual Safety Car


# ===========================================================================
# Lap time quality filters
# ===========================================================================

# Minimum physically plausible race lap time (seconds).
# Laps below this threshold are timing artefacts (red flag resets, etc.).
MIN_RACE_LAP_TIME_SEC: float = 60.0

# A lap is flagged anomalously slow if it exceeds:
#   (session_theoretical_best * SLOW_LAP_MULTIPLIER)
# This covers formation laps, SC queuing, and incident delays.
SLOW_LAP_MULTIPLIER: float = 2.5

# Fraction of TyreLife values that must be populated for us to trust
# FastF1's native column over our fallback derivation.
TYRE_LIFE_TRUST_THRESHOLD: float = 0.90

# Minimum tyre age included in degradation regression.
# Lap 1 = out-lap (cold tyres, artificially slow). Excluded from fitting.
# Regression starts at tyre_age == 2.
MIN_TYRE_AGE_FOR_REGRESSION: int = 2


# ===========================================================================
# Pit stop physics — circuit-specific
# ===========================================================================

# Pit stop stationary time (seconds).
# Covers: jack up/down, wheel gun x4, FIA minimum box time.
# Bahrain 2023 median from FastF1 analysis: ~2.4s.
PIT_STATIONARY_TIME_SEC: float = 2.5

# Total pit lane traverse time: entry line → exit line at 80 km/h limit.
# This value is CIRCUIT-SPECIFIC. The optimizer accepts it as a parameter.
# Default: Bahrain International Circuit.
PIT_LANE_DELTA_SEC_BAHRAIN: float = 19.0

# Inlap time penalty vs a representative lap.
# Driver lifts early entering pit lane (typically after final corner).
# Bahrain: pit entry after T4, costs ~1.2s in sector 3.
INLAP_TIME_PENALTY_SEC: float = 1.2

# Outlap time penalty vs a representative lap.
# First lap of new stint: cold tyres below operating temperature.
# 0.8s is the first-order approximation; compound-specific warm-up
# is modelled separately in the tyre module.
OUTLAP_TIME_PENALTY_SEC: float = 0.8


# ===========================================================================
# Visualisation
# ===========================================================================

FIGURE_DPI: int = 150
PLOT_STYLE: str = "dark_background"
