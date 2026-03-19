"""
src/tire_model/compound_profiles.py
=====================================
Baseline tyre compound degradation profiles per circuit.

Engineering responsibility:
    Define the prior degradation knowledge for each compound on each circuit.
    These profiles seed the degradation model with physically motivated
    starting parameters before any regression is performed.

    In a production F1 team, these profiles would be maintained by tyre
    engineers updating a database after every race weekend. Here they are
    derived from published FastF1 community analysis, Pirelli tyre briefings,
    and public post-race engineering debriefs.

Profile structure:
    PROFILES[compound][circuit] = {
        "baseline_deg_rate_sec_per_lap": float,
            The expected per-lap lap time increase on a representative
            dry stint. This is the LINEAR phase rate — before any cliff.
            Units: seconds per lap.

        "cliff_lap": int,
            The tyre age (laps into stint) at which degradation rate
            sharply accelerates. This is the "cliff threshold" — the
            maximum safe stint length under nominal conditions.
            The cliff is compound AND circuit specific because:
              - High-energy circuits (Silverstone, Suzuka) generate more
                tyre stress and trigger cliffs earlier.
              - Low-energy circuits (Monaco, Hungary) allow longer stints.
            None = no cliff detected historically on this circuit.

        "warmup_laps": int,
            Number of laps required for the tyre to reach full operating
            temperature from a cold state (out-lap).
            Softs warm faster (1-2 laps) than hards (3-5 laps).
            Laps within the warm-up window are excluded from the linear
            degradation fit but included in simulation with an outlap penalty.

        "cliff_rate_multiplier": float,
            How much faster the degradation rate becomes after the cliff.
            e.g. 2.5 means the post-cliff rate is 2.5× the linear rate.
    }

    Circuit keys match the `circuit` parameter used in fastf1_loader.py
    config dicts (lowercase, underscores for spaces).

Usage:
    >>> profile = get_compound_profile("SOFT", "bahrain")
    >>> print(profile["cliff_lap"])  # -> 16
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ===========================================================================
# Compound profiles
# ===========================================================================

# Type alias for clarity
CompoundProfile = dict[str, Any]

# Default profile used when a specific circuit entry is not present.
# These are circuit-agnostic estimates based on Pirelli's published
# expected stint lengths and degradation briefings.
_DEFAULT_PROFILES: dict[str, CompoundProfile] = {
    "SOFT": {
        "baseline_deg_rate_sec_per_lap": 0.065,
        "cliff_lap":                     18,
        "warmup_laps":                   1,
        "cliff_rate_multiplier":         2.8,
    },
    "MEDIUM": {
        "baseline_deg_rate_sec_per_lap": 0.038,
        "cliff_lap":                     28,
        "warmup_laps":                   2,
        "cliff_rate_multiplier":         2.2,
    },
    "HARD": {
        "baseline_deg_rate_sec_per_lap": 0.022,
        "cliff_lap":                     None,  # Hards rarely cliff on most circuits
        "warmup_laps":                   3,
        "cliff_rate_multiplier":         1.8,
    },
    "INTERMEDIATE": {
        "baseline_deg_rate_sec_per_lap": 0.080,
        "cliff_lap":                     20,
        "warmup_laps":                   2,
        "cliff_rate_multiplier":         2.0,
    },
    "WET": {
        "baseline_deg_rate_sec_per_lap": 0.100,
        "cliff_lap":                     15,
        "warmup_laps":                   2,
        "cliff_rate_multiplier":         1.5,
    },
}

# Circuit-specific overrides.
# nested dict: compound -> circuit_key -> profile
# Circuit keys are lowercased and space-replaced with underscores to
# match FastF1 event name normalisation.
PROFILES: dict[str, dict[str, CompoundProfile]] = {

    "SOFT": {
        "bahrain": {
            "baseline_deg_rate_sec_per_lap": 0.072,
            "cliff_lap":                     16,
            "warmup_laps":                   1,
            "cliff_rate_multiplier":         3.0,
            # Bahrain's abrasive surface and high thermal load on the rear
            # tyres (long slow-speed traction zones at T10-T12) produce
            # above-average soft degradation. Historical data: 2021-2023
            # average soft cliff at lap 15-17.
        },
        "monaco": {
            "baseline_deg_rate_sec_per_lap": 0.045,
            "cliff_lap":                     22,
            "warmup_laps":                   2,
            "cliff_rate_multiplier":         2.0,
            # Monaco's low speeds and low tyre energy produce the slowest
            # soft degradation on the calendar. Mechanical grip dominates;
            # thermal degradation is minimal.
        },
        "silverstone": {
            "baseline_deg_rate_sec_per_lap": 0.085,
            "cliff_lap":                     14,
            "warmup_laps":                   1,
            "cliff_rate_multiplier":         3.2,
            # Silverstone has the highest lateral-G corners on the calendar
            # (Copse, Maggotts-Becketts). Soft tyres are extremely stressed
            # and cliff early. Rarely used for more than 15 laps.
        },
        "spa": {
            "baseline_deg_rate_sec_per_lap": 0.068,
            "cliff_lap":                     17,
            "warmup_laps":                   1,
            "cliff_rate_multiplier":         2.6,
        },
        "monza": {
            "baseline_deg_rate_sec_per_lap": 0.055,
            "cliff_lap":                     20,
            "warmup_laps":                   1,
            "cliff_rate_multiplier":         2.3,
            # Monza is a low-downforce circuit with relatively low tyre
            # stress (long straights, slow chicanes). Softs last longer
            # than at high-energy circuits.
        },
        "suzuka": {
            "baseline_deg_rate_sec_per_lap": 0.078,
            "cliff_lap":                     15,
            "warmup_laps":                   1,
            "cliff_rate_multiplier":         3.0,
        },
        "singapore": {
            "baseline_deg_rate_sec_per_lap": 0.048,
            "cliff_lap":                     24,
            "warmup_laps":                   2,
            "cliff_rate_multiplier":         2.1,
        },
    },

    "MEDIUM": {
        "bahrain": {
            "baseline_deg_rate_sec_per_lap": 0.042,
            "cliff_lap":                     26,
            "warmup_laps":                   2,
            "cliff_rate_multiplier":         2.4,
        },
        "monaco": {
            "baseline_deg_rate_sec_per_lap": 0.025,
            "cliff_lap":                     35,
            "warmup_laps":                   2,
            "cliff_rate_multiplier":         1.8,
        },
        "silverstone": {
            "baseline_deg_rate_sec_per_lap": 0.052,
            "cliff_lap":                     24,
            "warmup_laps":                   2,
            "cliff_rate_multiplier":         2.5,
        },
        "spa": {
            "baseline_deg_rate_sec_per_lap": 0.040,
            "cliff_lap":                     28,
            "warmup_laps":                   2,
            "cliff_rate_multiplier":         2.0,
        },
        "monza": {
            "baseline_deg_rate_sec_per_lap": 0.032,
            "cliff_lap":                     32,
            "warmup_laps":                   2,
            "cliff_rate_multiplier":         1.9,
        },
        "suzuka": {
            "baseline_deg_rate_sec_per_lap": 0.048,
            "cliff_lap":                     25,
            "warmup_laps":                   2,
            "cliff_rate_multiplier":         2.3,
        },
        "singapore": {
            "baseline_deg_rate_sec_per_lap": 0.028,
            "cliff_lap":                     36,
            "warmup_laps":                   2,
            "cliff_rate_multiplier":         1.7,
        },
    },

    "HARD": {
        "bahrain": {
            "baseline_deg_rate_sec_per_lap": 0.025,
            "cliff_lap":                     None,
            "warmup_laps":                   3,
            "cliff_rate_multiplier":         1.6,
        },
        "monaco": {
            "baseline_deg_rate_sec_per_lap": 0.015,
            "cliff_lap":                     None,
            "warmup_laps":                   4,
            "cliff_rate_multiplier":         1.4,
        },
        "silverstone": {
            "baseline_deg_rate_sec_per_lap": 0.032,
            "cliff_lap":                     38,
            "warmup_laps":                   3,
            "cliff_rate_multiplier":         2.0,
            # Silverstone is one of the few circuits where Hards can cliff
            # under sustained high-energy cornering.
        },
        "spa": {
            "baseline_deg_rate_sec_per_lap": 0.020,
            "cliff_lap":                     None,
            "warmup_laps":                   3,
            "cliff_rate_multiplier":         1.5,
        },
        "monza": {
            "baseline_deg_rate_sec_per_lap": 0.018,
            "cliff_lap":                     None,
            "warmup_laps":                   3,
            "cliff_rate_multiplier":         1.4,
        },
        "suzuka": {
            "baseline_deg_rate_sec_per_lap": 0.028,
            "cliff_lap":                     40,
            "warmup_laps":                   3,
            "cliff_rate_multiplier":         1.9,
        },
        "singapore": {
            "baseline_deg_rate_sec_per_lap": 0.018,
            "cliff_lap":                     None,
            "warmup_laps":                   4,
            "cliff_rate_multiplier":         1.5,
        },
    },

    "INTERMEDIATE": {},  # Circuit-specific overrides not yet profiled — uses default

    "WET": {},           # Circuit-specific overrides not yet profiled — uses default
}


# ===========================================================================
# Public API
# ===========================================================================

def _normalise_circuit_key(circuit: str) -> str:
    """Normalise a circuit name to the key format used in PROFILES."""
    return circuit.lower().strip().replace(" ", "_").replace("-", "_")


def get_compound_profile(
    compound: str,
    circuit: str,
) -> CompoundProfile:
    """
    Retrieve the degradation profile for a compound on a specific circuit.

    Resolution order:
        1. Circuit-specific entry in PROFILES[compound][circuit]
        2. Default profile in _DEFAULT_PROFILES[compound]

    Both levels are logged so the caller knows which was used.

    Args:
        compound: Canonical compound name (e.g. "SOFT").
        circuit:  Circuit name (e.g. "Bahrain" or "bahrain").

    Returns:
        CompoundProfile dict with keys:
            baseline_deg_rate_sec_per_lap, cliff_lap,
            warmup_laps, cliff_rate_multiplier.

    Raises:
        KeyError: If compound is not in PROFILES or _DEFAULT_PROFILES.
    """
    compound_upper = compound.upper().strip()
    circuit_key    = _normalise_circuit_key(circuit)

    if compound_upper not in PROFILES:
        raise KeyError(
            f"get_compound_profile: unknown compound '{compound_upper}'. "
            f"Valid compounds: {sorted(PROFILES.keys())}"
        )

    circuit_profiles = PROFILES[compound_upper]
    if circuit_key in circuit_profiles:
        profile = circuit_profiles[circuit_key]
        logger.debug(
            "get_compound_profile: [%s][%s] — circuit-specific profile loaded.",
            compound_upper, circuit_key,
        )
    else:
        profile = _DEFAULT_PROFILES[compound_upper]
        logger.debug(
            "get_compound_profile: [%s][%s] — no circuit entry found, "
            "using default profile.",
            compound_upper, circuit_key,
        )

    return profile


def get_all_compound_profiles(circuit: str) -> dict[str, CompoundProfile]:
    """
    Retrieve profiles for all compounds at a given circuit.

    Args:
        circuit: Circuit name.

    Returns:
        Dict mapping compound name -> CompoundProfile.
    """
    compounds = list(PROFILES.keys())
    result = {c: get_compound_profile(c, circuit) for c in compounds}

    logger.info(
        "get_all_compound_profiles: loaded %d profiles for circuit='%s'  "
        "cliff_laps={%s}",
        len(result),
        circuit,
        ", ".join(
            f"{c}:{p['cliff_lap']}" for c, p in result.items()
        ),
    )
    return result


def list_profiled_circuits() -> list[str]:
    """Return sorted list of all circuits with at least one specific profile."""
    circuits: set[str] = set()
    for compound_data in PROFILES.values():
        circuits.update(compound_data.keys())
    return sorted(circuits)
