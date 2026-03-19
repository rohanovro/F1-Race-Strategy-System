"""
src/data_engineering/fastf1_loader.py
=======================================
Data ingestion boundary for the F1 Race Strategy System.

Engineering responsibility:
    Load raw FastF1 session objects for any circuit, year, and session type.
    This file owns exactly one job: get a fully populated session object from
    the FastF1 API into memory, with caching, logging, and robust error
    handling. It does NOT process, clean, or transform data — that is
    telemetry_processor.py's responsibility.

Design decisions:
    CACHING is mandatory. FastF1 makes HTTP requests to the OpenF1 timing
    API on every uncached load (~5-30 seconds per session). The cache stores
    pre-parsed data on local disk. Without it, every development iteration
    incurs a 30-second penalty and risks rate-limiting.

    EAGER LOADING (telemetry=True, weather=True, messages=True, laps=True)
    is chosen over lazy loading. Lazy loading can trigger additional I/O
    calls mid-pipeline with no warning, making profiling and error tracing
    nearly impossible. We pay the full load cost upfront and proceed with
    a fully populated object.

    CONFIG DICT PATTERN allows callers to store session specifications in
    YAML or JSON and pass them through without unpacking. This makes batch
    processing (loading 5 circuits for a model training run) trivial.

    SESSION TYPES supported: Race (R), Qualifying (Q), Sprint (S),
    Sprint Qualifying (SQ), Practice 1/2/3 (FP1, FP2, FP3).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import fastf1
import pandas as pd

from src.constants import FASTF1_CACHE_DIR

logger = logging.getLogger(__name__)


# ===========================================================================
# Constants
# ===========================================================================

VALID_SESSION_TYPES: tuple[str, ...] = (
    "R", "Q", "S", "SQ", "FP1", "FP2", "FP3"
)

# Env var allowing CI/CD pipelines to override the cache directory
# without modifying source code.
ENV_CACHE_DIR: str = "F1_CACHE_DIR"


# ===========================================================================
# Session configuration
# ===========================================================================

def validate_session_config(config: dict[str, Any]) -> None:
    """
    Validate a session configuration dictionary before use.

    Expected keys:
        circuit (str):      Event name (e.g. "Bahrain Grand Prix").
        year    (int):      Championship year (e.g. 2023).
        session_type (str): One of VALID_SESSION_TYPES. Defaults to "R".

    Args:
        config: Dictionary with session parameters.

    Raises:
        ValueError: If any required key is missing or has an invalid value.
        TypeError:  If year is not an integer or circuit is not a string.
    """
    if "circuit" not in config:
        raise ValueError(
            "Session config missing required key 'circuit'. "
            "Example: {'circuit': 'Bahrain Grand Prix', 'year': 2023}"
        )
    if "year" not in config:
        raise ValueError(
            "Session config missing required key 'year'. "
            "Example: {'circuit': 'Bahrain Grand Prix', 'year': 2023}"
        )
    if not isinstance(config["circuit"], str) or not config["circuit"].strip():
        raise TypeError(
            f"config['circuit'] must be a non-empty string, "
            f"got: {config['circuit']!r}"
        )
    if not isinstance(config["year"], int) or not (1950 <= config["year"] <= 2100):
        raise TypeError(
            f"config['year'] must be an integer in [1950, 2100], "
            f"got: {config['year']!r}"
        )

    session_type = config.get("session_type", "R")
    if session_type not in VALID_SESSION_TYPES:
        raise ValueError(
            f"config['session_type']={session_type!r} is not supported. "
            f"Valid values: {VALID_SESSION_TYPES}"
        )


# ===========================================================================
# Cache management
# ===========================================================================

def configure_cache(cache_dir: Optional[str | Path] = None) -> Path:
    """
    Initialise and activate the FastF1 local disk cache.

    Resolution order for cache directory:
        1. Explicit argument (highest priority — programmatic override)
        2. Environment variable F1_CACHE_DIR (CI/CD and Docker override)
        3. Project default: data/raw/fastf1_cache/

    FastF1 is idempotent on repeated cache.enable_cache() calls with the
    same directory — safe to call multiple times.

    Args:
        cache_dir: Optional path override.

    Returns:
        Resolved absolute Path of the active cache directory.

    Raises:
        OSError: If the directory cannot be created (permission denied, etc.).
    """
    resolved = Path(
        cache_dir
        or os.environ.get(ENV_CACHE_DIR, "")
        or FASTF1_CACHE_DIR
    ).resolve()

    resolved.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(resolved))

    logger.debug("FastF1 cache directory: %s", resolved)
    return resolved


# ===========================================================================
# Core session loader
# ===========================================================================

def load_session(
    config: dict[str, Any],
    cache_dir: Optional[str | Path] = None,
) -> fastf1.core.Session:
    """
    Load a fully populated FastF1 Session object from a config dict.

    This is the primary public function of this module. Every downstream
    component (telemetry_processor, degradation model, simulator) calls
    this to get a session object.

    Config dict keys:
        circuit (str):       Event name as FastF1 recognises it.
                             Accepts full name ("Bahrain Grand Prix")
                             or short form ("Bahrain") — FastF1 fuzzy-matches.
        year (int):          Championship year.
        session_type (str):  One of VALID_SESSION_TYPES. Defaults to "R".

    FastF1 async quirk handling:
        FastF1 internally uses threading and requests-cache. The .load()
        call is synchronous from the caller's perspective but may spawn
        background threads for parallel data fetching. We wrap the entire
        load in a try/except to catch both network errors and FastF1's
        internal exceptions (which use a non-standard exception hierarchy).
        The exception is re-raised as RuntimeError with full context so
        the caller always gets an actionable error message.

    Args:
        config:    Session configuration dict. See validate_session_config().
        cache_dir: Optional cache directory override.

    Returns:
        Fully loaded fastf1.core.Session with laps, telemetry,
        weather, and messages populated.

    Raises:
        ValueError:   Invalid config values.
        TypeError:    Wrong types in config.
        RuntimeError: FastF1 load failure (network, bad event name, etc.).

    Example:
        >>> session = load_session({
        ...     "circuit": "Bahrain Grand Prix",
        ...     "year": 2023,
        ...     "session_type": "R",
        ... })
    """
    validate_session_config(config)
    configure_cache(cache_dir)

    circuit      = config["circuit"].strip()
    year         = config["year"]
    session_type = config.get("session_type", "R")

    logger.info(
        "Loading session | year=%d  circuit='%s'  type=%s",
        year, circuit, session_type,
    )

    try:
        session = fastf1.get_session(year, circuit, session_type)
        session.load(
            laps      = True,
            telemetry = True,
            weather   = True,
            messages  = True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"FastF1 failed to load session "
            f"[year={year}, circuit='{circuit}', type={session_type}]. "
            f"Possible causes: invalid circuit name, no data for this year, "
            f"network error, or corrupted cache entry. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    n_laps    = len(session.laps)
    n_drivers = len(session.drivers)
    event_name = session.event.get("EventName", circuit)

    logger.info(
        "Session loaded  | event='%s'  drivers=%d  laps=%d",
        event_name, n_drivers, n_laps,
    )

    if n_laps == 0:
        logger.warning(
            "Session '%s' loaded but contains zero laps. "
            "This may indicate a session that was cancelled or "
            "has no timing data available in FastF1.",
            event_name,
        )

    return session


def load_multiple_sessions(
    configs: list[dict[str, Any]],
    cache_dir: Optional[str | Path] = None,
    skip_failures: bool = True,
) -> dict[str, fastf1.core.Session]:
    """
    Load multiple sessions from a list of config dicts.

    Used for multi-circuit model training: pass configs for 5+ races and
    receive a keyed dict. Failures are logged and skipped rather than
    aborting the entire batch (controlled by skip_failures).

    Session keys follow the pattern: "{year}_{circuit}_{session_type}"
    with spaces replaced by underscores and lowercased.

    Args:
        configs:       List of session config dicts.
        cache_dir:     Optional shared cache directory.
        skip_failures: If True, log errors and continue. If False, re-raise.

    Returns:
        Dict mapping session key -> loaded Session object.
        Failed sessions are absent from the dict.

    Example:
        >>> sessions = load_multiple_sessions([
        ...     {"circuit": "Bahrain Grand Prix", "year": 2023},
        ...     {"circuit": "Saudi Arabian Grand Prix", "year": 2023},
        ... ])
    """
    results: dict[str, fastf1.core.Session] = {}

    for i, config in enumerate(configs):
        try:
            validate_session_config(config)
        except (ValueError, TypeError) as exc:
            logger.error(
                "Config[%d] is invalid — skipping: %s", i, exc
            )
            if not skip_failures:
                raise
            continue

        circuit      = config["circuit"].strip()
        year         = config["year"]
        session_type = config.get("session_type", "R")
        key = f"{year}_{circuit.lower().replace(' ', '_')}_{session_type.lower()}"

        try:
            session = load_session(config, cache_dir=cache_dir)
            results[key] = session
            logger.info("Batch load [%d/%d]: '%s' OK", i + 1, len(configs), key)
        except RuntimeError as exc:
            logger.error(
                "Batch load [%d/%d]: '%s' FAILED — %s",
                i + 1, len(configs), key, exc,
            )
            if not skip_failures:
                raise

    logger.info(
        "Batch load complete: %d/%d sessions loaded successfully.",
        len(results), len(configs),
    )
    return results


# ===========================================================================
# Session metadata extraction
# ===========================================================================

def get_session_info(session: fastf1.core.Session) -> dict[str, Any]:
    """
    Extract key metadata from a loaded session into a plain dict.

    Useful for logging, dashboard headers, and DataFrame column values.

    Args:
        session: Loaded FastF1 Session object.

    Returns:
        Dict with keys: event_name, year, round_number, circuit_name,
        session_type, n_drivers, n_laps, event_date.
    """
    event = session.event
    return {
        "event_name":    event.get("EventName", "Unknown"),
        "year":          int(event.get("EventDate", "2023-01-01")[:4]),
        "round_number":  int(event.get("RoundNumber", 0)),
        "circuit_name":  event.get("Location", "Unknown"),
        "session_type":  session.name,
        "n_drivers":     len(session.drivers),
        "n_laps":        len(session.laps),
        "event_date":    str(event.get("EventDate", ""))[:10],
    }
