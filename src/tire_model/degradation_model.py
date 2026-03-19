"""
src/tire_model/degradation_model.py
=====================================
Piecewise polynomial tyre degradation regression model.

Engineering responsibility:
    Fit a per-compound degradation model using scikit-learn Pipelines.
    The model is a callable: (tyre_age: array) -> lap_delta_sec: array.
    It predicts the lap time INCREASE above a fresh tyre baseline as a
    function of tyre age on a specific compound and circuit.

Modelling rationale — why piecewise, not a single polynomial:
    A single degree-2 polynomial fitted over a full 30-lap stint produces
    curvature from lap 1 where the data is actually linear. This:
        - Overestimates pace loss in the first 10 laps (over-recommends
          early pits to the optimizer)
        - Underestimates the cliff severity (the quadratic is too smooth
          to capture the step-change in degradation rate)

    A piecewise model with a structural break at the cliff lap matches
    tyre physics far better:
        - Linear regime (age < cliff): constant per-lap degradation, well
          described by a degree-1 polynomial with zero intercept at age=1.
        - Cliff regime (age >= cliff): accelerating degradation, described
          by a degree-2 polynomial anchored at the cliff value for continuity.

    The two segments are joined with C0 continuity (value-continuous at the
    cliff) to prevent discontinuous jumps in predicted lap time that would
    confuse the race simulator's gradient-based search.

scikit-learn Pipeline design:
    Each compound model is a sklearn Pipeline:
        Step 1: PolynomialFeatures(degree) — expands tyre_age into polynomial
                features [age, age²] for the post-cliff segment.
        Step 2: LinearRegression(fit_intercept) — fits the regression.
    This allows the model to be serialised with joblib and embedded in a
    larger sklearn pipeline for the ML optimizer in Module 5.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from src.tire_model.cliff_detector import detect_cliff, detect_all_compound_cliffs
from src.tire_model.compound_profiles import get_compound_profile
from src.constants import (
    DRY_COMPOUNDS,
    MIN_TYRE_AGE_FOR_REGRESSION,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Constants
# ===========================================================================

# Minimum representative laps per compound for model fitting.
# Below this, regression coefficients are unreliable.
MIN_LAPS_FOR_FIT: int = 15

# Minimum distinct tyre age values for fitting.
# A compound seen only at ages 1,2,3 cannot extrapolate to age 25.
MIN_DISTINCT_AGES: int = 5

# Dense tyre age array for smooth model curve generation (for plots).
PLOT_AGE_MAX: int = 50


# ===========================================================================
# Model container
# ===========================================================================

@dataclass
class TyreDegradationModel:
    """
    Fitted degradation model for one compound on one circuit.

    Attributes:
        compound:         Compound name.
        circuit:          Circuit this model was fitted on.
        n_laps:           Number of representative laps used.
        n_stints:         Number of distinct driver-stints.
        cliff_lap:        Tyre age at cliff onset (None if not detected).
        deg_rate_linear:  Mean degradation rate in linear phase (s/lap).
        deg_rate_cliff:   Mean degradation rate in cliff phase (s/lap).
        r2:               R² of model against aggregated data.
        mae_sec:          Mean absolute error (seconds).
        predict:          Callable: (ages: np.ndarray) -> deltas: np.ndarray.
        fitted_ages:      Tyre ages used for fitting (for plotting).
        fitted_deltas:    Corresponding median deltas (for plotting).
        model_ages:       Dense ages for smooth curve (for plotting).
        model_deltas:     Predictions on model_ages (for plotting).
    """
    compound:          str
    circuit:           str
    n_laps:            int
    n_stints:          int
    cliff_lap:         Optional[int]
    deg_rate_linear:   float
    deg_rate_cliff:    Optional[float]
    r2:                float
    mae_sec:           float
    predict:           Callable[[np.ndarray], np.ndarray]
    fitted_ages:       np.ndarray
    fitted_deltas:     np.ndarray
    model_ages:        np.ndarray
    model_deltas:      np.ndarray

    def summary(self) -> str:
        cliff_str = (
            f"cliff@{self.cliff_lap}  post={self.deg_rate_cliff:+.3f}s/lap"
            if self.cliff_lap else "no cliff"
        )
        return (
            f"{self.compound:<12} | n={self.n_laps}/{self.n_stints}st | "
            f"deg={self.deg_rate_linear:+.3f}s/lap | {cliff_str} | "
            f"R²={self.r2:.3f} MAE={self.mae_sec:.3f}s"
        )


@dataclass
class DegradationModelSet:
    """
    Collection of TyreDegradationModel for all fitted compounds.

    This is the object consumed by the race simulator. The predict()
    method is the hot-path call — it must be fast.

    Attributes:
        circuit:           Circuit name.
        season:            Championship year.
        models:            Compound -> TyreDegradationModel mapping.
        compounds_fitted:  Successfully fitted compound names.
        compounds_skipped: Compounds with insufficient data.
    """
    circuit:           str
    season:            int
    models:            dict[str, TyreDegradationModel] = field(default_factory=dict)
    compounds_fitted:  list[str] = field(default_factory=list)
    compounds_skipped: list[str] = field(default_factory=list)

    def get(self, compound: str) -> Optional[TyreDegradationModel]:
        """Return model for compound, or None."""
        return self.models.get(compound.upper())

    def predict(self, compound: str, tyre_age: int) -> Optional[float]:
        """
        Predict lap time delta (s above fresh tyre) for compound at tyre_age.

        Returns None if the compound has no fitted model — the simulator
        handles this as zero degradation with a logged warning.

        Args:
            compound:  Compound name (e.g. "SOFT").
            tyre_age:  Tyre age in laps (1 = fresh).

        Returns:
            Predicted lap time delta in seconds, or None.
        """
        model = self.get(compound)
        if model is None:
            return None
        return float(model.predict(np.array([float(tyre_age)]))[0])

    def print_summary(self) -> None:
        print(f"\n{'='*72}")
        print(f"  DegradationModelSet — {self.circuit} {self.season}")
        print(f"{'='*72}")
        for c in self.compounds_fitted:
            print(f"  {self.models[c].summary()}")
        if self.compounds_skipped:
            print(f"  Skipped (insufficient data): {self.compounds_skipped}")
        print(f"{'='*72}\n")


# ===========================================================================
# Internal fitting helpers
# ===========================================================================

def _build_predict_fn(
    tyre_ages:  np.ndarray,
    deltas:     np.ndarray,
    cliff_lap:  Optional[int],
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Fit and return a piecewise prediction function.

    Segment structure:
        Pre-cliff (or full range if no cliff):
            Pipeline: PolynomialFeatures(1) + LinearRegression(no intercept)
            Fitted on ages shifted by -1 so that delta(age=1) = 0.

        Post-cliff (if cliff detected AND >= 3 post-cliff points):
            Pipeline: PolynomialFeatures(2) + LinearRegression(no intercept)
            Fitted on ages shifted by -cliff_lap, with delta shifted by
            the pre-cliff model's prediction at the cliff lap.
            This ensures C0 continuity (no jump at the cliff).

    Args:
        tyre_ages: Aggregated (median) tyre age values.
        deltas:    Corresponding lap time deltas.
        cliff_lap: Cliff tyre age, or None.

    Returns:
        Callable (ages_array) -> delta_array, always non-negative.
    """
    ages   = tyre_ages.astype(float)
    deltas = deltas.astype(float)

    # --- Pre-cliff (or no-cliff) linear pipeline ---
    use_cliff = (
        cliff_lap is not None
        and np.sum(ages >= cliff_lap) >= 3
        and np.sum(ages < cliff_lap) >= 2
    )

    pre_mask = (ages < cliff_lap) if use_cliff else np.ones(len(ages), dtype=bool)
    pre_ages = ages[pre_mask]
    pre_deltas = deltas[pre_mask]

    # Shift ages: degree-1 with no intercept, anchored at age=1 (delta=0)
    pre_pipe = Pipeline([
        ("poly",  PolynomialFeatures(degree=1, include_bias=False)),
        ("reg",   LinearRegression(fit_intercept=False)),
    ])
    pre_pipe.fit((pre_ages - 1.0).reshape(-1, 1), pre_deltas)

    if not use_cliff:
        def predict_linear(x: np.ndarray) -> np.ndarray:
            x_f = x.astype(float)
            return np.maximum(0.0, pre_pipe.predict((x_f - 1.0).reshape(-1, 1)).flatten())
        return predict_linear

    # --- Post-cliff quadratic pipeline ---
    cliff_anchor = float(pre_pipe.predict(np.array([[float(cliff_lap) - 1.0]]))[0])
    post_mask  = ages >= cliff_lap
    post_ages  = ages[post_mask]
    post_deltas = deltas[post_mask] - cliff_anchor  # shift for C0 continuity

    post_pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("reg",  LinearRegression(fit_intercept=False)),
    ])
    post_pipe.fit((post_ages - float(cliff_lap)).reshape(-1, 1), post_deltas)

    cliff_f = float(cliff_lap)

    def predict_piecewise(x: np.ndarray) -> np.ndarray:
        x_f = x.astype(float)
        linear_part = np.maximum(
            0.0,
            pre_pipe.predict((x_f - 1.0).reshape(-1, 1)).flatten(),
        )
        cliff_part = (
            cliff_anchor
            + post_pipe.predict((x_f - cliff_f).reshape(-1, 1)).flatten()
        )
        result = np.where(x_f < cliff_f, linear_part, cliff_part)
        return np.maximum(0.0, result)

    return predict_piecewise


# ===========================================================================
# Public fitting functions
# ===========================================================================

def fit_compound_model(
    feature_df: pd.DataFrame,
    compound: str,
    circuit: str = "unknown",
) -> Optional[TyreDegradationModel]:
    """
    Fit a TyreDegradationModel for a single compound.

    Uses lap_delta_from_baseline_sec aggregated by tyre_age (MEDIAN)
    as the fitting target. Median is used rather than mean because it is
    robust to outlier laps from traffic, driver errors, and mechanical issues.

    Args:
        feature_df: Output of feature_builder.build_feature_set().
                    Must contain: compound, tyre_age, lap_delta_from_baseline_sec,
                    is_representative, driver_code, stint_number.
        compound:   Compound to fit (e.g. "SOFT").
        circuit:    Circuit name for labelling.

    Returns:
        Fitted TyreDegradationModel, or None if insufficient data.
    """
    required = {
        "compound", "tyre_age", "lap_delta_from_baseline_sec",
        "is_representative", "driver_code", "stint_number",
    }
    missing = required - set(feature_df.columns)
    if missing:
        raise ValueError(
            f"fit_compound_model: missing columns {sorted(missing)}."
        )

    comp_data = feature_df[
        (feature_df["compound"] == compound.upper())
        & feature_df["is_representative"]
        & (feature_df["tyre_age"] > MIN_TYRE_AGE_FOR_REGRESSION)
    ].copy()

    n_laps = len(comp_data)
    if n_laps < MIN_LAPS_FOR_FIT:
        logger.warning(
            "fit_compound_model [%s]: only %d laps (min=%d) — skipping.",
            compound, n_laps, MIN_LAPS_FOR_FIT,
        )
        return None

    n_distinct_ages = comp_data["tyre_age"].nunique()
    if n_distinct_ages < MIN_DISTINCT_AGES:
        logger.warning(
            "fit_compound_model [%s]: only %d distinct tyre_ages (min=%d) — skipping.",
            compound, n_distinct_ages, MIN_DISTINCT_AGES,
        )
        return None

    n_stints = comp_data.groupby(["driver_code", "stint_number"]).ngroups

    # Aggregate by tyre_age using MEDIAN for outlier robustness
    agg = (
        comp_data.groupby("tyre_age")["lap_delta_from_baseline_sec"]
        .median()
        .reset_index()
        .rename(columns={"lap_delta_from_baseline_sec": "median_delta"})
        .sort_values("tyre_age")
    )

    ages_arr   = agg["tyre_age"].values.astype(float)
    deltas_arr = agg["median_delta"].values.astype(float)

    # Detect cliff
    cliff_lap = detect_cliff(
        tyre_ages       = ages_arr,
        lap_time_deltas = deltas_arr,
        compound        = compound,
    )

    # Fall back to compound profile cliff if detection fails
    if cliff_lap is None:
        profile = get_compound_profile(compound, circuit)
        cliff_lap = profile.get("cliff_lap")
        if cliff_lap is not None:
            logger.debug(
                "fit_compound_model [%s]: using profile cliff_lap=%d "
                "(data-driven detection found no cliff).",
                compound, cliff_lap,
            )

    # Fit piecewise model
    predict_fn = _build_predict_fn(ages_arr, deltas_arr, cliff_lap)

    # Evaluate
    predicted = predict_fn(ages_arr)
    model_r2  = float(r2_score(deltas_arr, predicted))
    model_mae = float(mean_absolute_error(deltas_arr, predicted))

    if model_r2 < 0.0:
        logger.warning(
            "fit_compound_model [%s]: R²=%.3f < 0 — model is worse than "
            "a constant predictor. Data may be too noisy for reliable fitting.",
            compound, model_r2,
        )

    # Degradation rates
    pre_mask = ages_arr < cliff_lap if cliff_lap else np.ones(len(ages_arr), dtype=bool)
    pre_a, pre_d = ages_arr[pre_mask], deltas_arr[pre_mask]
    deg_rate_linear = float(
        (pre_d[-1] - pre_d[0]) / (pre_a[-1] - pre_a[0])
        if len(pre_a) >= 2 else 0.0
    )

    deg_rate_cliff = None
    if cliff_lap is not None:
        post_mask = ages_arr >= cliff_lap
        post_a, post_d = ages_arr[post_mask], deltas_arr[post_mask]
        if len(post_a) >= 2:
            deg_rate_cliff = float(
                (post_d[-1] - post_d[0]) / (post_a[-1] - post_a[0])
            )

    # Dense curve for plotting
    max_age    = min(int(ages_arr.max()) + 5, PLOT_AGE_MAX)
    model_ages = np.arange(1.0, max_age + 1.0)
    model_deltas = predict_fn(model_ages)

    model = TyreDegradationModel(
        compound        = compound.upper(),
        circuit         = circuit,
        n_laps          = n_laps,
        n_stints        = n_stints,
        cliff_lap       = cliff_lap,
        deg_rate_linear = deg_rate_linear,
        deg_rate_cliff  = deg_rate_cliff,
        r2              = model_r2,
        mae_sec         = model_mae,
        predict         = predict_fn,
        fitted_ages     = ages_arr,
        fitted_deltas   = deltas_arr,
        model_ages      = model_ages,
        model_deltas    = model_deltas,
    )

    logger.info("fit_compound_model: %s", model.summary())
    return model


def fit_all_compounds(
    feature_df: pd.DataFrame,
    circuit: str = "unknown",
    season: int = 2023,
    target_compounds: Optional[frozenset[str]] = None,
) -> DegradationModelSet:
    """
    Fit degradation models for all available compounds.

    Args:
        feature_df:        Output of feature_builder.build_feature_set().
        circuit:           Circuit name.
        season:            Championship year.
        target_compounds:  Compounds to attempt. Defaults to DRY_COMPOUNDS.

    Returns:
        DegradationModelSet with all successfully fitted models.
    """
    compounds_to_try = (
        sorted(target_compounds)
        if target_compounds
        else sorted(DRY_COMPOUNDS)
    )
    present_compounds = set(feature_df["compound"].unique())

    model_set = DegradationModelSet(circuit=circuit, season=season)

    logger.info(
        "fit_all_compounds: circuit=%s  season=%d  attempting=%s",
        circuit, season, compounds_to_try,
    )

    for compound in compounds_to_try:
        if compound not in present_compounds:
            logger.warning(
                "fit_all_compounds: '%s' not in dataset — skipping.", compound
            )
            model_set.compounds_skipped.append(compound)
            continue

        model = fit_compound_model(feature_df, compound, circuit)
        if model is not None:
            model_set.models[compound] = model
            model_set.compounds_fitted.append(compound)
        else:
            model_set.compounds_skipped.append(compound)

    logger.info(
        "fit_all_compounds: fitted=%s  skipped=%s",
        model_set.compounds_fitted,
        model_set.compounds_skipped,
    )
    return model_set


def build_degradation_models(
    feature_df: pd.DataFrame,
    circuit: str = "unknown",
    season: int = 2023,
    target_compounds: Optional[frozenset[str]] = None,
) -> DegradationModelSet:
    """
    Top-level entry point: fit all compound degradation models.

    Intended to be called from the notebook or dashboard after the full
    feature pipeline (fastf1_loader → telemetry_processor → feature_builder)
    has been run.

    Args:
        feature_df:       Output of feature_builder.build_feature_set().
        circuit:          Circuit name (e.g. "Bahrain").
        season:           Year (e.g. 2023).
        target_compounds: Compounds to model. Defaults to DRY_COMPOUNDS.

    Returns:
        DegradationModelSet ready for consumption by race_simulator.

    Example:
        >>> model_set = build_degradation_models(features, "Bahrain", 2023)
        >>> delta = model_set.predict("SOFT", 20)
        >>> print(f"SOFT at lap 20: +{delta:.3f}s")
    """
    return fit_all_compounds(
        feature_df       = feature_df,
        circuit          = circuit,
        season           = season,
        target_compounds = target_compounds,
    )
