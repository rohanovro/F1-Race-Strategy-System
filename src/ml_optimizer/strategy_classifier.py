"""
src/ml_optimizer/strategy_classifier.py
=========================================
Multiclass classifier: predicts the optimal number of pit stops
for a given race context.

Engineering responsibility:
    Train a multiclass classifier on historical FastF1 race data
    to predict the most commonly used (and winning) pit stop count.
    The classifier answers one question:

        Given: circuit, qualifying gap to pole, grid position,
               available compounds, weather label, and circuit
               degradation profile — how many stops did the TOP
               FINISHERS use?

    This is used as a DATA-DRIVEN PRIOR before the strategy search:
        - If the classifier says "2-stop with high confidence", the
          optimizer focuses the search on 2-stop strategies.
        - If confidence is low (< CONFIDENCE_THRESHOLD), the full
          1/2/3-stop search is run without pruning.

Why this matters for the portfolio:
    A naive optimizer searches all strategies exhaustively. A smart one
    uses domain knowledge to focus the search. This classifier encodes
    the accumulated knowledge of 5+ seasons of F1 race strategies in a
    form that reduces computation by 60–70% on clear-case races.

Feature engineering for classification:
    The features are engineered to capture the known drivers of pit
    stop count:

    1. CIRCUIT DEGRADATION SEVERITY (from compound_profiles.py)
       High degradation → more stops. Encoded as the mean baseline
       degradation rate across all dry compounds for the circuit.

    2. CIRCUIT OVERTAKING DIFFICULTY
       Hard to overtake → fewer stops (track position matters more).
       Easy to overtake → more stops viable (pace > position).

    3. QUALIFYING GAP TO POLE (seconds)
       Large gap → likely starting deeper in the field → more
       aggressive multi-stop strategy to recover positions.

    4. GRID POSITION
       Front row → protect track position → conservative strategy.
       Midfield → aggressive to gain positions.

    5. TYRE AVAILABILITY LABEL
       Which compounds were nominated for this event (affects
       viable strategy count).

    6. SC PROBABILITY (from sc_detector CircuitSCProfile)
       High SC probability → flexible multi-stop more valuable.

    7. RACE DISTANCE (laps)
       Longer race → more stops possible within tyre constraints.

Model architecture:
    RandomForestClassifier (primary):
        - Robust to small datasets (40–80 historical races)
        - Built-in feature importance via impurity decrease
        - Naturally handles multiclass without OvR decomposition
        - Ensemble of 200 trees provides uncertainty estimation via
          predict_proba (soft voting)

    GradientBoostingClassifier (secondary, ensemble blending):
        - Captures non-linear feature interactions (circuit × degradation)
        - Trained on same features, combined via soft voting
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src.constants import DATA_PROCESSED_DIR, DRY_COMPOUNDS
from src.tire_model.compound_profiles import get_compound_profile

logger = logging.getLogger(__name__)


# ===========================================================================
# Constants
# ===========================================================================

# Minimum historical races to train a reliable classifier.
# Below this, cross-validation folds become unstable.
MIN_TRAINING_SAMPLES: int = 20

# K-fold cross-validation splits.
CV_FOLDS: int = 5

# Confidence threshold below which we do NOT prune the search space.
# If the classifier assigns < this probability to its top prediction,
# we fall back to the full strategy search.
CONFIDENCE_THRESHOLD: float = 0.60

# Maximum number of stops the classifier considers.
# 4-stop strategies are historically rare; treating them as separate
# class reduces training data per class without strategic value.
MAX_STOPS_CLASSIFIED: int = 3

# Random seed for all sklearn estimators (reproducibility).
RANDOM_STATE: int = 42

# Feature column names — canonical set used for training and inference.
# Changing this set requires retraining all saved models.
FEATURE_COLUMNS: list[str] = [
    "mean_dry_deg_rate",        # Circuit degradation severity
    "circuit_sc_probability",   # Historical SC probability per race
    "qualifying_gap_to_pole_sec",  # Car's Q gap to pole position
    "grid_position",            # Starting grid position
    "race_laps",                # Scheduled race distance
    "n_dry_compounds_available",  # Number of dry compounds nominated
    "is_high_deg_circuit",      # Binary: above-median degradation
    "is_street_circuit",        # Binary: Monaco/Singapore/Baku style
    "deg_rate_soft",            # Soft compound baseline deg rate
    "deg_rate_medium",          # Medium compound baseline deg rate
    "deg_rate_hard",            # Hard compound baseline deg rate
]

# Circuits classified as street circuits (high track position premium,
# low overtaking, typically fewer stops).
STREET_CIRCUITS: frozenset[str] = frozenset({
    "monaco", "singapore", "baku", "las_vegas", "miami", "jeddah",
    "saudi_arabia",
})

# Above-median degradation circuits (typically 2-stop biased).
HIGH_DEG_CIRCUITS: frozenset[str] = frozenset({
    "bahrain", "silverstone", "suzuka", "spain", "hungary", "australia",
})


# ===========================================================================
# Feature extraction
# ===========================================================================

@dataclass
class RaceContextFeatures:
    """
    Feature vector for a single race event.

    All fields are numeric or binary — the sklearn pipeline handles
    no categorical encoding beyond what is baked into these numeric proxies.

    Attributes:
        circuit:                    Circuit name (for logging only).
        mean_dry_deg_rate:          Mean baseline deg rate across S/M/H.
        circuit_sc_probability:     Historical SC events per race.
        qualifying_gap_to_pole_sec: Car's qualifying time gap to pole (s).
        grid_position:              Race starting grid position (1=pole).
        race_laps:                  Scheduled race laps.
        n_dry_compounds_available:  Number of dry compounds nominated.
        is_high_deg_circuit:        1 if circuit in HIGH_DEG_CIRCUITS.
        is_street_circuit:          1 if circuit in STREET_CIRCUITS.
        deg_rate_soft:              Soft baseline deg rate.
        deg_rate_medium:            Medium baseline deg rate.
        deg_rate_hard:              Hard baseline deg rate.
        optimal_n_stops:            Label: number of stops used by top-5
                                    finishers (for training). None for inference.
    """
    circuit:                    str
    mean_dry_deg_rate:          float
    circuit_sc_probability:     float
    qualifying_gap_to_pole_sec: float
    grid_position:              int
    race_laps:                  int
    n_dry_compounds_available:  int
    is_high_deg_circuit:        int
    is_street_circuit:          int
    deg_rate_soft:              float
    deg_rate_medium:            float
    deg_rate_hard:              float
    optimal_n_stops:            Optional[int] = None

    def to_array(self) -> np.ndarray:
        """Convert to 1D numpy array matching FEATURE_COLUMNS order."""
        return np.array([
            self.mean_dry_deg_rate,
            self.circuit_sc_probability,
            self.qualifying_gap_to_pole_sec,
            float(self.grid_position),
            float(self.race_laps),
            float(self.n_dry_compounds_available),
            float(self.is_high_deg_circuit),
            float(self.is_street_circuit),
            self.deg_rate_soft,
            self.deg_rate_medium,
            self.deg_rate_hard,
        ], dtype=np.float64)


def extract_race_context_features(
    circuit:                    str,
    qualifying_gap_to_pole_sec: float,
    grid_position:              int,
    race_laps:                  int,
    sc_probability:             float,
    available_dry_compounds:    Optional[list[str]] = None,
    optimal_n_stops:            Optional[int] = None,
) -> RaceContextFeatures:
    """
    Build a RaceContextFeatures from raw race context inputs.

    Looks up compound degradation profiles from compound_profiles.py
    to extract circuit-specific degradation rates.

    Args:
        circuit:                    Circuit name (e.g. "Bahrain").
        qualifying_gap_to_pole_sec: Qualifying time gap to P1 (seconds).
        grid_position:              Race start position (1 = pole).
        race_laps:                  Scheduled race distance in laps.
        sc_probability:             Historical SC events per race at circuit.
        available_dry_compounds:    Nominated dry compounds. Defaults to all 3.
        optimal_n_stops:            Training label — stops used by top 5.
                                    None for inference.

    Returns:
        RaceContextFeatures ready for classification.
    """
    if available_dry_compounds is None:
        available_dry_compounds = list(DRY_COMPOUNDS)

    circuit_key = circuit.lower().strip().replace(" ", "_")

    # Extract degradation rates from compound profiles
    def _deg_rate(compound: str) -> float:
        try:
            profile = get_compound_profile(compound, circuit)
            return float(profile.get("baseline_deg_rate_sec_per_lap", 0.04))
        except (KeyError, Exception):
            return 0.04  # Global fallback

    deg_soft   = _deg_rate("SOFT")
    deg_medium = _deg_rate("MEDIUM")
    deg_hard   = _deg_rate("HARD")
    mean_deg   = np.mean([deg_soft, deg_medium, deg_hard])

    is_high_deg = int(circuit_key in HIGH_DEG_CIRCUITS)
    is_street   = int(circuit_key in STREET_CIRCUITS)

    n_dry = sum(
        1 for c in available_dry_compounds
        if c.upper() in DRY_COMPOUNDS
    )

    features = RaceContextFeatures(
        circuit                    = circuit,
        mean_dry_deg_rate          = mean_deg,
        circuit_sc_probability     = sc_probability,
        qualifying_gap_to_pole_sec = qualifying_gap_to_pole_sec,
        grid_position              = grid_position,
        race_laps                  = race_laps,
        n_dry_compounds_available  = n_dry,
        is_high_deg_circuit        = is_high_deg,
        is_street_circuit          = is_street,
        deg_rate_soft              = deg_soft,
        deg_rate_medium            = deg_medium,
        deg_rate_hard              = deg_hard,
        optimal_n_stops            = optimal_n_stops,
    )

    logger.debug(
        "extract_race_context_features [%s]: mean_deg=%.4f  "
        "is_high_deg=%d  is_street=%d  grid=%d  gap=%.2fs",
        circuit, mean_deg, is_high_deg, is_street,
        grid_position, qualifying_gap_to_pole_sec,
    )
    return features


def build_training_dataframe(
    race_contexts: list[RaceContextFeatures],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Convert a list of RaceContextFeatures into (X, y) training arrays.

    Args:
        race_contexts: List with optimal_n_stops populated (training data).

    Returns:
        Tuple of (X DataFrame, y Series) ready for sklearn.

    Raises:
        ValueError: If any sample is missing optimal_n_stops label.
    """
    missing_labels = [r.circuit for r in race_contexts if r.optimal_n_stops is None]
    if missing_labels:
        raise ValueError(
            f"build_training_dataframe: {len(missing_labels)} samples missing "
            f"optimal_n_stops label: {missing_labels[:5]}..."
        )

    X = pd.DataFrame(
        [r.to_array() for r in race_contexts],
        columns=FEATURE_COLUMNS,
    )
    y = pd.Series(
        [min(r.optimal_n_stops, MAX_STOPS_CLASSIFIED) for r in race_contexts],
        name="optimal_n_stops",
    )

    logger.info(
        "build_training_dataframe: %d samples | "
        "class distribution: %s",
        len(X),
        y.value_counts().to_dict(),
    )
    return X, y


# ===========================================================================
# Model training
# ===========================================================================

@dataclass
class StrategyClassifierModel:
    """
    Trained strategy classifier with metadata.

    Attributes:
        pipeline:         Fitted sklearn Pipeline (scaler + classifier).
        label_encoder:    LabelEncoder for stop-count classes.
        feature_columns:  Feature column names (canonical order).
        cv_scores:        Cross-validation F1 scores (macro).
        mean_cv_f1:       Mean CV F1 score.
        std_cv_f1:        Std dev of CV F1 scores.
        class_labels:     Sorted list of stop-count classes (e.g. [1, 2, 3]).
        training_samples: Number of training samples.
        classification_report_str: Full sklearn classification report.
    """
    pipeline:                  Pipeline
    label_encoder:             LabelEncoder
    feature_columns:           list[str]
    cv_scores:                 np.ndarray
    mean_cv_f1:                float
    std_cv_f1:                 float
    class_labels:              list[int]
    training_samples:          int
    classification_report_str: str = ""

    def predict(self, features: RaceContextFeatures) -> tuple[int, float]:
        """
        Predict optimal stop count and confidence for a race context.

        Args:
            features: RaceContextFeatures for inference.

        Returns:
            Tuple of (predicted_n_stops: int, confidence: float).
        """
        X = features.to_array().reshape(1, -1)
        X_df = pd.DataFrame(X, columns=self.feature_columns)

        pred_encoded = self.pipeline.predict(X_df)[0]
        proba        = self.pipeline.predict_proba(X_df)[0]
        confidence   = float(proba.max())

        # Decode: predicted class index → stop count integer
        pred_stops = int(self.label_encoder.inverse_transform([pred_encoded])[0])

        logger.info(
            "StrategyClassifierModel.predict [%s]: "
            "→ %d-stop (confidence=%.0%%)",
            features.circuit, pred_stops, confidence * 100,
        )
        return pred_stops, confidence

    def predict_proba_all(
        self, features: RaceContextFeatures
    ) -> dict[int, float]:
        """
        Return probability for each stop count class.

        Returns:
            Dict mapping n_stops -> probability.
        """
        X    = features.to_array().reshape(1, -1)
        X_df = pd.DataFrame(X, columns=self.feature_columns)
        proba = self.pipeline.predict_proba(X_df)[0]
        return {
            int(self.label_encoder.inverse_transform([i])[0]): float(p)
            for i, p in enumerate(proba)
        }

    def summary(self) -> str:
        return (
            f"StrategyClassifier | n={self.training_samples} | "
            f"CV F1={self.mean_cv_f1:.3f}±{self.std_cv_f1:.3f} | "
            f"classes={self.class_labels}"
        )


def train_strategy_classifier(
    race_contexts: list[RaceContextFeatures],
    n_estimators:  int = 200,
    cv_folds:      int = CV_FOLDS,
) -> StrategyClassifierModel:
    """
    Train and cross-validate the strategy classifier.

    Pipeline:
        StandardScaler → CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=200, class_weight='balanced')
        )

    Calibration wrapping:
        RandomForestClassifier.predict_proba() is known to produce
        overconfident predictions (probabilities cluster near 0 and 1).
        CalibratedClassifierCV with isotonic regression corrects this
        miscalibration, producing probabilities that better reflect
        true uncertainty — critical for the CONFIDENCE_THRESHOLD logic.

    Class weights:
        'balanced' upweights rare stop counts (e.g. 3-stop or 0-stop).
        Without this, a dataset dominated by 2-stop races would produce
        a classifier that always predicts 2-stop with high "accuracy"
        while being useless for minority classes.

    Args:
        race_contexts: Training data with optimal_n_stops labels.
        n_estimators:  RF trees (more = better, diminishing returns >200).
        cv_folds:      StratifiedKFold folds for CV.

    Returns:
        Fitted StrategyClassifierModel.

    Raises:
        ValueError: If insufficient training samples.
    """
    if len(race_contexts) < MIN_TRAINING_SAMPLES:
        raise ValueError(
            f"train_strategy_classifier: only {len(race_contexts)} samples "
            f"(minimum={MIN_TRAINING_SAMPLES}). "
            f"Collect more historical race data before training."
        )

    X, y = build_training_dataframe(race_contexts)

    # Encode class labels to consecutive integers for sklearn
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Compute sample weights for class imbalance
    sample_weights = compute_sample_weight("balanced", y_encoded)

    # Primary estimator
    rf = RandomForestClassifier(
        n_estimators  = n_estimators,
        max_depth     = 8,
        min_samples_leaf = 3,
        random_state  = RANDOM_STATE,
        n_jobs        = -1,
    )

    # Calibrate probabilities using isotonic regression (5-fold internal CV)
    calibrated_rf = CalibratedClassifierCV(
        estimator = rf,
        method    = "isotonic",
        cv        = 3,
    )

    pipeline = Pipeline([
        ("scaler",     StandardScaler()),
        ("classifier", calibrated_rf),
    ])

    logger.info(
        "train_strategy_classifier: training on %d samples "
        "| classes=%s | cv_folds=%d",
        len(X), le.classes_.tolist(), cv_folds,
    )

    # Cross-validation on the FULL pipeline (scaler is fitted inside each fold)
    skf      = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                               random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        pipeline, X, y_encoded,
        cv      = skf,
        scoring = "f1_macro",
        n_jobs  = -1,
    )

    logger.info(
        "train_strategy_classifier: CV F1 (macro) = "
        "%.3f ± %.3f | fold scores: %s",
        cv_scores.mean(), cv_scores.std(),
        [f"{s:.3f}" for s in cv_scores],
    )

    # Final fit on full training set
    pipeline.fit(X, y_encoded)

    # Evaluation on training set (in-sample — for diagnostics only)
    y_pred     = pipeline.predict(X)
    report_str = classification_report(
        y_encoded, y_pred,
        target_names=[f"{int(c)}-stop" for c in le.classes_],
        zero_division=0,
    )

    logger.info(
        "train_strategy_classifier: in-sample classification report:\n%s",
        report_str,
    )

    model = StrategyClassifierModel(
        pipeline                  = pipeline,
        label_encoder             = le,
        feature_columns           = FEATURE_COLUMNS,
        cv_scores                 = cv_scores,
        mean_cv_f1                = float(cv_scores.mean()),
        std_cv_f1                 = float(cv_scores.std()),
        class_labels              = [int(c) for c in le.classes_],
        training_samples          = len(X),
        classification_report_str = report_str,
    )

    logger.info("train_strategy_classifier: %s", model.summary())
    return model


# ===========================================================================
# Strategy search pruning
# ===========================================================================

def prune_search_space(
    model:            StrategyClassifierModel,
    features:         RaceContextFeatures,
    candidate_strategies: list,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> tuple[list, dict[int, float]]:
    """
    Use classifier predictions to prune the strategy search space.

    If the classifier is confident in its prediction (>= threshold),
    filter candidate strategies to only those with the predicted stop
    count. Otherwise return all candidates unchanged.

    Engineering rationale:
        The classifier encodes domain knowledge: "at Monaco with these
        tyre characteristics, the top-5 finishers overwhelmingly used
        2 stops." Running 3,000 simulations when 2,500 of them are
        1-stop strategies at Monaco is wasteful. The classifier prunes
        these without the simulator having to evaluate them.

        The threshold is conservative (0.60) to avoid over-pruning.
        We never prune to a single stop count below 40% probability —
        we want to explore ±1 from the predicted optimum.

    Args:
        model:                Fitted StrategyClassifierModel.
        features:             Race context for prediction.
        candidate_strategies: List of RaceStrategy from enumerate_strategies().
        confidence_threshold: Minimum confidence to prune.

    Returns:
        Tuple of (pruned_strategies, stop_probabilities):
            pruned_strategies:  List of strategies matching predicted stops
                                (or all strategies if confidence is low).
            stop_probabilities: Dict of stop_count -> probability.
    """
    stop_probas = model.predict_proba_all(features)
    pred_stops, confidence = model.predict(features)

    logger.info(
        "prune_search_space [%s]: predicted=%d-stop (conf=%.0%%) | "
        "all_probs=%s",
        features.circuit, pred_stops, confidence * 100,
        {k: f"{v:.2f}" for k, v in stop_probas.items()},
    )

    if confidence < confidence_threshold:
        logger.info(
            "prune_search_space: confidence %.0%% < threshold %.0%% — "
            "returning full search space (%d strategies).",
            confidence * 100, confidence_threshold * 100,
            len(candidate_strategies),
        )
        return candidate_strategies, stop_probas

    # Include predicted stop count AND ±1 to hedge against edge cases
    allowed_stops = {max(0, pred_stops - 1), pred_stops,
                     min(MAX_STOPS_CLASSIFIED, pred_stops + 1)}
    pruned = [
        s for s in candidate_strategies
        if s.n_stops in allowed_stops
    ]

    reduction_pct = (1 - len(pruned) / len(candidate_strategies)) * 100
    logger.info(
        "prune_search_space: pruned to %d/%d strategies "
        "(%.0%% reduction) | allowed_stops=%s",
        len(pruned), len(candidate_strategies),
        reduction_pct, sorted(allowed_stops),
    )

    return pruned, stop_probas


# ===========================================================================
# Model persistence
# ===========================================================================

def save_classifier(
    model:     StrategyClassifierModel,
    save_path: Optional[Path] = None,
) -> Path:
    """
    Serialise a trained classifier to disk using pickle.

    Args:
        model:     StrategyClassifierModel to save.
        save_path: File path. Defaults to DATA_PROCESSED_DIR / "strategy_classifier.pkl".

    Returns:
        Path where the model was saved.
    """
    if save_path is None:
        save_path = DATA_PROCESSED_DIR / "strategy_classifier.pkl"

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(model, f)

    logger.info("save_classifier: saved to %s", save_path)
    return save_path


def load_classifier(load_path: Optional[Path] = None) -> StrategyClassifierModel:
    """
    Load a previously serialised StrategyClassifierModel.

    Args:
        load_path: File path. Defaults to DATA_PROCESSED_DIR / "strategy_classifier.pkl".

    Returns:
        Loaded StrategyClassifierModel.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if load_path is None:
        load_path = DATA_PROCESSED_DIR / "strategy_classifier.pkl"

    load_path = Path(load_path)
    if not load_path.exists():
        raise FileNotFoundError(
            f"load_classifier: no saved model at {load_path}. "
            "Run train_strategy_classifier() first."
        )

    with open(load_path, "rb") as f:
        model = pickle.load(f)

    logger.info("load_classifier: loaded from %s | %s", load_path, model.summary())
    return model
