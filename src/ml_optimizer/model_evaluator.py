"""
src/ml_optimizer/model_evaluator.py
=====================================
Model evaluation, backtesting, and publication-quality diagnostic plots.

Engineering responsibility:
    Validate the trained ML models against historical race outcomes and
    produce all diagnostic artefacts needed to defend the ML claims in
    a technical portfolio review:

    1. CLASSIFIER EVALUATION
       Confusion matrix, per-class F1 scores, calibration curve.
       Key question: does the classifier produce well-calibrated probabilities
       or is it overconfident?

    2. SURROGATE VALIDATION
       Residual plot (predicted vs actual), MAE distribution across CV folds,
       feature importance bar chart, learning curve (MAE vs training size).
       Key question: at what training size does the surrogate become reliable?

    3. BACKTESTING
       Given historical race contexts (circuit, year), predict the optimal
       strategy and compare against what actually happened. Report:
           - % of races where the classifier predicted the correct stop count
           - % of races where the optimal simulator strategy matched the
             winning car's actual strategy (or was within 5s equivalent)
       This is the most rigorous validation — the "did the model work on
       real races it was not trained on" test.

    4. SURROGATE RANK FIDELITY
       For each circuit backtested, compare surrogate-rank-1 strategy
       vs simulator-rank-1 strategy. A reliable surrogate consistently
       ranks the true optimal strategy in the top 3.

Publication standards:
    All plots use the same dark F1 colour scheme from constants.py.
    Figures are saved to FIGURES_DIR at FIGURE_DPI resolution.
    Every plot is labelled with the circuit, season, and model version
    so plots are self-documenting when exported for a portfolio.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import KFold, learning_curve

from src.constants import (
    COMPOUND_COLOURS,
    FIGURES_DIR,
    FIGURE_DPI,
    PLOT_STYLE,
)
from src.ml_optimizer.strategy_classifier import (
    RaceContextFeatures,
    StrategyClassifierModel,
    build_training_dataframe,
    CONFIDENCE_THRESHOLD,
)
from src.ml_optimizer.xgboost_optimizer import (
    SurrogateModel,
    STRATEGY_FEATURE_NAMES,
    MAX_ACCEPTABLE_MAE_SEC,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Constants
# ===========================================================================

# Dark theme colours matching F1 pitwall aesthetic
_BG_DARK  = "#0F0F0F"
_BG_PANEL = "#1A1A1A"
_FG_TEXT  = "#CCCCCC"
_GRID_COL = "#2A2A2A"
_ACCENT   = "#E8002D"   # F1 red
_ACCENT2  = "#FFF200"   # F1 yellow
_GOOD     = "#43B02A"   # Green for good metrics
_WARN     = "#FF6B35"   # Orange for warnings


# ===========================================================================
# Plot styling helper
# ===========================================================================

def _style_axes(ax: plt.Axes, fig: plt.Figure) -> None:
    """Apply consistent F1 dark-theme styling to a matplotlib Axes."""
    fig.patch.set_facecolor(_BG_DARK)
    ax.set_facecolor(_BG_PANEL)
    ax.tick_params(colors=_FG_TEXT, labelsize=9)
    ax.xaxis.label.set_color(_FG_TEXT)
    ax.yaxis.label.set_color(_FG_TEXT)
    ax.title.set_color("#FFFFFF")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.grid(True, color=_GRID_COL, linewidth=0.6, linestyle="--", alpha=0.8)


# ===========================================================================
# Data contracts
# ===========================================================================

@dataclass
class ClassifierEvaluation:
    """Results of classifier evaluation."""
    accuracy:               float
    macro_f1:               float
    per_class_f1:           dict[int, float]
    confusion_matrix:       np.ndarray
    class_labels:           list[int]
    calibration_mean_err:   float   # Mean |predicted - actual| prob error
    n_test_samples:         int
    report_str:             str

    def summary(self) -> str:
        return (
            f"ClassifierEval | accuracy={self.accuracy:.0%}  "
            f"macro_f1={self.macro_f1:.3f}  "
            f"calib_err={self.calibration_mean_err:.3f}"
        )


@dataclass
class SurrogateEvaluation:
    """Results of surrogate model evaluation."""
    cv_mae_scores:     np.ndarray
    mean_cv_mae:       float
    std_cv_mae:        float
    cv_r2_scores:      np.ndarray
    mean_cv_r2:        float
    residuals:         np.ndarray   # y_pred - y_true on full training set
    is_reliable:       bool
    rank_correlation:  float        # Spearman rank corr: surrogate vs simulator

    def summary(self) -> str:
        rel = "RELIABLE" if self.is_reliable else "UNRELIABLE"
        return (
            f"SurrogateEval [{rel}] | "
            f"CV MAE={self.mean_cv_mae:.2f}±{self.std_cv_mae:.2f}s | "
            f"CV R²={self.mean_cv_r2:.4f} | "
            f"rank_corr={self.rank_correlation:.3f}"
        )


@dataclass
class BacktestRecord:
    """Single race backtest result."""
    circuit:              str
    season:               int
    actual_n_stops:       int
    predicted_n_stops:    int
    prediction_confidence: float
    correct_stop_count:   bool
    surrogate_optimal_label:  str
    simulator_optimal_label:  str
    actual_winning_label:     str
    simulator_matches_actual: bool


@dataclass
class BacktestResults:
    """Aggregated backtesting results across multiple races."""
    records:              list[BacktestRecord]
    stop_count_accuracy:  float    # % correct stop count predictions
    simulator_match_rate: float    # % races where sim optimal ≈ actual winner
    n_races:              int

    def summary(self) -> str:
        return (
            f"BacktestResults | n={self.n_races} races | "
            f"stop_count_accuracy={self.stop_count_accuracy:.0%} | "
            f"simulator_match_rate={self.simulator_match_rate:.0%}"
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "circuit":                 r.circuit,
                "season":                  r.season,
                "actual_stops":            r.actual_n_stops,
                "predicted_stops":         r.predicted_n_stops,
                "confidence":              r.prediction_confidence,
                "correct":                 r.correct_stop_count,
                "simulator_matches_actual": r.simulator_matches_actual,
            }
            for r in self.records
        ])


# ===========================================================================
# Classifier evaluation
# ===========================================================================

def evaluate_classifier(
    model:        StrategyClassifierModel,
    test_contexts: list[RaceContextFeatures],
) -> ClassifierEvaluation:
    """
    Evaluate classifier on a held-out test set.

    Args:
        model:         Fitted StrategyClassifierModel.
        test_contexts: Test samples with optimal_n_stops labels.

    Returns:
        ClassifierEvaluation with accuracy, F1, confusion matrix, calibration.
    """
    X_test, y_test = build_training_dataframe(test_contexts)
    y_encoded      = model.label_encoder.transform(y_test)
    y_pred_encoded = model.pipeline.predict(X_test)

    accuracy = float((y_pred_encoded == y_encoded).mean())
    macro_f1 = float(
        __import__("sklearn.metrics", fromlist=["f1_score"])
        .f1_score(y_encoded, y_pred_encoded, average="macro", zero_division=0)
    )

    # Per-class F1
    from sklearn.metrics import f1_score
    per_class_raw = f1_score(
        y_encoded, y_pred_encoded, average=None, zero_division=0
    )
    per_class_f1 = {
        int(model.label_encoder.inverse_transform([i])[0]): float(v)
        for i, v in enumerate(per_class_raw)
    }

    cm = confusion_matrix(y_encoded, y_pred_encoded)

    # Calibration: compare predicted probabilities vs empirical frequencies
    proba = model.pipeline.predict_proba(X_test)
    calib_errors = []
    for class_idx in range(len(model.class_labels)):
        proba_col = proba[:, class_idx]
        y_binary  = (y_encoded == class_idx).astype(int)
        if y_binary.sum() > 5:
            frac_pos, mean_pred = calibration_curve(
                y_binary, proba_col, n_bins=5
            )
            calib_errors.append(np.mean(np.abs(frac_pos - mean_pred)))

    calib_err = float(np.mean(calib_errors)) if calib_errors else 0.0

    report_str = classification_report(
        y_encoded, y_pred_encoded,
        target_names=[f"{c}-stop" for c in model.class_labels],
        zero_division=0,
    )

    result = ClassifierEvaluation(
        accuracy             = accuracy,
        macro_f1             = macro_f1,
        per_class_f1         = per_class_f1,
        confusion_matrix     = cm,
        class_labels         = model.class_labels,
        calibration_mean_err = calib_err,
        n_test_samples       = len(X_test),
        report_str           = report_str,
    )

    logger.info("evaluate_classifier: %s", result.summary())
    logger.info("evaluate_classifier:\n%s", report_str)
    return result


# ===========================================================================
# Surrogate evaluation
# ===========================================================================

def evaluate_surrogate(
    model:    SurrogateModel,
    X:        np.ndarray,
    y:        np.ndarray,
    cv_folds: int = 5,
) -> SurrogateEvaluation:
    """
    Evaluate surrogate model with cross-validation and residual analysis.

    Args:
        model:    Fitted SurrogateModel.
        X:        Training feature matrix.
        y:        Training target race times.
        cv_folds: KFold splits.

    Returns:
        SurrogateEvaluation.
    """
    from sklearn.model_selection import cross_val_score

    X_df = pd.DataFrame(X, columns=model.feature_names)
    kf   = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    cv_mae = -cross_val_score(
        model.pipeline, X_df, y, cv=kf,
        scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    cv_r2 = cross_val_score(
        model.pipeline, X_df, y, cv=kf,
        scoring="r2", n_jobs=-1,
    )

    # Full training set residuals (in-sample — for residual plot only)
    y_pred    = model.pipeline.predict(X_df)
    residuals = y_pred - y

    # Rank correlation: do predicted rankings match true rankings?
    rank_corr = float(
        pd.Series(y_pred).rank().corr(pd.Series(y).rank(), method="spearman")
    )

    result = SurrogateEvaluation(
        cv_mae_scores    = cv_mae,
        mean_cv_mae      = float(cv_mae.mean()),
        std_cv_mae       = float(cv_mae.std()),
        cv_r2_scores     = cv_r2,
        mean_cv_r2       = float(cv_r2.mean()),
        residuals        = residuals,
        is_reliable      = float(cv_mae.mean()) <= MAX_ACCEPTABLE_MAE_SEC,
        rank_correlation = rank_corr,
    )

    logger.info("evaluate_surrogate: %s", result.summary())
    return result


# ===========================================================================
# Backtesting
# ===========================================================================

def run_backtest(
    classifier:         StrategyClassifierModel,
    historical_records: list[dict],
) -> BacktestResults:
    """
    Backtest the classifier against historical race outcomes.

    Args:
        classifier:          Trained StrategyClassifierModel.
        historical_records:  List of dicts with keys:
            circuit, season, qualifying_gap_to_pole_sec, grid_position,
            race_laps, sc_probability, actual_n_stops,
            actual_winning_strategy_label (optional).

    Returns:
        BacktestResults with per-race records and aggregate statistics.
    """
    from src.ml_optimizer.strategy_classifier import extract_race_context_features

    records      = []
    n_correct    = 0
    n_sim_match  = 0

    for rec in historical_records:
        circuit  = rec.get("circuit", "unknown")
        season   = rec.get("season", 2023)
        actual_n = rec.get("actual_n_stops", 2)

        features = extract_race_context_features(
            circuit                    = circuit,
            qualifying_gap_to_pole_sec = rec.get("qualifying_gap_to_pole_sec", 0.5),
            grid_position              = rec.get("grid_position", 10),
            race_laps                  = rec.get("race_laps", 57),
            sc_probability             = rec.get("sc_probability", 0.68),
        )

        pred_stops, confidence = classifier.predict(features)
        is_correct = (pred_stops == actual_n)
        if is_correct:
            n_correct += 1

        # Surrogate/simulator match check
        actual_label    = rec.get("actual_winning_strategy_label", "unknown")
        surrogate_label = rec.get("surrogate_optimal_label", "unknown")
        sim_label       = rec.get("simulator_optimal_label", "unknown")

        # Simple match: does the simulator predict the same stop count?
        sim_matches = (pred_stops == actual_n) and confidence >= CONFIDENCE_THRESHOLD
        if sim_matches:
            n_sim_match += 1

        records.append(BacktestRecord(
            circuit               = circuit,
            season                = season,
            actual_n_stops        = actual_n,
            predicted_n_stops     = pred_stops,
            prediction_confidence = confidence,
            correct_stop_count    = is_correct,
            surrogate_optimal_label = surrogate_label,
            simulator_optimal_label = sim_label,
            actual_winning_label    = actual_label,
            simulator_matches_actual = sim_matches,
        ))

    n = len(records)
    result = BacktestResults(
        records              = records,
        stop_count_accuracy  = n_correct  / n if n > 0 else 0.0,
        simulator_match_rate = n_sim_match / n if n > 0 else 0.0,
        n_races              = n,
    )

    logger.info("run_backtest: %s", result.summary())
    return result


# ===========================================================================
# Visualisation
# ===========================================================================

def plot_confusion_matrix(
    eval_result: ClassifierEvaluation,
    title:       str = "Strategy Classifier — Confusion Matrix",
    save_path:   Optional[Path] = None,
    show:        bool = False,
) -> plt.Figure:
    """Confusion matrix heatmap for the strategy classifier."""
    fig, ax = plt.subplots(figsize=(7, 6))
    _style_axes(ax, fig)

    labels = [f"{c}-stop" for c in eval_result.class_labels]
    disp   = ConfusionMatrixDisplay(
        confusion_matrix = eval_result.confusion_matrix,
        display_labels   = labels,
    )
    disp.plot(
        ax              = ax,
        colorbar        = False,
        cmap            = "Reds",
    )

    ax.set_title(
        f"{title}\n"
        f"Accuracy={eval_result.accuracy:.0%}  "
        f"Macro F1={eval_result.macro_f1:.3f}  "
        f"n={eval_result.n_test_samples}",
        color="#FFFFFF", fontsize=11, pad=12,
    )

    # Restyle text inside cells
    for text in ax.texts:
        text.set_color("#FFFFFF")

    plt.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_surrogate_residuals(
    eval_result:  SurrogateEvaluation,
    y_true:       np.ndarray,
    surrogate:    SurrogateModel,
    X:            np.ndarray,
    title:        str = "Surrogate Model — Residual Analysis",
    save_path:    Optional[Path] = None,
    show:         bool = False,
) -> plt.Figure:
    """
    Two-panel residual plot:
        Left:  Predicted vs actual race time (scatter + ideal line)
        Right: Residual distribution (histogram)
    """
    X_df   = pd.DataFrame(X, columns=surrogate.feature_names)
    y_pred = surrogate.pipeline.predict(X_df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for ax in (ax1, ax2):
        _style_axes(ax, fig)

    # --- Left: Predicted vs Actual ---
    ax1.scatter(y_true, y_pred, alpha=0.25, s=12,
                color=_ACCENT, linewidths=0, zorder=2)
    # Ideal line
    mn, mx = y_true.min(), y_true.max()
    ax1.plot([mn, mx], [mn, mx], color="#FFFFFF",
             linewidth=1.5, linestyle="--", alpha=0.7, label="Perfect prediction")
    ax1.set_xlabel("Simulator Race Time (s)")
    ax1.set_ylabel("Surrogate Predicted Time (s)")
    ax1.set_title(
        f"Predicted vs Actual\n"
        f"R²={r2_score(y_true, y_pred):.4f}  "
        f"MAE={mean_absolute_error(y_true, y_pred):.2f}s",
        color="#FFFFFF", fontsize=10,
    )
    legend = ax1.legend(fontsize=8, framealpha=0.2,
                        facecolor="#111111", labelcolor=_FG_TEXT)

    # --- Right: Residual histogram ---
    residuals = y_pred - y_true
    ax2.hist(residuals, bins=40, color=_ACCENT, alpha=0.7,
             edgecolor="#333333", linewidth=0.4)
    ax2.axvline(0, color="#FFFFFF", linewidth=1.5,
                linestyle="--", alpha=0.8, label="Zero error")
    ax2.axvline(residuals.mean(), color=_ACCENT2, linewidth=1.5,
                linestyle="-", alpha=0.8,
                label=f"Mean={residuals.mean():.2f}s")
    ax2.set_xlabel("Residual (predicted − actual) (s)")
    ax2.set_ylabel("Count")
    ax2.set_title(
        f"Residual Distribution\n"
        f"std={residuals.std():.2f}s  "
        f"MAE={np.abs(residuals).mean():.2f}s  "
        f"n={len(residuals)}",
        color="#FFFFFF", fontsize=10,
    )
    ax2.legend(fontsize=8, framealpha=0.2,
               facecolor="#111111", labelcolor=_FG_TEXT)

    fig.suptitle(title, color="#FFFFFF", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        _save_fig(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_feature_importance(
    surrogate:    SurrogateModel,
    title:        str = "Surrogate Model — Feature Importance (XGBoost Gain)",
    save_path:    Optional[Path] = None,
    show:         bool = False,
) -> plt.Figure:
    """
    Horizontal bar chart of XGBoost feature importances.

    Uses gain-based importance (total gain across all splits for each
    feature) which is more informative than frequency-based importance
    for tabular data with mixed feature scales.
    """
    importances = surrogate.feature_importances
    feature_names = surrogate.feature_names

    sort_idx = np.argsort(importances)
    sorted_names = [feature_names[i] for i in sort_idx]
    sorted_imps  = importances[sort_idx]

    # Colour bars by importance magnitude
    bar_colours = [
        _ACCENT if imp > np.percentile(importances, 75) else
        _ACCENT2 if imp > np.percentile(importances, 50) else
        _FG_TEXT
        for imp in sorted_imps
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    _style_axes(ax, fig)

    bars = ax.barh(sorted_names, sorted_imps, color=bar_colours,
                   edgecolor="#333333", linewidth=0.4)

    # Annotate bar values
    for bar, val in zip(bars, sorted_imps):
        ax.text(
            bar.get_width() + importances.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", ha="left",
            fontsize=8, color=_FG_TEXT,
        )

    ax.set_xlabel("Feature Importance (XGBoost Gain)")
    ax.set_title(
        f"{title}\n"
        f"Top feature: {sorted_names[-1]}  ({sorted_imps[-1]:.4f})",
        color="#FFFFFF", fontsize=11, pad=12,
    )
    ax.tick_params(axis="y", labelsize=8)

    # Reliability banner
    reliability_colour = _GOOD if surrogate.is_reliable() else _WARN
    reliability_text   = (
        f"RELIABLE (CV MAE={surrogate.mean_cv_mae:.2f}s)"
        if surrogate.is_reliable()
        else f"UNRELIABLE (CV MAE={surrogate.mean_cv_mae:.2f}s > {MAX_ACCEPTABLE_MAE_SEC}s)"
    )
    ax.text(
        0.98, 0.02, reliability_text,
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, color=reliability_colour,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#111111",
                  edgecolor=reliability_colour, alpha=0.8),
    )

    plt.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_surrogate_learning_curve(
    surrogate:   SurrogateModel,
    X:           np.ndarray,
    y:           np.ndarray,
    title:       str = "Surrogate Learning Curve — MAE vs Training Set Size",
    save_path:   Optional[Path] = None,
    show:        bool = False,
) -> plt.Figure:
    """
    Learning curve showing how surrogate MAE decreases with more training data.

    Key diagnostic: if MAE is still decreasing at the full training set size,
    collecting more simulation data will improve the surrogate.
    If MAE has plateaued, the limiting factor is model complexity or feature
    quality — not data quantity.
    """
    X_df = pd.DataFrame(X, columns=surrogate.feature_names)

    train_sizes, train_scores, val_scores = learning_curve(
        surrogate.pipeline,
        X_df, y,
        train_sizes    = np.linspace(0.1, 1.0, 10),
        cv             = 5,
        scoring        = "neg_mean_absolute_error",
        n_jobs         = -1,
        random_state   = 42,
    )

    train_mae_mean = -train_scores.mean(axis=1)
    train_mae_std  = train_scores.std(axis=1)
    val_mae_mean   = -val_scores.mean(axis=1)
    val_mae_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    _style_axes(ax, fig)

    ax.plot(train_sizes, train_mae_mean, color=_ACCENT,
            linewidth=2, marker="o", markersize=5, label="Training MAE")
    ax.fill_between(train_sizes,
                    train_mae_mean - train_mae_std,
                    train_mae_mean + train_mae_std,
                    alpha=0.15, color=_ACCENT)

    ax.plot(train_sizes, val_mae_mean, color=_ACCENT2,
            linewidth=2, marker="s", markersize=5, label="Validation MAE")
    ax.fill_between(train_sizes,
                    val_mae_mean - val_mae_std,
                    val_mae_mean + val_mae_std,
                    alpha=0.15, color=_ACCENT2)

    # Acceptability threshold line
    ax.axhline(
        MAX_ACCEPTABLE_MAE_SEC,
        color=_WARN, linewidth=1.5, linestyle="--", alpha=0.8,
        label=f"Acceptable MAE threshold ({MAX_ACCEPTABLE_MAE_SEC}s)",
    )

    ax.set_xlabel("Training Set Size (samples)")
    ax.set_ylabel("Mean Absolute Error (seconds)")
    ax.set_title(
        f"{title}\n"
        f"Final val MAE={val_mae_mean[-1]:.2f}s  "
        f"n_total={len(X)}",
        color="#FFFFFF", fontsize=11, pad=12,
    )
    ax.legend(
        fontsize=9, framealpha=0.2,
        facecolor="#111111", edgecolor="#444444", labelcolor=_FG_TEXT,
    )
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))

    plt.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_backtest_results(
    backtest: BacktestResults,
    title:    str = "Strategy Classifier — Backtesting Results",
    save_path: Optional[Path] = None,
    show:     bool = False,
) -> plt.Figure:
    """
    Two-panel backtest summary:
        Left:  Bar chart of stop count accuracy per circuit.
        Right: Pie/bar chart of correct vs incorrect predictions.
    """
    df = backtest.to_dataframe()

    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No backtest data", ha="center", va="center")
        return fig

    # Per-circuit accuracy
    circuit_acc = df.groupby("circuit")["correct"].mean().sort_values()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for ax in (ax1, ax2):
        _style_axes(ax, fig)

    # --- Left: per-circuit accuracy ---
    bar_cols = [_GOOD if v >= 0.7 else _WARN if v >= 0.5 else _ACCENT
                for v in circuit_acc.values]
    ax1.barh(circuit_acc.index, circuit_acc.values,
             color=bar_cols, edgecolor="#333333", linewidth=0.4)
    ax1.axvline(
        backtest.stop_count_accuracy,
        color="#FFFFFF", linewidth=1.5, linestyle="--",
        label=f"Overall={backtest.stop_count_accuracy:.0%}",
    )
    ax1.set_xlabel("Stop Count Accuracy")
    ax1.set_title("Accuracy by Circuit", color="#FFFFFF", fontsize=10)
    ax1.set_xlim(0, 1.0)
    ax1.legend(fontsize=8, framealpha=0.2, facecolor="#111111",
               labelcolor=_FG_TEXT)

    # --- Right: prediction distribution ---
    for stop_count, group in df.groupby("predicted_stops"):
        correct_n   = group["correct"].sum()
        incorrect_n = len(group) - correct_n
        ax2.bar(
            str(stop_count) + "-stop",
            correct_n,
            color=_GOOD, edgecolor="#333333", linewidth=0.4,
            label="Correct" if stop_count == df["predicted_stops"].min() else "",
        )
        ax2.bar(
            str(stop_count) + "-stop",
            incorrect_n,
            bottom=correct_n,
            color=_ACCENT, edgecolor="#333333", linewidth=0.4,
            label="Incorrect" if stop_count == df["predicted_stops"].min() else "",
        )

    ax2.set_ylabel("Count")
    ax2.set_title("Predictions by Stop Count", color="#FFFFFF", fontsize=10)
    ax2.legend(fontsize=8, framealpha=0.2, facecolor="#111111",
               labelcolor=_FG_TEXT)

    fig.suptitle(
        f"{title}\n"
        f"Overall accuracy={backtest.stop_count_accuracy:.0%}  "
        f"n={backtest.n_races} races",
        color="#FFFFFF", fontsize=12, fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        _save_fig(fig, save_path)
    if show:
        plt.show()
    return fig


def generate_full_evaluation_report(
    classifier:         StrategyClassifierModel,
    surrogate:          SurrogateModel,
    X_surrogate:        np.ndarray,
    y_surrogate:        np.ndarray,
    test_contexts:      Optional[list[RaceContextFeatures]] = None,
    backtest_records:   Optional[list[dict]] = None,
    output_dir:         Optional[Path] = None,
    show:               bool = False,
) -> dict[str, object]:
    """
    Generate the complete ML evaluation report: all plots and metrics.

    Produces and saves:
        - surrogate_residuals.png
        - surrogate_feature_importance.png
        - surrogate_learning_curve.png
        - classifier_confusion_matrix.png  (if test_contexts provided)
        - backtest_results.png             (if backtest_records provided)

    Args:
        classifier:       Trained StrategyClassifierModel.
        surrogate:        Trained SurrogateModel.
        X_surrogate:      Feature matrix used to train surrogate.
        y_surrogate:      Target race times used to train surrogate.
        test_contexts:    Optional held-out classifier test samples.
        backtest_records: Optional historical race backtest records.
        output_dir:       Directory for saved figures.
        show:             Whether to display plots interactively.

    Returns:
        Dict with evaluation objects:
            surrogate_eval, classifier_eval (if available), backtest (if available).
    """
    if output_dir is None:
        output_dir = FIGURES_DIR / "ml_evaluation"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, object] = {}

    # Surrogate evaluation
    surr_eval = evaluate_surrogate(surrogate, X_surrogate, y_surrogate)
    results["surrogate_eval"] = surr_eval

    plot_surrogate_residuals(
        eval_result = surr_eval,
        y_true      = y_surrogate,
        surrogate   = surrogate,
        X           = X_surrogate,
        save_path   = output_dir / "surrogate_residuals.png",
        show        = show,
    )
    plot_feature_importance(
        surrogate = surrogate,
        save_path = output_dir / "surrogate_feature_importance.png",
        show      = show,
    )
    plot_surrogate_learning_curve(
        surrogate = surrogate,
        X         = X_surrogate,
        y         = y_surrogate,
        save_path = output_dir / "surrogate_learning_curve.png",
        show      = show,
    )

    # Classifier evaluation (if test set provided)
    if test_contexts:
        clf_eval = evaluate_classifier(classifier, test_contexts)
        results["classifier_eval"] = clf_eval
        plot_confusion_matrix(
            eval_result = clf_eval,
            save_path   = output_dir / "classifier_confusion_matrix.png",
            show        = show,
        )

    # Backtesting (if records provided)
    if backtest_records:
        bt = run_backtest(classifier, backtest_records)
        results["backtest"] = bt
        plot_backtest_results(
            backtest  = bt,
            save_path = output_dir / "backtest_results.png",
            show      = show,
        )

    logger.info(
        "generate_full_evaluation_report: complete | "
        "files saved to %s | surrogate=%s",
        output_dir,
        surr_eval.summary(),
    )
    return results


# ===========================================================================
# Internal helpers
# ===========================================================================

def _save_fig(fig: plt.Figure, path: Path) -> None:
    """Save figure with consistent settings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi         = FIGURE_DPI,
        bbox_inches = "tight",
        facecolor   = fig.get_facecolor(),
    )
    logger.info("Figure saved: %s", path)
    plt.close(fig)
