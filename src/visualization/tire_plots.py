"""
src/visualization/tire_plots.py
=================================
Publication-quality tyre degradation visualisations.

Engineering responsibility:
    Translate DegradationModelSet outputs into plots that communicate
    four specific insights to the tyre engineer and race strategist:

    1. COMPOUND DEGRADATION CURVES
       How fast does each compound degrade? Where is the cliff?
       Audience: tyre engineer, race strategist.
       Answer: "Softs cliff at lap 16, mediums at lap 26."

    2. PACE DROP HEATMAP
       Which drivers/stints had above-average degradation?
       Identifies outlier stints (damage, contact, unusual management).
       Audience: race engineer reviewing post-session data.
       Answer: "HAM stint 2 shows 40% above-median degradation — investigate."

    3. STINT COMPARISON SCATTER
       Individual driver-stint degradation vs fitted model.
       Shows model fit quality and driver-to-driver variance.
       Audience: senior engineer validating model reliability.
       Answer: "Model fits 87% of variance; outlier is Russell stint 3."

    4. COMPOUND COMPARISON SUMMARY
       Side-by-side degradation rate bars with cliff annotations.
       Quick-reference chart for strategic compound choice meetings.
       Audience: head of strategy, pre-race briefing.
       Answer: "Medium degradation rate is 58% of soft — second stint
                medium gains 0.4s/lap vs soft from lap 20."

Design rules applied consistently:
    - Dark background (#0F0F0F / #1A1A1A panels) — pitwall standard.
    - FOM compound colours (red=S, yellow=M, grey=H) — instantly readable.
    - Cliff markers in #FF6B35 (orange) — high visibility against dark.
    - All axes labelled with units. All titles include n= sample sizes.
    - Figures saved at FIGURE_DPI (150) for print-quality export.
    - No chart junk: no box borders, minimal tick marks, informative grid.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.constants import (
    COMPOUND_COLOURS,
    COMPOUND_ABBREV,
    FIGURES_DIR,
    FIGURE_DPI,
    DRY_COMPOUNDS,
)
from src.tire_model.degradation_model import DegradationModelSet, TyreDegradationModel

logger = logging.getLogger(__name__)


# ===========================================================================
# Theme constants
# ===========================================================================

_BG_FIGURE  = "#0F0F0F"
_BG_PANEL   = "#1A1A1A"
_FG_TEXT    = "#CCCCCC"
_FG_WHITE   = "#FFFFFF"
_GRID       = "#2A2A2A"
_CLIFF_COL  = "#FF6B35"
_SCATTER_ALPHA = 0.22
_CURVE_LW      = 2.4


# ===========================================================================
# Internal helpers
# ===========================================================================

def _style(ax: plt.Axes, fig: plt.Figure) -> None:
    """Apply consistent F1 dark-theme styling."""
    fig.patch.set_facecolor(_BG_FIGURE)
    ax.set_facecolor(_BG_PANEL)
    ax.tick_params(colors=_FG_TEXT, labelsize=9)
    ax.xaxis.label.set_color(_FG_TEXT)
    ax.yaxis.label.set_color(_FG_TEXT)
    ax.title.set_color(_FG_WHITE)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.grid(True, color=_GRID, linewidth=0.55, linestyle="--", alpha=0.75)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.grid(True, which="minor", color="#212121", linewidth=0.3, linestyle=":")


def _save(fig: plt.Figure, path: Optional[Path], close: bool = True) -> None:
    """Save figure to disk if path is provided, then optionally close."""
    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        logger.info("Figure saved: %s", p)
    if close:
        plt.close(fig)


def _add_cliff_annotation(
    ax:       plt.Axes,
    model:    TyreDegradationModel,
    colour:   str,
) -> None:
    """
    Add a vertical dashed line and text annotation at the cliff lap.

    The cliff annotation is the single most strategically important
    element on the degradation chart — it marks the MAXIMUM safe stint
    length. The orange colour (distinct from all compound colours)
    ensures it is immediately visible.
    """
    if model.cliff_lap is None:
        return

    cliff_x    = float(model.cliff_lap)
    cliff_y    = float(model.predict(np.array([cliff_x]))[0])
    y_range    = ax.get_ylim()
    panel_top  = y_range[1] if y_range[1] > 0 else 1.5

    ax.axvline(cliff_x, color=_CLIFF_COL, linewidth=1.6,
               linestyle="--", alpha=0.88, zorder=5)

    ax.annotate(
        f"Cliff\nL{int(cliff_x)}",
        xy       = (cliff_x, cliff_y),
        xytext   = (cliff_x + 1.5, cliff_y + panel_top * 0.08),
        fontsize = 8,
        color    = _CLIFF_COL,
        arrowprops=dict(arrowstyle="->", color=_CLIFF_COL, lw=1.1),
        zorder   = 6,
    )


def _legend(ax: plt.Axes, **kwargs) -> None:
    """Styled legend for dark background panels."""
    ax.legend(
        fontsize     = 8,
        framealpha   = 0.25,
        facecolor    = "#111111",
        edgecolor    = "#444444",
        labelcolor   = _FG_TEXT,
        **kwargs,
    )


# ===========================================================================
# 1. Compound degradation curves
# ===========================================================================

def plot_compound_degradation_curves(
    model_set:      DegradationModelSet,
    feature_df:     pd.DataFrame,
    compounds:      Optional[list[str]] = None,
    title:          Optional[str] = None,
    save_path:      Optional[Path] = None,
    show:           bool = False,
) -> plt.Figure:
    """
    Publication-quality compound degradation curve comparison.

    Three-layer plot per compound:
        Layer 1 (scatter, low alpha): individual driver-stint lap deltas.
                 Shows the raw distribution — model fit quality is visible
                 from how tight the scatter is around the curve.
        Layer 2 (filled scatter): median delta per tyre age — the signal
                 the model was actually fitted to.
        Layer 3 (smooth curve): fitted piecewise model — the clean
                 degradation signal with cliff annotation.

    Why three layers:
        Showing only the model curve would hide uncertainty.
        Showing only the scatter would hide the fitted signal.
        All three together let the engineer read both the estimate
        and the confidence interval simultaneously.

    Args:
        model_set:   DegradationModelSet from degradation_model.py.
        feature_df:  Output of feature_builder.build_feature_set().
                     Used for individual lap scatter points.
        compounds:   Compounds to plot. Defaults to model_set.compounds_fitted.
        title:       Plot suptitle override.
        save_path:   File path for saving.
        show:        Display interactively.

    Returns:
        Matplotlib Figure.
    """
    compounds = compounds or model_set.compounds_fitted
    if not compounds:
        logger.warning("plot_compound_degradation_curves: no fitted compounds.")
        return plt.figure()

    n  = len(compounds)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 6), squeeze=False)
    axes = axes[0]

    suptitle = title or (
        f"Tyre Degradation Model — {model_set.circuit} {model_set.season}\n"
        r"Fuel & track-evolution corrected  |  Δt vs fresh tyre baseline"
    )
    fig.suptitle(suptitle, color=_FG_WHITE, fontsize=13,
                 fontweight="bold", y=1.02)
    fig.patch.set_facecolor(_BG_FIGURE)

    required = {"compound", "tyre_age", "lap_delta_from_baseline_sec", "is_representative"}
    has_feature_df = (
        feature_df is not None
        and not feature_df.empty
        and required.issubset(feature_df.columns)
    )

    for ax, compound in zip(axes, compounds):
        _style(ax, fig)
        model  = model_set.models.get(compound)
        colour = COMPOUND_COLOURS.get(compound, "#888888")

        if model is None:
            ax.set_title(f"{compound} — no model", color=_FG_WHITE)
            continue

        # --- Layer 1: individual lap scatter ---
        if has_feature_df:
            comp_laps = feature_df[
                (feature_df["compound"] == compound)
                & feature_df["is_representative"]
            ]
            if not comp_laps.empty:
                ax.scatter(
                    comp_laps["tyre_age"],
                    comp_laps["lap_delta_from_baseline_sec"],
                    alpha     = _SCATTER_ALPHA,
                    s         = 14,
                    color     = colour,
                    linewidths= 0,
                    zorder    = 2,
                    label     = "Individual laps",
                )

        # --- Layer 2: median per tyre age (fitting signal) ---
        ax.scatter(
            model.fitted_ages,
            model.fitted_deltas,
            color      = colour,
            s          = 58,
            zorder     = 4,
            edgecolors = _FG_WHITE,
            linewidths = 0.9,
            label      = "Median per tyre age",
        )

        # --- Layer 3: smooth fitted curve ---
        ax.plot(
            model.model_ages,
            model.model_deltas,
            color     = colour,
            linewidth = _CURVE_LW,
            zorder    = 3,
            label     = f"Model (R²={model.r2:.3f}  MAE={model.mae_sec:.3f}s)",
        )

        # Refresh y-limits before cliff annotation so annotation is positioned correctly
        ax.relim()
        ax.autoscale_view()
        _add_cliff_annotation(ax, model, colour)

        # Linear phase degradation rate annotation
        ax.text(
            0.04, 0.96,
            f"deg = {model.deg_rate_linear:+.3f} s/lap (linear phase)",
            transform = ax.transAxes,
            fontsize  = 8,
            color     = colour,
            va        = "top",
            ha        = "left",
            bbox      = dict(boxstyle="round,pad=0.3", facecolor="#111111",
                             edgecolor=colour, alpha=0.7),
            zorder    = 7,
        )

        ax.set_xlabel("Tyre Age (laps into stint)", fontsize=10)
        ax.set_ylabel(r"$\Delta$t vs fresh tyre (seconds)", fontsize=10)
        ax.set_title(
            f"{compound}\n"
            f"n={model.n_laps} laps / {model.n_stints} stints",
            fontsize=10, color=_FG_WHITE, pad=10,
        )
        ax.set_xlim(left=1)
        ax.set_ylim(bottom=-0.05)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
        _legend(ax)

    plt.tight_layout()
    _save(fig, save_path, close=False)
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ===========================================================================
# 2. Pace drop heatmap
# ===========================================================================

def plot_pace_drop_heatmap(
    feature_df:  pd.DataFrame,
    compound:    str = "SOFT",
    title:       Optional[str] = None,
    save_path:   Optional[Path] = None,
    show:        bool = False,
) -> plt.Figure:
    """
    Heatmap of per-lap pace drop by driver and stint.

    Rows = drivers sorted by mean pace drop (best at top).
    Columns = tyre age (laps into stint).
    Cell colour = pace_drop_sec relative to compound median.

    Engineering value:
        Identifies drivers managing tyres unusually (positive = faster
        than expected → tyre saving) or degrading unusually fast (negative
        → tyre damage, aggressive driving, different set). Post-session
        this feeds into strategy adjustments for the next race.

    Args:
        feature_df: Output of feature_builder.build_feature_set().
                    Must contain: driver_code, stint_number, tyre_age,
                    pace_drop_sec, compound, is_representative.
        compound:   Compound to visualise (one chart per compound).
        title:      Suptitle override.
        save_path:  File save path.
        show:       Interactive display.

    Returns:
        Matplotlib Figure.
    """
    required = {"driver_code", "stint_number", "tyre_age",
                "pace_drop_sec", "compound", "is_representative"}
    missing = required - set(feature_df.columns)
    if missing:
        logger.warning(
            "plot_pace_drop_heatmap: missing columns %s — skipping.", sorted(missing)
        )
        return plt.figure()

    comp_data = feature_df[
        (feature_df["compound"] == compound)
        & feature_df["is_representative"]
        & feature_df["pace_drop_sec"].notna()
    ].copy()

    if comp_data.empty:
        logger.warning(
            "plot_pace_drop_heatmap: no data for compound '%s'.", compound
        )
        return plt.figure()

    # Create (driver+stint) label
    comp_data["driver_stint"] = (
        comp_data["driver_code"] + " S" + comp_data["stint_number"].astype(str)
    )

    # Pivot: rows = driver_stint, cols = tyre_age, values = pace_drop_sec
    pivot = comp_data.pivot_table(
        index   = "driver_stint",
        columns = "tyre_age",
        values  = "pace_drop_sec",
        aggfunc = "mean",
    )

    # Sort rows by mean pace drop (most degrading at bottom)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    colour = COMPOUND_COLOURS.get(compound, "#888888")
    fig, ax = plt.subplots(figsize=(max(10, pivot.shape[1] * 0.55),
                                    max(5, pivot.shape[0] * 0.45)))
    _style(ax, fig)

    import matplotlib.colors as mcolors

    # Diverging colormap centred at 0 (green = improving, red = degrading)
    cmap = plt.get_cmap("RdYlGn_r")
    vmax = float(np.nanpercentile(np.abs(pivot.values), 95))
    vmax = max(vmax, 0.05)

    im = ax.imshow(
        pivot.values,
        aspect = "auto",
        cmap   = cmap,
        vmin   = -vmax,
        vmax   =  vmax,
        interpolation = "nearest",
    )

    # Axis labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.astype(int), fontsize=8, color=_FG_TEXT)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8, color=_FG_TEXT)

    ax.set_xlabel("Tyre Age (laps)", fontsize=10)
    ax.set_ylabel("Driver / Stint", fontsize=10)

    # Colourbar
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Pace Drop (s/lap)  [+ve = slower]",
                   color=_FG_TEXT, fontsize=9)
    cbar.ax.tick_params(colors=_FG_TEXT, labelsize=8)

    # Annotate cells with value (only where data exists)
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            val = pivot.values[r, c]
            if not np.isnan(val):
                ax.text(c, r, f"{val:+.2f}",
                        ha="center", va="center",
                        fontsize=6, color="#111111",
                        fontweight="bold")

    t = title or f"Pace Drop Heatmap — {compound}  |  {feature_df['driver_code'].nunique()} drivers"
    ax.set_title(t, color=_FG_WHITE, fontsize=11, pad=12)

    plt.tight_layout()
    _save(fig, save_path, close=False)
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ===========================================================================
# 3. Stint comparison scatter
# ===========================================================================

def plot_stint_comparison(
    model_set:   DegradationModelSet,
    feature_df:  pd.DataFrame,
    compound:    str,
    highlight_drivers: Optional[list[str]] = None,
    title:       Optional[str] = None,
    save_path:   Optional[Path] = None,
    show:        bool = False,
) -> plt.Figure:
    """
    Scatter of individual driver-stint degradation curves vs the fitted model.

    Each driver-stint is drawn as a thin line; highlighted drivers are
    drawn in full opacity. The fitted model curve is overlaid in the
    FOM compound colour. Outlier stints are immediately visible as lines
    that deviate far from the model.

    Args:
        model_set:         DegradationModelSet.
        feature_df:        feature_builder output.
        compound:          Compound to plot (one figure per compound).
        highlight_drivers: Driver codes to emphasise (e.g. ["VER", "PER"]).
        title:             Suptitle override.
        save_path:         Save path.
        show:              Interactive display.

    Returns:
        Matplotlib Figure.
    """
    model = model_set.models.get(compound)
    if model is None:
        logger.warning("plot_stint_comparison: no model for '%s'.", compound)
        return plt.figure()

    required = {"driver_code", "stint_number", "tyre_age",
                "lap_delta_from_baseline_sec", "is_representative", "compound"}
    if not required.issubset(feature_df.columns):
        logger.warning("plot_stint_comparison: missing required columns.")
        return plt.figure()

    comp_data = feature_df[
        (feature_df["compound"] == compound)
        & feature_df["is_representative"]
    ].copy().sort_values(["driver_code", "stint_number", "tyre_age"])

    colour = COMPOUND_COLOURS.get(compound, "#888888")
    fig, ax = plt.subplots(figsize=(11, 6))
    _style(ax, fig)

    drawn_drivers: set[str] = set()

    for (driver, stint_n), stint_laps in comp_data.groupby(
        ["driver_code", "stint_number"]
    ):
        is_highlighted = highlight_drivers and driver in highlight_drivers
        alpha = 0.80 if is_highlighted else 0.25
        lw    = 1.8  if is_highlighted else 0.8
        label = f"{driver} S{stint_n}" if is_highlighted and driver not in drawn_drivers else None
        if is_highlighted and driver not in drawn_drivers:
            drawn_drivers.add(driver)

        ax.plot(
            stint_laps["tyre_age"],
            stint_laps["lap_delta_from_baseline_sec"],
            alpha     = alpha,
            linewidth = lw,
            color     = colour if is_highlighted else "#888888",
            label     = label,
            zorder    = 3 if is_highlighted else 2,
        )

    # Fitted model curve (always on top)
    ax.plot(
        model.model_ages,
        model.model_deltas,
        color     = colour,
        linewidth = _CURVE_LW + 0.4,
        zorder    = 5,
        linestyle = "-",
        label     = f"Fitted model (R²={model.r2:.3f})",
    )

    _add_cliff_annotation(ax, model, colour)

    ax.set_xlabel("Tyre Age (laps into stint)", fontsize=10)
    ax.set_ylabel(r"$\Delta$t vs driver stint baseline (seconds)", fontsize=10)

    t = title or (
        f"Driver Stint Comparison — {compound} | "
        f"{model_set.circuit} {model_set.season}"
    )
    n_stints = comp_data.groupby(["driver_code", "stint_number"]).ngroups
    ax.set_title(f"{t}\n{n_stints} stints shown", color=_FG_WHITE,
                 fontsize=11, pad=10)
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=-0.1)
    _legend(ax, loc="upper left")

    plt.tight_layout()
    _save(fig, save_path, close=False)
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ===========================================================================
# 4. Compound comparison summary bar chart
# ===========================================================================

def plot_compound_summary(
    model_set:  DegradationModelSet,
    title:      Optional[str] = None,
    save_path:  Optional[Path] = None,
    show:       bool = False,
) -> plt.Figure:
    """
    Side-by-side bar chart comparing key degradation metrics across compounds.

    Three sub-panels:
        (A) Baseline degradation rate (s/lap) — linear phase.
        (B) Post-cliff degradation rate (s/lap) — with a clear visual
            break between linear and cliff rates.
        (C) Predicted total stint delta at cliff lap (seconds above fresh).

    This is the pre-race briefing chart: one glance tells the strategist
    which compound is most durable, when to expect the cliff, and how
    much time is saved by pitting before vs after the cliff.

    Args:
        model_set:  DegradationModelSet.
        title:      Suptitle override.
        save_path:  Save path.
        show:       Interactive display.

    Returns:
        Matplotlib Figure.
    """
    if not model_set.compounds_fitted:
        logger.warning("plot_compound_summary: no fitted compounds.")
        return plt.figure()

    compounds    = model_set.compounds_fitted
    colours      = [COMPOUND_COLOURS.get(c, "#888888") for c in compounds]
    x            = np.arange(len(compounds))
    bar_width    = 0.6

    fig = plt.figure(figsize=(11, 6))
    fig.patch.set_facecolor(_BG_FIGURE)

    gs   = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax in axes:
        _style(ax, fig)

    # --- Panel A: Linear phase degradation rate ---
    linear_rates = [model_set.models[c].deg_rate_linear for c in compounds]
    bars_a = axes[0].bar(x, linear_rates, width=bar_width,
                         color=colours, edgecolor="#333333", linewidth=0.5, zorder=2)
    for bar, val in zip(bars_a, linear_rates):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.001,
                     f"{val:+.3f}", ha="center", va="bottom",
                     fontsize=8, color=_FG_TEXT)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([COMPOUND_ABBREV.get(c, c) for c in compounds],
                             fontsize=9, color=_FG_TEXT)
    axes[0].set_ylabel("Degradation Rate (s/lap)", fontsize=9)
    axes[0].set_title("Linear Phase\nDeg Rate", color=_FG_WHITE, fontsize=10)

    # --- Panel B: Cliff lap + post-cliff rate (dual axis) ---
    cliff_laps = [
        model_set.models[c].cliff_lap or 0
        for c in compounds
    ]
    cliff_rates = [
        model_set.models[c].deg_rate_cliff or 0.0
        for c in compounds
    ]

    bars_b1 = axes[1].bar(x - 0.18, cliff_laps, width=0.35,
                          color=colours, edgecolor="#333333",
                          linewidth=0.5, alpha=0.7, zorder=2,
                          label="Cliff lap")
    ax1_twin = axes[1].twinx()
    ax1_twin.set_facecolor(_BG_PANEL)
    ax1_twin.tick_params(colors=_FG_TEXT, labelsize=8)
    ax1_twin.yaxis.label.set_color(_FG_TEXT)

    bars_b2 = ax1_twin.bar(x + 0.18, cliff_rates, width=0.35,
                           color=_CLIFF_COL, edgecolor="#333333",
                           linewidth=0.5, alpha=0.9, zorder=2,
                           label="Post-cliff rate")

    for bar, val in zip(bars_b1, cliff_laps):
        if val > 0:
            axes[1].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.3,
                         f"L{int(val)}", ha="center", va="bottom",
                         fontsize=7, color=_FG_TEXT)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([COMPOUND_ABBREV.get(c, c) for c in compounds],
                             fontsize=9, color=_FG_TEXT)
    axes[1].set_ylabel("Cliff Lap (tyre age)", fontsize=9)
    ax1_twin.set_ylabel("Post-cliff Rate (s/lap)", fontsize=8, color=_CLIFF_COL)
    axes[1].set_title("Cliff Detection", color=_FG_WHITE, fontsize=10)

    patch_cliff = mpatches.Patch(color=colours[0], alpha=0.7, label="Cliff lap")
    patch_rate  = mpatches.Patch(color=_CLIFF_COL, label="Post-cliff rate")
    axes[1].legend(handles=[patch_cliff, patch_rate], fontsize=7,
                   framealpha=0.2, facecolor="#111111", labelcolor=_FG_TEXT,
                   loc="upper right")

    # --- Panel C: Cumulative delta at cliff lap ---
    cum_deltas = []
    for c in compounds:
        m    = model_set.models[c]
        cval = m.cliff_lap if m.cliff_lap else int(m.model_ages.max())
        delta = float(m.predict(np.array([float(cval)]))[0])
        cum_deltas.append(delta)

    bars_c = axes[2].bar(x, cum_deltas, width=bar_width,
                         color=colours, edgecolor="#333333",
                         linewidth=0.5, zorder=2)
    for bar, val in zip(bars_c, cum_deltas):
        axes[2].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.01,
                     f"+{val:.2f}s", ha="center", va="bottom",
                     fontsize=8, color=_FG_TEXT)

    axes[2].set_xticks(x)
    axes[2].set_xticklabels([COMPOUND_ABBREV.get(c, c) for c in compounds],
                             fontsize=9, color=_FG_TEXT)
    axes[2].set_ylabel("Total Δt at Cliff Lap (seconds)", fontsize=9)
    axes[2].set_title("Cumulative Loss\nat Cliff Lap", color=_FG_WHITE, fontsize=10)

    t = title or f"Compound Summary — {model_set.circuit} {model_set.season}"
    fig.suptitle(t, color=_FG_WHITE, fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    _save(fig, save_path, close=False)
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ===========================================================================
# 5. Multi-compound overlay
# ===========================================================================

def plot_degradation_overlay(
    model_set:  DegradationModelSet,
    max_age:    int = 40,
    title:      Optional[str] = None,
    save_path:  Optional[Path] = None,
    show:       bool = False,
) -> plt.Figure:
    """
    All fitted compound degradation curves on a single axes for direct comparison.

    The overlay answers: "At tyre age N, how much faster is a fresh tyre
    vs the current set — and which compound has the smallest delta?"

    This is the chart shown at pit window decision time: which compound
    choice gives the best race time for the remaining laps available?

    Args:
        model_set:  DegradationModelSet.
        max_age:    Maximum tyre age to display.
        title:      Suptitle override.
        save_path:  Save path.
        show:       Interactive display.

    Returns:
        Matplotlib Figure.
    """
    if not model_set.compounds_fitted:
        logger.warning("plot_degradation_overlay: no fitted compounds.")
        return plt.figure()

    fig, ax = plt.subplots(figsize=(11, 6))
    _style(ax, fig)
    ages = np.arange(1.0, max_age + 1.0)

    for compound in model_set.compounds_fitted:
        model  = model_set.models[compound]
        colour = COMPOUND_COLOURS.get(compound, "#888888")
        deltas = model.predict(ages)

        ax.plot(ages, deltas, color=colour,
                linewidth=_CURVE_LW, zorder=3,
                label=f"{compound}  "
                      f"(deg={model.deg_rate_linear:+.3f}s/lap  "
                      f"cliff={f'L{model.cliff_lap}' if model.cliff_lap else 'none'})")

        # Shade the cliff phase
        if model.cliff_lap is not None:
            cliff_ages = ages[ages >= model.cliff_lap]
            if len(cliff_ages) > 0:
                cliff_deltas = model.predict(cliff_ages)
                ax.fill_between(cliff_ages, cliff_deltas,
                                alpha=0.12, color=colour, zorder=2)
            ax.axvline(model.cliff_lap, color=colour,
                       linewidth=0.9, linestyle=":", alpha=0.6, zorder=2)

    ax.set_xlabel("Tyre Age (laps into stint)", fontsize=11)
    ax.set_ylabel(r"Lap time increase vs fresh tyre (Δ seconds)", fontsize=11)

    t = title or (
        f"Compound Degradation Overlay — {model_set.circuit} {model_set.season}"
    )
    ax.set_title(t, color=_FG_WHITE, fontsize=12, pad=12)
    ax.set_xlim(1, max_age)
    ax.set_ylim(bottom=-0.05)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    _legend(ax, loc="upper left")

    # Annotation: "Shaded region = cliff phase"
    ax.text(0.98, 0.04,
            "Shaded region = cliff phase (accelerating degradation)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#888888",
            style="italic")

    plt.tight_layout()
    _save(fig, save_path, close=False)
    if show:
        plt.show()
    plt.close(fig)
    return fig
