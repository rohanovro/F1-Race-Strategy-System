"""
src/visualization/strategy_plots.py
=====================================
Race strategy timeline, pit window, and leaderboard visualisations.

Engineering responsibility:
    Translate OptimizationResult and SimulationResult objects into
    the three charts that every F1 strategy team displays:

    1. STRATEGY TIMELINE
       Race lap (x) vs cumulative time delta to optimal (y), one line
       per strategy, coloured by compound with pit stop markers.
       Answers: "When does strategy A overtake strategy B on track?"

    2. PIT WINDOW SENSITIVITY CURVE
       Total race time (y) vs pit lap number (x) for a fixed compound
       pair. Shows the optimal pit lap and how wide the forgiving window is.
       Answers: "We must pit between lap 18 and 24 or we lose more than 0.5s."

    3. STRATEGY LEADERBOARD WATERFALL
       Horizontal bar chart of time gaps to optimal, colour-coded by
       stop count. Shows clearly which strategies are competitive.
       Answers: "1-stop is 12s slower than optimal. All 2-stops within 3s
                 of each other — pit timing matters more than compound choice."

    4. LAP TIME COMPONENT BREAKDOWN
       Stacked area chart of fuel delta + deg delta + pit loss per lap
       for a single strategy. Shows exactly why each lap is the time it is.
       Answers: "Lap 18 is slow because of the pit stop (19s) plus the
                 out-lap cold-tyre penalty (0.8s)."

    5. MONTE CARLO RACE TIME DISTRIBUTION
       Violin or box plot of race time distributions across strategies.
       Shows risk as well as expected value.
       Answers: "Strategy A is faster on average but has higher variance —
                 Strategy B is safer."
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

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
)
from src.strategy_engine.race_simulator import (
    RaceStrategy,
    SimulationResult,
    MonteCarloResult,
)
from src.strategy_engine.pit_window_optimizer import OptimizationResult

logger = logging.getLogger(__name__)


# ===========================================================================
# Theme constants
# ===========================================================================

_BG_FIGURE = "#0F0F0F"
_BG_PANEL  = "#1A1A1A"
_FG_TEXT   = "#CCCCCC"
_FG_WHITE  = "#FFFFFF"
_GRID      = "#2A2A2A"
_ACCENT    = "#E8002D"
_ACCENT2   = "#FFF200"
_NEUTRAL   = "#555555"

_STOP_COUNT_COLOURS = {0: "#888888", 1: "#4FC3F7", 2: "#E8002D", 3: "#FF9800"}


# ===========================================================================
# Shared helpers
# ===========================================================================

def _style(ax: plt.Axes, fig: plt.Figure) -> None:
    fig.patch.set_facecolor(_BG_FIGURE)
    ax.set_facecolor(_BG_PANEL)
    ax.tick_params(colors=_FG_TEXT, labelsize=9)
    ax.xaxis.label.set_color(_FG_TEXT)
    ax.yaxis.label.set_color(_FG_TEXT)
    ax.title.set_color(_FG_WHITE)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.grid(True, color=_GRID, linewidth=0.55, linestyle="--", alpha=0.75)


def _save(fig: plt.Figure, path: Optional[Path]) -> None:
    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        logger.info("Figure saved: %s", p)
    plt.close(fig)


def _legend(ax: plt.Axes, **kwargs) -> None:
    ax.legend(fontsize=8, framealpha=0.25,
              facecolor="#111111", edgecolor="#444444",
              labelcolor=_FG_TEXT, **kwargs)


def _pit_marker(ax: plt.Axes, x: float, y: float, colour: str) -> None:
    """Draw a pit stop marker (inverted triangle) at (x, y)."""
    ax.plot(x, y, marker="v", color=colour, markersize=9,
            markeredgecolor=_FG_WHITE, markeredgewidth=0.8, zorder=5)


# ===========================================================================
# 1. Strategy timeline
# ===========================================================================

def plot_strategy_timeline(
    results:        list[SimulationResult],
    reference_idx:  int = 0,
    max_strategies: int = 8,
    title:          Optional[str] = None,
    save_path:      Optional[Path] = None,
    show:           bool = False,
) -> plt.Figure:
    """
    Cumulative time delta vs the reference strategy, lap-by-lap.

    x-axis: race lap number.
    y-axis: cumulative time delta to the reference strategy (seconds).
            Positive = reference is ahead. Negative = this strategy leads.

    Each line is coloured by the STARTING compound; stint changes are
    shown by compound-colour transitions along the line. Pit stops are
    marked with inverted triangles.

    Engineering purpose:
        This is the "live strategy screen" chart. At any race lap you can
        read: "If we pitted on lap 18 (blue line), we would currently be
        3.2s behind the optimal strategy (dashed white)."

    Args:
        results:        List of SimulationResult to compare.
                        Must include at least one result.
        reference_idx:  Index in results to use as the zero-delta reference.
                        Defaults to 0 (typically the optimal strategy).
        max_strategies: Maximum number of strategies to plot.
        title:          Suptitle override.
        save_path:      Save path.
        show:           Interactive display.

    Returns:
        Matplotlib Figure.
    """
    valid = [r for r in results if r.is_valid and r.lap_results]
    if not valid:
        logger.warning("plot_strategy_timeline: no valid simulation results.")
        return plt.figure()

    valid = valid[:max_strategies]
    ref   = valid[reference_idx] if reference_idx < len(valid) else valid[0]
    ref_df = ref.to_dataframe()
    ref_cum = ref_df["predicted_lap_sec"].cumsum().values

    fig, ax = plt.subplots(figsize=(14, 7))
    _style(ax, fig)

    for i, result in enumerate(valid):
        df     = result.to_dataframe()
        cum    = df["predicted_lap_sec"].cumsum().values
        delta  = cum - ref_cum
        laps   = df["lap_number"].values

        # Line colour: stop count
        line_colour = _STOP_COUNT_COLOURS.get(result.strategy.n_stops, "#888888")
        lw          = 2.5 if result == ref else 1.4
        ls          = "--" if result == ref else "-"
        alpha       = 1.0 if result == ref else 0.70
        lbl         = result.strategy.label + (" [ref]" if result == ref else "")

        ax.plot(laps, delta, color=line_colour,
                linewidth=lw, linestyle=ls, alpha=alpha,
                label=lbl, zorder=3)

        # Pit stop markers
        for pit_lap in result.strategy.pit_laps:
            pit_mask = df["lap_number"] == pit_lap
            if pit_mask.any():
                pit_delta = float(delta[pit_mask][0])
                pit_colour = COMPOUND_COLOURS.get(
                    df.loc[pit_mask, "compound"].values[0], "#888888"
                )
                _pit_marker(ax, pit_lap, pit_delta, pit_colour)

    # Reference zero line
    ax.axhline(0, color="#888888", linewidth=0.9, linestyle="--", alpha=0.6)
    ax.text(laps[-1] + 0.5, 0.1, "Reference", fontsize=7,
            color="#888888", va="bottom")

    ax.set_xlabel("Race Lap", fontsize=11)
    ax.set_ylabel("Cumulative Time Delta to Reference (seconds)", fontsize=11)

    # Stop count legend patches
    stop_patches = [
        mpatches.Patch(color=_STOP_COUNT_COLOURS[n], label=f"{n}-stop")
        for n in sorted(_STOP_COUNT_COLOURS)
        if any(r.strategy.n_stops == n for r in valid)
    ]
    strategy_legend = ax.legend(fontsize=7, framealpha=0.2,
                                facecolor="#111111", edgecolor="#444444",
                                labelcolor=_FG_TEXT, loc="upper left",
                                ncol=2)
    ax.add_artist(strategy_legend)
    ax.legend(handles=stop_patches, fontsize=8,
              framealpha=0.2, facecolor="#111111",
              labelcolor=_FG_TEXT, loc="lower right",
              title="Stop count", title_fontsize=8)

    t = title or "Strategy Timeline — Cumulative Time Delta"
    ax.set_title(t, color=_FG_WHITE, fontsize=12, pad=12)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))

    plt.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


# ===========================================================================
# 2. Pit window sensitivity curve
# ===========================================================================

def plot_pit_window_sensitivity(
    sensitivity_df:   pd.DataFrame,
    start_compound:   str,
    next_compound:    str,
    circuit:          str = "",
    title:            Optional[str] = None,
    save_path:        Optional[Path] = None,
    show:             bool = False,
) -> plt.Figure:
    """
    Pit window sensitivity curve: total race time vs pit lap number.

    The curve shows the cost (in total race time) of each possible pit lap.
    The "optimal window" (within EQUIVALENCE_THRESHOLD of best) is shaded.
    A flat curve → wide forgiving window. A sharp V-shape → timing critical.

    Engineering purpose:
        The pit wall engineer sets the EARLIEST and LATEST acceptable pit
        lap before the race, then adjusts within that window based on
        live SC probability, competitor position, and tyre condition.
        This plot defines those boundaries.

    Args:
        sensitivity_df:  Output of pit_window_optimizer.compute_pit_window_sensitivity().
                         Must contain: pit_lap, total_time_sec,
                         gap_to_optimal_sec, is_optimal_window.
        start_compound:  First stint compound (for labelling).
        next_compound:   Second stint compound (for labelling).
        circuit:         Circuit name for title.
        title:           Override title.
        save_path:       Save path.
        show:            Interactive display.

    Returns:
        Matplotlib Figure.
    """
    if sensitivity_df.empty:
        logger.warning("plot_pit_window_sensitivity: empty DataFrame.")
        return plt.figure()

    required = {"pit_lap", "total_time_sec", "gap_to_optimal_sec", "is_optimal_window"}
    if not required.issubset(sensitivity_df.columns):
        logger.warning("plot_pit_window_sensitivity: missing required columns.")
        return plt.figure()

    df = sensitivity_df.copy().sort_values("pit_lap")

    start_col  = COMPOUND_COLOURS.get(start_compound, "#888888")
    next_col   = COMPOUND_COLOURS.get(next_compound, "#888888")
    opt_lap    = int(df.loc[df["gap_to_optimal_sec"] == 0, "pit_lap"].values[0])
    window_df  = df[df["is_optimal_window"]]
    win_lo     = int(window_df["pit_lap"].min())
    win_hi     = int(window_df["pit_lap"].max())

    fig, ax = plt.subplots(figsize=(12, 5))
    _style(ax, fig)

    # Optimal window shading
    ax.axvspan(win_lo, win_hi, alpha=0.12, color=next_col, zorder=1,
               label=f"Optimal window (L{win_lo}–L{win_hi})")

    # Main curve
    ax.plot(df["pit_lap"], df["gap_to_optimal_sec"],
            color=_ACCENT2, linewidth=2.2, zorder=3, label="Gap to optimal (s)")

    # Fill below curve to zero
    ax.fill_between(df["pit_lap"], df["gap_to_optimal_sec"],
                    alpha=0.08, color=_ACCENT2, zorder=2)

    # Optimal pit lap vertical marker
    ax.axvline(opt_lap, color=_FG_WHITE, linewidth=1.5,
               linestyle="--", alpha=0.7, zorder=4)
    ax.text(opt_lap + 0.4, df["gap_to_optimal_sec"].max() * 0.92,
            f"Optimal: L{opt_lap}", fontsize=9, color=_FG_WHITE, va="top")

    # Compound colour swatches as x-axis annotation
    ax.annotate(
        "", xy=(win_lo, -0.035), xytext=(win_hi, -0.035),
        xycoords=("data", "axes fraction"),
        textcoords=("data", "axes fraction"),
        arrowprops=dict(arrowstyle="<->", color=next_col, lw=1.5),
        annotation_clip=False,
    )

    ax.set_xlabel("Pit Lap", fontsize=11)
    ax.set_ylabel("Gap to Optimal Race Time (seconds)", fontsize=11)
    ax.set_ylim(bottom=-0.1)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))

    abbrev_start = COMPOUND_ABBREV.get(start_compound, start_compound[0])
    abbrev_next  = COMPOUND_ABBREV.get(next_compound,  next_compound[0])
    t = title or (
        f"Pit Window Sensitivity — {abbrev_start}→{abbrev_next} 1-stop  |  "
        f"{circuit}"
    )
    ax.set_title(t, color=_FG_WHITE, fontsize=12, pad=12)
    _legend(ax, loc="upper right")

    plt.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


# ===========================================================================
# 3. Strategy leaderboard waterfall
# ===========================================================================

def plot_leaderboard_waterfall(
    opt_result:  OptimizationResult,
    top_n:       int = 20,
    title:       Optional[str] = None,
    save_path:   Optional[Path] = None,
    show:        bool = False,
) -> plt.Figure:
    """
    Horizontal waterfall chart: strategies ranked by time gap to optimal.

    Bars are coloured by stop count. The optimal strategy sits at 0 on
    the x-axis; all others extend to the right. Strategies within the
    equivalence threshold are labelled "~equivalent".

    Engineering purpose:
        Pre-race strategy meeting chart. At a glance: how many viable
        strategies exist, what is the cost of each stop count, and which
        compound combinations are equivalent.

    Args:
        opt_result:  OptimizationResult from optimise_strategy().
        top_n:       Strategies to display.
        title:       Override title.
        save_path:   Save path.
        show:        Interactive display.

    Returns:
        Matplotlib Figure.
    """
    if opt_result.leaderboard.empty:
        logger.warning("plot_leaderboard_waterfall: empty leaderboard.")
        return plt.figure()

    df = opt_result.leaderboard.head(top_n).copy()

    fig_h = max(5, len(df) * 0.42)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    _style(ax, fig)

    y_pos    = np.arange(len(df))
    gaps     = df["gap_to_optimal_sec"].values
    n_stops  = df["n_stops"].values
    labels   = df["label"].values
    equiv    = df["is_equivalent"].values

    bar_colours = [_STOP_COUNT_COLOURS.get(int(n), "#888888") for n in n_stops]

    bars = ax.barh(y_pos, gaps, color=bar_colours,
                   edgecolor="#333333", linewidth=0.4,
                   height=0.72, zorder=2)

    # Annotations
    for i, (bar, gap, eq) in enumerate(zip(bars, gaps, equiv)):
        label_x = max(gap + 0.1, 0.15)
        suffix  = "  ~equiv" if eq and gap > 0 else ""
        ax.text(label_x, i, f"+{gap:.3f}s{suffix}",
                va="center", fontsize=7, color=_FG_TEXT)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7.5, color=_FG_TEXT)
    ax.set_xlabel("Gap to Optimal Strategy (seconds)", fontsize=10)
    ax.set_xlim(left=-0.5)
    ax.invert_yaxis()

    # Stop count legend
    patches = [
        mpatches.Patch(color=_STOP_COUNT_COLOURS.get(n, "#888888"),
                       label=f"{n}-stop")
        for n in sorted(set(int(x) for x in n_stops))
    ]
    ax.legend(handles=patches, fontsize=8, framealpha=0.2,
              facecolor="#111111", edgecolor="#444444",
              labelcolor=_FG_TEXT, loc="lower right",
              title="Stop count", title_fontsize=8)

    # Equivalence window annotation
    from src.strategy_engine.pit_window_optimizer import EQUIVALENCE_THRESHOLD_SEC
    ax.axvline(EQUIVALENCE_THRESHOLD_SEC, color="#888888",
               linewidth=0.9, linestyle=":", alpha=0.6)
    ax.text(EQUIVALENCE_THRESHOLD_SEC + 0.05, 0.5,
            f"Equiv. threshold ({EQUIVALENCE_THRESHOLD_SEC}s)",
            fontsize=7, color="#888888", va="bottom",
            transform=ax.get_xaxis_transform())

    t = title or (
        f"Strategy Leaderboard — {opt_result.circuit} {opt_result.season}\n"
        f"Base pace={opt_result.base_lap_time_sec:.3f}s  |  "
        f"{opt_result.n_evaluated} strategies evaluated  |  "
        f"Optimal: {opt_result.optimal.strategy.label if opt_result.optimal else 'N/A'}"
    )
    ax.set_title(t, color=_FG_WHITE, fontsize=11, pad=10)

    plt.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


# ===========================================================================
# 4. Lap time component breakdown
# ===========================================================================

def plot_lap_time_breakdown(
    result:     SimulationResult,
    title:      Optional[str] = None,
    save_path:  Optional[Path] = None,
    show:       bool = False,
) -> plt.Figure:
    """
    Stacked area chart: per-lap time components for a single strategy.

    Components stacked from bottom:
        Base lap time (invisible — serves as y-offset for readability)
        Fuel delta (decreasing over race as fuel burns)
        Degradation delta (increasing within each stint; resets at pit)
        Pit loss (spike at pit stop laps only)
        In/out lap penalties

    Engineering purpose:
        Post-race debrief and pre-race modelling. Shows exactly where time
        is being lost each lap. "The pit stop on lap 18 cost 21.5s total
        (stationary=2.5s, traverse=19s)."

    Args:
        result:     SimulationResult from simulate_strategy().
        title:      Override title.
        save_path:  Save path.
        show:       Interactive display.

    Returns:
        Matplotlib Figure.
    """
    if not result.is_valid or not result.lap_results:
        logger.warning("plot_lap_time_breakdown: invalid or empty result.")
        return plt.figure()

    df   = result.to_dataframe()
    laps = df["lap_number"].values

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]}
    )
    for ax in (ax_top, ax_bot):
        _style(ax, fig)

    # --- Top panel: stacked components ---
    fuel_delta  = df["fuel_delta_sec"].values
    deg_delta   = df["deg_delta_sec"].values
    pit_loss    = df["pit_loss_sec"].values
    inlap_pen   = df["inlap_penalty_sec"].values
    outlap_pen  = df["outlap_penalty_sec"].values

    # Stack order (bottom to top): fuel → deg → pit → penalties
    ax_top.stackplot(
        laps,
        fuel_delta, deg_delta, pit_loss,
        inlap_pen + outlap_pen,
        labels  = ["Fuel penalty", "Tyre degradation", "Pit stop loss", "In/out lap penalty"],
        colors  = ["#4FC3F7", _ACCENT, "#FF9800", "#CE93D8"],
        alpha   = 0.82,
        zorder  = 2,
    )

    # Predicted total lap time line
    ax_top.plot(laps, df["predicted_lap_sec"] - df["base_lap_sec"],
                color=_FG_WHITE, linewidth=1.4, linestyle="--",
                alpha=0.7, zorder=3, label="Total delta above base pace")

    # Pit stop vertical markers
    for pl in result.strategy.pit_laps:
        ax_top.axvline(pl, color=_FG_WHITE, linewidth=0.9,
                       linestyle=":", alpha=0.45, zorder=4)

    ax_top.set_ylabel("Time Delta above Base Pace (seconds)", fontsize=10)
    ax_top.xaxis.set_major_locator(mticker.MultipleLocator(5))
    _legend(ax_top, loc="upper right")

    # Compound colour band along x-axis
    compounds = df["compound"].values
    for i in range(len(laps) - 1):
        col = COMPOUND_COLOURS.get(compounds[i], "#555555")
        ax_top.axvspan(laps[i], laps[i + 1], ymin=0, ymax=0.018,
                       color=col, alpha=0.9, zorder=5)

    # --- Bottom panel: tyre age ---
    tyre_ages = df["tyre_age"].values
    ax_bot.plot(laps, tyre_ages, color=_ACCENT2,
                linewidth=1.6, zorder=3)
    ax_bot.fill_between(laps, tyre_ages, alpha=0.15,
                        color=_ACCENT2, zorder=2)

    for pl in result.strategy.pit_laps:
        ax_bot.axvline(pl, color=_FG_WHITE, linewidth=0.9,
                       linestyle=":", alpha=0.45, zorder=4)

    ax_bot.set_xlabel("Race Lap", fontsize=10)
    ax_bot.set_ylabel("Tyre Age", fontsize=9)
    ax_bot.xaxis.set_major_locator(mticker.MultipleLocator(5))

    t = title or (
        f"Lap Time Component Breakdown — {result.strategy.label}\n"
        f"Total race time: {result.total_time_formatted}  |  "
        f"Base pace: {result.lap_results[0].base_lap_sec:.3f}s"
    )
    ax_top.set_title(t, color=_FG_WHITE, fontsize=11, pad=10)

    plt.tight_layout(h_pad=1.5)
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


# ===========================================================================
# 5. Monte Carlo race time distribution
# ===========================================================================

def plot_monte_carlo_distributions(
    mc_results:  list[MonteCarloResult],
    title:       Optional[str] = None,
    save_path:   Optional[Path] = None,
    show:        bool = False,
) -> plt.Figure:
    """
    Violin + box plot of Monte Carlo race time distributions per strategy.

    The violin shows the full distribution shape; the box shows P25/P75
    and the median. Overlaid dots show the P10/P90 (optimistic/pessimistic).

    Engineering purpose:
        Risk-adjusted strategy selection. Strategy A may have a lower mean
        but higher variance — in a championship battle, the lower-risk
        strategy B may be preferable. This chart makes the risk/reward
        tradeoff explicit.

    Args:
        mc_results:  List of MonteCarloResult from monte_carlo_simulate().
        title:       Override title.
        save_path:   Save path.
        show:        Interactive display.

    Returns:
        Matplotlib Figure.
    """
    if not mc_results:
        logger.warning("plot_monte_carlo_distributions: no results provided.")
        return plt.figure()

    n   = len(mc_results)
    fig, ax = plt.subplots(figsize=(max(10, n * 2.2), 7))
    _style(ax, fig)

    positions = np.arange(n)
    labels    = [r.strategy.label for r in mc_results]
    colours   = [
        _STOP_COUNT_COLOURS.get(r.strategy.n_stops, "#888888")
        for r in mc_results
    ]

    # Shift all distributions relative to the best mean for readability
    best_mean = min(r.mean_time_sec for r in mc_results)

    violin_data = [r.sample_times - best_mean for r in mc_results]
    parts = ax.violinplot(
        violin_data,
        positions = positions,
        showmedians=False, showextrema=False, widths=0.7,
    )

    for i, (body, col) in enumerate(zip(parts["bodies"], colours)):
        body.set_facecolor(col)
        body.set_alpha(0.35)
        body.set_edgecolor("#333333")
        body.set_linewidth(0.5)

    # Box plot overlay (P25/P75)
    for i, mc in enumerate(mc_results):
        shifted = mc.sample_times - best_mean
        p25, p50, p75 = np.percentile(shifted, [25, 50, 75])
        p10, p90      = np.percentile(shifted, [10, 90])
        col = colours[i]

        # IQR box
        ax.add_patch(plt.Rectangle(
            (i - 0.18, p25), 0.36, p75 - p25,
            facecolor="#333333", edgecolor=col,
            linewidth=1.2, zorder=3,
        ))
        # Median line
        ax.hlines(p50, i - 0.18, i + 0.18, colors=col,
                  linewidth=2.0, zorder=4)
        # Whiskers to P10/P90
        ax.vlines(i, p10, p25, colors=col, linewidth=1.0,
                  linestyle="--", alpha=0.6, zorder=3)
        ax.vlines(i, p75, p90, colors=col, linewidth=1.0,
                  linestyle="--", alpha=0.6, zorder=3)
        # P10/P90 dots
        ax.scatter([i, i], [p10, p90], color=col,
                   s=22, zorder=5, edgecolors="#111111", linewidths=0.5)

        # Mean label above each violin
        mean_shifted = mc.mean_time_sec - best_mean
        ax.text(i, p90 + 0.3, f"+{mean_shifted:.1f}s",
                ha="center", va="bottom", fontsize=7, color=_FG_TEXT)

    ax.axhline(0, color="#888888", linewidth=0.9,
               linestyle="--", alpha=0.6, zorder=1)
    ax.text(n - 0.5, 0.15, "Reference (fastest mean)",
            fontsize=7, color="#888888", ha="right")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right",
                       fontsize=7.5, color=_FG_TEXT)
    ax.set_ylabel("Race Time Delta vs Fastest Mean (seconds)", fontsize=10)

    # SC fraction badges below each label
    for i, mc in enumerate(mc_results):
        ax.text(i, ax.get_ylim()[0] - 0.5,
                f"SC: {mc.sc_affected_fraction:.0%}",
                ha="center", va="top", fontsize=6,
                color="#888888")

    t = title or "Monte Carlo Race Time Distributions"
    ax.set_title(
        f"{t}\n"
        f"n={mc_results[0].n_samples} samples per strategy  |  "
        f"Box=IQR  |  Dots=P10/P90  |  Dashes=whiskers",
        color=_FG_WHITE, fontsize=11, pad=10,
    )

    plt.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig
