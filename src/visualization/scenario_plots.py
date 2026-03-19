"""
src/visualization/scenario_plots.py
=====================================
Safety Car and VSC scenario decision visualisations.

Engineering responsibility:
    Translate SCDecision outputs from sc_scenario_analyzer.py into
    three charts that communicate the SC/VSC strategy decision
    under real-time pressure:

    1. SC OPTION COMPARISON (primary decision chart)
       Side-by-side Monte Carlo distributions for PIT_NOW / PIT_NEXT /
       STAY_OUT, with the recommended option highlighted.
       Answers: "Pit now saves an expected 4.2s vs staying out,
                 with P10-P90 interval [2.1s, 6.8s]."

    2. GAP COMPRESSION TIMELINE
       On-track gap vs laps into SC period, showing how quickly the
       field bunches under SC vs VSC.
       Answers: "After 3 SC laps, the 8-second gap has compressed to 1.2s
                 — the undercut window is closing fast."

    3. SC PROBABILITY HEAT MAP
       Historical SC deployment probability by race lap (from
       CircuitSCProfile KDE). Shows when the SC is most likely to arrive.
       Answers: "There is a 12% per-lap probability of SC between laps
                 20-30 at this circuit — the high-risk window."

    4. SCENARIO EXPECTED VALUE SUMMARY
       Table-style annotated chart: one row per option, columns for
       mean/P10/P50/P90 race time, pit cost, and recommendation badge.
       Answers: everything at once, for the debrief slide deck.

Design principles specific to SC charts:
    - Time pressure context: these plots are read in real time on a
      race engineer's screen. Information density must be high but
      the key recommendation must be immediately readable.
    - The recommended option is always in the FOM red (#E8002D) with
      a "RECOMMENDED" badge. Non-recommended options are in muted greys.
    - All time axes are in SECONDS RELATIVE TO THE FASTEST OPTION,
      not absolute race time — this makes the comparison immediately
      quantitative without mental arithmetic.
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
    FIGURES_DIR,
    FIGURE_DPI,
)
from src.safety_car_engine.sc_scenario_analyzer import (
    SCDecision,
    SCResponse,
    OptionOutcome,
    SCRaceState,
)
from src.safety_car_engine.sc_detector import CircuitSCProfile
from src.safety_car_engine.vsc_handler import (
    NeutralisationTimeDelta,
    apply_gap_compression,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Theme constants
# ===========================================================================

_BG_FIGURE   = "#0F0F0F"
_BG_PANEL    = "#1A1A1A"
_FG_TEXT     = "#CCCCCC"
_FG_WHITE    = "#FFFFFF"
_GRID        = "#2A2A2A"
_ACCENT      = "#E8002D"   # FOM red — RECOMMENDED option
_ACCENT2     = "#FFF200"   # FOM yellow
_NEUTRAL     = "#4A4A4A"   # Non-recommended options
_SC_COLOUR   = "#FF9800"   # Orange for SC annotations
_VSC_COLOUR  = "#4FC3F7"   # Blue for VSC annotations
_GOOD        = "#43B02A"   # Green for favourable metrics

# Response-specific colours
_RESPONSE_COLOURS: dict[SCResponse, str] = {
    SCResponse.PIT_NOW:  "#E8002D",
    SCResponse.PIT_NEXT: "#FF9800",
    SCResponse.STAY_OUT: "#4FC3F7",
}


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


def _recommended_badge(ax: plt.Axes, x: float, y: float) -> None:
    """Draw a RECOMMENDED badge at the given axes-fraction coordinates."""
    ax.text(
        x, y, "◀ RECOMMENDED",
        transform=ax.transAxes,
        fontsize=8.5, color=_ACCENT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#111111",
                  edgecolor=_ACCENT, alpha=0.85),
        va="center", ha="left",
    )


# ===========================================================================
# 1. SC option comparison (primary decision chart)
# ===========================================================================

def plot_sc_option_comparison(
    decision:   SCDecision,
    title:      Optional[str] = None,
    save_path:  Optional[Path] = None,
    show:       bool = False,
) -> plt.Figure:
    """
    Three-panel Monte Carlo distribution comparison for SC/VSC options.

    One violin per option (PIT_NOW / PIT_NEXT / STAY_OUT).
    Distributions shifted relative to the fastest mean option.
    Recommended option highlighted in FOM red.
    Key metrics annotated on each violin: mean, P10, P90, net pit cost.

    Args:
        decision:   SCDecision from sc_scenario_analyzer.evaluate_sc_scenario().
        title:      Override title.
        save_path:  Save path.
        show:       Interactive display.

    Returns:
        Matplotlib Figure.
    """
    outcomes   = decision.option_outcomes
    valid_opts = {r: o for r, o in outcomes.items()
                  if o.mean_time_sec < float("inf")}

    if not valid_opts:
        logger.warning("plot_sc_option_comparison: no valid options.")
        return plt.figure()

    best_mean = min(o.mean_time_sec for o in valid_opts.values())
    responses = list(valid_opts.keys())
    n         = len(responses)

    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 7), sharey=True,
                             squeeze=False)
    axes = axes[0]
    fig.patch.set_facecolor(_BG_FIGURE)

    nt      = decision.race_state.neutralisation_type
    nt_col  = _SC_COLOUR if nt == "SC" else _VSC_COLOUR

    for ax, response in zip(axes, responses):
        _style(ax, fig)
        outcome    = valid_opts[response]
        is_rec     = (response == decision.recommended)
        base_col   = _RESPONSE_COLOURS.get(response, "#888888")
        panel_col  = _ACCENT if is_rec else base_col

        # Violin — need sample data
        # Build a synthetic distribution from P10/P50/P90 if no raw samples
        # (outcomes from OptionOutcome may not store raw samples)
        mean_shifted = outcome.mean_time_sec - best_mean
        p10  = outcome.p10_time_sec - best_mean
        p50  = outcome.p50_time_sec - best_mean
        p90  = outcome.p90_time_sec - best_mean
        std  = outcome.std_time_sec

        # Synthesise distribution from moments (for violin shape)
        rng     = np.random.default_rng(42)
        samples = rng.normal(mean_shifted, std, 300)
        samples = np.clip(samples, p10 - 1, p90 + 1)

        parts = ax.violinplot([samples], positions=[0.5],
                              showmedians=False, showextrema=False, widths=0.7)
        for body in parts["bodies"]:
            body.set_facecolor(panel_col)
            body.set_alpha(0.30 if not is_rec else 0.55)
            body.set_edgecolor(panel_col)
            body.set_linewidth(1.2)

        # IQR box
        iqr_lo = mean_shifted - 0.674 * std
        iqr_hi = mean_shifted + 0.674 * std
        ax.add_patch(plt.Rectangle(
            (0.32, iqr_lo), 0.36, iqr_hi - iqr_lo,
            facecolor="#2A2A2A", edgecolor=panel_col, linewidth=1.4, zorder=3,
        ))
        # Median line
        ax.hlines(p50, 0.32, 0.68, colors=panel_col, linewidth=2.2, zorder=4)
        # P10/P90 whiskers
        ax.vlines(0.5, p10, iqr_lo, colors=panel_col, linewidth=1.0,
                  linestyle="--", alpha=0.7, zorder=3)
        ax.vlines(0.5, iqr_hi, p90, colors=panel_col, linewidth=1.0,
                  linestyle="--", alpha=0.7, zorder=3)
        ax.scatter([0.5, 0.5], [p10, p90], color=panel_col,
                   s=28, zorder=5, edgecolors="#111111", linewidths=0.6)

        # Key metrics text
        metrics = [
            f"Mean: +{mean_shifted:.1f}s",
            f"P10:  +{p10:.1f}s",
            f"P50:  +{p50:.1f}s",
            f"P90:  +{p90:.1f}s",
            f"Pit cost: {outcome.net_pit_cost_sec:.1f}s",
        ]
        y_text = p90 + 0.8
        for line in metrics:
            ax.text(0.5, y_text, line, ha="center", va="bottom",
                    fontsize=8, color=_FG_TEXT,
                    transform=ax.get_xaxis_transform()
                    if False else ax.transData)
            y_text += std * 0.25

        # RECOMMENDED badge
        if is_rec:
            ax.text(0.5, 0.96, "✓ RECOMMENDED",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=10, color=_ACCENT, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#111111",
                              edgecolor=_ACCENT, alpha=0.9),
                    zorder=6)

        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_title(
            f"{response.value.replace('_', ' ')}\n"
            f"conf={decision.confidence:.0%}" if is_rec else response.value.replace("_", " "),
            color=panel_col if is_rec else _FG_TEXT,
            fontsize=11, fontweight="bold" if is_rec else "normal",
            pad=10,
        )
        ax.set_ylabel("Race Time Delta vs Fastest Option (seconds)", fontsize=9)

    # Shared reference line
    for ax in axes:
        ax.axhline(0, color="#555555", linewidth=0.9, linestyle="--", alpha=0.5)

    nt_label = "Safety Car" if nt == "SC" else "Virtual Safety Car"
    t = title or (
        f"{nt_label} Strategy Decision — {decision.race_state.circuit.title()}\n"
        f"Lap {decision.race_state.current_lap}  |  "
        f"{decision.race_state.our_compound} age={decision.race_state.our_tyre_age}  |  "
        f"Gap={decision.race_state.gap_to_leader_sec:+.1f}s  |  "
        f"{'FREE STOP' if decision.is_free_stop else f'Net cost={decision.neutralisation.net_pit_cost_sec:.1f}s'}"
    )
    fig.suptitle(t, color=_FG_WHITE, fontsize=12, fontweight="bold", y=1.01)

    plt.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


# ===========================================================================
# 2. Gap compression timeline
# ===========================================================================

def plot_gap_compression_timeline(
    initial_gap_sec:      float,
    neutralisation_delta: NeutralisationTimeDelta,
    n_sc_laps:            int = 8,
    pit_lap:              Optional[int] = 2,
    title:                Optional[str] = None,
    save_path:            Optional[Path] = None,
    show:                 bool = False,
) -> plt.Figure:
    """
    On-track gap vs laps into neutralisation period.

    Shows:
        - SC gap compression curve (exponential decay)
        - VSC gap compression curve for comparison
        - "Pit NOW" vertical marker at pit_lap
        - The gap at which overtaking becomes impossible (circuit threshold)

    Engineering purpose:
        Gap compression is TIME-SENSITIVE. A strategist watching a 10-second
        gap with 2 laps of SC remaining has about 40 seconds to decide.
        This chart shows exactly how much gap remains at each potential
        pit lap — the decision has an expiry time.

    Args:
        initial_gap_sec:      Gap to car ahead/behind at SC deployment (seconds).
        neutralisation_delta: NeutralisationTimeDelta (SC or VSC).
        n_sc_laps:            Number of laps to model.
        pit_lap:              Optional lap to mark as "pit now" (relative to SC start).
        title:                Override title.
        save_path:            Save path.
        show:                 Interactive display.

    Returns:
        Matplotlib Figure.
    """
    laps = np.arange(0, n_sc_laps + 1, dtype=float)

    # Compute gap compression for each lap under this neutralisation
    sc_gaps = [
        apply_gap_compression(initial_gap_sec, neutralisation_delta, int(n))
        for n in laps
    ]

    # Also compute the "other" neutralisation for comparison
    from src.safety_car_engine.vsc_handler import compute_neutralisation_delta
    other_type  = "VSC" if neutralisation_delta.neutralisation_type == "SC" else "SC"
    other_delta = compute_neutralisation_delta(
        other_type,
        neutralisation_delta.base_racing_lap_sec,
        neutralisation_delta.circuit,
    )
    other_gaps = [
        apply_gap_compression(initial_gap_sec, other_delta, int(n))
        for n in laps
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    _style(ax, fig)

    nt     = neutralisation_delta.neutralisation_type
    col    = _SC_COLOUR if nt == "SC" else _VSC_COLOUR
    o_col  = _VSC_COLOUR if nt == "SC" else _SC_COLOUR

    ax.plot(laps, sc_gaps, color=col, linewidth=2.4, zorder=3,
            label=f"{nt} (this event)")
    ax.fill_between(laps, sc_gaps, alpha=0.15, color=col, zorder=2)

    ax.plot(laps, other_gaps, color=o_col, linewidth=1.6,
            linestyle="--", alpha=0.65, zorder=3,
            label=f"{other_type} (for comparison)")

    # Initial gap reference
    ax.axhline(initial_gap_sec, color="#555555", linewidth=0.9,
               linestyle=":", alpha=0.5)
    ax.text(n_sc_laps * 0.02, initial_gap_sec + 0.15,
            f"Initial gap: {initial_gap_sec:.1f}s",
            fontsize=8, color="#888888")

    # Pit lap marker
    if pit_lap is not None and 0 < pit_lap <= n_sc_laps:
        gap_at_pit = float(sc_gaps[int(pit_lap)])
        ax.axvline(pit_lap, color=_ACCENT, linewidth=1.8,
                   linestyle="-", alpha=0.9, zorder=4)
        ax.scatter([pit_lap], [gap_at_pit], color=_ACCENT,
                   s=60, zorder=5, edgecolors=_FG_WHITE, linewidths=0.8)
        ax.text(pit_lap + 0.15, gap_at_pit + 0.15,
                f"Pit L{int(pit_lap)}: {gap_at_pit:.1f}s",
                fontsize=8.5, color=_ACCENT)

    # Overtaking threshold line (approx 1s for DRS circuits)
    ax.axhline(1.0, color=_GOOD, linewidth=0.9, linestyle="--",
               alpha=0.6, label="~Overtaking threshold (1.0s)")

    ax.set_xlabel("Laps into Neutralisation Period", fontsize=11)
    ax.set_ylabel("On-track Gap (seconds)", fontsize=11)
    ax.set_xlim(0, n_sc_laps)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    _legend(ax, loc="upper right")

    t = title or (
        f"Gap Compression Timeline — {nt}  |  Initial gap: {initial_gap_sec:.1f}s"
    )
    ax.set_title(t, color=_FG_WHITE, fontsize=11, pad=12)

    plt.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


# ===========================================================================
# 3. SC probability heatmap (by race lap)
# ===========================================================================

def plot_sc_probability_by_lap(
    sc_profile:      CircuitSCProfile,
    highlight_laps:  Optional[list[int]] = None,
    title:           Optional[str] = None,
    save_path:       Optional[Path] = None,
    show:            bool = False,
) -> plt.Figure:
    """
    Per-lap SC deployment probability from the CircuitSCProfile KDE.

    x-axis: race lap number (1 to total_race_laps).
    y-axis: conditional probability of SC starting on that lap.

    The area under the curve = 1.0 (properly normalised conditional
    probability). Historical deployment laps are shown as rug marks
    along the x-axis.

    Engineering purpose:
        Pre-race risk briefing. The strategy team uses this to identify
        the "high risk window" — the lap range where SC probability is
        above the circuit average — and pre-plan the response strategy.

    Args:
        sc_profile:      CircuitSCProfile from sc_detector.py.
        highlight_laps:  Optional list of laps to highlight
                         (e.g. planned pit windows, competitor pit laps).
        title:           Override title.
        save_path:       Save path.
        show:            Interactive display.

    Returns:
        Matplotlib Figure.
    """
    total_laps = sc_profile.total_race_laps
    laps       = np.arange(1, total_laps + 1, dtype=float)

    sc_probs  = np.array([sc_profile.sc_probability_at_lap(int(l)) for l in laps])
    vsc_probs = np.zeros_like(sc_probs)  # simplified: focus on SC

    fig, ax = plt.subplots(figsize=(13, 5))
    _style(ax, fig)

    # Smooth SC probability fill
    ax.fill_between(laps, sc_probs, alpha=0.35, color=_SC_COLOUR, zorder=2)
    ax.plot(laps, sc_probs, color=_SC_COLOUR, linewidth=1.8, zorder=3,
            label=f"SC probability  (freq={sc_profile.sc_frequency:.2f}/race)")

    # Uniform baseline (1/total_laps)
    uniform = 1.0 / total_laps
    ax.axhline(uniform, color="#555555", linewidth=0.9, linestyle="--",
               alpha=0.6, label=f"Uniform baseline ({uniform:.4f}/lap)")

    # Rug marks for historical SC deployment laps
    if len(sc_profile.sc_deployment_laps) > 0:
        ax.plot(
            sc_profile.sc_deployment_laps,
            np.full_like(sc_profile.sc_deployment_laps, -uniform * 0.4),
            "|", color=_SC_COLOUR, markersize=8, alpha=0.6,
            label="Historical SC laps",
            clip_on=False,
        )

    # Highlight laps (e.g. planned pit windows)
    if highlight_laps:
        for hl in highlight_laps:
            ax.axvline(hl, color=_ACCENT2, linewidth=1.2,
                       linestyle=":", alpha=0.7)
        ax.plot([], [], color=_ACCENT2, linewidth=1.2,
                linestyle=":", label="Highlighted laps")

    # High-risk zone: above 1.5× baseline
    high_risk = sc_probs > 1.5 * uniform
    if high_risk.any():
        ax.fill_between(laps, sc_probs, uniform,
                        where=high_risk,
                        alpha=0.22, color=_ACCENT, zorder=2,
                        label="High-risk zone (>1.5× baseline)")

    ax.set_xlabel("Race Lap Number", fontsize=11)
    ax.set_ylabel("SC Deployment Probability (conditional)", fontsize=11)
    ax.set_xlim(1, total_laps)
    ax.set_ylim(bottom=-uniform * 0.5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    _legend(ax, loc="upper right")

    n_sess = sc_profile.n_sessions
    t = title or (
        f"SC Deployment Probability — {sc_profile.circuit.title()}\n"
        f"Based on {n_sess} historical sessions  |  "
        f"SC: {sc_profile.sc_frequency:.2f}/race  |  "
        f"Avg duration: {sc_profile.sc_duration_laps.mean():.1f} laps"
        if n_sess > 0 else
        f"SC Deployment Probability — {sc_profile.circuit.title()} (global prior)"
    )
    ax.set_title(t, color=_FG_WHITE, fontsize=11, pad=12)

    plt.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


# ===========================================================================
# 4. SC scenario expected value summary table
# ===========================================================================

def plot_sc_decision_summary(
    decision:    SCDecision,
    title:       Optional[str] = None,
    save_path:   Optional[Path] = None,
    show:        bool = False,
) -> plt.Figure:
    """
    Annotated table-style summary of the SC/VSC decision analysis.

    Each row is a response option. Columns:
        Option | Mean time | P10 | P50 | P90 | Pit cost | Verdict

    Colour coding:
        Recommended row: FOM red highlight.
        Best value per column: bold.
        Worst value per column: muted.

    Engineering purpose:
        The debrief and portfolio chart. Combines all numerical outputs
        of sc_scenario_analyzer into a single readable table that can
        be screenshotted and presented without context.

    Args:
        decision:   SCDecision from sc_scenario_analyzer.
        title:      Override title.
        save_path:  Save path.
        show:       Interactive display.

    Returns:
        Matplotlib Figure.
    """
    outcomes = {r: o for r, o in decision.option_outcomes.items()
                if o.mean_time_sec < float("inf")}

    if not outcomes:
        logger.warning("plot_sc_decision_summary: no valid outcomes.")
        return plt.figure()

    best_mean = min(o.mean_time_sec for o in outcomes.values())

    rows  = []
    for response, outcome in outcomes.items():
        is_rec = response == decision.recommended
        rows.append({
            "option":     response.value.replace("_", " "),
            "mean_delta": outcome.mean_time_sec - best_mean,
            "p10_delta":  outcome.p10_time_sec  - best_mean,
            "p50_delta":  outcome.p50_time_sec  - best_mean,
            "p90_delta":  outcome.p90_time_sec  - best_mean,
            "pit_cost":   outcome.net_pit_cost_sec,
            "verdict":    "✓ RECOMMENDED" if is_rec else "",
            "_is_rec":    is_rec,
            "_colour":    _RESPONSE_COLOURS.get(response, "#888888"),
        })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 1.5 + len(df) * 0.9))
    _style(ax, fig)
    ax.axis("off")

    col_labels = ["Option", "Mean +Δ", "P10 +Δ", "P50 +Δ", "P90 +Δ",
                  "Pit cost", "Verdict"]
    col_keys   = ["option", "mean_delta", "p10_delta", "p50_delta",
                  "p90_delta", "pit_cost", "verdict"]

    col_widths = [0.20, 0.11, 0.11, 0.11, 0.11, 0.13, 0.23]
    x_starts   = [sum(col_widths[:i]) for i in range(len(col_widths))]
    y_header   = 0.90
    row_h      = 0.75 / max(len(df), 1)

    # Header row
    for j, (label, xs) in enumerate(zip(col_labels, x_starts)):
        ax.text(xs + col_widths[j] / 2, y_header, label,
                ha="center", va="center", fontsize=9,
                fontweight="bold", color=_FG_WHITE,
                transform=ax.transAxes)

    # Separator
    ax.axhline(y_header - 0.05, color="#333333", linewidth=0.8,
               transform=ax.transAxes, xmin=0, xmax=1)

    for i, row in df.iterrows():
        y_row = y_header - 0.12 - i * row_h
        bg    = "#2A1010" if row["_is_rec"] else _BG_PANEL

        # Row background
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.01, y_row - row_h * 0.4), 0.98, row_h * 0.85,
            boxstyle="round,pad=0.01",
            facecolor=bg, edgecolor=row["_colour"],
            linewidth=1.2 if row["_is_rec"] else 0.3,
            transform=ax.transAxes, zorder=2,
        ))

        for j, key in enumerate(col_keys):
            val = row[key]
            if isinstance(val, float):
                text = f"+{val:.2f}s" if key != "pit_cost" else f"{val:.1f}s"
            else:
                text = str(val)

            col  = _ACCENT if row["_is_rec"] and key == "verdict" else row["_colour"]
            fw   = "bold" if row["_is_rec"] else "normal"
            ax.text(x_starts[j] + col_widths[j] / 2, y_row,
                    text, ha="center", va="center",
                    fontsize=8.5, color=col, fontweight=fw,
                    transform=ax.transAxes, zorder=3)

    nt = decision.race_state.neutralisation_type
    t = title or (
        f"{'Safety Car' if nt == 'SC' else 'VSC'} Decision Summary — "
        f"{decision.race_state.circuit.title()}  |  "
        f"Lap {decision.race_state.current_lap}  |  "
        f"{decision.race_state.our_compound} age={decision.race_state.our_tyre_age}  |  "
        f"Confidence={decision.confidence:.0%}"
    )
    ax.set_title(t, color=_FG_WHITE, fontsize=11, pad=8)

    reasoning_lines = decision.reasoning[:120] + "…" if len(decision.reasoning) > 120 \
                      else decision.reasoning
    fig.text(0.5, 0.01, reasoning_lines, ha="center", va="bottom",
             fontsize=7.5, color="#888888", style="italic",
             wrap=True)

    plt.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


# ===========================================================================
# 5. Full SC scenario dashboard (composite 2×2)
# ===========================================================================

def plot_sc_scenario_dashboard(
    decision:             SCDecision,
    sc_profile:           CircuitSCProfile,
    title:                Optional[str] = None,
    save_path:            Optional[Path] = None,
    show:                 bool = False,
) -> plt.Figure:
    """
    Composite 2×2 dashboard combining all four SC scenario charts.

    Layout:
        [Option comparison violin] [Gap compression timeline]
        [SC probability by lap   ] [Decision summary table  ]

    This is the portfolio showcase chart — all SC scenario information
    on one page, arranged for immediate readability.

    Args:
        decision:   SCDecision from sc_scenario_analyzer.
        sc_profile: CircuitSCProfile for the probability chart.
        title:      Override suptitle.
        save_path:  Save path.
        show:       Interactive display.

    Returns:
        Matplotlib Figure.
    """
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor(_BG_FIGURE)

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.42, wspace=0.32)

    # --- [0,0] Option comparison (reuse existing chart as inset) ---
    ax_opts = fig.add_subplot(gs[0, 0])
    _style(ax_opts, fig)
    outcomes = {r: o for r, o in decision.option_outcomes.items()
                if o.mean_time_sec < float("inf")}
    best_mean = min(o.mean_time_sec for o in outcomes.values()) if outcomes else 0

    for i, (response, outcome) in enumerate(outcomes.items()):
        col       = _RESPONSE_COLOURS.get(response, "#888888")
        is_rec    = response == decision.recommended
        x_pos     = float(i)
        mean_d    = outcome.mean_time_sec - best_mean
        p10_d     = outcome.p10_time_sec  - best_mean
        p90_d     = outcome.p90_time_sec  - best_mean
        std       = outcome.std_time_sec

        samples = np.random.default_rng(i + 42).normal(mean_d, std, 200)
        parts   = ax_opts.violinplot([samples], positions=[x_pos],
                                     showmedians=False, showextrema=False,
                                     widths=0.6)
        for body in parts["bodies"]:
            body.set_facecolor(col)
            body.set_alpha(0.45 if is_rec else 0.20)
            body.set_edgecolor(col)

        ax_opts.hlines(mean_d, x_pos - 0.25, x_pos + 0.25,
                       colors=col, linewidth=2.0, zorder=4)
        ax_opts.scatter([x_pos, x_pos], [p10_d, p90_d],
                        color=col, s=18, zorder=5)

        label_extra = "\n✓REC" if is_rec else ""
        ax_opts.text(x_pos, p90_d + std * 0.3,
                     f"{response.value.replace('_', ' ')}{label_extra}",
                     ha="center", va="bottom", fontsize=7.5,
                     color=col, fontweight="bold" if is_rec else "normal")

    ax_opts.axhline(0, color="#555555", linewidth=0.9, linestyle="--", alpha=0.5)
    ax_opts.set_xticks([])
    ax_opts.set_ylabel("Race Time Delta vs Best (s)", fontsize=9)
    ax_opts.set_title("Option Comparison", color=_FG_WHITE, fontsize=10)

    # --- [0,1] Gap compression ---
    ax_gap = fig.add_subplot(gs[0, 1])
    _style(ax_gap, fig)
    initial_gap = decision.race_state.gap_to_leader_sec
    nd          = decision.neutralisation
    laps_arr    = np.arange(0, 9, dtype=float)
    gaps_arr    = [apply_gap_compression(initial_gap, nd, int(n)) for n in laps_arr]
    nt_col      = _SC_COLOUR if nd.neutralisation_type == "SC" else _VSC_COLOUR

    ax_gap.plot(laps_arr, gaps_arr, color=nt_col, linewidth=2.0, zorder=3,
                label=nd.neutralisation_type)
    ax_gap.fill_between(laps_arr, gaps_arr, alpha=0.15, color=nt_col)
    ax_gap.axhline(1.0, color=_GOOD, linewidth=0.8, linestyle="--",
                   alpha=0.6, label="~1s overtaking threshold")
    ax_gap.set_xlabel("Laps into SC/VSC", fontsize=9)
    ax_gap.set_ylabel("Gap (seconds)", fontsize=9)
    ax_gap.set_title("Gap Compression", color=_FG_WHITE, fontsize=10)
    ax_gap.set_xlim(0, 8)
    ax_gap.set_ylim(bottom=0)
    _legend(ax_gap, fontsize=7)

    # --- [1,0] SC probability ---
    ax_prob = fig.add_subplot(gs[1, 0])
    _style(ax_prob, fig)
    total_laps = sc_profile.total_race_laps
    lap_ns     = np.arange(1, total_laps + 1, dtype=float)
    probs      = np.array([sc_profile.sc_probability_at_lap(int(l)) for l in lap_ns])
    uniform    = 1.0 / total_laps

    ax_prob.fill_between(lap_ns, probs, alpha=0.30, color=_SC_COLOUR)
    ax_prob.plot(lap_ns, probs, color=_SC_COLOUR, linewidth=1.5, zorder=3)
    ax_prob.axhline(uniform, color="#555555", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_prob.axvline(decision.race_state.current_lap,
                    color=_ACCENT, linewidth=1.4, linestyle="--", alpha=0.8,
                    label=f"Current lap {decision.race_state.current_lap}")
    ax_prob.set_xlabel("Race Lap", fontsize=9)
    ax_prob.set_ylabel("SC Prob. (conditional)", fontsize=9)
    ax_prob.set_title("Historical SC Probability by Lap", color=_FG_WHITE, fontsize=10)
    ax_prob.set_xlim(1, total_laps)
    _legend(ax_prob, fontsize=7)

    # --- [1,1] Decision summary text block ---
    ax_sum = fig.add_subplot(gs[1, 1])
    _style(ax_sum, fig)
    ax_sum.axis("off")

    summary_lines = [
        f"CIRCUIT:     {decision.race_state.circuit.upper()}",
        f"EVENT:       {nd.neutralisation_type} on lap {decision.race_state.current_lap}",
        f"COMPOUND:    {decision.race_state.our_compound}  age={decision.race_state.our_tyre_age}",
        f"GAP:         {initial_gap:+.1f}s  → compressed: {decision.gap_after_compression:.1f}s",
        f"NET PIT COST:{nd.net_pit_cost_sec:.1f}s",
        "",
        f"RECOMMENDED: {decision.recommended.value}",
        f"CONFIDENCE:  {decision.confidence:.0%}",
        f"FREE STOP:   {'YES ✓' if decision.is_free_stop else 'NO'}",
        "",
        "REASONING:",
    ]
    reasoning_wrapped = [decision.reasoning[i:i+52]
                         for i in range(0, min(len(decision.reasoning), 210), 52)]
    summary_lines.extend(reasoning_wrapped)

    for j, line in enumerate(summary_lines):
        is_rec_line  = "RECOMMENDED" in line
        is_header    = ":" in line and j < 9 and line.strip()
        col   = _ACCENT if is_rec_line else (_FG_WHITE if is_header else _FG_TEXT)
        fw    = "bold" if is_rec_line or is_header else "normal"
        ax_sum.text(0.04, 0.94 - j * 0.072, line,
                    transform=ax_sum.transAxes,
                    fontsize=8.5, color=col, fontweight=fw, va="top",
                    fontfamily="monospace")

    ax_sum.set_title("Decision Summary", color=_FG_WHITE, fontsize=10)

    nt_full = "Safety Car" if nd.neutralisation_type == "SC" else "Virtual Safety Car"
    t = title or (
        f"F1 {nt_full} Strategy Dashboard — "
        f"{decision.race_state.circuit.title()} | "
        f"Lap {decision.race_state.current_lap}"
    )
    fig.suptitle(t, color=_FG_WHITE, fontsize=13, fontweight="bold", y=1.01)

    _save(fig, save_path)
    if show:
        plt.show()
    return fig
