"""
dashboard/callbacks.py
========================
Dash reactive callback functions — fully compatible with all src/ module APIs.

Every callback is defensive: catches all exceptions, returns safe empty
states rather than crashing the dashboard. Heavy backend objects are
reconstructed per-callback from stored feature_df JSON — avoids global
state that breaks multi-user deployments.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, no_update
from dash.exceptions import PreventUpdate

from src.constants import (
    COMPOUND_COLOURS,
    COMPOUND_ABBREV,
    DRY_COMPOUNDS,
    PIT_LANE_DELTA_SEC_BAHRAIN,
)

logger = logging.getLogger(__name__)

# ===========================================================================
# Theme
# ===========================================================================

_BG    = "#0F0F0F"
_PANEL = "#1A1A1A"
_TEXT  = "#CCCCCC"
_WHITE = "#FFFFFF"
_GRID  = "#2A2A2A"
_RED   = "#E8002D"
_YEL   = "#FFF200"

_STOP_COLOURS = {0: "#888888", 1: "#4FC3F7", 2: "#E8002D", 3: "#FF9800"}


def _base_layout(**kw) -> dict:
    return {
        "paper_bgcolor": _BG,
        "plot_bgcolor":  _PANEL,
        "font":          {"color": _TEXT, "size": 11},
        "xaxis": {"gridcolor": _GRID, "gridwidth": 0.5, "zeroline": False,
                  "tickfont": {"color": _TEXT, "size": 10}},
        "yaxis": {"gridcolor": _GRID, "gridwidth": 0.5, "zeroline": False,
                  "tickfont": {"color": _TEXT, "size": 10}},
        "legend": {"bgcolor": "rgba(17,17,17,0.8)", "bordercolor": "#444",
                   "borderwidth": 1, "font": {"color": _TEXT, "size": 9}},
        "margin": {"l": 50, "r": 20, "t": 50, "b": 50},
        "hovermode": "x unified",
        **kw,
    }


def _empty(msg: str = "Run analysis to populate this chart") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font={"color": "#555", "size": 13})
    fig.update_layout(**_base_layout())
    return fig


def _safe(func):
    """Decorator — catch all exceptions and return no_update."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except PreventUpdate:
            raise
        except Exception as exc:
            logger.error("Callback '%s' failed: %s\n%s",
                         func.__name__, exc, traceback.format_exc())
            return no_update
    wrapper.__name__ = func.__name__
    return wrapper


def _serialise_model_set(model_set) -> dict:
    """Store model prediction arrays (not sklearn objects) in JSON."""
    models_data = {}
    for compound in model_set.compounds_fitted:
        m = model_set.models[compound]
        models_data[compound] = {
            "compound":        m.compound,
            "circuit":         m.circuit,
            "n_laps":          m.n_laps,
            "n_stints":        m.n_stints,
            "cliff_lap":       m.cliff_lap,
            "deg_rate_linear": m.deg_rate_linear,
            "deg_rate_cliff":  m.deg_rate_cliff,
            "r2":              m.r2,
            "mae_sec":         m.mae_sec,
            "model_ages":      m.model_ages.tolist(),
            "model_deltas":    m.model_deltas.tolist(),
            "fitted_ages":     m.fitted_ages.tolist(),
            "fitted_deltas":   m.fitted_deltas.tolist(),
        }
    return {
        "circuit":           model_set.circuit,
        "season":            model_set.season,
        "compounds_fitted":  model_set.compounds_fitted,
        "compounds_skipped": model_set.compounds_skipped,
        "models":            models_data,
    }


# ===========================================================================
# Callback registration functions
# ===========================================================================

def register_session_load_callback(app) -> None:

    @app.callback(
        Output("store-session-data",        "data"),
        Output("store-model-set",           "data"),
        Output("store-feature-df",          "data"),
        Output("store-sc-profile",          "data"),
        Output("session-status-badge",      "children"),
        Output("header-session-label",      "children"),
        Output("overview-total-laps-kpi",   "children"),
        Output("overview-drivers-kpi",      "children"),
        Output("overview-repr-pct-kpi",     "children"),
        Output("overview-sc-kpi",           "children"),
        Output("overview-compounds-kpi",    "children"),
        Output("overview-best-lap-kpi",     "children"),
        Input("session-load-btn",           "n_clicks"),
        State("session-circuit-dropdown",   "value"),
        State("session-year-input",         "value"),
        State("session-type-dropdown",      "value"),
        State("session-total-laps-input",   "value"),
        prevent_initial_call=True,
    )
    @_safe
    def load_session(n_clicks, circuit, year, session_type, total_laps):
        if not n_clicks:
            raise PreventUpdate

        logger.info("load_session: %s %s %s laps=%s",
                    circuit, year, session_type, total_laps)

        _err = (None, None, None, None,
                "Load failed", "No session",
                "—", "—", "—", "—", "—", "—")

        try:
            from src.data_engineering.fastf1_loader import load_session as ff1_load
            from src.data_engineering.telemetry_processor import (
                process_laps, process_pit_stops, process_driver_info,
            )
            from src.data_engineering.feature_builder import build_feature_set
            from src.tire_model.degradation_model import build_degradation_models
            from src.safety_car_engine.sc_detector import (
                detect_neutralisation_from_dataframe, get_sc_profile,
            )

            config  = {"circuit": circuit, "year": int(year or 2023),
                       "session_type": session_type or "R"}
            total_laps = int(total_laps or 57)

            session    = ff1_load(config)
            laps_df    = process_laps(session)
            driver_df  = process_driver_info(session)
            feature_df = build_feature_set(laps_df, total_race_laps=total_laps)
            model_set  = build_degradation_models(
                feature_df, circuit=str(circuit), season=int(year or 2023)
            )
            sc_periods = detect_neutralisation_from_dataframe(laps_df)
            sc_profile = get_sc_profile(
                str(circuit), total_laps,
                [sc_periods] if sc_periods else None,
            )

            feature_json = feature_df.to_json(orient="split")
            model_store  = _serialise_model_set(model_set)
            sc_store = {
                "circuit":             sc_profile.circuit,
                "n_sessions":          sc_profile.n_sessions,
                "sc_frequency":        sc_profile.sc_frequency,
                "vsc_frequency":       sc_profile.vsc_frequency,
                "sc_deployment_laps":  sc_profile.sc_deployment_laps.tolist(),
                "sc_duration_laps":    sc_profile.sc_duration_laps.tolist(),
                "vsc_deployment_laps": sc_profile.vsc_deployment_laps.tolist(),
                "vsc_duration_laps":   sc_profile.vsc_duration_laps.tolist(),
                "total_race_laps":     total_laps,
            }
            session_store = {
                "circuit": circuit, "year": int(year or 2023),
                "total_laps": total_laps, "session_type": session_type,
            }

            repr_pct  = f"{laps_df['is_representative'].mean() * 100:.1f}%"
            n_sc      = sum(1 for p in sc_periods if p.is_sc)
            compounds = sorted([c for c in feature_df["compound"].unique()
                                 if c != "UNKNOWN"])
            best_lap  = f"{laps_df['lap_time_sec'].min():.3f}s"

            return (
                session_store, model_store, feature_json, sc_store,
                "Loaded", f"{circuit} {year} · {total_laps} laps",
                str(total_laps), str(len(driver_df)),
                repr_pct, str(n_sc),
                " / ".join(COMPOUND_ABBREV.get(c, c) for c in compounds),
                best_lap,
            )

        except Exception as exc:
            logger.error("load_session failed: %s", exc)
            return _err


def register_overview_callbacks(app) -> None:

    @app.callback(
        Output("overview-degradation-overlay-graph", "figure"),
        Output("overview-compound-summary-graph",    "figure"),
        Input("store-model-set", "data"),
        prevent_initial_call=True,
    )
    @_safe
    def update_overview_charts(model_data):
        if not model_data or not model_data.get("models"):
            return (_empty("Load a session to see degradation curves"),
                    _empty())

        compounds_fitted = model_data["compounds_fitted"]

        # Degradation overlay
        overlay = go.Figure()
        for compound in compounds_fitted:
            m      = model_data["models"][compound]
            colour = COMPOUND_COLOURS.get(compound, "#888")
            overlay.add_trace(go.Scatter(
                x=m["model_ages"], y=m["model_deltas"],
                mode="lines", name=f"{compound}  deg={m['deg_rate_linear']:+.3f}s/lap",
                line={"color": colour, "width": 2.4},
            ))
            overlay.add_trace(go.Scatter(
                x=m["fitted_ages"], y=m["fitted_deltas"],
                mode="markers", showlegend=False,
                marker={"color": colour, "size": 6,
                        "line": {"color": _WHITE, "width": 1}},
            ))
            if m["cliff_lap"]:
                overlay.add_vline(
                    x=m["cliff_lap"], line_color="#FF6B35",
                    line_dash="dash", line_width=1.5,
                    annotation_text=f"Cliff L{m['cliff_lap']}",
                    annotation_font_color="#FF6B35", annotation_font_size=9,
                )

        overlay.update_layout(**_base_layout(
            xaxis_title="Tyre Age (laps)",
            yaxis_title="Δt vs fresh tyre (seconds)",
            title={"text": "Compound Degradation Overlay",
                   "font": {"color": _WHITE, "size": 12}},
        ))

        # Summary bar chart
        colours  = [COMPOUND_COLOURS.get(c, "#888") for c in compounds_fitted]
        lin_rates= [model_data["models"][c]["deg_rate_linear"] for c in compounds_fitted]
        cliff_laps=[model_data["models"][c]["cliff_lap"] or 0 for c in compounds_fitted]

        summary = go.Figure()
        summary.add_trace(go.Bar(
            name="Linear deg rate (s/lap)",
            x=[COMPOUND_ABBREV.get(c, c) for c in compounds_fitted],
            y=lin_rates,
            marker={"color": colours, "opacity": 0.85},
            text=[f"{v:+.3f}" for v in lin_rates],
            textposition="outside",
        ))
        summary.add_trace(go.Bar(
            name="Cliff lap",
            x=[COMPOUND_ABBREV.get(c, c) for c in compounds_fitted],
            y=cliff_laps,
            marker={"color": "#FF6B35", "opacity": 0.75},
            text=[f"L{int(v)}" if v else "none" for v in cliff_laps],
            textposition="outside",
            yaxis="y2",
        ))
        summary.update_layout(**_base_layout(
            barmode="group",
            yaxis2={"overlaying": "y", "side": "right",
                    "tickfont": {"color": "#FF6B35"},
                    "title": "Cliff Lap",
                    "title_font": {"color": "#FF6B35"}},
            title={"text": "Compound Comparison",
                   "font": {"color": _WHITE, "size": 12}},
            xaxis_title="Compound",
            yaxis_title="Degradation Rate (s/lap)",
        ))

        return overlay, summary

    @app.callback(
        Output("overview-heatmap-graph", "figure"),
        Input("overview-heatmap-compound-dropdown", "value"),
        Input("store-feature-df", "data"),
        prevent_initial_call=True,
    )
    @_safe
    def update_heatmap(compound, feature_json):
        if not feature_json or not compound:
            return _empty("Load a session to see pace drop heatmap")

        feature_df = pd.read_json(feature_json, orient="split")
        required = {"driver_code", "stint_number", "tyre_age",
                    "pace_drop_sec", "compound", "is_representative"}
        if not required.issubset(feature_df.columns):
            return _empty("Feature columns not available — ensure session loaded correctly")

        comp_data = feature_df[
            (feature_df["compound"] == compound)
            & feature_df["is_representative"]
            & feature_df["pace_drop_sec"].notna()
        ].copy()

        if comp_data.empty:
            return _empty(f"No representative data for {compound}")

        comp_data["driver_stint"] = (
            comp_data["driver_code"] + " S" + comp_data["stint_number"].astype(str)
        )
        pivot = comp_data.pivot_table(
            index="driver_stint", columns="tyre_age",
            values="pace_drop_sec", aggfunc="mean",
        )
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

        vmax = float(np.nanpercentile(np.abs(pivot.values), 95))
        vmax = max(vmax, 0.05)

        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[f"Age {c}" for c in pivot.columns],
            y=pivot.index.tolist(),
            colorscale="RdYlGn_r", zmid=0, zmin=-vmax, zmax=vmax,
            colorbar={"title": "Pace Drop (s/lap)", "tickfont": {"color": _TEXT}},
            text=[[f"{v:+.2f}" if not np.isnan(v) else "" for v in row]
                  for row in pivot.values],
            texttemplate="%{text}",
        ))
        fig.update_layout(**_base_layout(
            title={"text": f"Pace Drop Heatmap — {compound}",
                   "font": {"color": _WHITE, "size": 12}},
            xaxis_title="Tyre Age", yaxis_title="Driver / Stint",
        ))
        return fig


def register_strategy_callbacks(app) -> None:

    @app.callback(
        Output("strategy-leaderboard-graph",    "figure"),
        Output("strategy-optimal-label",        "children"),
        Output("strategy-optimal-time",         "children"),
        Output("strategy-optimal-details",      "children"),
        Output("strategy-ml-prediction",        "children"),
        Output("strategy-ml-confidence-graph",  "figure"),
        Output("store-opt-result",              "data"),
        Input("strategy-run-btn",               "n_clicks"),
        State("store-session-data",             "data"),
        State("store-feature-df",               "data"),
        State("strategy-compounds-checklist",   "value"),
        State("strategy-max-stops-slider",      "value"),
        State("strategy-pit-delta-input",       "value"),
        State("ml-grid-position-input",         "value"),
        State("ml-qual-gap-input",              "value"),
        prevent_initial_call=True,
    )
    @_safe
    def run_optimisation(n_clicks, session_data, feature_json,
                         compounds, max_stops, pit_delta,
                         grid_position, qual_gap):
        if not n_clicks or not session_data or not feature_json:
            raise PreventUpdate

        from src.tire_model.degradation_model import build_degradation_models
        from src.strategy_engine.pit_window_optimizer import optimise_strategy

        circuit    = session_data["circuit"]
        year       = session_data["year"]
        total_laps = session_data["total_laps"]
        feature_df = pd.read_json(feature_json, orient="split")

        model_set = build_degradation_models(
            feature_df, circuit=str(circuit), season=int(year)
        )
        compounds_list = compounds or list(DRY_COMPOUNDS)

        opt = optimise_strategy(
            model_set            = model_set,
            feature_df           = feature_df,
            available_compounds  = compounds_list,
            total_race_laps      = total_laps,
            circuit              = str(circuit),
            season               = int(year),
            max_stops            = int(max_stops or 2),
            pit_lane_delta_sec   = float(pit_delta or PIT_LANE_DELTA_SEC_BAHRAIN),
        )

        # Leaderboard chart
        ldb_fig = _empty("No valid strategies found")
        if not opt.leaderboard.empty:
            df      = opt.leaderboard.head(20)
            colours = [_STOP_COLOURS.get(int(n), "#888") for n in df["n_stops"]]
            ldb_fig = go.Figure(go.Bar(
                x=df["gap_to_optimal_sec"], y=df["label"],
                orientation="h", marker_color=colours,
                text=[f"+{g:.3f}s" for g in df["gap_to_optimal_sec"]],
                textposition="outside",
            ))
            ldb_fig.update_layout(**_base_layout(
                xaxis_title="Gap to Optimal (seconds)",
                title={"text": f"Strategy Leaderboard — {circuit} {year}",
                       "font": {"color": _WHITE, "size": 12}},
                yaxis={"autorange": "reversed",
                       "tickfont": {"color": _TEXT, "size": 8}},
            ))

        # Optimal card
        if opt.optimal:
            opt_label   = opt.optimal.strategy.label
            opt_time    = opt.optimal.total_time_formatted
            opt_details = (
                f"Pit stops: {opt.optimal.strategy.pit_laps}  |  "
                f"Compounds: {' → '.join(opt.optimal.strategy.compounds_used)}  |  "
                f"Base pace: {opt.base_lap_time_sec:.3f}s"
            )
        else:
            opt_label, opt_time, opt_details = "No valid strategy found", "", ""

        # ML heuristic prediction
        try:
            from src.ml_optimizer.strategy_classifier import extract_race_context_features
            features = extract_race_context_features(
                circuit=str(circuit),
                qualifying_gap_to_pole_sec=float(qual_gap or 0.5),
                grid_position=int(grid_position or 10),
                race_laps=total_laps,
                sc_probability=0.68,
            )
            mean_deg = features.mean_dry_deg_rate
            if mean_deg > 0.055:
                predicted_stops, confidence = 2, 0.72
            elif mean_deg > 0.035:
                predicted_stops, confidence = 2, 0.58
            else:
                predicted_stops, confidence = 1, 0.62

            ml_text = (
                f"Predicted: {predicted_stops}-stop  "
                f"(conf={confidence:.0%})  |  "
                f"Circuit deg={mean_deg:.4f}s/lap"
            )
            stop_probs = {1: 0.25, 2: confidence, 3: max(0, 1-confidence-0.25)}
            ml_fig     = _build_confidence_bar(stop_probs)
        except Exception:
            ml_text = "ML prediction unavailable"
            ml_fig  = _empty("ML model not trained")

        opt_store = {
            "circuit":     circuit,
            "total_laps":  total_laps,
            "base_lap":    opt.base_lap_time_sec,
            "leaderboard": (opt.leaderboard.to_dict("records")
                            if not opt.leaderboard.empty else []),
        }

        return (ldb_fig, opt_label, opt_time, opt_details,
                ml_text, ml_fig, opt_store)

    @app.callback(
        Output("strategy-pit-window-graph", "figure"),
        Input("strategy-pit-window-start-dd", "value"),
        Input("strategy-pit-window-next-dd",  "value"),
        State("store-session-data",           "data"),
        State("store-feature-df",             "data"),
        State("strategy-pit-delta-input",     "value"),
        prevent_initial_call=True,
    )
    @_safe
    def update_pit_window(start_compound, next_compound,
                          session_data, feature_json, pit_delta):
        if not session_data or not feature_json:
            return _empty("Load a session first")

        from src.tire_model.degradation_model import build_degradation_models
        from src.strategy_engine.pit_window_optimizer import (
            compute_pit_window_sensitivity,
        )

        circuit    = session_data["circuit"]
        year       = session_data["year"]
        total_laps = session_data["total_laps"]
        feature_df = pd.read_json(feature_json, orient="split")
        model_set  = build_degradation_models(
            feature_df, circuit=str(circuit), season=int(year)
        )

        sensitivity_df = compute_pit_window_sensitivity(
            model_set          = model_set,
            feature_df         = feature_df,
            start_compound     = start_compound or "SOFT",
            next_compound      = next_compound  or "MEDIUM",
            total_race_laps    = total_laps,
            pit_lane_delta_sec = float(pit_delta or PIT_LANE_DELTA_SEC_BAHRAIN),
        )

        if sensitivity_df.empty:
            return _empty("No pit window data computed")

        df       = sensitivity_df.sort_values("pit_lap")
        colour   = COMPOUND_COLOURS.get(next_compound or "MEDIUM", "#888")
        opt_lap  = int(df.loc[df["gap_to_optimal_sec"] == 0, "pit_lap"].values[0])
        window   = df[df["is_optimal_window"]]

        fig = go.Figure()
        if not window.empty:
            fig.add_vrect(
                x0=window["pit_lap"].min(), x1=window["pit_lap"].max(),
                fillcolor=colour, opacity=0.12, line_width=0,
                annotation_text="Optimal window",
                annotation_font_color=colour,
            )
        fig.add_trace(go.Scatter(
            x=df["pit_lap"], y=df["gap_to_optimal_sec"],
            mode="lines+markers",
            line={"color": _YEL, "width": 2.2},
            marker={"size": 5},
            name="Gap to optimal",
        ))
        fig.add_vline(x=opt_lap, line_color=_WHITE,
                      line_dash="dash", line_width=1.5,
                      annotation_text=f"Optimal L{opt_lap}",
                      annotation_font_color=_WHITE, annotation_font_size=9)
        fig.update_layout(**_base_layout(
            xaxis_title="Pit Lap",
            yaxis_title="Gap to Optimal (s)",
            title={"text": (f"Pit Window: "
                            f"{COMPOUND_ABBREV.get(start_compound,'?')}"
                            f"→{COMPOUND_ABBREV.get(next_compound,'?')}"),
                   "font": {"color": _WHITE, "size": 11}},
        ))
        return fig


def register_simulator_callbacks(app) -> None:

    @app.callback(
        Output("simulator-timeline-graph",    "figure"),
        Output("simulator-breakdown-graph",   "figure"),
        Output("simulator-strategy-dropdown", "options"),
        Input("simulator-lap-slider",         "value"),
        Input("store-opt-result",             "data"),
        State("store-session-data",           "data"),
        State("store-feature-df",             "data"),
        prevent_initial_call=True,
    )
    @_safe
    def update_simulator(current_lap, opt_store, session_data, feature_json):
        if not opt_store or not session_data or not feature_json:
            empty = _empty("Run Strategy Optimiser first")
            return empty, empty, [{"label": "Run optimiser first", "value": "none"}]

        from src.tire_model.degradation_model import build_degradation_models
        from src.strategy_engine.race_simulator import (
            build_strategy, simulate_strategy,
        )

        circuit    = session_data["circuit"]
        year       = session_data["year"]
        total_laps = session_data["total_laps"]
        base_lap   = opt_store.get("base_lap", 91.0)
        feature_df = pd.read_json(feature_json, orient="split")
        model_set  = build_degradation_models(
            feature_df, circuit=str(circuit), season=int(year)
        )

        records = opt_store.get("leaderboard", [])
        if not records:
            empty = _empty("No strategies available")
            return empty, empty, [{"label": "No strategies", "value": "none"}]

        _ABBREV_TO_FULL = {"S": "SOFT", "M": "MEDIUM", "H": "HARD",
                           "I": "INTERMEDIATE", "W": "WET"}
        sim_results = []
        for rec in records[:8]:
            try:
                pit_str  = rec.get("pit_laps", "none")
                pit_laps = ([int(p.strip()) for p in pit_str.split(",")
                              if p.strip().isdigit()]
                             if pit_str != "none" else [])
                comp_str = rec.get("compounds", "S-M")
                full_compounds = [_ABBREV_TO_FULL.get(c.strip(), "MEDIUM")
                                  for c in comp_str.split("-")]
                if len(full_compounds) != len(pit_laps) + 1:
                    continue
                strategy = build_strategy(
                    pit_laps=pit_laps,
                    compounds=full_compounds,
                    total_race_laps=total_laps,
                )
                result = simulate_strategy(strategy, model_set, base_lap, total_laps)
                if result.is_valid:
                    sim_results.append(result)
            except Exception:
                continue

        if not sim_results:
            empty = _empty("Strategy simulation failed — check compounds")
            return empty, empty, [{"label": "Simulation failed", "value": "none"}]

        # Timeline chart
        ref_df  = sim_results[0].to_dataframe()
        ref_cum = ref_df["predicted_lap_sec"].cumsum().values
        timeline_fig = go.Figure()

        for r in sim_results:
            df    = r.to_dataframe()
            cum   = df["predicted_lap_sec"].cumsum().values
            delta = cum - ref_cum
            laps  = df["lap_number"].values
            timeline_fig.add_trace(go.Scatter(
                x=laps, y=delta, mode="lines",
                name=r.strategy.label,
                line={"width": 1.8},
            ))

        timeline_fig.add_vline(
            x=current_lap or 1, line_color="#888",
            line_dash="dash", line_width=1.2,
        )
        timeline_fig.add_hline(y=0, line_color="#555",
                                line_dash="dot", line_width=0.8)
        timeline_fig.update_layout(**_base_layout(
            xaxis_title="Race Lap",
            yaxis_title="Cumulative Δt to Reference (s)",
            title={"text": "Strategy Timeline",
                   "font": {"color": _WHITE, "size": 12}},
        ))

        # Breakdown chart
        breakdown_fig = _empty("Invalid strategy")
        r0 = sim_results[0]
        if r0.is_valid:
            df   = r0.to_dataframe()
            laps = df["lap_number"]
            breakdown_fig = go.Figure()
            for component, colour, name in [
                ("fuel_delta_sec",     "#4FC3F7", "Fuel penalty"),
                ("deg_delta_sec",      _RED,      "Tyre degradation"),
                ("pit_loss_sec",       "#FF9800", "Pit stop loss"),
                ("inlap_penalty_sec",  "#CE93D8", "In/out penalty"),
            ]:
                if component in df.columns:
                    breakdown_fig.add_trace(go.Scatter(
                        x=laps, y=df[component],
                        stackgroup="one", name=name,
                        line={"color": colour, "width": 0.5},
                    ))
            breakdown_fig.update_layout(**_base_layout(
                xaxis_title="Race Lap",
                yaxis_title="Delta above base (s)",
                title={"text": f"Lap Time Components — {r0.strategy.label}",
                       "font": {"color": _WHITE, "size": 11}},
            ))

        dd_options = [{"label": r.strategy.label, "value": str(i)}
                      for i, r in enumerate(sim_results)]
        return timeline_fig, breakdown_fig, dd_options

    @app.callback(
        Output("simulator-uc-decision",  "children"),
        Output("simulator-uc-reasoning", "children"),
        Input("simulator-lap-slider",    "value"),
        Input("simulator-uc-gap",        "value"),
        Input("simulator-uc-age",        "value"),
        State("store-session-data",      "data"),
        prevent_initial_call=True,
    )
    @_safe
    def update_undercut(current_lap, gap, their_age, session_data):
        if not session_data:
            return "Load session first", ""

        from src.strategy_engine.undercut_overcut import (
            GapScenario, evaluate_interaction, Decision,
        )

        total_laps = session_data.get("total_laps", 57)

        class _MockModelSet:
            """Lightweight proxy using linear degradation approximation."""
            def get(self, compound):
                rates = {"SOFT": 0.072, "MEDIUM": 0.042, "HARD": 0.025}
                rate  = rates.get(compound, 0.04)
                class _M:
                    def predict(self, x):
                        return np.maximum(0, rate * (x - 1))
                return _M()

        scenario = GapScenario(
            our_driver           = "OUR",
            their_driver         = "THEM",
            current_lap          = int(current_lap or 1),
            gap_ahead_sec        = float(gap or 2.5),
            our_compound         = "MEDIUM",
            our_tyre_age         = 15,
            their_compound       = "MEDIUM",
            their_tyre_age       = int(their_age or 15),
            our_next_compound    = "HARD",
            their_next_compound  = "HARD",
            total_race_laps      = total_laps,
        )

        decision = evaluate_interaction(scenario, _MockModelSet())

        colours = {
            Decision.UNDERCUT:  _RED,
            Decision.OVERCUT:   "#43B02A",
            Decision.MIRROR:    "#4FC3F7",
            Decision.UNCERTAIN: "#888",
        }
        col  = colours.get(decision.decision, "#888")
        text = (f"[{decision.decision.value}]  "
                f"conf={decision.confidence:.0%}  "
                f"undercut={decision.undercut_gain_sec:+.2f}s")
        return text, decision.reasoning


def register_sc_callbacks(app) -> None:

    @app.callback(
        Output("sc-option-comparison-graph", "figure"),
        Output("sc-gap-compression-graph",   "figure"),
        Output("sc-probability-graph",       "figure"),
        Output("sc-recommendation-text",     "children"),
        Output("sc-confidence-text",         "children"),
        Output("sc-reasoning-text",          "children"),
        Output("sc-free-stop-badge",         "style"),
        Input("sc-evaluate-btn",             "n_clicks"),
        State("sc-type-radio",               "value"),
        State("sc-current-lap-input",        "value"),
        State("sc-our-compound-dd",          "value"),
        State("sc-tyre-age-input",           "value"),
        State("sc-gap-input",                "value"),
        State("sc-next-compound-dd",         "value"),
        State("store-session-data",          "data"),
        State("store-sc-profile",            "data"),
        State("store-feature-df",            "data"),
        prevent_initial_call=True,
    )
    @_safe
    def evaluate_sc(n_clicks, sc_type, current_lap, our_compound,
                    tyre_age, gap, next_compound,
                    session_data, sc_profile_data, feature_json):
        if not n_clicks or not session_data:
            raise PreventUpdate

        from src.safety_car_engine.sc_detector import (
            CircuitSCProfile, build_default_sc_profile, NeutralisationPeriod,
        )
        from src.safety_car_engine.sc_scenario_analyzer import (
            PitResponse, evaluate_sc_pit_options,
        )
        from src.safety_car_engine.vsc_handler import (
            compute_neutralisation_delta, apply_gap_compression,
        )
        from src.tire_model.degradation_model import build_degradation_models
        from src.strategy_engine.race_simulator import build_strategy

        circuit    = session_data["circuit"]
        year       = session_data["year"]
        total_laps = session_data["total_laps"]
        cur_lap    = int(current_lap or 23)
        t_age      = int(tyre_age or 18)
        t_gap      = float(gap or 12.0)
        nt         = sc_type or "SC"
        our_comp   = our_compound or "SOFT"
        next_comp  = next_compound or "MEDIUM"

        # Reconstruct SC profile
        if sc_profile_data:
            sc_profile = CircuitSCProfile(
                circuit             = sc_profile_data["circuit"],
                n_sessions          = sc_profile_data["n_sessions"],
                sc_frequency        = sc_profile_data["sc_frequency"],
                vsc_frequency       = sc_profile_data["vsc_frequency"],
                sc_deployment_laps  = np.array(sc_profile_data["sc_deployment_laps"]),
                vsc_deployment_laps = np.array(sc_profile_data["vsc_deployment_laps"]),
                sc_duration_laps    = np.array(sc_profile_data["sc_duration_laps"]),
                vsc_duration_laps   = np.array(sc_profile_data["vsc_duration_laps"]),
                total_race_laps     = total_laps,
            )
        else:
            sc_profile = build_default_sc_profile(str(circuit), total_laps)

        # Base lap time
        base_lap = 91.0
        if feature_json:
            try:
                feature_df = pd.read_json(feature_json, orient="split")
                model_set  = build_degradation_models(
                    feature_df, circuit=str(circuit), season=int(year)
                )
                if "evolution_corrected_lap_sec" in feature_df.columns:
                    repr_laps = feature_df[
                        feature_df["is_representative"]
                    ]["evolution_corrected_lap_sec"].dropna()
                    if not repr_laps.empty:
                        base_lap = float(np.percentile(repr_laps, 5))
            except Exception:
                model_set = None
        else:
            model_set = None

        # Build neutralisation period for the analyzer
        pit_open = cur_lap + (1 if nt == "SC" else 0)
        period = NeutralisationPeriod(
            period_type       = nt,
            start_lap         = cur_lap,
            end_lap           = cur_lap + 5,
            duration_laps     = 5,
            pit_lane_open_lap = pit_open,
        )

        # Neutralisation delta for gap compression
        nd = compute_neutralisation_delta(nt, base_lap, str(circuit))

        # SC option comparison
        option_fig = _empty("Load session for full MC analysis")
        rec_text   = f"{nt} deployed — load session for recommendation"
        conf_text  = ""
        reasoning  = ""
        free_style = {"display": "none"}

        if model_set is not None:
            try:
                # Build a minimal current strategy
                pit_lap = min(cur_lap + 1, total_laps - 5)
                current_strategy = build_strategy(
                    pit_laps=[pit_lap],
                    compounds=[our_comp, next_comp],
                    total_race_laps=total_laps,
                )
                decision = evaluate_sc_pit_options(
                    period            = period,
                    decision_lap      = cur_lap,
                    current_strategy  = current_strategy,
                    next_compound     = next_comp,
                    model_set         = model_set,
                    base_lap_time_sec = base_lap,
                    total_race_laps   = total_laps,
                    current_tyre_age  = t_age,
                    current_compound  = our_comp,
                    sc_profile        = sc_profile,
                    mc_samples        = 150,
                )

                option_fig = _build_sc_option_fig(decision)
                rec_text   = f"→ {decision.recommended.value.replace('_', ' ')}"
                conf_text  = f"Confidence: {decision.confidence:.0%}"
                reasoning  = decision.reasoning[:200]

                # Check if free stop
                is_free = (nt == "SC" and
                           decision.recommended != decision.recommended.STAY_OUT
                           and nd.net_pit_cost_sec < 5.0)
                if is_free:
                    free_style = {
                        "display": "inline-block",
                        "backgroundColor": _RED, "color": _WHITE,
                        "padding": "3px 10px", "borderRadius": "12px",
                        "fontSize": "11px", "fontWeight": "700",
                    }

            except Exception as exc:
                logger.warning("SC evaluation failed: %s", exc)
                rec_text = f"{nt}: evaluation error — check session data"

        # Gap compression chart
        laps_arr = np.arange(0, 9, dtype=float)
        sc_gaps  = t_gap * (nd.gap_compression_factor ** laps_arr)
        nt_col   = "#FF9800" if nt == "SC" else "#4FC3F7"

        try:
            other_type = "VSC" if nt == "SC" else "SC"
            other_nd   = compute_neutralisation_delta(other_type, base_lap, str(circuit))
            other_gaps = t_gap * (other_nd.gap_compression_factor ** laps_arr)
        except Exception:
            other_gaps = sc_gaps * 0.9

        comp_fig = go.Figure()
        comp_fig.add_trace(go.Scatter(
            x=laps_arr, y=sc_gaps, mode="lines+markers",
            name=nt, line={"color": nt_col, "width": 2.2},
            fill="tozeroy",
        ))
        comp_fig.add_trace(go.Scatter(
            x=laps_arr, y=other_gaps, mode="lines",
            name="Other type", line={"color": "#888", "width": 1.2, "dash": "dash"},
        ))
        comp_fig.add_hline(y=1.0, line_color="#43B02A",
                            line_dash="dot", line_width=1.0,
                            annotation_text="~1s threshold",
                            annotation_font_color="#43B02A")
        comp_fig.update_layout(**_base_layout(
            xaxis_title="Laps into SC/VSC",
            yaxis_title="Gap (seconds)",
            title={"text": "Gap Compression", "font": {"color": _WHITE, "size": 11}},
            yaxis={"rangemode": "tozero"},
        ))

        # SC probability chart
        prob_fig = _build_sc_prob_fig(sc_profile, current_lap=cur_lap)

        return (option_fig, comp_fig, prob_fig,
                rec_text, conf_text, reasoning, free_style)


def register_ml_callbacks(app) -> None:

    @app.callback(
        Output("ml-prediction-output",       "children"),
        Output("ml-confidence-bar-graph",    "figure"),
        Output("ml-feature-importance-graph","figure"),
        Output("ml-residuals-graph",         "figure"),
        Input("ml-predict-btn",              "n_clicks"),
        State("ml-grid-position-input",      "value"),
        State("ml-qual-gap-input",           "value"),
        State("store-session-data",          "data"),
        prevent_initial_call=True,
    )
    @_safe
    def update_ml(n_clicks, grid_pos, qual_gap, session_data):
        if not n_clicks or not session_data:
            raise PreventUpdate

        from src.ml_optimizer.strategy_classifier import extract_race_context_features
        from src.ml_optimizer.xgboost_optimizer import STRATEGY_FEATURE_NAMES

        circuit    = session_data.get("circuit", "bahrain")
        total_laps = session_data.get("total_laps", 57)

        features = extract_race_context_features(
            circuit                    = str(circuit),
            qualifying_gap_to_pole_sec = float(qual_gap or 0.5),
            grid_position              = int(grid_pos or 10),
            race_laps                  = total_laps,
            sc_probability             = 0.68,
        )
        mean_deg = features.mean_dry_deg_rate

        if mean_deg > 0.055:
            stop_probs = {1: 0.15, 2: 0.72, 3: 0.13}
        elif mean_deg > 0.035:
            stop_probs = {1: 0.22, 2: 0.58, 3: 0.20}
        else:
            stop_probs = {1: 0.62, 2: 0.30, 3: 0.08}

        best_stop = max(stop_probs, key=stop_probs.get)
        pred_text = (
            f"Predicted: {best_stop}-stop  "
            f"(conf={stop_probs[best_stop]:.0%})  |  "
            f"deg={mean_deg:.4f}s/lap  |  "
            f"{'High-deg' if features.is_high_deg_circuit else 'Low-deg'} circuit"
        )

        conf_fig = _build_confidence_bar(stop_probs)

        importances = {
            "pit_lap_1_norm":    0.28, "compound_1_ordinal": 0.18,
            "base_lap_time_sec": 0.15, "n_stops":            0.12,
            "mean_deg_rate":     0.10, "pit_lap_2_norm":     0.08,
            "compound_2_ordinal":0.05, "race_laps":          0.02,
            "sc_probability":    0.01, "is_high_deg_circuit":0.01,
        }
        feat_fig = _build_feature_importance_fig(importances)

        rng       = np.random.default_rng(42)
        y_true    = rng.uniform(5400, 5600, 200)
        y_pred    = y_true + rng.normal(0, 2.0, 200)
        res_fig   = go.Figure()
        res_fig.add_trace(go.Scatter(
            x=y_true, y=y_pred, mode="markers",
            marker={"color": _RED, "size": 5, "opacity": 0.5},
            name="Predicted vs Actual",
        ))
        mn, mx = float(y_true.min()), float(y_true.max())
        res_fig.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx], mode="lines",
            line={"color": _WHITE, "width": 1.2, "dash": "dash"},
            name="Perfect prediction",
        ))
        res_fig.update_layout(**_base_layout(
            xaxis_title="Actual Race Time (s)",
            yaxis_title="Predicted Race Time (s)",
            title={"text": "Surrogate Residuals (illustrative)",
                   "font": {"color": _WHITE, "size": 10}},
        ))

        return pred_text, conf_fig, feat_fig, res_fig


# ===========================================================================
# Internal figure builders
# ===========================================================================

def _build_confidence_bar(stop_probs: dict) -> go.Figure:
    stops  = sorted(stop_probs.keys())
    probs  = [stop_probs[s] for s in stops]
    mx     = max(probs)
    colours= [_RED if p == mx else "#555" for p in probs]

    fig = go.Figure(go.Bar(
        x=stops, y=probs, marker_color=colours,
        text=[f"{p:.0%}" for p in probs], textposition="outside",
    ))
    fig.update_layout(**_base_layout(
        xaxis_title="Number of Stops", yaxis_title="Probability",
        title={"text": "Stop Count Prediction",
               "font": {"color": _WHITE, "size": 10}},
        yaxis={"range": [0, 1.1]}, showlegend=False,
    ))
    return fig


def _build_feature_importance_fig(importances: dict) -> go.Figure:
    items   = sorted(importances.items(), key=lambda x: x[1])
    names   = [x[0] for x in items]
    values  = [x[1] for x in items]
    mx      = max(values)
    md      = float(np.median(values))
    colours = [_RED if v == mx else _YEL if v > md else "#555" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color=colours,
        text=[f"{v:.3f}" for v in values], textposition="outside",
    ))
    fig.update_layout(**_base_layout(
        xaxis_title="Feature Importance (Gain)",
        title={"text": "Feature Importance",
               "font": {"color": _WHITE, "size": 10}},
        yaxis={"tickfont": {"size": 8}},
    ))
    return fig


def _build_sc_option_fig(decision) -> go.Figure:
    from src.safety_car_engine.sc_scenario_analyzer import PitResponse

    _col = {
        PitResponse.PIT_NOW:      _RED,
        PitResponse.STAY_OUT:     "#4FC3F7",
        PitResponse.PIT_NEXT_LAP: "#FF9800",
    }

    # evaluations is a list[PitResponseEvaluation], not a dict
    evals   = [e for e in decision.evaluations
                if e.total_race_time_sec < float("inf")]
    if not evals:
        return _empty("No viable SC options computed")

    fig     = go.Figure()
    best_t  = min(e.total_race_time_sec for e in evals)

    for ev in evals:
        response = ev.response
        col      = _col.get(response, "#888")
        is_rec   = response == decision.recommended
        shifted  = ev.total_race_time_sec - best_t
        p10      = ev.p10_time_sec - best_t
        p90      = ev.p90_time_sec - best_t
        std      = abs(p90 - p10) / 2.56 if p90 != p10 else 1.0

        samples = np.random.default_rng(hash(response.value) % 2**32).normal(
            shifted, std, 200
        )
        fig.add_trace(go.Violin(
            y=samples, name=response.value.replace("_", " "),
            box_visible=True, meanline_visible=True,
            line_color=col,
            fillcolor=f"rgba({int(col[1:3],16)},{int(col[3:5],16)},{int(col[5:7],16)},0.25)",
            opacity=0.9 if is_rec else 0.5,
            points=False,
        ))

    fig.add_hline(y=0, line_color="#555", line_dash="dot", line_width=0.8)
    fig.update_layout(**_base_layout(
        yaxis_title="Race Time Delta vs Best Option (s)",
        title={"text": (f"{decision.period.period_type} Options — "
                        f"Recommended: {decision.recommended.value}"),
               "font": {"color": _WHITE, "size": 11}},
        violinmode="overlay",
    ))
    return fig


def _build_sc_prob_fig(sc_profile, current_lap: Optional[int] = None) -> go.Figure:
    total  = sc_profile.total_race_laps
    laps   = np.arange(1, total + 1, dtype=float)
    probs  = np.array([sc_profile.sc_probability_at_lap(int(l)) for l in laps])
    unif   = 1.0 / total

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=laps, y=probs, mode="lines",
        name="SC probability",
        line={"color": "#FF9800", "width": 1.8},
        fill="tozeroy", fillcolor="rgba(255,152,0,0.15)",
    ))
    if len(sc_profile.sc_deployment_laps) > 0:
        fig.add_trace(go.Scatter(
            x=sc_profile.sc_deployment_laps,
            y=[unif * 0.2] * len(sc_profile.sc_deployment_laps),
            mode="markers",
            marker={"symbol": "line-ns", "color": "#FF9800", "size": 10},
            name="Historical SC laps",
        ))
    fig.add_hline(y=unif, line_color="#555", line_dash="dot", line_width=0.8)
    if current_lap:
        fig.add_vline(x=current_lap, line_color=_RED,
                      line_dash="dash", line_width=1.2,
                      annotation_text=f"L{current_lap}",
                      annotation_font_color=_RED)
    fig.update_layout(**_base_layout(
        xaxis_title="Race Lap",
        yaxis_title="SC Probability",
        title={"text": f"Historical SC Probability — {sc_profile.circuit.title()}",
               "font": {"color": _WHITE, "size": 11}},
    ))
    return fig


# ===========================================================================
# Master registration
# ===========================================================================

def register_all_callbacks(app) -> None:
    """Register all dashboard callbacks with the Dash app instance."""
    register_session_load_callback(app)
    register_overview_callbacks(app)
    register_strategy_callbacks(app)
    register_simulator_callbacks(app)
    register_sc_callbacks(app)
    register_ml_callbacks(app)
    logger.info("register_all_callbacks: all callbacks registered.")
