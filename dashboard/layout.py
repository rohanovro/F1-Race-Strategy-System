"""
dashboard/layout.py
=====================
Dash UI component tree for the F1 Race Strategy Dashboard.

Engineering responsibility:
    Define the complete visual structure of the dashboard as a pure
    Dash component tree. This file contains ZERO business logic,
    ZERO data processing, and ZERO callback registration.

    Layout and callbacks are separated because:
        - Layout is testable independently of data (snapshot tests).
        - Callbacks can be unit-tested without rendering the full UI.
        - Future layout changes (adding a panel, reorganising tabs)
          never risk breaking callback logic.

UI architecture:
    The dashboard is organised as a 5-tab application:

    Tab 1 — RACE OVERVIEW
        Circuit / driver / session selector + data load trigger.
        Tyre degradation overlay plot (all compounds).
        Session summary stats (laps, representative %, SC periods).

    Tab 2 — STRATEGY OPTIMISER
        Compound availability checkboxes + pit lane delta input.
        Strategy leaderboard waterfall chart.
        Pit window sensitivity curve.
        Optimal strategy display card.

    Tab 3 — LIVE RACE SIMULATOR
        Race lap slider (current_lap 1→total_race_laps).
        Strategy timeline chart (cumulative delta to optimal).
        Lap time component breakdown for selected strategy.
        Undercut/overcut decision panel for a competitor gap.

    Tab 4 — SAFETY CAR ANALYSER
        SC/VSC deployment toggle + current lap input.
        Three-option comparison chart (PIT_NOW / PIT_NEXT / STAY_OUT).
        Historical SC probability by lap chart.
        Recommendation card with confidence badge.

    Tab 5 — ML INSIGHTS
        Classifier stop-count prediction with confidence bars.
        Surrogate model residual plot.
        Feature importance chart.
        Backtest accuracy summary.

Design decisions:
    - dcc.Store components hold all session-level computed state
      (model_set, feature_df, opt_result) as JSON-serialisable dicts.
      This avoids global Python state that breaks multi-user deployments.
    - All IDs follow the convention: "{tab}-{component}-{type}"
      e.g. "strategy-leaderboard-graph", "overview-circuit-dropdown".
      Consistent naming makes callbacks self-documenting.
    - Loading spinners (dcc.Loading) wrap every graph that calls a
      backend computation — the UI never appears frozen.
    - The layout uses dash-bootstrap-components (dbc) for a professional
      grid system and pre-styled cards, avoiding bespoke CSS.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from src.constants import (
    COMPOUND_COLOURS,
    COMPOUND_ABBREV,
    DRY_COMPOUNDS,
    PIT_LANE_DELTA_SEC_BAHRAIN,
)
from src.tire_model.compound_profiles import list_profiled_circuits

# ===========================================================================
# Theme tokens — used for consistent inline styling
# ===========================================================================

_BG     = "#0F0F0F"
_PANEL  = "#1A1A1A"
_BORDER = "#2A2A2A"
_TEXT   = "#CCCCCC"
_WHITE  = "#FFFFFF"
_ACCENT = "#E8002D"   # FOM red

_CARD_STYLE = {
    "backgroundColor": _PANEL,
    "border":          f"1px solid {_BORDER}",
    "borderRadius":    "8px",
    "padding":         "16px",
    "marginBottom":    "14px",
    "color":           _TEXT,
}

_LABEL_STYLE = {
    "color":      _TEXT,
    "fontSize":   "12px",
    "fontWeight": "600",
    "marginBottom": "4px",
    "textTransform": "uppercase",
    "letterSpacing": "0.05em",
}

_METRIC_LABEL = {
    "color":    "#888888",
    "fontSize": "11px",
    "textTransform": "uppercase",
    "letterSpacing": "0.04em",
}

_METRIC_VALUE = {
    "color":      _WHITE,
    "fontSize":   "22px",
    "fontWeight": "700",
    "lineHeight": "1.1",
}


# ===========================================================================
# Helper component factories
# ===========================================================================

def _section_label(text: str) -> html.P:
    return html.P(text, style=_LABEL_STYLE)


def _metric_card(
    label: str,
    value_id: str,
    value_default: str = "—",
    accent_colour: str = _WHITE,
) -> dbc.Card:
    """Small KPI card: label above, large metric value below."""
    return dbc.Card(
        dbc.CardBody([
            html.P(label,       style=_METRIC_LABEL),
            html.H4(value_default,
                    id=value_id,
                    style={**_METRIC_VALUE, "color": accent_colour}),
        ]),
        style={**_CARD_STYLE, "padding": "12px"},
    )


def _graph(
    graph_id: str,
    height:   int  = 420,
    spinner:  bool = True,
) -> html.Div:
    """Graph wrapped in a loading spinner."""
    g = dcc.Graph(
        id     = graph_id,
        style  = {"height": f"{height}px"},
        config = {"displayModeBar": True, "scrollZoom": True,
                  "toImageButtonOptions": {"format": "png", "scale": 2}},
    )
    return dcc.Loading(g, type="circle", color=_ACCENT) if spinner else html.Div(g)


def _dropdown(
    dd_id:       str,
    options:     list[dict],
    value,
    multi:       bool  = False,
    clearable:   bool  = False,
    placeholder: str   = "Select…",
) -> dcc.Dropdown:
    return dcc.Dropdown(
        id          = dd_id,
        options     = options,
        value       = value,
        multi       = multi,
        clearable   = clearable,
        placeholder = placeholder,
        style={
            "backgroundColor": "#222222",
            "color":           _TEXT,
            "border":          f"1px solid {_BORDER}",
            "borderRadius":    "4px",
            "fontSize":        "13px",
        },
    )


def _slider(
    slider_id: str,
    min_val:   int,
    max_val:   int,
    value:     int,
    step:      int = 1,
) -> dcc.Slider:
    return dcc.Slider(
        id    = slider_id,
        min   = min_val,
        max   = max_val,
        step  = step,
        value = value,
        marks = {i: {"label": str(i), "style": {"color": _TEXT, "fontSize": "11px"}}
                 for i in range(min_val, max_val + 1, max(1, (max_val - min_val) // 10))},
        tooltip = {"placement": "bottom", "always_visible": True},
    )


def _alert_card(
    card_id:    str,
    icon:       str = "🏎",
    text:       str = "Awaiting data…",
    colour:     str = _ACCENT,
) -> dbc.Alert:
    return dbc.Alert(
        [html.Span(icon, style={"fontSize": "18px", "marginRight": "8px"}),
         html.Span(text, id=card_id)],
        color    = "dark",
        style    = {**_CARD_STYLE,
                    "borderLeft": f"4px solid {colour}",
                    "padding":    "12px 16px"},
    )


# ===========================================================================
# Compound checkbox group
# ===========================================================================

def _compound_checklist(checklist_id: str) -> dbc.Card:
    options = [
        {
            "label": html.Span(
                [
                    html.Span(
                        "●",
                        style={"color": COMPOUND_COLOURS.get(c, "#888"),
                               "marginRight": "6px", "fontSize": "14px"},
                    ),
                    html.Span(COMPOUND_ABBREV.get(c, c), style={"color": _TEXT}),
                ],
                style={"display": "inline-flex", "alignItems": "center"},
            ),
            "value": c,
        }
        for c in sorted(DRY_COMPOUNDS)
    ]
    return dbc.Card(
        dbc.CardBody([
            _section_label("Available Compounds"),
            dcc.Checklist(
                id      = checklist_id,
                options = options,
                value   = sorted(DRY_COMPOUNDS),
                inline  = True,
                style   = {"display": "flex", "gap": "18px"},
            ),
        ]),
        style=_CARD_STYLE,
    )


# ===========================================================================
# Session selector (shared across tabs via dcc.Store)
# ===========================================================================

def _session_selector_row() -> dbc.Card:
    """Top-of-page session selector: circuit, year, session type, load button."""
    circuit_options = [
        {"label": c.replace("_", " ").title(), "value": c}
        for c in list_profiled_circuits()
    ] or [{"label": "Bahrain", "value": "bahrain"}]

    return dbc.Card(
        dbc.CardBody(
            dbc.Row([
                dbc.Col([
                    _section_label("Circuit"),
                    _dropdown(
                        "session-circuit-dropdown",
                        circuit_options,
                        value="bahrain",
                    ),
                ], md=3),

                dbc.Col([
                    _section_label("Season"),
                    dcc.Input(
                        id          = "session-year-input",
                        type        = "number",
                        value       = 2023,
                        min         = 2018,
                        max         = 2025,
                        step        = 1,
                        debounce    = True,
                        style       = {"width": "100%", "backgroundColor": "#222",
                                       "color": _TEXT, "border": f"1px solid {_BORDER}",
                                       "borderRadius": "4px", "padding": "6px",
                                       "fontSize": "13px"},
                    ),
                ], md=2),

                dbc.Col([
                    _section_label("Session Type"),
                    _dropdown(
                        "session-type-dropdown",
                        [{"label": t, "value": t} for t in ["R", "Q", "FP1", "FP2", "FP3"]],
                        value="R",
                    ),
                ], md=2),

                dbc.Col([
                    _section_label("Total Race Laps"),
                    dcc.Input(
                        id       = "session-total-laps-input",
                        type     = "number",
                        value    = 57,
                        min      = 10,
                        max      = 80,
                        step     = 1,
                        debounce = True,
                        style    = {"width": "100%", "backgroundColor": "#222",
                                    "color": _TEXT, "border": f"1px solid {_BORDER}",
                                    "borderRadius": "4px", "padding": "6px",
                                    "fontSize": "13px"},
                    ),
                ], md=2),

                dbc.Col([
                    _section_label("\u00a0"),   # non-breaking space for alignment
                    dbc.Button(
                        "⚡ Load Session",
                        id        = "session-load-btn",
                        color     = "danger",
                        n_clicks  = 0,
                        style     = {"width": "100%", "fontWeight": "700",
                                     "letterSpacing": "0.05em"},
                    ),
                ], md=2),

                dbc.Col([
                    html.Div(
                        id    = "session-status-badge",
                        style = {"paddingTop": "24px", "fontSize": "12px",
                                 "color": "#888"},
                    ),
                ], md=1),
            ])
        ),
        style={**_CARD_STYLE, "marginBottom": "6px"},
    )


# ===========================================================================
# Tab 1 — Race Overview
# ===========================================================================

def _tab_overview() -> dbc.Tab:
    return dbc.Tab(
        label     = "🏁 Race Overview",
        tab_id    = "tab-overview",
        label_style = {"color": _TEXT},
        active_label_style = {"color": _WHITE, "fontWeight": "700"},
        children  = [
            html.Br(),

            # KPI row
            dbc.Row([
                dbc.Col(_metric_card("Total Laps",          "overview-total-laps-kpi"),   md=2),
                dbc.Col(_metric_card("Drivers",             "overview-drivers-kpi"),       md=2),
                dbc.Col(_metric_card("Representative Laps", "overview-repr-pct-kpi",
                                     accent_colour="#43B02A"),                             md=2),
                dbc.Col(_metric_card("SC Periods",          "overview-sc-kpi",
                                     accent_colour="#FF9800"),                             md=2),
                dbc.Col(_metric_card("Compounds Used",      "overview-compounds-kpi"),     md=2),
                dbc.Col(_metric_card("Session Best",        "overview-best-lap-kpi"),      md=2),
            ], className="mb-3"),

            # Degradation overlay + compound summary side-by-side
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(
                            "Compound Degradation Overlay",
                            style={"backgroundColor": "#222", "color": _WHITE,
                                   "fontWeight": "600", "fontSize": "13px"},
                        ),
                        dbc.CardBody(_graph("overview-degradation-overlay-graph",
                                            height=380)),
                    ], style=_CARD_STYLE),
                ], md=7),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(
                            "Compound Summary",
                            style={"backgroundColor": "#222", "color": _WHITE,
                                   "fontWeight": "600", "fontSize": "13px"},
                        ),
                        dbc.CardBody(_graph("overview-compound-summary-graph",
                                            height=380)),
                    ], style=_CARD_STYLE),
                ], md=5),
            ]),

            # Pace drop heatmap
            dbc.Card([
                dbc.CardHeader(
                    html.Div([
                        html.Span("Pace Drop Heatmap — Compound: ",
                                  style={"color": _WHITE, "fontWeight": "600"}),
                        _dropdown(
                            "overview-heatmap-compound-dropdown",
                            [{"label": c, "value": c} for c in sorted(DRY_COMPOUNDS)],
                            value="SOFT",
                            clearable=False,
                        ),
                    ], style={"display": "flex", "alignItems": "center", "gap": "10px"}),
                    style={"backgroundColor": "#222"},
                ),
                dbc.CardBody(_graph("overview-heatmap-graph", height=300)),
            ], style=_CARD_STYLE),
        ],
    )


# ===========================================================================
# Tab 2 — Strategy Optimiser
# ===========================================================================

def _tab_strategy() -> dbc.Tab:
    return dbc.Tab(
        label      = "⚙️ Strategy Optimiser",
        tab_id     = "tab-strategy",
        label_style= {"color": _TEXT},
        active_label_style={"color": _WHITE, "fontWeight": "700"},
        children   = [
            html.Br(),
            dbc.Row([

                # Left column — controls
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            _section_label("Available Compounds"),
                            _compound_checklist("strategy-compounds-checklist"),
                            html.Br(),

                            _section_label("Max Pit Stops"),
                            dcc.Slider(
                                id="strategy-max-stops-slider",
                                min=0, max=3, step=1, value=2,
                                marks={i: {"label": str(i), "style": {"color": _TEXT}}
                                       for i in range(4)},
                                tooltip={"placement": "bottom",
                                         "always_visible": True},
                            ),
                            html.Br(),

                            _section_label("Pit Lane Delta (seconds)"),
                            dcc.Input(
                                id       = "strategy-pit-delta-input",
                                type     = "number",
                                value    = round(PIT_LANE_DELTA_SEC_BAHRAIN, 1),
                                min      = 10.0,
                                max      = 30.0,
                                step     = 0.1,
                                debounce = True,
                                style    = {"width": "100%", "backgroundColor": "#222",
                                            "color": _TEXT, "border": f"1px solid {_BORDER}",
                                            "borderRadius": "4px", "padding": "6px",
                                            "fontSize": "13px"},
                            ),
                            html.Br(),

                            dbc.Button(
                                "🔍 Run Optimisation",
                                id       = "strategy-run-btn",
                                color    = "danger",
                                n_clicks = 0,
                                style    = {"width": "100%", "fontWeight": "700",
                                            "marginTop": "10px"},
                            ),
                        ])
                    ], style=_CARD_STYLE),

                    # Optimal strategy card
                    dbc.Card([
                        dbc.CardHeader("Optimal Strategy",
                                       style={"backgroundColor": "#222",
                                              "color": _WHITE, "fontWeight": "600"}),
                        dbc.CardBody([
                            html.H5(id="strategy-optimal-label",
                                    children="Run optimisation →",
                                    style={"color": _ACCENT, "fontWeight": "700"}),
                            html.P(id="strategy-optimal-time",
                                   children="",
                                   style={"color": _TEXT, "fontSize": "13px"}),
                            html.Hr(style={"borderColor": _BORDER}),
                            html.Div(id="strategy-optimal-details",
                                     style={"color": _TEXT, "fontSize": "12px"}),
                        ]),
                    ], style=_CARD_STYLE),

                    # ML classifier prediction
                    dbc.Card([
                        dbc.CardHeader("ML Stop-Count Prediction",
                                       style={"backgroundColor": "#222",
                                              "color": _WHITE, "fontWeight": "600"}),
                        dbc.CardBody([
                            html.Div(id="strategy-ml-prediction",
                                     style={"color": _TEXT, "fontSize": "13px"}),
                            _graph("strategy-ml-confidence-graph", height=120, spinner=False),
                        ]),
                    ], style=_CARD_STYLE),

                ], md=3),

                # Right column — charts
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Strategy Leaderboard",
                                       style={"backgroundColor": "#222",
                                              "color": _WHITE, "fontWeight": "600"}),
                        dbc.CardBody(_graph("strategy-leaderboard-graph", height=460)),
                    ], style=_CARD_STYLE),

                    dbc.Card([
                        dbc.CardHeader(
                            html.Div([
                                html.Span("Pit Window Sensitivity  ",
                                          style={"color": _WHITE, "fontWeight": "600"}),
                                _dropdown(
                                    "strategy-pit-window-start-dd",
                                    [{"label": c, "value": c} for c in sorted(DRY_COMPOUNDS)],
                                    value="SOFT", clearable=False,
                                ),
                                html.Span(" → ", style={"color": _TEXT}),
                                _dropdown(
                                    "strategy-pit-window-next-dd",
                                    [{"label": c, "value": c} for c in sorted(DRY_COMPOUNDS)],
                                    value="MEDIUM", clearable=False,
                                ),
                            ], style={"display": "flex", "alignItems": "center",
                                      "gap": "8px"}),
                            style={"backgroundColor": "#222"},
                        ),
                        dbc.CardBody(_graph("strategy-pit-window-graph", height=280)),
                    ], style=_CARD_STYLE),
                ], md=9),
            ]),
        ],
    )


# ===========================================================================
# Tab 3 — Live Race Simulator
# ===========================================================================

def _tab_simulator() -> dbc.Tab:
    return dbc.Tab(
        label      = "🏎 Live Simulator",
        tab_id     = "tab-simulator",
        label_style= {"color": _TEXT},
        active_label_style={"color": _WHITE, "fontWeight": "700"},
        children   = [
            html.Br(),
            # Race lap slider
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            _section_label("Current Race Lap"),
                            _slider("simulator-lap-slider", 1, 57, 1),
                        ], md=8),
                        dbc.Col([
                            _section_label("Strategy to Inspect"),
                            _dropdown(
                                "simulator-strategy-dropdown",
                                [{"label": "Run optimisation first", "value": "none"}],
                                value="none",
                            ),
                        ], md=4),
                    ])
                ]),
            ], style=_CARD_STYLE),

            # Strategy timeline
            dbc.Card([
                dbc.CardHeader("Strategy Timeline",
                               style={"backgroundColor": "#222", "color": _WHITE,
                                      "fontWeight": "600"}),
                dbc.CardBody(_graph("simulator-timeline-graph", height=380)),
            ], style=_CARD_STYLE),

            dbc.Row([
                # Lap time breakdown
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Lap Time Component Breakdown",
                                       style={"backgroundColor": "#222", "color": _WHITE,
                                              "fontWeight": "600"}),
                        dbc.CardBody(_graph("simulator-breakdown-graph", height=300)),
                    ], style=_CARD_STYLE),
                ], md=7),

                # Undercut/Overcut panel
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Undercut / Overcut Analyser",
                                       style={"backgroundColor": "#222", "color": _WHITE,
                                              "fontWeight": "600"}),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    _section_label("Their Driver"),
                                    dcc.Input(
                                        id="simulator-uc-driver", type="text",
                                        value="ALO", debounce=True,
                                        style={"width": "100%", "backgroundColor": "#222",
                                               "color": _TEXT, "border": f"1px solid {_BORDER}",
                                               "borderRadius": "4px", "padding": "6px",
                                               "fontSize": "13px"},
                                    ),
                                ], md=4),
                                dbc.Col([
                                    _section_label("Gap to Them (s)"),
                                    dcc.Input(
                                        id="simulator-uc-gap", type="number",
                                        value=2.5, step=0.1, debounce=True,
                                        style={"width": "100%", "backgroundColor": "#222",
                                               "color": _TEXT, "border": f"1px solid {_BORDER}",
                                               "borderRadius": "4px", "padding": "6px",
                                               "fontSize": "13px"},
                                    ),
                                ], md=4),
                                dbc.Col([
                                    _section_label("Their Tyre Age"),
                                    dcc.Input(
                                        id="simulator-uc-age", type="number",
                                        value=15, min=1, max=50, step=1, debounce=True,
                                        style={"width": "100%", "backgroundColor": "#222",
                                               "color": _TEXT, "border": f"1px solid {_BORDER}",
                                               "borderRadius": "4px", "padding": "6px",
                                               "fontSize": "13px"},
                                    ),
                                ], md=4),
                            ]),
                            html.Br(),
                            _alert_card("simulator-uc-decision",
                                        icon="🔄", text="Configure inputs above →",
                                        colour="#FF9800"),
                            html.Div(id="simulator-uc-reasoning",
                                     style={"color": "#888", "fontSize": "11px",
                                            "marginTop": "8px"}),
                        ]),
                    ], style=_CARD_STYLE),
                ], md=5),
            ]),
        ],
    )


# ===========================================================================
# Tab 4 — Safety Car Analyser
# ===========================================================================

def _tab_safety_car() -> dbc.Tab:
    return dbc.Tab(
        label      = "🚨 Safety Car",
        tab_id     = "tab-sc",
        label_style= {"color": _TEXT},
        active_label_style={"color": _WHITE, "fontWeight": "700"},
        children   = [
            html.Br(),
            dbc.Row([
                # SC controls
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            _section_label("Neutralisation Type"),
                            dcc.RadioItems(
                                id="sc-type-radio",
                                options=[
                                    {"label": html.Span("Full SC",
                                                        style={"color": "#FF9800",
                                                               "marginLeft": "6px"}),
                                     "value": "SC"},
                                    {"label": html.Span("VSC",
                                                        style={"color": "#4FC3F7",
                                                               "marginLeft": "6px"}),
                                     "value": "VSC"},
                                ],
                                value  = "SC",
                                inline = True,
                                style  = {"gap": "18px", "marginBottom": "12px"},
                            ),

                            _section_label("Deployment Lap"),
                            dcc.Input(
                                id="sc-current-lap-input", type="number",
                                value=23, min=1, max=57, step=1, debounce=True,
                                style={"width": "100%", "backgroundColor": "#222",
                                       "color": _TEXT, "border": f"1px solid {_BORDER}",
                                       "borderRadius": "4px", "padding": "6px",
                                       "fontSize": "13px"},
                            ),
                            html.Br(),

                            _section_label("Our Compound"),
                            _dropdown("sc-our-compound-dd",
                                      [{"label": c, "value": c} for c in sorted(DRY_COMPOUNDS)],
                                      value="SOFT"),
                            html.Br(),

                            _section_label("Our Tyre Age"),
                            dcc.Input(
                                id="sc-tyre-age-input", type="number",
                                value=18, min=1, max=50, step=1, debounce=True,
                                style={"width": "100%", "backgroundColor": "#222",
                                       "color": _TEXT, "border": f"1px solid {_BORDER}",
                                       "borderRadius": "4px", "padding": "6px",
                                       "fontSize": "13px"},
                            ),
                            html.Br(),

                            _section_label("Gap to Leader (s)"),
                            dcc.Input(
                                id="sc-gap-input", type="number",
                                value=12.5, step=0.1, debounce=True,
                                style={"width": "100%", "backgroundColor": "#222",
                                       "color": _TEXT, "border": f"1px solid {_BORDER}",
                                       "borderRadius": "4px", "padding": "6px",
                                       "fontSize": "13px"},
                            ),
                            html.Br(),

                            _section_label("Next Compound (if pit)"),
                            _dropdown("sc-next-compound-dd",
                                      [{"label": c, "value": c} for c in sorted(DRY_COMPOUNDS)],
                                      value="MEDIUM"),
                            html.Br(),

                            dbc.Button(
                                "🚦 Evaluate SC Options",
                                id="sc-evaluate-btn", color="warning",
                                n_clicks=0,
                                style={"width": "100%", "fontWeight": "700",
                                       "color": "#000"},
                            ),
                        ])
                    ], style=_CARD_STYLE),

                    # Recommendation card
                    dbc.Card([
                        dbc.CardHeader("Recommendation",
                                       style={"backgroundColor": "#222", "color": _WHITE,
                                              "fontWeight": "600"}),
                        dbc.CardBody([
                            html.H4(id="sc-recommendation-text",
                                    children="Awaiting analysis…",
                                    style={"color": _ACCENT, "fontWeight": "700"}),
                            html.P(id="sc-confidence-text",
                                   style={"color": "#888", "fontSize": "12px"}),
                            html.Hr(style={"borderColor": _BORDER}),
                            html.P(id="sc-reasoning-text",
                                   style={"color": _TEXT, "fontSize": "11px",
                                          "lineHeight": "1.5"}),
                            html.Div([
                                html.Span("FREE STOP",
                                          id="sc-free-stop-badge",
                                          style={"display": "none",
                                                 "backgroundColor": _ACCENT,
                                                 "color": _WHITE,
                                                 "padding": "3px 10px",
                                                 "borderRadius": "12px",
                                                 "fontSize": "11px",
                                                 "fontWeight": "700"}),
                            ]),
                        ]),
                    ], style={**_CARD_STYLE,
                               "borderLeft": f"3px solid {_ACCENT}"}),
                ], md=3),

                # SC charts
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Option Comparison (Monte Carlo)",
                                       style={"backgroundColor": "#222", "color": _WHITE,
                                              "fontWeight": "600"}),
                        dbc.CardBody(_graph("sc-option-comparison-graph", height=360)),
                    ], style=_CARD_STYLE),

                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Gap Compression Timeline",
                                               style={"backgroundColor": "#222",
                                                      "color": _WHITE, "fontWeight": "600"}),
                                dbc.CardBody(_graph("sc-gap-compression-graph",
                                                    height=250)),
                            ], style=_CARD_STYLE),
                        ], md=6),

                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Historical SC Probability by Lap",
                                               style={"backgroundColor": "#222",
                                                      "color": _WHITE, "fontWeight": "600"}),
                                dbc.CardBody(_graph("sc-probability-graph", height=250)),
                            ], style=_CARD_STYLE),
                        ], md=6),
                    ]),
                ], md=9),
            ]),
        ],
    )


# ===========================================================================
# Tab 5 — ML Insights
# ===========================================================================

def _tab_ml() -> dbc.Tab:
    return dbc.Tab(
        label      = "🤖 ML Insights",
        tab_id     = "tab-ml",
        label_style= {"color": _TEXT},
        active_label_style={"color": _WHITE, "fontWeight": "700"},
        children   = [
            html.Br(),
            dbc.Row([
                dbc.Col([
                    # Classifier inputs
                    dbc.Card([
                        dbc.CardHeader("Strategy Classifier Inputs",
                                       style={"backgroundColor": "#222", "color": _WHITE,
                                              "fontWeight": "600"}),
                        dbc.CardBody([
                            _section_label("Grid Position"),
                            dcc.Input(
                                id="ml-grid-position-input", type="number",
                                value=5, min=1, max=20, step=1, debounce=True,
                                style={"width": "100%", "backgroundColor": "#222",
                                       "color": _TEXT, "border": f"1px solid {_BORDER}",
                                       "borderRadius": "4px", "padding": "6px"},
                            ),
                            html.Br(),
                            _section_label("Qualifying Gap to Pole (s)"),
                            dcc.Input(
                                id="ml-qual-gap-input", type="number",
                                value=0.45, min=0, max=5.0, step=0.01, debounce=True,
                                style={"width": "100%", "backgroundColor": "#222",
                                       "color": _TEXT, "border": f"1px solid {_BORDER}",
                                       "borderRadius": "4px", "padding": "6px"},
                            ),
                            html.Br(),
                            dbc.Button(
                                "🔮 Predict Stop Count",
                                id="ml-predict-btn", color="secondary",
                                n_clicks=0,
                                style={"width": "100%", "fontWeight": "700",
                                       "marginTop": "8px"},
                            ),
                            html.Br(), html.Br(),
                            html.Div(id="ml-prediction-output",
                                     style={"color": _TEXT, "fontSize": "13px"}),
                        ]),
                    ], style=_CARD_STYLE),
                ], md=3),

                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Stop Count Probability Breakdown",
                                               style={"backgroundColor": "#222",
                                                      "color": _WHITE, "fontWeight": "600"}),
                                dbc.CardBody(_graph("ml-confidence-bar-graph",
                                                    height=220, spinner=False)),
                            ], style=_CARD_STYLE),
                        ], md=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Surrogate Model Feature Importance",
                                               style={"backgroundColor": "#222",
                                                      "color": _WHITE, "fontWeight": "600"}),
                                dbc.CardBody(_graph("ml-feature-importance-graph",
                                                    height=220, spinner=False)),
                            ], style=_CARD_STYLE),
                        ], md=6),
                    ]),
                    dbc.Card([
                        dbc.CardHeader("Surrogate Residuals (Predicted vs Actual)",
                                       style={"backgroundColor": "#222", "color": _WHITE,
                                              "fontWeight": "600"}),
                        dbc.CardBody(_graph("ml-residuals-graph", height=280)),
                    ], style=_CARD_STYLE),
                ], md=9),
            ]),
        ],
    )


# ===========================================================================
# Root layout
# ===========================================================================

def build_layout() -> html.Div:
    """
    Assemble and return the complete Dash app layout.

    All dcc.Store components for shared state are declared here.
    Stores use session-level storage to isolate users in multi-user
    deployments.

    Returns:
        html.Div containing the full application layout.
    """
    return html.Div(
        style={"backgroundColor": _BG, "minHeight": "100vh",
               "fontFamily": "'Inter', 'Helvetica Neue', sans-serif"},
        children=[

            # ---- Shared state stores ----
            dcc.Store(id="store-session-data",    storage_type="session"),
            dcc.Store(id="store-model-set",       storage_type="session"),
            dcc.Store(id="store-feature-df",      storage_type="session"),
            dcc.Store(id="store-opt-result",      storage_type="session"),
            dcc.Store(id="store-sc-profile",      storage_type="session"),
            dcc.Store(id="store-classifier",      storage_type="memory"),
            dcc.Store(id="store-surrogate",       storage_type="memory"),

            # ---- Header ----
            html.Div(
                style={"backgroundColor": "#0A0A0A",
                       "borderBottom":    f"2px solid {_ACCENT}",
                       "padding":         "12px 24px",
                       "display":         "flex",
                       "alignItems":      "center",
                       "justifyContent":  "space-between"},
                children=[
                    html.Div([
                        html.Span("F1",
                                  style={"color": _ACCENT, "fontWeight": "900",
                                         "fontSize": "22px", "letterSpacing": "-0.02em"}),
                        html.Span(" RACE STRATEGY SYSTEM",
                                  style={"color": _WHITE, "fontWeight": "700",
                                         "fontSize": "16px", "letterSpacing": "0.1em",
                                         "marginLeft": "4px"}),
                    ]),
                    html.Div(
                        id="header-session-label",
                        children="No session loaded",
                        style={"color": "#888", "fontSize": "12px"},
                    ),
                ],
            ),

            # ---- Session selector ----
            html.Div(
                style={"padding": "12px 24px 0 24px"},
                children=[_session_selector_row()],
            ),

            # ---- Main tabs ----
            html.Div(
                style={"padding": "0 24px 24px 24px"},
                children=[
                    dbc.Tabs(
                        id       = "main-tabs",
                        active_tab="tab-overview",
                        style    = {"borderBottom": f"1px solid {_BORDER}"},
                        children = [
                            _tab_overview(),
                            _tab_strategy(),
                            _tab_simulator(),
                            _tab_safety_car(),
                            _tab_ml(),
                        ],
                    ),
                ],
            ),
        ],
    )
