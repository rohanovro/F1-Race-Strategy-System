"""
dashboard/app.py
==================
Dash application entry point and server configuration.

Engineering responsibility:
    - Initialise the Dash app with Bootstrap theme and metadata.
    - Wire layout and callbacks.
    - Configure production-grade logging.
    - Expose the WSGI server object for Gunicorn/uWSGI deployment.
    - Provide a CLI entry point for development mode.

Architecture principle:
    app.py knows about layout.py and callbacks.py.
    layout.py and callbacks.py know about src/ modules.
    src/ modules know nothing about the dashboard.
    This dependency direction is strictly one-way.

Production deployment:
    Gunicorn: gunicorn "dashboard.app:server" --workers 4 --bind 0.0.0.0:8050
    uWSGI:    uwsgi --http 0.0.0.0:8050 --module dashboard.app:server

Development:
    python dashboard/app.py
    or: python -m dashboard.app
    Dashboard will be available at http://127.0.0.1:8050

Environment variables:
    DASH_DEBUG:     "true" / "false"  (default: "false" in production)
    DASH_PORT:      Port number       (default: 8050)
    DASH_HOST:      Bind host         (default: "0.0.0.0")
    F1_CACHE_DIR:   FastF1 cache dir  (default: data/raw/fastf1_cache)
    LOG_LEVEL:      Logging level     (default: "INFO")
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Ensure src/ is importable when running from the project root or
# from the dashboard/ directory directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import dash
import dash_bootstrap_components as dbc

# ===========================================================================
# Logging configuration
# ===========================================================================

def _configure_logging() -> None:
    """
    Configure application-wide logging.

    Uses a structured format readable by both humans and log aggregators
    (Datadog, CloudWatch). Sets the root logger level from the LOG_LEVEL
    env var so production deployments can tune verbosity without code changes.
    """
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level   = getattr(logging, log_level, logging.INFO),
        format  = "%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Suppress overly verbose third-party loggers
    for noisy in ("werkzeug", "urllib3", "fastf1", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging configured | level=%s", log_level
    )


_configure_logging()
logger = logging.getLogger(__name__)


# ===========================================================================
# App initialisation
# ===========================================================================

def _create_app() -> dash.Dash:
    """
    Create and configure the Dash application.

    Bootstrap theme: DARKLY provides a clean dark base that
    matches the F1 pitwall aesthetic without requiring bespoke CSS.
    The custom inline styles in layout.py override Bootstrap where
    the F1 colour palette diverges from the generic dark theme.

    suppress_callback_exceptions=True:
        Required because the tab-based layout dynamically shows/hides
        components. Dash would raise exceptions for callbacks referencing
        components not currently visible if this were False.
    """
    app = dash.Dash(
        __name__,
        external_stylesheets  = [dbc.themes.DARKLY],
        suppress_callback_exceptions=True,
        meta_tags             = [
            {"name": "viewport",
             "content": "width=device-width, initial-scale=1"},
            {"name": "description",
             "content": "F1 Intelligent Race Strategy System"},
        ],
        title = "F1 Race Strategy System",
        update_title=None,
    )

    # Expose the Flask server for WSGI deployment
    app.server.secret_key = os.environ.get(
        "DASH_SECRET_KEY", "f1-strategy-system-dev-key"
    )

    logger.info("Dash app initialised | theme=DARKLY | suppress_exceptions=True")
    return app


# ===========================================================================
# Wiring — layout + callbacks
# ===========================================================================

def _wire_app(app: dash.Dash) -> None:
    """
    Attach layout and register all callbacks.

    Importing callbacks here (not at module level) avoids circular imports:
        app.py → layout.py  (layout imports src/ constants, not app or callbacks)
        app.py → callbacks.py (callbacks import src/ modules, not app or layout)
    """
    from dashboard.layout import build_layout
    from dashboard.callbacks import register_all_callbacks

    app.layout = build_layout()
    register_all_callbacks(app)

    logger.info("App wired | layout=build_layout() | callbacks=registered")


# ===========================================================================
# Create the app (module-level — needed for WSGI import)
# ===========================================================================

app    = _create_app()
_wire_app(app)
server = app.server   # WSGI entry point: gunicorn "dashboard.app:server"

logger.info(
    "F1 Race Strategy Dashboard initialised | "
    "WSGI server exposed as 'server'"
)


# ===========================================================================
# Development CLI entry point
# ===========================================================================

if __name__ == "__main__":
    debug = os.environ.get("DASH_DEBUG", "false").lower() == "true"
    port  = int(os.environ.get("DASH_PORT", "8050"))
    host  = os.environ.get("DASH_HOST", "127.0.0.1")

    logger.info(
        "Starting development server | host=%s port=%d debug=%s",
        host, port, debug,
    )
    logger.info(
        "Dashboard available at: http://%s:%d", host, port
    )

    app.run(
        debug      = debug,
        host       = host,
        port       = port,
        use_reloader=debug,
    )
