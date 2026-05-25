"""
Turbomole Remote Orchestrator — Dash application
=================================================

Local Dash GUI that:
  • Builds Turbomole inputs locally with ASE
  • Pushes them to an HPC cluster via SSH/SFTP
  • Submits SLURM jobs and tracks status (on-demand refresh)
  • Persists job metadata in a local SQLite DB (survives shutdown)
  • Allows downloading partial/finished outputs and remote cleanup

Run:
    python app.py
"""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc

from app_ui.layout import build_layout
from app_ui.callbacks import register_callbacks
from backend.db import init_db
from backend.config import load_config


def create_app() -> dash.Dash:
    cfg = load_config()
    init_db(cfg.db_path)

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.SLATE, dbc.icons.BOOTSTRAP],
        suppress_callback_exceptions=True,
        title="Turbomole Orchestrator",
        update_title=None,
        assets_folder="assets",
    )

    app.layout = build_layout(cfg)
    register_callbacks(app, cfg)
    return app


app = create_app()
server = app.server  # for gunicorn / production

if __name__ == "__main__":
    # debug=False so the reloader doesn't duplicate state
    app.run(host="127.0.0.1", port=8050, debug=False)
