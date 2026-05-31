"""
Dash layout. Four tabs + two modals (confirmation + job detail).
"""

from __future__ import annotations

from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc

from backend.config import AppConfig
from backend.turbomole_io import (
    available_constraint_algorithms,
    available_functionals, available_basis_sets,
    AU_TIME_TO_FS,
    TASK_TYPES, SUPPORTED_INPUT_FORMATS,
)


def build_layout(cfg: AppConfig) -> html.Div:
    return html.Div(
        [
            _navbar(),
            dbc.Container(
                [
                    dcc.Store(id="job-staging-store"),
                    dcc.Store(id="last-action"),
                    dcc.Store(id="pending-action"),
                    dcc.Store(id="tab-visited", data={}),
                    dcc.Store(id="active-ops", data={}),       # in-flight uploads/downloads
                    dcc.Store(id="protocols-version", data=0), # bumped to force dropdown refresh
                    dcc.Interval(id="active-ops-tick",
                                 interval=2000,
                                 disabled=True,
                                 n_intervals=0),
                    _confirm_modal(),
                    _job_detail_modal(),
                    _save_protocol_modal(),
                    dbc.Tabs(
                        [
                            dbc.Tab(_tab_new_job(cfg), label="New job", tab_id="tab-new"),
                            dbc.Tab(_tab_jobs(), label="Job manager", tab_id="tab-jobs"),
                            dbc.Tab(_tab_analysis(), label="Analysis", tab_id="tab-analysis"),
                            dbc.Tab(_tab_clusters(cfg), label="Clusters", tab_id="tab-clusters"),
                            dbc.Tab(_tab_database(), label="Database", tab_id="tab-db"),
                        ],
                        id="main-tabs",
                        active_tab="tab-new",
                        className="mt-3",
                    ),
                ],
                fluid=True,
                className="pb-5",
            ),
            html.Div(id="toast-container",
                     style={"position": "fixed", "top": 70, "right": 20, "zIndex": 1080}),
        ]
    )


def _navbar() -> dbc.Navbar:
    return dbc.Navbar(
        dbc.Container(
            [
                dbc.NavbarBrand(
                    [html.I(className="bi bi-cpu me-2"),
                     "Turbomole Orchestrator"],
                    className="fw-bold"
                ),
                html.Span("local <-> HPC bridge", className="text-muted small"),
            ],
            fluid=True,
        ),
        color="dark", dark=True, sticky="top",
    )


def _confirm_modal() -> dbc.Modal:
    return dbc.Modal(
        [
            dbc.ModalHeader(
                dbc.ModalTitle(
                    [html.I(className="bi bi-exclamation-triangle me-2 text-warning"),
                     "Confirm action"]
                ),
                close_button=False,
            ),
            dbc.ModalBody(id="confirm-modal-body"),
            dbc.ModalFooter(
                [
                    dbc.Button("Cancel", id="btn-confirm-cancel",
                               color="secondary", outline=True),
                    dbc.Button("Confirm", id="btn-confirm-ok", color="danger"),
                ]
            ),
        ],
        id="confirm-modal",
        is_open=False,
        backdrop="static",
        keyboard=True,
        centered=True,
    )


def _job_detail_modal() -> dbc.Modal:
    """Modal that shows extended info about a single job: parameters, results
    extracted from ridft.out, resource usage and list of available files."""
    return dbc.Modal(
        [
            dbc.ModalHeader(
                dbc.ModalTitle(id="detail-modal-title"),
                close_button=True,
            ),
            dbc.ModalBody(
                dbc.Spinner(
                    html.Div(id="detail-modal-body"),
                    color="primary",
                    size="sm",
                ),
            ),
            dbc.ModalFooter(
                dbc.Button("Close", id="btn-detail-close",
                           color="secondary", outline=True),
            ),
        ],
        id="detail-modal",
        is_open=False,
        size="xl",
        scrollable=True,
        centered=True,
    )


# ---------------------------------------------------------------------------
# Tab 1 — New job
# ---------------------------------------------------------------------------

def _protocols_bar(cfg: AppConfig) -> dbc.Card:
    """Top bar of the New job tab: load / save / delete protocols, plus
    a reset-to-defaults button."""
    return dbc.Card(
        dbc.CardBody(
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Protocol",
                                      className="small text-muted mb-1"),
                            dcc.Dropdown(
                                id="dd-protocol",
                                placeholder="-- select a saved protocol to load --",
                                clearable=True,
                            ),
                        ],
                        md=7,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                dbc.Button(
                                    [html.I(className="bi bi-floppy me-1"),
                                     "Save as..."],
                                    id="btn-proto-save",
                                    color="primary",
                                    outline=True,
                                    size="sm",
                                    className="me-1",
                                ),
                                dbc.Button(
                                    [html.I(className="bi bi-trash me-1"),
                                     "Delete"],
                                    id="btn-proto-delete",
                                    color="danger",
                                    outline=True,
                                    size="sm",
                                    disabled=True,    # enabled once a protocol is selected
                                    className="me-1",
                                ),
                                dbc.Button(
                                    [html.I(className="bi bi-arrow-counterclockwise me-1"),
                                     "Reset"],
                                    id="btn-proto-reset",
                                    color="secondary",
                                    outline=True,
                                    size="sm",
                                ),
                                dbc.Tooltip(
                                    "Save the current parameters as a reusable "
                                    "protocol (JSON). The molecular structure is "
                                    "not part of the protocol.",
                                    target="btn-proto-save",
                                    delay={"show": 400, "hide": 100},
                                ),
                                dbc.Tooltip(
                                    "Delete the selected protocol from disk.",
                                    target="btn-proto-delete",
                                    delay={"show": 400, "hide": 100},
                                ),
                                dbc.Tooltip(
                                    "Reset all fields to their factory defaults.",
                                    target="btn-proto-reset",
                                    delay={"show": 400, "hide": 100},
                                ),
                            ],
                            className="d-flex align-items-end h-100 pb-1",
                        ),
                        md=5,
                    ),
                ],
                className="g-2 align-items-end",
            ),
        ),
        className="shadow-sm mt-3",
    )


def _save_protocol_modal() -> dbc.Modal:
    """Modal asking for protocol name and optional description before
    persisting the current form state as a JSON protocol."""
    return dbc.Modal(
        [
            dbc.ModalHeader(
                dbc.ModalTitle(
                    [html.I(className="bi bi-floppy me-2"),
                     "Save protocol"]
                ),
            ),
            dbc.ModalBody(
                [
                    dbc.Label("Protocol name"),
                    dbc.Input(
                        id="inp-proto-name",
                        placeholder="e.g. DFT opt organometallics",
                        autoFocus=True,
                    ),
                    dbc.Label("Description (optional)", className="mt-3"),
                    dbc.Textarea(
                        id="inp-proto-description",
                        placeholder="Anything that helps you recognize this protocol later",
                        rows=2,
                    ),
                    html.Div(id="save-proto-warning", className="mt-2"),
                ],
            ),
            dbc.ModalFooter(
                [
                    dbc.Button("Cancel", id="btn-proto-save-cancel",
                               color="secondary", outline=True),
                    dbc.Button("Save", id="btn-proto-save-ok", color="primary"),
                ]
            ),
        ],
        id="proto-save-modal",
        is_open=False,
        backdrop="static",
        centered=True,
    )


def _tab_new_job(cfg: AppConfig) -> html.Div:
    cluster_options = [{"label": c, "value": c} for c in cfg.clusters.keys()]

    return html.Div(
        [
            _protocols_bar(cfg),
            dbc.Row(
                [
                    dbc.Col(_card_structure(), md=6),
                    dbc.Col(_card_method(), md=6),
                ],
                className="g-3 mt-1",
            ),
            dbc.Row(
                [
                    dbc.Col(_card_task(), md=6),
                    dbc.Col(_card_submission(cluster_options), md=6),
                ],
                className="g-3",
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Preview (define.inp)", className="card-title"),
                                html.Pre(id="preview-control",
                                         style={"maxHeight": "300px", "overflow": "auto",
                                                "fontSize": "0.78rem"}),
                            ]
                        ),
                        className="shadow-sm",
                    ),
                ),
                className="g-3",
            ),
            dbc.Row(
                dbc.Col(
                    html.Div(
                        [
                            dbc.Button("Preview define.inp", id="btn-preview",
                                       color="secondary", outline=True, className="me-2"),
                            dbc.Button([html.I(className="bi bi-rocket-takeoff me-1"),
                                        "Build & submit"],
                                       id="btn-submit", color="success"),
                        ],
                        className="text-end",
                    )
                ),
                className="g-3 mt-2",
            ),
        ]
    )


def _card_structure() -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5([html.I(className="bi bi-diagram-3 me-2"), "Structure"],
                        className="card-title"),
                dcc.Upload(
                    id="upload-structure",
                    children=html.Div(
                        ["Drop structure file or ", html.A("browse")],
                        className="text-center text-muted",
                    ),
                    style={
                        "borderWidth": "1px", "borderStyle": "dashed",
                        "borderRadius": "8px", "padding": "1.5rem",
                        "cursor": "pointer",
                    },
                    multiple=False,
                ),
                html.Small(
                    f"Accepted: {', '.join(SUPPORTED_INPUT_FORMATS)}",
                    className="text-muted",
                ),
                html.Div(id="structure-summary", className="mt-2"),
                dbc.Row(
                    [
                        dbc.Col(dbc.Label("Charge"), width=4),
                        dbc.Col(dbc.Input(id="inp-charge", type="number", value=0, step=1), width=2),
                        dbc.Col(dbc.Label("Multiplicity"), width=4),
                        dbc.Col(dbc.Input(id="inp-mult", type="number", value=1, min=1, step=1), width=2),
                    ],
                    className="mt-3 g-2 align-items-center",
                ),
            ]
        ),
        className="shadow-sm h-100",
    )


def _card_method() -> dbc.Card:
    from backend.turbomole_io import available_dispersions
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5([html.I(className="bi bi-sliders me-2"), "Method"],
                        className="card-title"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Functional"),
                                dcc.Dropdown(
                                    id="dd-functional",
                                    options=[{"label": f, "value": f}
                                             for f in available_functionals()],
                                    value="BP86", clearable=False,
                                ),
                            ],
                            md=7,
                        ),
                        dbc.Col(
                            [
                                dbc.Label([
                                    "Dispersion ",
                                    html.Span("(optional)",
                                              className="text-muted small"),
                                ]),
                                dcc.Dropdown(
                                    id="dd-dispersion",
                                    options=[{"label": lab, "value": val}
                                             for val, lab in available_dispersions()],
                                    value="none", clearable=False,
                                ),
                                dbc.Tooltip(
                                    "Adds an empirical dispersion correction "
                                    "(D3 / D3-BJ / D4) by inserting the "
                                    "appropriate $disp* keyword into the control "
                                    "file before ridft runs. Highly recommended "
                                    "for non-covalent interactions, organometallics, "
                                    "and large/flexible systems.",
                                    target="dd-dispersion",
                                    delay={"show": 400, "hide": 100},
                                ),
                            ],
                            md=5,
                        ),
                    ],
                    className="g-2",
                ),
                dbc.Label("Basis set", className="mt-3"),
                dcc.Dropdown(
                    id="dd-basis",
                    options=[{"label": b, "value": b} for b in available_basis_sets()],
                    value="def2-SVP", clearable=False,
                ),
                dbc.Checklist(
                    options=[{"label": " Use RI (recommended for DFT)", "value": "ri"}],
                    value=["ri"], id="chk-ri", switch=True, className="mt-3",
                ),
                dbc.Label("SCF grid", className="mt-2"),
                dcc.Dropdown(
                    id="dd-grid",
                    options=[{"label": g, "value": g} for g in ("m3", "m4", "m5")],
                    value="m4", clearable=False,
                ),
            ]
        ),
        className="shadow-sm h-100",
    )


def _card_task() -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5([html.I(className="bi bi-list-task me-2"), "Task"],
                        className="card-title"),
                dbc.RadioItems(
                    id="rad-task",
                    options=[
                        {"label": " Single point",       "value": "single_point"},
                        {"label": " Geometry optimization", "value": "optimization"},
                        {"label": " Frequencies (aoforce)", "value": "frequencies"},
                        {"label": " Ab initio MD (frog)", "value": "aimd"},
                    ],
                    value="single_point",
                    inline=False,
                ),
                _aimd_params_block(),
            ]
        ),
        className="shadow-sm h-100",
    )


def _aimd_params_block() -> html.Div:
    """Collapsible parameter group rendered only when the AIMD task is
    selected. Drives the contents of mdprep.inp on the cluster.

    Field order mirrors the mdprep menu walked by the renderer:
      4) distance constraints (switch + algorithm + textarea)
      5) initial velocities  → temperature (K)
      6) timestep            → a.u. (with live fs equivalent)
      7) number of MD steps

    A separator at the bottom marks the space reserved for the
    additional AIMD options that will be added next.
    """
    return html.Div(
        [
            html.Hr(),
            html.H6(
                [html.I(className="bi bi-broadcast-pin me-2"),
                 "Ab initio MD parameters"],
                className="text-muted small mb-2",
            ),

            # --- (4) Distance constraints --------------------------------
            dbc.Checklist(
                options=[{"label": " Apply distance constraints", "value": "on"}],
                value=[],
                id="chk-aimd-constraints",
                switch=True,
            ),
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(dbc.Label("Algorithm"), width=4),
                            dbc.Col(
                                dcc.Dropdown(
                                    id="dd-aimd-constraint-alg",
                                    options=[{"label": a, "value": a}
                                             for a in available_constraint_algorithms()],
                                    value="shake",
                                    clearable=False,
                                ),
                                width=8,
                            ),
                        ],
                        className="g-2 mt-2 align-items-center",
                    ),
                    dbc.Label(
                        ["Constraints ",
                         html.Span("(at1 at2 distance_Å; separated by ';')",
                                   className="text-muted small")],
                        className="mt-2",
                    ),
                    dbc.Textarea(
                        id="inp-aimd-constraints",
                        placeholder="e.g. 1 2 1.09; 3 4 1.54",
                        rows=2,
                    ),
                    dbc.Tooltip(
                        "Atom indices are 1-based (same as Turbomole's $coord). "
                        "Distances are entered in Angstrom and converted to Bohr "
                        "internally before being written to mdprep.inp.",
                        target="inp-aimd-constraints",
                        delay={"show": 400, "hide": 100},
                    ),
                ],
                id="aimd-constraints-group",
                style={"display": "none"},
            ),

            # --- (5) Temperature ----------------------------------------
            dbc.Row(
                [
                    dbc.Col(dbc.Label("Temperature (K)"), width=4),
                    dbc.Col(
                        dbc.Input(id="inp-aimd-T", type="number",
                                  value=300, min=0.01, step=1),
                        width=8,
                    ),
                ],
                className="g-2 mt-3 align-items-center",
            ),

            # --- (6) Timestep -------------------------------------------
            dbc.Row(
                [
                    dbc.Col(dbc.Label("Timestep (a.u.)"), width=4),
                    dbc.Col(
                        html.Div(
                            [
                                dbc.Input(id="inp-aimd-dt-au", type="number",
                                          value=80.0, min=0.01, step=0.1,
                                          style={"display": "inline-block",
                                                 "width": "8rem",
                                                 "verticalAlign": "middle"}),
                                html.Span(id="aimd-dt-fs-label",
                                          className="text-muted small ms-2"),
                            ],
                            className="d-flex align-items-center",
                        ),
                        width=8,
                    ),
                ],
                className="g-2 mt-2 align-items-center",
            ),

            # --- (7) Number of MD steps ---------------------------------
            dbc.Row(
                [
                    dbc.Col(dbc.Label("MD steps"), width=4),
                    dbc.Col(
                        dbc.Input(id="inp-aimd-steps", type="number",
                                  value=256, min=1, step=1),
                        width=8,
                    ),
                ],
                className="g-2 mt-2 align-items-center",
            ),

            # --- Space reserved for additional AIMD options ------------
            html.Hr(className="mt-3 mb-2"),
            # more AIMD options below

        ],
        id="aimd-params", style={"display": "none"},
    )


def _card_submission(cluster_options: list) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5([html.I(className="bi bi-hdd-network me-2"), "Submission"],
                        className="card-title"),
                dbc.Label("Job name"),
                dbc.Input(id="inp-job-name", placeholder="e.g. ferrocene_opt"),
                dbc.Label("Cluster", className="mt-2"),
                dcc.Dropdown(
                    id="dd-cluster", options=cluster_options,
                    value=cluster_options[0]["value"] if cluster_options else None,
                    clearable=False,
                ),
                dbc.Row(
                    [
                        dbc.Col([dbc.Label("Partition"),
                                 dbc.Input(id="inp-partition")], md=6),
                        dbc.Col([dbc.Label("Walltime"),
                                 dbc.Input(id="inp-walltime")], md=6),
                    ],
                    className="g-2 mt-2",
                ),
                dbc.Row(
                    [
                        dbc.Col([dbc.Label("Nodes"),
                                 dbc.Input(id="inp-nodes", type="number", min=1)], md=4),
                        dbc.Col([dbc.Label("ntasks"),
                                 dbc.Input(id="inp-ntasks", type="number", min=1)], md=4),
                        dbc.Col([dbc.Label("Memory"),
                                 dbc.Input(id="inp-mem")], md=4),
                    ],
                    className="g-2 mt-2",
                ),
                # Optional SLURM reservation. When empty no directive is emitted.
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label([
                                    "Reservation ",
                                    html.Span("(optional)", className="text-muted small"),
                                ]),
                                dbc.Input(
                                    id="inp-reservation",
                                    placeholder="leave empty for none",
                                ),
                                dbc.Tooltip(
                                    "If your SLURM admin gave you a reservation "
                                    "name, enter it here and the script will "
                                    "include #SBATCH --reservation=<name>. "
                                    "Leave empty to omit the directive.",
                                    target="inp-reservation",
                                    placement="top",
                                    delay={"show": 400, "hide": 100},
                                ),
                            ],
                            md=12,
                        ),
                    ],
                    className="g-2 mt-2",
                ),
            ]
        ),
        className="shadow-sm h-100",
    )


# ---------------------------------------------------------------------------
# Tab 2 — Job manager
# ---------------------------------------------------------------------------

def _tab_jobs() -> html.Div:
    return html.Div(
        [
            dbc.Alert(
                [
                    html.I(className="bi bi-info-circle me-2"),
                    "Click a row to see job details. Use ",
                    html.B("Refresh status"),
                    " to query SLURM. ",
                    html.Em("First visit to this tab auto-refreshes once."),
                ],
                color="info",
                className="mt-3 py-2 small",
            ),
            html.Div(
                [
                    dbc.Button(
                        [html.I(className="bi bi-arrow-clockwise me-1"),
                         "Refresh status"],
                        id="btn-refresh-jobs",
                        color="primary",
                    ),
                    html.Span(
                        id="jobs-last-updated",
                        className="ms-3 text-muted small",
                    ),
                ],
                className="mb-3 d-flex align-items-center",
            ),
            html.Div(id="jobs-table-container"),
        ]
    )


# ---------------------------------------------------------------------------
# Tab 3 — Clusters
# ---------------------------------------------------------------------------

def _tab_clusters(cfg: AppConfig) -> html.Div:
    cards = []
    for name, c in cfg.clusters.items():
        cards.append(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5(name, className="card-title"),
                            html.Div([html.Strong("Host: "), f"{c.user}@{c.host}:{c.port}"]),
                            html.Div([html.Strong("Workdir: "), html.Code(c.remote_workdir)]),
                            html.Div([html.Strong("Turbomole version: "),
                                      dbc.Badge(c.turbomole_version, color="primary",
                                                className="ms-1")]),
                            html.Div([html.Strong("Default partition: "), c.default_partition]),
                            html.Div([html.Strong("Modules: "),
                                      ", ".join(c.module_load) or html.Em("(none)")]),
                            html.Div([html.Strong("env_setup: "),
                                      html.Em("(none)") if not c.env_setup
                                      else html.Code(f"{len(c.env_setup)} line(s)")]),
                            dbc.Button([html.I(className="bi bi-plug me-1"), "Test connection"],
                                       id={"type": "btn-test-cluster", "name": name},
                                       color="primary", outline=True, size="sm",
                                       className="mt-2"),
                            html.Span(id={"type": "test-cluster-out", "name": name},
                                      className="ms-2 small"),
                        ]
                    ),
                    className="shadow-sm h-100",
                ),
                md=6, className="mt-3",
            )
        )
    return html.Div(dbc.Row(cards, className="g-3"))


# ---------------------------------------------------------------------------
# Tab 4 — Database
# ---------------------------------------------------------------------------

def _tab_database() -> html.Div:
    return html.Div(
        [
            dbc.Alert(
                [
                    html.I(className="bi bi-info-circle me-2"),
                    "Read-only view of the SQLite job store. ",
                    "Use the buttons to refresh, vacuum, or open the file in ",
                    html.B("DB Browser for SQLite"),
                    " (external, read-only).",
                ],
                color="info",
                className="mt-3 py-2 small",
            ),
            html.Div(
                [
                    dbc.Button(
                        [html.I(className="bi bi-arrow-clockwise me-1"), "Refresh table"],
                        id="btn-db-refresh", color="primary", className="me-2",
                    ),
                    dbc.Button(
                        [html.I(className="bi bi-box-arrow-up-right me-1"),
                         "Open in SQLite Browser (read-only)"],
                        id="btn-db-open-external", color="secondary",
                        outline=True, className="me-2",
                    ),
                    dbc.Button(
                        [html.I(className="bi bi-archive me-1"), "Vacuum"],
                        id="btn-db-vacuum", color="warning",
                        outline=True, className="me-2",
                    ),
                    dbc.Tooltip(
                        "Reclaim disk space after deletions. The DB is locked briefly.",
                        target="btn-db-vacuum", placement="top",
                        delay={"show": 400, "hide": 100},
                    ),
                ],
                className="mb-3 d-flex flex-wrap gap-2 align-items-center",
            ),
            html.Div(id="db-status-line", className="small text-muted mb-2"),
            html.Div(id="db-table-container"),
        ]
    )

# ---------------------------------------------------------------------------
# Tab 5 — Analysis
# ---------------------------------------------------------------------------
def _tab_analysis() -> html.Div:
    """Post-processing of downloaded jobs. Renders different content per
    task type (optimization / single_point / frequencies / aimd) and
    exposes paths + launchers for VMD / COSMOBuild / COSMOQuick."""
    return html.Div(
        [
            dbc.Alert(
                [
                    html.I(className="bi bi-info-circle me-2"),
                    "Analysis works on ",
                    html.B("locally downloaded"),
                    " jobs. Use ",
                    html.B("Download"),
                    " or ",
                    html.B("Partial"),
                    " from the Job manager to pull files first.",
                ],
                color="info",
                className="mt-3 py-2 small",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Job",
                                      className="small text-muted mb-1"),
                            dcc.Dropdown(
                                id="analysis-job-dropdown",
                                placeholder="-- select a downloaded job --",
                                clearable=True,
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(_card_external_tools(), md=6),
                ],
                className="g-3 mb-3",
            ),
            html.Div(id="analysis-content"),
        ]
    )


def _card_external_tools() -> dbc.Card:
    """Three inputs + three launch buttons for VMD / COSMOBuild / COSMOQuick.

    Paths are persisted to ~/turbomole_orchestrator/app_settings.json
    on blur. Launchers operate on the currently selected job's local
    download directory."""
    rows = []
    for key, label, default_cmd in (
        ("vmd",        "VMD",        "vmd"),
        ("cosmobuild", "COSMOBuild", "cosmobuild"),
        ("cosmoquick", "COSMOQuick", "cosmoquick"),
    ):
        rows.append(
            dbc.Row(
                [
                    dbc.Col(dbc.Label(label, className="small mb-0"),
                            md=3, className="d-flex align-items-center"),
                    dbc.Col(
                        dbc.Input(
                            id=f"inp-tool-{key}",
                            placeholder=f"absolute path or '{default_cmd}'",
                            type="text", size="sm",
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        dbc.Button(
                            [html.I(className="bi bi-play-fill me-1"),
                             "Launch"],
                            id=f"btn-launch-{key}",
                            color="primary", outline=True, size="sm",
                            className="w-100",
                        ),
                        md=3,
                    ),
                ],
                className="g-2 mb-2 align-items-center",
            )
        )
    return dbc.Card(
        dbc.CardBody(
            [
                html.H6(
                    [html.I(className="bi bi-tools me-2"),
                     "External tools"],
                    className="card-subtitle text-muted mb-2",
                ),
                html.Div(rows),
                html.Small(
                    "Leave empty to look up the command on PATH. Paths "
                    "save automatically when you leave the field.",
                    className="text-muted",
                ),
            ]
        ),
        className="shadow-sm h-100",
    )