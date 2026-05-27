"""
Callbacks for the Analysis tab.

Operates exclusively on locally-downloaded job directories. For each job
that has been downloaded (DOWNLOADED state, or any state with at least
one partial/full snapshot in cfg.download_dir), the user can:

  - optimization:  energy vs cycle chart, convergence criteria table,
                   last geometry as .xyz, full trajectory as multi-frame
                   .xyz, CSV export, gnuplot scripts
  - single_point:  parsed ridft.out summary, CSV export
  - frequencies:   IR stick plot, imaginary modes highlighted, CSV +
                   gnuplot script

In addition the tab exposes paths and launch buttons for three external
desktop tools (VMD, COSMOBuild, COSMOQuick). Paths persist in
~/.turbomole_orchestrator/app_settings.json.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import dash
from dash import Input, Output, State, ctx, dcc, html, no_update
import dash_bootstrap_components as dbc

from backend import app_settings
from backend.config import AppConfig
from backend.db import list_jobs
from backend.opt_trajectory import (
    BOHR_TO_ANGSTROM, OptTrajectory, load_trajectory, trajectory_xyz,
)
from backend.result_parser import parse_aoforce, parse_ridft
from backend.analysis_exports import (
    JobMeta,
    gnuplot_optimization, gnuplot_spectrum, gnuplot_trajectory,
    single_point_to_csv, spectrum_to_csv, trajectory_to_csv,
)


log = logging.getLogger("analysis")


# Tools that can be launched. Display name, settings field, default
# executable to look up on PATH if no absolute path is configured.
_TOOLS = [
    ("vmd",        "VMD",        "vmd"),
    ("cosmobuild", "COSMOBuild", "cosmobuild"),
    ("cosmoquick", "COSMOQuick", "cosmoquick"),
]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_analysis_callbacks(app: dash.Dash, cfg: AppConfig) -> None:

    # --- Job selector ------------------------------------------------------
    @app.callback(
        Output("analysis-job-dropdown", "options"),
        Input("main-tabs", "active_tab"),
        Input("last-action", "data"),
    )
    def _populate_analysis_jobs(_active_tab, _action):
        items = _downloadable_jobs(cfg)
        return [
            {"label": f"#{j['id']} — {j['name']}  [{j['task_type']}]  "
                      f"({j['state']})",
             "value": j["id"]}
            for j in items
        ]

    # --- Main content ------------------------------------------------------
    @app.callback(
        Output("analysis-content", "children"),
        Input("analysis-job-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _render_analysis(job_id):
        if not job_id:
            return _placeholder("Select a downloaded job above to inspect.")
        job = next((j for j in list_jobs() if j["id"] == job_id), None)
        if job is None:
            return _placeholder("Job not found.")
        local_dir = _resolve_local_dir(cfg, job)
        if local_dir is None:
            return dbc.Alert(
                ["Job #", str(job_id),
                 " has no local files yet. Download it first from the "
                 "Job manager tab."],
                color="warning",
            )

        task = job["task_type"]
        if task == "optimization":
            return _render_optimization(job, local_dir)
        if task == "single_point":
            return _render_single_point(job, local_dir)
        if task == "frequencies":
            return _render_frequencies(job, local_dir)
        if task == "aimd":
            return _render_aimd(job, local_dir)
        return dbc.Alert(
            f"Analysis view for task type '{task}' is not implemented yet.",
            color="secondary",
        )

    # --- Downloads (CSV / xyz / gnuplot) -----------------------------------
    @app.callback(
        Output("analysis-file-download", "data"),
        Input("btn-dl-last-xyz",       "n_clicks"),
        Input("btn-dl-traj-xyz",       "n_clicks"),
        Input("btn-dl-csv",            "n_clicks"),
        Input("btn-dl-gnuplot-energy", "n_clicks"),
        Input("btn-dl-gnuplot-grad",   "n_clicks"),
        Input("btn-dl-gnuplot-spec",   "n_clicks"),
        State("analysis-job-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _serve_download(_n1, _n2, _n3, _n4, _n5, _n6, job_id):
        trig = ctx.triggered_id
        if trig is None or not job_id:
            return no_update
        job = next((j for j in list_jobs() if j["id"] == job_id), None)
        if job is None:
            return no_update
        local_dir = _resolve_local_dir(cfg, job)
        if local_dir is None:
            return no_update

        meta = JobMeta(name=job["name"],
                       functional=job.get("functional"),
                       basis_set=job.get("basis_set"))

        if trig == "btn-dl-last-xyz":
            traj = load_trajectory(local_dir)
            xyz = traj.to_xyz(
                comment=(f"{job['name']} cycle={traj.n_cycles} "
                         f"E={traj.cycles[-1].scf_energy:.10f} Ha")
                if traj.cycles else job["name"]
            )
            if not xyz:
                return no_update
            return dict(
                content=xyz,
                filename=f"{job['name']}_cycle{traj.n_cycles}_last.xyz",
            )

        if trig == "btn-dl-traj-xyz":
            xyz = trajectory_xyz(local_dir, name=job["name"])
            if not xyz:
                return no_update
            return dict(
                content=xyz,
                filename=f"{job['name']}_trajectory.xyz",
            )

        if trig == "btn-dl-csv":
            return _csv_for_job(job, local_dir)

        if trig == "btn-dl-gnuplot-energy":
            traj = load_trajectory(local_dir)
            if traj.n_cycles == 0:
                return no_update
            return dict(
                content=gnuplot_optimization(
                    csv_filename=f"{job['name']}_trajectory.csv",
                    meta=meta,
                ),
                filename=f"{job['name']}_energy.gp",
            )

        if trig == "btn-dl-gnuplot-grad":
            traj = load_trajectory(local_dir)
            if traj.n_cycles == 0:
                return no_update
            return dict(
                content=gnuplot_trajectory(
                    csv_filename=f"{job['name']}_trajectory.csv",
                    meta=meta,
                ),
                filename=f"{job['name']}_gradient.gp",
            )

        if trig == "btn-dl-gnuplot-spec":
            aoforce = local_dir / "aoforce.out"
            if not aoforce.exists():
                return no_update
            return dict(
                content=gnuplot_spectrum(
                    csv_filename=f"{job['name']}_spectrum.csv",
                    meta=meta,
                ),
                filename=f"{job['name']}_spectrum.gp",
            )

        return no_update

    # --- External tools: load saved paths into the inputs ------------------
    @app.callback(
        Output("inp-tool-vmd",        "value"),
        Output("inp-tool-cosmobuild", "value"),
        Output("inp-tool-cosmoquick", "value"),
        Input("main-tabs", "active_tab"),
    )
    def _load_tool_paths(active_tab):
        if active_tab != "tab-analysis":
            return no_update, no_update, no_update
        s = app_settings.load_settings(cfg.settings_dir)
        et = s.external_tools
        return et.vmd, et.cosmobuild, et.cosmoquick

    # --- External tools: persist paths on blur (each input independently) --
    @app.callback(
        Output("last-action", "data", allow_duplicate=True),
        Input("inp-tool-vmd",        "n_blur"),
        Input("inp-tool-cosmobuild", "n_blur"),
        Input("inp-tool-cosmoquick", "n_blur"),
        State("inp-tool-vmd",        "value"),
        State("inp-tool-cosmobuild", "value"),
        State("inp-tool-cosmoquick", "value"),
        prevent_initial_call=True,
    )
    def _save_tool_paths(_b1, _b2, _b3, vmd, cosmobuild, cosmoquick):
        s = app_settings.load_settings(cfg.settings_dir)
        s.external_tools.vmd        = (vmd or "").strip()
        s.external_tools.cosmobuild = (cosmobuild or "").strip()
        s.external_tools.cosmoquick = (cosmoquick or "").strip()
        try:
            app_settings.save_settings(cfg.settings_dir, s)
            return {"ok": True, "msg": "External tool paths saved.",
                    "ts": time.time()}
        except OSError as exc:
            return {"ok": False, "msg": f"Could not save settings: {exc}",
                    "ts": time.time()}

    # --- External tools: launch buttons ------------------------------------
    @app.callback(
        Output("last-action", "data", allow_duplicate=True),
        Input("btn-launch-vmd",        "n_clicks"),
        Input("btn-launch-cosmobuild", "n_clicks"),
        Input("btn-launch-cosmoquick", "n_clicks"),
        State("analysis-job-dropdown", "value"),
        State("inp-tool-vmd",        "value"),
        State("inp-tool-cosmobuild", "value"),
        State("inp-tool-cosmoquick", "value"),
        prevent_initial_call=True,
    )
    def _launch_tool(_n1, _n2, _n3, job_id, vmd_path, cb_path, cq_path):
        trig = ctx.triggered_id
        if trig is None or not job_id:
            return {"ok": False, "ts": time.time(),
                    "msg": "Select a downloaded job before launching a tool."}
        job = next((j for j in list_jobs() if j["id"] == job_id), None)
        if job is None:
            return {"ok": False, "ts": time.time(), "msg": "Job not found."}
        local_dir = _resolve_local_dir(cfg, job)
        if local_dir is None:
            return {"ok": False, "ts": time.time(),
                    "msg": "Job has no local files. Download it first."}

        mapping = {
            "btn-launch-vmd":        ("vmd",        vmd_path,
                                      _vmd_args(local_dir)),
            "btn-launch-cosmobuild": ("cosmobuild", cb_path,
                                      [str(local_dir)]),
            "btn-launch-cosmoquick": ("cosmoquick", cq_path,
                                      [str(local_dir)]),
        }
        if trig not in mapping:
            return no_update
        key, configured_path, args = mapping[trig]
        return _launch_subprocess(key, configured_path, args)


# ---------------------------------------------------------------------------
# Job discovery
# ---------------------------------------------------------------------------

def _downloadable_jobs(cfg: AppConfig) -> list[dict]:
    out: list[dict] = []
    for j in list_jobs():
        if _resolve_local_dir(cfg, j) is not None:
            out.append(j)
    return out


def _resolve_local_dir(cfg: AppConfig, job: dict) -> Path | None:
    prefix = f"job_{job['id']}_{job['name']}"
    candidates = sorted(cfg.download_dir.glob(f"{prefix}*"), reverse=True)
    full = [p for p in candidates if "_partial_" not in p.name and p.is_dir()]
    if full:
        return full[0]
    partial = [p for p in candidates if p.is_dir()]
    if partial:
        return partial[0]
    return None


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _render_optimization(job: dict, local_dir: Path) -> html.Div:
    traj = load_trajectory(local_dir)
    if traj.n_cycles == 0:
        return dbc.Alert(
            "No `energy` / `gradient` data found in the downloaded "
            "directory. The optimization may not have produced any cycle "
            "yet, or the files were excluded.", color="warning",
        )

    has_traj_xyz = (local_dir / "gradient").exists()

    energy_card = dbc.Card(
        dbc.CardBody(
            [
                html.H5([html.I(className="bi bi-graph-up me-2"),
                         "SCF energy per cycle"], className="card-title"),
                dcc.Graph(figure=_build_energy_figure(traj),
                          config={"displaylogo": False}),
            ]
        ),
        className="shadow-sm",
    )

    conv_card = dbc.Card(
        dbc.CardBody(
            [
                html.H5([html.I(className="bi bi-check2-square me-2"),
                         "Convergence"], className="card-title"),
                _build_convergence_table(traj),
                html.Hr(),
                html.Div(
                    [
                        html.Strong("Status: "),
                        dbc.Badge("Converged", color="success")
                        if traj.converged()
                        else dbc.Badge("Not converged", color="warning"),
                        html.Span(f"   cycles: {traj.n_cycles}",
                                  className="text-muted ms-3 small"),
                    ],
                    className="mt-2",
                ),
            ]
        ),
        className="shadow-sm",
    )

    geom_card = dbc.Card(
        dbc.CardBody(
            [
                html.H5([html.I(className="bi bi-box me-2"),
                         "Geometry / data exports"], className="card-title"),
                html.Div(
                    [
                        _dl_btn("btn-dl-last-xyz", "bi-download",
                                "Last geometry (.xyz)",
                                "Single-frame xyz of the last cycle"),
                        _dl_btn("btn-dl-traj-xyz", "bi-collection-play",
                                "Trajectory (.xyz)",
                                "Multi-frame xyz with every cycle",
                                disabled=not has_traj_xyz),
                        _dl_btn("btn-dl-csv", "bi-filetype-csv",
                                "Trajectory (.csv)",
                                "cycle, energy, ΔE, |grad|, |grad|max"),
                        _dl_btn("btn-dl-gnuplot-energy", "bi-graph-up-arrow",
                                "Gnuplot: energy",
                                "Script to plot ΔE vs cycle (kcal/mol)"),
                        _dl_btn("btn-dl-gnuplot-grad", "bi-bar-chart-steps",
                                "Gnuplot: gradient",
                                "Script to plot |grad| vs cycle (log y)"),
                    ],
                    className="d-flex flex-wrap gap-2 mb-2",
                ),
            ]
        ),
        className="shadow-sm",
    )

    return html.Div(
        [
            dbc.Row(
                [dbc.Col(energy_card, md=8),
                 dbc.Col(conv_card, md=4)],
                className="g-3 mt-1",
            ),
            dbc.Row(dbc.Col(geom_card), className="g-3"),
        ]
    )


def _render_single_point(job: dict, local_dir: Path) -> html.Div:
    ridft = local_dir / "ridft.out"
    if not ridft.exists():
        return dbc.Alert("No ridft.out in the downloaded directory.",
                         color="warning")
    summary = parse_ridft(ridft)
    items = [_kv(k, v) for k, v in summary.as_display_dict().items()]
    if not items:
        items = [html.Em("ridft.out present but no parseable fields.")]
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5([html.I(className="bi bi-lightning-charge me-2"),
                         "Single-point results"], className="card-title"),
                html.Div(items),
                html.Hr(),
                html.Div(
                    _dl_btn("btn-dl-csv", "bi-filetype-csv",
                            "Summary (.csv)",
                            "Parsed key/value table"),
                ),
            ]
        ),
        className="shadow-sm",
    )


def _render_frequencies(job: dict, local_dir: Path) -> html.Div:
    aoforce = local_dir / "aoforce.out"
    if not aoforce.exists():
        return dbc.Alert("No aoforce.out in the downloaded directory.",
                         color="warning")
    summary = parse_aoforce(aoforce)
    if not summary.frequencies_cm1:
        return dbc.Alert("aoforce.out present but no frequencies parsed.",
                         color="warning")

    items = [_kv(k, v) for k, v in summary.as_display_dict().items()]
    n_imag = summary.n_imaginary or 0

    alert = (dbc.Alert(
        [html.I(className="bi bi-exclamation-triangle me-2"),
         f"{n_imag} imaginary mode(s) detected — structure is NOT a minimum."],
        color="warning", className="py-2 small",
    ) if n_imag > 0 else html.Div())

    exports = html.Div(
        [
            _dl_btn("btn-dl-csv", "bi-filetype-csv",
                    "Spectrum (.csv)",
                    "mode, frequency, kind"),
            _dl_btn("btn-dl-gnuplot-spec", "bi-graph-up-arrow",
                    "Gnuplot: spectrum",
                    "Script to plot the stick spectrum"),
        ],
        className="d-flex flex-wrap gap-2 mt-2",
    )

    return html.Div(
        [
            alert,
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5([html.I(className="bi bi-soundwave me-2"),
                                             "Vibrational spectrum"],
                                            className="card-title"),
                                    dcc.Graph(
                                        figure=_build_spectrum_figure(summary),
                                        config={"displaylogo": False},
                                    ),
                                ]
                            ),
                            className="shadow-sm",
                        ),
                        md=8,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Thermochemistry",
                                            className="card-title"),
                                    html.Div(items),
                                    html.Hr(),
                                    exports,
                                ]
                            ),
                            className="shadow-sm",
                        ),
                        md=4,
                    ),
                ],
                className="g-3 mt-1",
            ),
        ]
    )


def _render_aimd(job: dict, local_dir: Path) -> html.Div:
    """AIMD reuses the trajectory export from gradient when frog leaves
    geometry snapshots there. If not, only the .xyz download is offered."""
    has_traj = (local_dir / "gradient").exists()
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5([html.I(className="bi bi-broadcast-pin me-2"),
                         "AIMD trajectory"], className="card-title"),
                html.P("Download the recorded trajectory for visualization.",
                       className="text-muted small"),
                html.Div(
                    [
                        _dl_btn("btn-dl-traj-xyz", "bi-collection-play",
                                "Trajectory (.xyz)",
                                "Multi-frame xyz", disabled=not has_traj),
                        _dl_btn("btn-dl-last-xyz", "bi-download",
                                "Last frame (.xyz)",
                                "Single-frame xyz of the last step",
                                disabled=not has_traj),
                    ],
                    className="d-flex flex-wrap gap-2",
                ),
            ]
        ),
        className="shadow-sm",
    )


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def _build_energy_figure(traj: OptTrajectory):
    cycles = [c.cycle for c in traj.cycles]
    energies = [c.scf_energy for c in traj.cycles]
    e_min = min(energies)
    rel = [(e - e_min) for e in energies]

    return {
        "data": [
            {"x": cycles, "y": energies,
             "type": "scatter", "mode": "lines+markers",
             "name": "SCF energy (Ha)",
             "line": {"color": "#58a6ff", "width": 2},
             "marker": {"size": 7},
             "hovertemplate": "cycle %{x}<br>E = %{y:.8f} Ha<extra></extra>"},
            {"x": cycles, "y": rel,
             "type": "scatter", "mode": "lines",
             "name": "ΔE vs min (Ha)",
             "yaxis": "y2",
             "line": {"color": "#ff7b72", "width": 1, "dash": "dot"},
             "hovertemplate": "cycle %{x}<br>ΔE = %{y:.6f} Ha<extra></extra>"},
        ],
        "layout": {
            "template": "plotly_dark",
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor":  "rgba(0,0,0,0)",
            "xaxis": {"title": "Optimization cycle",
                      "dtick": 1 if traj.n_cycles <= 20 else None,
                      "gridcolor": "#30363d"},
            "yaxis": {"title": "E (Ha)", "gridcolor": "#30363d"},
            "yaxis2": {"title": "ΔE vs min (Ha)", "overlaying": "y",
                       "side": "right", "showgrid": False, "type": "log"},
            "legend": {"orientation": "h", "y": -0.2},
            "margin": {"l": 60, "r": 50, "t": 20, "b": 60},
            "height": 360,
        },
    }


def _build_spectrum_figure(summary):
    freqs = summary.frequencies_cm1
    real_modes = [f for f in freqs if abs(f) >= 1.0]
    x_vals, y_vals = [], []
    for f in real_modes:
        x_vals.extend([f, f, None])
        y_vals.extend([0, 1, None])

    return {
        "data": [
            {"x": x_vals, "y": y_vals,
             "type": "scatter", "mode": "lines",
             "line": {"color": "#58a6ff", "width": 1.5},
             "name": "Modes", "hoverinfo": "skip"},
            {"x": real_modes,
             "y": [1.0] * len(real_modes),
             "type": "scatter", "mode": "markers",
             "marker": {"size": 8,
                        "color": ["#ff7b72" if f < 0 else "#58a6ff"
                                  for f in real_modes]},
             "name": "Wavenumbers",
             "hovertemplate": "%{x:.1f} cm⁻¹<extra></extra>"},
        ],
        "layout": {
            "template": "plotly_dark",
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor":  "rgba(0,0,0,0)",
            "xaxis": {"title": "Wavenumber (cm⁻¹)",
                      "gridcolor": "#30363d", "autorange": "reversed"},
            "yaxis": {"visible": False, "range": [0, 1.1]},
            "showlegend": False,
            "margin": {"l": 40, "r": 30, "t": 20, "b": 60},
            "height": 360,
        },
    }


# ---------------------------------------------------------------------------
# CSV dispatcher
# ---------------------------------------------------------------------------

def _csv_for_job(job: dict, local_dir: Path):
    task = job["task_type"]
    if task == "optimization":
        traj = load_trajectory(local_dir)
        if traj.n_cycles == 0:
            return no_update
        return dict(content=trajectory_to_csv(traj),
                    filename=f"{job['name']}_trajectory.csv")
    if task == "single_point":
        ridft = local_dir / "ridft.out"
        if not ridft.exists():
            return no_update
        return dict(content=single_point_to_csv(parse_ridft(ridft)),
                    filename=f"{job['name']}_singlepoint.csv")
    if task == "frequencies":
        aoforce = local_dir / "aoforce.out"
        if not aoforce.exists():
            return no_update
        return dict(content=spectrum_to_csv(parse_aoforce(aoforce)),
                    filename=f"{job['name']}_spectrum.csv")
    return no_update


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _build_convergence_table(traj: OptTrajectory) -> dbc.Table:
    status = traj.convergence_status()
    rows = []
    labels = {
        "energy_change": "|ΔE|  (Hartree)",
        "gradient_max":  "|grad|max  (Ha/Bohr)",
        "gradient_norm": "|grad|     (Ha/Bohr)",
    }
    for key, label in labels.items():
        rec = status[key]
        val = rec["value"]
        thr = rec["threshold"]
        ok = rec["ok"]
        if val is None:
            val_str = "—"
            badge = dbc.Badge("n/a", color="secondary")
        else:
            val_str = f"{val:.2e}"
            badge = (dbc.Badge("✓", color="success") if ok
                     else dbc.Badge("✗", color="danger"))
        rows.append(html.Tr([
            html.Td(label),
            html.Td(val_str, className="font-monospace"),
            html.Td(f"{thr:.1e}", className="font-monospace text-muted"),
            html.Td(badge),
        ]))
    header = html.Thead(html.Tr([
        html.Th("Criterion"), html.Th("Value"),
        html.Th("Threshold"), html.Th(""),
    ]))
    return dbc.Table([header, html.Tbody(rows)],
                     bordered=False, hover=False, size="sm",
                     className="mb-0 small")


def _dl_btn(btn_id: str, icon: str, label: str, tooltip: str,
            disabled: bool = False):
    btn = dbc.Button(
        [html.I(className=f"bi {icon} me-1"), label],
        id=btn_id, color="primary", outline=True, size="sm",
        disabled=disabled,
    )
    return html.Span(
        [btn, dbc.Tooltip(tooltip, target=btn_id, placement="top",
                          delay={"show": 400, "hide": 100})]
    )


def _placeholder(msg: str) -> html.Div:
    return dbc.Alert(msg, color="secondary", className="mt-3")


def _kv(key, value) -> html.Div:
    return html.Div(
        [html.Strong(f"{key}: ", className="me-1"), str(value)],
        className="mb-1",
    )


# ---------------------------------------------------------------------------
# External tool launching
# ---------------------------------------------------------------------------

def _vmd_args(local_dir: Path) -> list[str]:
    """Prefer the trajectory .xyz if Turbomole produced one, otherwise
    fall back to the static `coord` file. VMD reads both."""
    traj = local_dir / "gradient"
    coord = local_dir / "coord"
    if traj.exists():
        # VMD doesn't read Turbomole's `gradient` directly; convert on
        # the fly to a temp .xyz next to the data so VMD can open it.
        xyz_path = local_dir / "trajectory.xyz"
        try:
            xyz_text = trajectory_xyz(local_dir, name=local_dir.name)
            if xyz_text:
                xyz_path.write_text(xyz_text)
                return [str(xyz_path)]
        except OSError:
            pass
    if coord.exists():
        return [str(coord)]
    return [str(local_dir)]


def _resolve_binary(configured: str, default_name: str) -> str | None:
    """Return a usable command for subprocess. Configured absolute path
    wins; otherwise look up `default_name` on PATH. Returns None if
    neither resolves."""
    if configured:
        p = Path(os.path.expanduser(configured))
        if p.exists() and os.access(p, os.X_OK):
            return str(p)
        # Allow a bare command name even if not directly on disk (some
        # binaries are wrappers on PATH but in non-standard locations).
        which = shutil.which(configured)
        if which:
            return which
        return None
    return shutil.which(default_name)


def _launch_subprocess(key: str, configured_path: str | None,
                       args: list[str]) -> dict:
    display = next((d for k, d, _ in _TOOLS if k == key), key)
    default_name = next((dflt for k, _, dflt in _TOOLS if k == key), key)
    binary = _resolve_binary((configured_path or "").strip(), default_name)
    if binary is None:
        return {
            "ok": False, "ts": time.time(),
            "msg": (f"{display} not found. Set an absolute path in the "
                    "External tools section, or make sure the binary is "
                    "on your PATH."),
        }
    try:
        subprocess.Popen(
            [binary, *args],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=(platform.system() != "Windows"),
        )
    except OSError as exc:
        log.exception("Launch of %s failed", display)
        return {"ok": False, "ts": time.time(),
                "msg": f"Could not launch {display}: {exc}"}
    return {"ok": True, "ts": time.time(),
            "msg": f"Launched {display}."}