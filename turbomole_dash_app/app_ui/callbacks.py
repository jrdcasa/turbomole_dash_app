"""
Dash callbacks.
"""

from __future__ import annotations

import base64
import json
import logging
import pickle
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dash
from dash import Input, Output, State, ALL, ctx, dash_table, dcc, html, no_update
import dash_bootstrap_components as dbc

from backend.config import AppConfig
from backend.db import (
    JobRecord, insert_job, list_jobs, list_jobs_raw, get_job, update_job,
    delete_job, get_db_path, vacuum_db,
)
from backend.db_inspector import open_in_sqlite_browser
from backend import op_tracker
from backend.result_parser import parse_ridft, RidftSummary
from backend.slurm import SlurmParams, build_slurm_script
from backend.turbomole_io import (
    SUPPORTED_INPUT_FORMATS, build_control_file, load_structure_from_bytes,
    turbomole_driver_commands,
)
from remote import slurm_remote, ssh_client
from workers.poller import refresh_active_jobs, refresh_single_job


log = logging.getLogger("callbacks")
_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tm-io")


def register_callbacks(app: dash.Dash, cfg: AppConfig) -> None:
    _register_new_job_callbacks(app, cfg)
    _register_jobs_table_callbacks(app, cfg)
    _register_job_detail_callbacks(app, cfg)
    _register_cluster_test_callback(app, cfg)
    _register_db_inspector_callbacks(app, cfg)


# ===========================================================================
# Tab 1 — New job  (unchanged)
# ===========================================================================

def _register_new_job_callbacks(app: dash.Dash, cfg: AppConfig) -> None:

    @app.callback(
        Output("inp-partition", "value"),
        Output("inp-walltime",  "value"),
        Output("inp-nodes",     "value"),
        Output("inp-ntasks",    "value"),
        Output("inp-mem",       "value"),
        Input("dd-cluster", "value"),
    )
    def _fill_cluster_defaults(name):
        if not name or name not in cfg.clusters:
            return no_update, no_update, no_update, no_update, no_update
        c = cfg.clusters[name]
        return c.default_partition, c.default_time, c.default_nodes, c.default_ntasks, c.default_mem

    @app.callback(
        Output("aimd-params", "style"),
        Input("rad-task", "value"),
    )
    def _toggle_aimd(task):
        return {"display": "block"} if task == "aimd" else {"display": "none"}

    @app.callback(
        Output("job-staging-store", "data"),
        Output("structure-summary", "children"),
        Input("upload-structure", "contents"),
        State("upload-structure", "filename"),
        prevent_initial_call=True,
    )
    def _parse_structure(contents, filename):
        if contents is None:
            return no_update, no_update
        try:
            _, b64 = contents.split(",", 1)
            data = base64.b64decode(b64)
            fmt = (Path(filename).suffix.lstrip(".") or "xyz").lower()
            if fmt not in SUPPORTED_INPUT_FORMATS:
                fmt = "xyz"
            atoms = load_structure_from_bytes(data, fmt)
            payload = base64.b64encode(pickle.dumps(atoms)).decode()
            summary = dbc.Alert(
                [
                    html.I(className="bi bi-check-circle me-2"),
                    f"{filename} — {len(atoms)} atoms, formula ",
                    html.Code(atoms.get_chemical_formula()),
                ],
                color="success", className="py-2 mb-0",
            )
            return {"atoms_b64": payload, "filename": filename}, summary
        except Exception as exc:
            return no_update, dbc.Alert(
                f"Could not parse: {exc}", color="danger", className="py-2 mb-0",
            )

    @app.callback(
        Output("preview-control", "children"),
        Input("btn-preview", "n_clicks"),
        State("job-staging-store", "data"),
        State("dd-cluster", "value"),
        State("dd-functional", "value"),
        State("dd-basis", "value"),
        State("rad-task", "value"),
        State("inp-charge", "value"),
        State("inp-mult", "value"),
        State("chk-ri", "value"),
        State("dd-grid", "value"),
        State("inp-aimd-steps", "value"),
        State("inp-aimd-dt", "value"),
        State("inp-aimd-T", "value"),
        prevent_initial_call=True,
    )
    def _preview(n, staging, cluster_name,
                 functional, basis, task, charge, mult,
                 ri, grid, aimd_steps, aimd_dt, aimd_T):
        if not staging:
            return "Upload a structure first."
        tm_version = "7.8"
        if cluster_name and cluster_name in cfg.clusters:
            tm_version = cfg.clusters[cluster_name].turbomole_version

        atoms = pickle.loads(base64.b64decode(staging["atoms_b64"]))
        tmp = cfg.local_workdir / f".preview_{int(time.time())}"
        try:
            build_control_file(
                atoms, tmp,
                functional=functional, basis_set=basis, task_type=task,
                charge=int(charge or 0), multiplicity=int(mult or 1),
                grid=grid, use_ri="ri" in (ri or []),
                aimd_steps=int(aimd_steps or 500),
                aimd_timestep_fs=float(aimd_dt or 0.5),
                aimd_temperature_K=float(aimd_T or 300),
                turbomole_version=tm_version,
            )
            define_text = (tmp / "define.inp").read_text()
            header = (
                f"# Turbomole {tm_version} — define.inp\n"
                f"# (the control file is generated on the cluster\n"
                f"#  by running:  define < define.inp)\n"
                f"#\n"
            )
            return header + define_text
        except Exception as exc:
            return f"# Error: {exc}"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @app.callback(
        Output("last-action", "data", allow_duplicate=True),
        Output("main-tabs", "active_tab"),
        Input("btn-submit", "n_clicks"),
        State("job-staging-store", "data"),
        State("inp-job-name", "value"),
        State("dd-cluster", "value"),
        State("dd-functional", "value"),
        State("dd-basis", "value"),
        State("rad-task", "value"),
        State("inp-charge", "value"),
        State("inp-mult", "value"),
        State("chk-ri", "value"),
        State("dd-grid", "value"),
        State("inp-aimd-steps", "value"),
        State("inp-aimd-dt", "value"),
        State("inp-aimd-T", "value"),
        State("inp-partition", "value"),
        State("inp-walltime", "value"),
        State("inp-nodes", "value"),
        State("inp-ntasks", "value"),
        State("inp-mem", "value"),
        State("inp-reservation", "value"),
        prevent_initial_call=True,
    )
    def _submit(n, staging, job_name, cluster_name,
                functional, basis, task, charge, mult, ri, grid,
                aimd_steps, aimd_dt, aimd_T,
                partition, walltime, nodes, ntasks, mem, reservation):
        if not n:
            return no_update, no_update
        try:
            if not staging:
                raise ValueError("No structure uploaded.")
            if not job_name:
                raise ValueError("Please provide a job name.")
            cluster = cfg.clusters[cluster_name]
            atoms = pickle.loads(base64.b64decode(staging["atoms_b64"]))

            use_ri = "ri" in (ri or [])

            ts = time.strftime("%Y%m%d-%H%M%S")
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in job_name)
            local_dir = cfg.local_workdir / f"{safe_name}_{ts}"
            local_dir.mkdir(parents=True, exist_ok=False)

            build_control_file(
                atoms, local_dir,
                functional=functional, basis_set=basis, task_type=task,
                charge=int(charge or 0), multiplicity=int(mult or 1),
                grid=grid, use_ri=use_ri,
                aimd_steps=int(aimd_steps or 500),
                aimd_timestep_fs=float(aimd_dt or 0.5),
                aimd_temperature_K=float(aimd_T or 300),
                turbomole_version=cluster.turbomole_version,
            )

            # Normalize reservation: empty/whitespace -> None so the SLURM
            # builder simply omits the --reservation directive.
            reservation_clean = (reservation or "").strip() or None

            sl = SlurmParams(
                job_name=safe_name,
                partition=partition or cluster.default_partition,
                time=walltime or cluster.default_time,
                nodes=int(nodes or cluster.default_nodes),
                ntasks=int(ntasks or cluster.default_ntasks),
                mem=mem or cluster.default_mem,
                reservation=reservation_clean,
            )
            base_remote = ssh_client.expand_remote(cluster, cluster.remote_workdir)
            remote_dir = f"{base_remote.rstrip('/')}/{safe_name}_{ts}"

            script = build_slurm_script(
                sl, remote_dir, cluster.module_load,
                turbomole_driver_commands(
                    task,
                    turbomole_version=cluster.turbomole_version,
                    functional=functional,
                    basis_set=basis,
                    use_ri=use_ri,
                ),
                env_setup=cluster.env_setup,
            )
            (local_dir / "submit.slurm").write_text(script)

            rec = JobRecord(
                name=job_name, cluster=cluster_name, state="DRAFT",
                task_type=task, local_dir=str(local_dir), remote_dir=remote_dir,
                functional=functional, basis_set=basis,
                charge=int(charge or 0), multiplicity=int(mult or 1),
                submit_meta={
                    "partition": sl.partition, "time": sl.time,
                    "nodes": sl.nodes, "ntasks": sl.ntasks, "mem": sl.mem,
                    "reservation": sl.reservation,
                    "turbomole_version": cluster.turbomole_version,
                },
            )
            job_id = insert_job(rec)

            # Mark the job as "uploading" in the in-memory tracker so the
            # UI immediately shows a spinner on the row. The executor
            # thread will clear this when SFTP+sbatch finish.
            op_tracker.mark_started(job_id, "uploading")
            _EXECUTOR.submit(_upload_and_submit, cfg, job_id)

            return {
                "ok": True,
                "msg": (f"Job '{job_name}' queued for upload to {cluster_name}. "
                        "Watch the Job manager — the row updates automatically "
                        "when SFTP finishes."),
                "ts": time.time(),
            }, "tab-jobs"
        except Exception as exc:
            log.exception("Submit failed")
            return {"ok": False, "msg": f"Submit failed: {exc}", "ts": time.time()}, no_update


def _upload_and_submit(cfg: AppConfig, job_id: int) -> None:
    job = get_job(job_id)
    if job is None:
        op_tracker.mark_finished(job_id)
        return
    cluster = cfg.clusters[job["cluster"]]
    try:
        ssh_client.upload_dir(cluster, Path(job["local_dir"]), job["remote_dir"])
        update_job(job_id, state="UPLOADED")
        slurm_id = slurm_remote.submit(cluster, job["remote_dir"], "submit.slurm")
        update_job(job_id, state="SUBMITTED", slurm_id=slurm_id)
        log.info("Job %s submitted as SLURM %s", job_id, slurm_id)
    except Exception as exc:
        log.exception("Upload/submit failed for job %s", job_id)
        update_job(job_id, state="FAILED",
                   extra={"error": str(exc), "trace": traceback.format_exc()})
    finally:
        op_tracker.mark_finished(job_id)


# ===========================================================================
# Tab 2 — Job manager
# ===========================================================================

_STATE_COLORS = {
    "DRAFT": "secondary", "UPLOADED": "info", "SUBMITTED": "info",
    "PENDING": "warning", "RUNNING": "primary",
    "COMPLETED": "success", "FAILED": "danger",
    "DOWNLOADED": "success", "CLEANED": "dark",
}


def _register_jobs_table_callbacks(app: dash.Dash, cfg: AppConfig) -> None:

    @app.callback(
        Output("jobs-table-container", "children"),
        Output("jobs-last-updated", "children"),
        Output("last-action", "data", allow_duplicate=True),
        Output("tab-visited", "data"),
        Input("btn-refresh-jobs", "n_clicks"),
        Input("last-action", "data"),
        Input("main-tabs", "active_tab"),
        Input("active-ops", "data"),
        State("tab-visited", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def _render_jobs_table(_n_click, _action, active_tab, active_ops, visited):
        """Re-render the jobs table.

        Auto-refresh against SLURM happens in two cases:
          1) The user pressed the Refresh button explicitly.
          2) The user just entered the 'Job manager' tab for the FIRST time
             in this browser session (A1: refresh on tab focus).
        """
        triggered = ctx.triggered_id
        feedback = no_update
        visited = visited or {}
        active_ops = active_ops or {}

        first_visit_to_jobs = (
            triggered == "main-tabs"
            and active_tab == "tab-jobs"
            and not visited.get("tab-jobs")
        )
        explicit_refresh = (triggered == "btn-refresh-jobs")

        if explicit_refresh or first_visit_to_jobs:
            report = refresh_active_jobs(cfg)
            if explicit_refresh or report.updated > 0 or report.errors > 0:
                feedback = {
                    "ok":  report.errors == 0,
                    "msg": report.summary(),
                    "ts":  time.time(),
                }
            if report.details:
                log.info("Refresh details: %s", report.details)

        if first_visit_to_jobs:
            visited = {**visited, "tab-jobs": True}

        jobs = list_jobs()
        if not jobs:
            return (
                dbc.Alert("No jobs yet. Submit one from the 'New job' tab.",
                          color="secondary"),
                "",
                feedback,
                visited,
            )

        header = html.Thead(html.Tr([
            html.Th("ID"), html.Th("Name"), html.Th("Cluster"),
            html.Th("SLURM"), html.Th("Task"), html.Th("State"),
            html.Th("Created"), html.Th("Actions"),
        ]))
        rows = [_job_row(j, active_ops, cfg) for j in jobs]
        table = dbc.Table(
            [header, html.Tbody(rows)],
            bordered=False, hover=True, responsive=True, striped=True,
            className="align-middle jobs-table",
        )
        return table, f"Last refreshed at {time.strftime('%H:%M:%S')}", feedback, visited

    # ----- Destructive actions: open confirmation modal first --------------
    @app.callback(
        Output("confirm-modal", "is_open", allow_duplicate=True),
        Output("confirm-modal-body", "children"),
        Output("pending-action", "data"),
        Input({"type": "job-action", "action": ALL, "id": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def _maybe_open_confirm(_clicks):
        trig = ctx.triggered_id
        if not trig or not any(_clicks):
            return no_update, no_update, no_update
        action = trig["action"]
        if action not in ("delete-remote", "clean-job"):
            return no_update, no_update, no_update
        job_id = int(trig["id"])
        job = get_job(job_id)
        if job is None:
            return no_update, no_update, no_update

        if action == "delete-remote":
            body = [
                html.P([
                    "Remove the remote directory for job ",
                    html.B(f"#{job_id} ({job['name']})"),
                    " on cluster ", html.Code(job["cluster"]), "?",
                ]),
                html.P(html.Code(job["remote_dir"]),
                       className="small text-muted"),
                html.P("Database row and local files will be kept.",
                       className="mb-0"),
            ]
        else:
            body = [
                html.P([
                    "Full cleanup of job ",
                    html.B(f"#{job_id} ({job['name']})"),
                    ":",
                ]),
                html.Ul([
                    html.Li("Database row will be deleted"),
                    html.Li(["Local workspace ", html.Code(job["local_dir"])]),
                    html.Li("Any partial/full downloads of this job"),
                    html.Li(["Remote directory ", html.Code(job["remote_dir"]),
                             " on ", html.Code(job["cluster"])]),
                ]),
                html.P("This cannot be undone.",
                       className="text-danger fw-bold mb-0"),
            ]
        return True, body, {"action": action, "job_id": job_id}

    @app.callback(
        Output("confirm-modal", "is_open", allow_duplicate=True),
        Output("last-action", "data", allow_duplicate=True),
        Input("btn-confirm-cancel", "n_clicks"),
        Input("btn-confirm-ok", "n_clicks"),
        State("pending-action", "data"),
        prevent_initial_call=True,
    )
    def _resolve_confirm(_cancel_n, _ok_n, pending):
        trig = ctx.triggered_id
        if trig == "btn-confirm-cancel" or pending is None:
            return False, no_update
        action = pending["action"]
        job_id = pending["job_id"]
        try:
            if action == "delete-remote":
                msg = _action_delete_remote(cfg, job_id)
            elif action == "clean-job":
                msg = _action_clean_job(cfg, job_id)
            else:
                msg = f"Unknown pending action: {action}"
            return False, {"ok": True, "msg": msg, "ts": time.time()}
        except Exception as exc:
            log.exception("Confirmed action %s failed", action)
            return False, {"ok": False, "msg": f"{action} failed: {exc}",
                           "ts": time.time()}

    @app.callback(
        Output("last-action", "data", allow_duplicate=True),
        Input({"type": "job-action", "action": ALL, "id": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def _handle_action(_clicks):
        trig = ctx.triggered_id
        if not trig or not any(_clicks):
            return no_update
        action = trig["action"]
        if action in ("delete-remote", "clean-job"):
            return no_update
        job_id = int(trig["id"])
        try:
            if action in ("download-final", "download-and-clean"):
                refresh_single_job(cfg, job_id)

            if action == "download-partial":
                op_tracker.mark_started(job_id, "downloading")
                _EXECUTOR.submit(_action_download, cfg, job_id,
                                 partial=True, clean=False)
                msg = f"Streaming partial outputs for job {job_id}..."
            elif action == "download-final":
                op_tracker.mark_started(job_id, "downloading")
                _EXECUTOR.submit(_action_download, cfg, job_id,
                                 partial=False, clean=False)
                msg = f"Downloading finished outputs for job {job_id}..."
            elif action == "download-and-clean":
                op_tracker.mark_started(job_id, "downloading")
                _EXECUTOR.submit(_action_download, cfg, job_id,
                                 partial=False, clean=True)
                msg = f"Downloading + cleaning remote dir for job {job_id}..."
            elif action == "cancel":
                _action_cancel(cfg, job_id)
                msg = f"Cancelled job {job_id}."
            elif action == "delete-db":
                msg = _action_delete_db(job_id)
            else:
                msg = f"Unknown action: {action}"
            return {"ok": True, "msg": msg, "ts": time.time()}
        except Exception as exc:
            log.exception("Action %s on job %s failed", action, job_id)
            return {"ok": False, "msg": f"{action} failed: {exc}",
                    "ts": time.time()}

    # ----- In-flight operations: sync from tracker to store, drive the
    # local Interval, and refresh the table when ops change ---------------
    @app.callback(
        Output("active-ops", "data"),
        Output("active-ops-tick", "disabled"),
        Input("active-ops-tick", "n_intervals"),
        Input("last-action", "data"),
        State("active-ops", "data"),
    )
    def _sync_active_ops(_tick, _action, current):
        """Reads op_tracker (process memory) and pushes its snapshot to a
        dcc.Store. When there are no ops in flight, the polling Interval
        is disabled to avoid useless work.

        Triggered by:
          - the Interval itself while ops are active (auto-refresh)
          - last-action data changes (a submit or download was just
            kicked off — gives the store an instant update without
            waiting for the next tick).
        """
        snapshot = op_tracker.active()
        # Normalize keys to strings (JSON dict keys cannot be ints)
        new_data = {str(k): v for k, v in snapshot.items()}
        disabled = not snapshot
        # If nothing changed AND interval is already in the right state,
        # signal no_update so the table doesn't redraw needlessly.
        if new_data == (current or {}):
            return no_update, disabled
        return new_data, disabled

    @app.callback(
        Output("toast-container", "children"),
        Input("last-action", "data"),
        prevent_initial_call=True,
    )
    def _show_toast(data):
        if not data:
            return no_update
        return dbc.Toast(
            data.get("msg", ""),
            header="Turbomole Orchestrator",
            icon="success" if data.get("ok") else "danger",
            duration=4500, is_open=True, dismissable=True,
            style={"minWidth": "320px"},
        )


def _job_row(j: dict, active_ops: dict | None = None, cfg=None) -> html.Tr:
    state = j["state"]
    job_id = j["id"]
    active_ops = active_ops or {}

    # If this job has an in-flight upload/download, show a transient
    # badge with a spinner INSTEAD of the persisted DB state. The
    # transient state lives only in the frontend store and disappears
    # when the executor thread finishes.
    op = active_ops.get(str(job_id))
    if op:
        kind = op.get("kind", "")
        label = "UPLOADING..." if kind == "uploading" else "DOWNLOADING..."
        color = "info" if kind == "uploading" else "primary"
        badge = dbc.Badge(
            [dbc.Spinner(size="sm", color="light",
                         spinner_style={"width": "0.8rem", "height": "0.8rem"},
                         spinnerClassName="me-1"),
             label],
            color=color, className="px-2 py-1 d-inline-flex align-items-center",
        )
    else:
        badge = dbc.Badge(state, color=_STATE_COLORS.get(state, "secondary"),
                          className="px-2 py-1")

    actions = []

    if state in ("RUNNING", "PENDING", "SUBMITTED", "UPLOADED"):
        actions.append(_action_btn(
            "download-partial", job_id, "bi-download", "Partial", "info",
            tooltip="Download a snapshot of the current remote directory.",
        ))
        actions.append(_action_btn(
            "cancel", job_id, "bi-x-circle", "Cancel", "danger",
            tooltip="Send scancel to SLURM.",
        ))
    if state in ("COMPLETED", "FAILED"):
        actions.append(_action_btn(
            "download-final", job_id, "bi-download", "Download", "success",
            tooltip="Pull all files. Remote directory preserved.",
        ))
        actions.append(_action_btn(
            "download-and-clean", job_id, "bi-cloud-download",
            "DL + clean", "warning",
            tooltip="Download all files AND remove the remote directory.",
        ))
    if state in ("DOWNLOADED", "CLEANED"):
        actions.append(_path_with_copy(cfg, j))

    if state not in ("UPLOADED", "SUBMITTED", "PENDING", "RUNNING"):
        actions.append(_action_btn(
            "delete-db", job_id, "bi-database-x",
            "Delete from DB", "secondary",
            tooltip="Remove this job from the local database. Files preserved.",
        ))
        if state in ("DRAFT", "COMPLETED", "FAILED", "DOWNLOADED"):
            actions.append(_action_btn(
                "delete-remote", job_id, "bi-cloud-slash",
                "Delete Remote Only", "warning",
                tooltip="Remove only the directory on the cluster. Asks for confirmation.",
            ))
        actions.append(_action_btn(
            "clean-job", job_id, "bi-trash3",
            "Clean Job", "danger",
            tooltip="Full cleanup. Asks for confirmation.",
        ))

    # Cells *outside* the Actions column carry the row id so a click on
    # them opens the detail modal. The Actions column is excluded so its
    # buttons don't double-trigger.
    row_id = {"type": "job-row", "id": str(job_id)}
    clickable = {"cursor": "pointer"}

    return html.Tr([
        html.Td(job_id, id={**row_id, "col": "id"},      style=clickable),
        html.Td(j["name"], id={**row_id, "col": "name"}, style=clickable),
        html.Td(j["cluster"], id={**row_id, "col": "cl"}, style=clickable),
        html.Td(j.get("slurm_id") or "—",
                id={**row_id, "col": "sid"}, style=clickable),
        html.Td(j["task_type"], id={**row_id, "col": "task"}, style=clickable),
        html.Td(badge, id={**row_id, "col": "state"}, style=clickable),
        html.Td(time.strftime("%Y-%m-%d %H:%M",
                              time.localtime(j["created_at"])),
                id={**row_id, "col": "ts"}, style=clickable),
        html.Td(html.Div(actions, className="d-flex gap-1 flex-wrap")),
    ])


def _action_btn(action: str, job_id: int, icon: str, label: str,
                color: str, tooltip: str | None = None):
    btn_id = {"type": "job-action", "action": action, "id": str(job_id)}
    btn = dbc.Button(
        [html.I(className=f"bi {icon} me-1"), label],
        id=btn_id, size="sm", color=color, outline=True,
    )
    if tooltip:
        return html.Span(
            [
                btn,
                dbc.Tooltip(tooltip, target=btn_id, placement="top",
                            delay={"show": 400, "hide": 100}),
            ]
        )
    return btn


def _best_local_path(cfg: AppConfig, job: dict) -> str:
    """Return the most useful local path for this job.

    Prefer the download dir (which contains the actual results pulled from
    the cluster) over the local workspace (which may have been deleted by
    a Clean Job action, and in any case only contained the inputs).
    """
    prefix = f"job_{job['id']}_{job['name']}"
    candidates = sorted(cfg.download_dir.glob(f"{prefix}*"), reverse=True)
    # Skip "_partial_*" snapshots when a full download exists
    full = [p for p in candidates if "_partial_" not in p.name]
    if full and full[0].exists():
        return str(full[0])
    if candidates and candidates[0].exists():
        return str(candidates[0])
    return job["local_dir"]


def _path_with_copy(cfg: AppConfig, job: dict):
    """Render the local results path with a 'Copy' button next to it.

    Browsers block file:// links from http:// pages for security, so we
    don't link directly. Instead we show the path and offer one-click
    copy to the clipboard, so the user can paste it in their file
    manager or terminal.
    """
    path = _best_local_path(cfg, job)
    short = path
    if len(path) > 48:
        # Show only the last segments so the row stays compact
        short = ".../" + "/".join(path.split("/")[-2:])

    target_id = {"type": "copy-path-target", "id": str(job["id"])}
    btn_id    = {"type": "copy-path-btn",    "id": str(job["id"])}

    return html.Span(
        [
            html.Code(
                short,
                id=target_id,
                # Full path on hover via native browser tooltip
                title=path,
                className="small me-1",
                style={"userSelect": "all"},
            ),
            dcc.Clipboard(
                target_id=target_id,
                content=path,
                id=btn_id,
                title="Copy path to clipboard",
                style={"display": "inline-block", "cursor": "pointer",
                       "verticalAlign": "middle", "fontSize": "0.9rem"},
            ),
        ],
        className="d-inline-flex align-items-center me-2",
    )


# ===========================================================================
# Action implementations
# ===========================================================================

def _action_download(cfg: AppConfig, job_id: int, *, partial: bool, clean: bool) -> None:
    job = get_job(job_id)
    if not job:
        op_tracker.mark_finished(job_id)
        return
    cluster = cfg.clusters.get(job["cluster"])
    if cluster is None:
        update_job(job_id, extra={**job["extra"],
                                  "error": f"Cluster {job['cluster']} not configured"})
        op_tracker.mark_finished(job_id)
        return
    dest = cfg.download_dir / f"job_{job_id}_{job['name']}"
    if partial:
        dest = dest.with_name(dest.name + "_partial_" + time.strftime("%H%M%S"))
    try:
        ssh_client.download_dir(cluster, job["remote_dir"], dest, partial=partial)
        log.info("Downloaded job %s -> %s", job_id, dest)
        if not partial:
            update_job(job_id, state="DOWNLOADED",
                       extra={**job["extra"], "downloaded_to": str(dest)})
        if clean:
            ssh_client.remove_remote_dir(cluster, job["remote_dir"])
            update_job(job_id, state="CLEANED")
    except Exception as exc:
        log.exception("Download failed for job %s", job_id)
        update_job(job_id, extra={**job["extra"],
                                  "error": f"download failed: {exc}"})
    finally:
        op_tracker.mark_finished(job_id)


def _action_cancel(cfg: AppConfig, job_id: int) -> None:
    job = get_job(job_id)
    if not job or not job.get("slurm_id"):
        return
    cluster = cfg.clusters.get(job["cluster"])
    if cluster is None:
        return
    slurm_remote.cancel(cluster, job["slurm_id"])
    update_job(job_id, state="FAILED",
               extra={**job["extra"], "cancelled_by_user": True})


def _action_delete_db(job_id: int) -> str:
    job = get_job(job_id)
    if job is None:
        return f"Job {job_id} not found."
    if job["state"] in ("UPLOADED", "SUBMITTED", "PENDING", "RUNNING"):
        return (f"Job {job_id} is still active ({job['state']}). "
                "Cancel it before deleting.")
    if delete_job(job_id):
        log.info("Deleted DB row for job %s (files preserved)", job_id)
        return (f"Job {job_id} removed from database. "
                "Local and remote files preserved.")
    return f"Job {job_id} could not be removed."


def _action_delete_remote(cfg: AppConfig, job_id: int) -> str:
    job = get_job(job_id)
    if job is None:
        return f"Job {job_id} not found."
    if job["state"] in ("UPLOADED", "SUBMITTED", "PENDING", "RUNNING"):
        return (f"Job {job_id} is still active ({job['state']}). "
                "Cancel it before deleting.")
    if job["state"] == "CLEANED":
        return f"Job {job_id} remote dir was already cleaned."
    cluster = cfg.clusters.get(job["cluster"])
    if cluster is None:
        return f"Cluster '{job['cluster']}' is no longer configured."
    ssh_client.remove_remote_dir(cluster, job["remote_dir"])
    update_job(job_id, state="CLEANED",
               extra={**job["extra"], "remote_deleted_by_user": True})
    log.info("Deleted remote dir for job %s: %s", job_id, job["remote_dir"])
    return f"Remote directory for job {job_id} removed."


def _action_clean_job(cfg: AppConfig, job_id: int) -> str:
    job = get_job(job_id)
    if job is None:
        return f"Job {job_id} not found."
    if job["state"] in ("UPLOADED", "SUBMITTED", "PENDING", "RUNNING"):
        return (f"Job {job_id} is still active ({job['state']}). "
                "Cancel it before deleting.")

    errors: list[str] = []

    if job["state"] != "CLEANED":
        cluster = cfg.clusters.get(job["cluster"])
        if cluster is not None:
            try:
                ssh_client.remove_remote_dir(cluster, job["remote_dir"])
            except Exception as exc:
                errors.append(f"remote: {exc}")

    local_dir = Path(job["local_dir"])
    if local_dir.exists():
        try:
            shutil.rmtree(local_dir)
        except Exception as exc:
            errors.append(f"local workspace: {exc}")

    prefix = f"job_{job_id}_{job['name']}"
    for entry in cfg.download_dir.glob(f"{prefix}*"):
        try:
            shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
        except Exception as exc:
            errors.append(f"download {entry.name}: {exc}")

    delete_job(job_id)

    if errors:
        return f"Job {job_id} cleaned with warnings: " + "; ".join(errors)
    return f"Job {job_id} fully cleaned (DB + local + remote)."


# ===========================================================================
# A2 — Job detail modal
# ===========================================================================

def _register_job_detail_callbacks(app: dash.Dash, cfg: AppConfig) -> None:

    @app.callback(
        Output("detail-modal", "is_open"),
        Output("detail-modal-title", "children"),
        Output("detail-modal-body", "children"),
        Input({"type": "job-row", "id": ALL, "col": ALL}, "n_clicks"),
        Input("btn-detail-close", "n_clicks"),
        prevent_initial_call=True,
    )
    def _open_or_close(_row_clicks, _close_n):
        trig = ctx.triggered_id
        if trig is None:
            return no_update, no_update, no_update
        if trig == "btn-detail-close":
            return False, no_update, no_update

        # A row cell was clicked. Build the modal content.
        if not isinstance(trig, dict) or trig.get("type") != "job-row":
            return no_update, no_update, no_update
        # Guard against the "phantom" initial click event Dash fires when
        # the pattern-matched IDs first appear in the layout.
        if not any(_row_clicks):
            return no_update, no_update, no_update

        try:
            job_id = int(trig["id"])
        except (ValueError, KeyError):
            return no_update, no_update, no_update

        job = get_job(job_id)
        if job is None:
            return True, "Job not found", html.Div("This job no longer exists.")

        title = [
            html.I(className="bi bi-folder2-open me-2"),
            f"Job #{job_id} — {job['name']} ",
            dbc.Badge(job["state"],
                      color=_STATE_COLORS.get(job["state"], "secondary"),
                      className="ms-2 align-middle"),
        ]
        body = _build_detail_body(cfg, job)
        return True, title, body


def _build_detail_body(cfg: AppConfig, job: dict) -> html.Div:
    """Assemble the modal content for one job: physical parameters,
    parsed results, resource usage and the list of available files."""
    rows: list = []

    # ----- Header info -----
    sub = job.get("submit_meta") or {}
    rows.append(_section_card(
        "Job",
        [
            _kv("Cluster", job["cluster"]),
            _kv("SLURM id", job.get("slurm_id") or "—"),
            _kv("Created", time.strftime("%Y-%m-%d %H:%M:%S",
                                         time.localtime(job["created_at"]))),
            _kv("Updated", time.strftime("%Y-%m-%d %H:%M:%S",
                                         time.localtime(job["updated_at"]))),
            _kv("Local dir", html.Code(job["local_dir"]), small=True),
            _kv("Remote dir", html.Code(job["remote_dir"]), small=True),
        ],
    ))

    # ----- Calculation parameters -----
    rows.append(_section_card(
        "Calculation",
        [
            _kv("Task", job["task_type"]),
            _kv("Functional", job.get("functional") or "—"),
            _kv("Basis", job.get("basis_set") or "—"),
            _kv("Charge", job.get("charge", 0)),
            _kv("Multiplicity", job.get("multiplicity", 1)),
            _kv("Turbomole version", sub.get("turbomole_version", "—")),
        ],
    ))

    # ----- Resources requested -----
    res_items = [
        _kv("Partition", sub.get("partition", "—")),
        _kv("Walltime", sub.get("time", "—")),
        _kv("Nodes", sub.get("nodes", "—")),
        _kv("ntasks", sub.get("ntasks", "—")),
        _kv("Memory", sub.get("mem", "—")),
    ]
    # Show reservation only when one was actually set
    if sub.get("reservation"):
        res_items.append(_kv("Reservation", sub["reservation"]))
    rows.append(_section_card("Resources requested", res_items))

    # ----- Results (parsed from ridft.out) -----
    results = _gather_results(cfg, job)
    if results is None:
        rows.append(_section_card(
            "Results",
            [html.Div(html.Em("Not available — job has not produced output yet."),
                      className="text-muted small")],
        ))
    else:
        summary, source_note = results
        items = [
            _kv(k, v)
            for k, v in summary.as_display_dict().items()
        ]
        if not items:
            items = [html.Em("Could not parse ridft.out yet.")]
        items.insert(0, html.Div(source_note,
                                 className="small text-muted mb-2"))
        rows.append(_section_card("Results", items))

    # ----- Files -----
    files = _gather_files(cfg, job)
    if files["entries"]:
        items = [
            html.Div(
                [
                    html.Code(f["name"]),
                    html.Span(
                        f"  {_human_size(f['size'])}",
                        className="text-muted small ms-2",
                    ),
                ],
                className="mb-1",
            )
            for f in files["entries"][:60]   # cap at 60 to avoid huge modals
        ]
        more = ""
        if len(files["entries"]) > 60:
            more = f"  (+{len(files['entries']) - 60} more)"
        rows.append(_section_card(
            f"Files ({files['source']}){more}",
            items,
        ))
    else:
        rows.append(_section_card(
            "Files",
            [html.Div(html.Em(files["note"] or "No files available."),
                      className="text-muted small")],
        ))

    # ----- Logs (tails) -----
    log_blocks = []
    if results and results[0] is not None:
        # Show tail of ridft.out, if we have it (whether local or remote)
        tail_ridft = _gather_log_tail(cfg, job, "ridft.out", n_lines=40)
        if tail_ridft:
            log_blocks.append(("ridft.out (last 40 lines)", tail_ridft))
    tail_err = _gather_log_tail(cfg, job, glob_remote="slurm-*.err",
                                local_glob="slurm-*.err", n_lines=40)
    if tail_err:
        log_blocks.append(("slurm-*.err (last 40 lines)", tail_err))

    if log_blocks:
        log_children = []
        for title, content in log_blocks:
            log_children.append(html.H6(title, className="mt-2"))
            log_children.append(
                html.Pre(content,
                         style={"maxHeight": "240px", "overflow": "auto",
                                "fontSize": "0.75rem"})
            )
        rows.append(_section_card("Logs", log_children))

    return html.Div(rows)


def _section_card(title: str, children) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([html.H6(title, className="card-subtitle text-muted mb-2"),
                      html.Div(children)]),
        className="mb-2",
    )


def _kv(key, value, small: bool = False) -> html.Div:
    cls = "small" if small else ""
    return html.Div(
        [html.Strong(f"{key}: ", className="me-1"),
         value if not isinstance(value, (str, int, float)) else str(value)],
        className=f"mb-1 {cls}".strip(),
    )


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# ----- Gathering helpers ----------------------------------------------------

def _gather_results(cfg: AppConfig, job: dict) -> tuple[RidftSummary, str] | None:
    """Try to obtain a parsed RidftSummary, either from local files (if
    downloaded) or via a light SSH tail of the remote ridft.out."""
    # 1) Prefer local: check if we have a downloaded copy
    local_ridft = _find_local_ridft(cfg, job)
    if local_ridft and local_ridft.exists():
        return parse_ridft(local_ridft), "Parsed from local file."

    # 2) Otherwise try remote tail (only if the job has progressed)
    if job["state"] in ("RUNNING", "COMPLETED", "FAILED", "CLEANED"):
        cluster = cfg.clusters.get(job["cluster"])
        if cluster is None:
            return None
        try:
            tail = ssh_client.read_remote_file_tail(
                cluster, f"{job['remote_dir']}/ridft.out", n_lines=500,
            )
        except Exception as exc:
            log.warning("Remote tail failed for job %s: %s", job["id"], exc)
            return None
        if tail.strip():
            return parse_ridft(tail), "Parsed from remote ridft.out (last 500 lines)."
    return None


def _find_local_ridft(cfg: AppConfig, job: dict) -> Path | None:
    """Look for a ridft.out among the downloaded folders of this job."""
    prefix = f"job_{job['id']}_{job['name']}"
    for entry in sorted(cfg.download_dir.glob(f"{prefix}*"), reverse=True):
        candidate = entry / "ridft.out"
        if candidate.exists():
            return candidate
    return None


def _gather_files(cfg: AppConfig, job: dict) -> dict:
    """Return a dict with 'entries' (list of {name,size}), 'source' label,
    and an optional 'note' explaining why nothing was found."""
    # If we have a downloaded copy, list local files
    prefix = f"job_{job['id']}_{job['name']}"
    candidates = sorted(cfg.download_dir.glob(f"{prefix}*"), reverse=True)
    for entry in candidates:
        if entry.is_dir():
            files = sorted(entry.iterdir())
            return {
                "entries": [
                    {"name": p.name, "size": p.stat().st_size}
                    for p in files if p.is_file()
                ],
                "source": f"local: {entry.name}",
                "note": "",
            }

    # Otherwise try remote ls
    if job["state"] == "CLEANED":
        return {"entries": [], "source": "remote",
                "note": "Remote directory was cleaned."}
    cluster = cfg.clusters.get(job["cluster"])
    if cluster is None:
        return {"entries": [], "source": "remote",
                "note": "Cluster not configured."}
    try:
        remote = ssh_client.list_remote_files(cluster, job["remote_dir"])
    except Exception as exc:
        return {"entries": [], "source": "remote",
                "note": f"ls failed: {exc}"}
    if not remote:
        return {"entries": [], "source": "remote",
                "note": "Remote directory empty or unreachable."}
    return {"entries": remote, "source": "remote", "note": ""}


def _gather_log_tail(cfg: AppConfig, job: dict, filename: str | None = None,
                     *, glob_remote: str | None = None,
                     local_glob: str | None = None,
                     n_lines: int = 40) -> str:
    """Return the tail of a log file (local if downloaded, else remote)."""
    # 1) Local
    prefix = f"job_{job['id']}_{job['name']}"
    for entry in sorted(cfg.download_dir.glob(f"{prefix}*"), reverse=True):
        if not entry.is_dir():
            continue
        if local_glob:
            matches = list(entry.glob(local_glob))
            if matches:
                try:
                    text = matches[0].read_text(errors="replace")
                    return "\n".join(text.splitlines()[-n_lines:])
                except OSError:
                    pass
        elif filename:
            f = entry / filename
            if f.exists():
                try:
                    text = f.read_text(errors="replace")
                    return "\n".join(text.splitlines()[-n_lines:])
                except OSError:
                    pass

    # 2) Remote
    if job["state"] == "CLEANED":
        return ""
    cluster = cfg.clusters.get(job["cluster"])
    if cluster is None:
        return ""
    try:
        target = filename
        if glob_remote and not target:
            # Find a matching remote file via ls
            rc, out, _ = ssh_client.run(
                cluster,
                f"ls {job['remote_dir']}/{glob_remote} 2>/dev/null | head -1",
                timeout=10,
            )
            if rc != 0 or not out.strip():
                return ""
            target = out.strip().rsplit("/", 1)[-1]
        if not target:
            return ""
        return ssh_client.read_remote_file_tail(
            cluster, f"{job['remote_dir']}/{target}", n_lines=n_lines,
        )
    except Exception as exc:
        log.warning("Remote tail failed for %s: %s", filename, exc)
        return ""


# ===========================================================================
# Tab 3 — Clusters
# ===========================================================================

def _register_cluster_test_callback(app: dash.Dash, cfg: AppConfig) -> None:

    @app.callback(
        Output({"type": "test-cluster-out", "name": dash.MATCH}, "children"),
        Input({"type": "btn-test-cluster", "name": dash.MATCH}, "n_clicks"),
        State({"type": "btn-test-cluster", "name": dash.MATCH}, "id"),
        prevent_initial_call=True,
    )
    def _test(n, btn_id):
        if not n:
            return no_update
        name = btn_id["name"]
        cluster = cfg.clusters.get(name)
        if cluster is None:
            return html.Span("not configured", className="text-danger")
        try:
            rc, out, _ = ssh_client.run(cluster, "hostname && squeue --version", timeout=15)
            if rc == 0:
                return html.Span(f"OK {out.strip().splitlines()[0]}", className="text-success")
            return html.Span("connected but squeue failed", className="text-warning")
        except Exception as exc:
            return html.Span(f"FAIL {exc}", className="text-danger")


# ===========================================================================
# Tab 4 — Database inspector
# ===========================================================================

def _register_db_inspector_callbacks(app: dash.Dash, cfg: AppConfig) -> None:

    @app.callback(
        Output("db-table-container", "children"),
        Output("db-status-line", "children"),
        Input("btn-db-refresh", "n_clicks"),
        Input("main-tabs", "active_tab"),
        Input("last-action", "data"),
        prevent_initial_call=False,
    )
    def _render_db_table(_n, active_tab, _action):
        triggered = ctx.triggered_id
        if active_tab != "tab-db" and triggered != "btn-db-refresh":
            return no_update, no_update

        rows = list_jobs_raw()
        db_path = get_db_path()
        size_kb = (db_path.stat().st_size / 1024.0) if (db_path and db_path.exists()) else 0
        status = (
            f"DB file: {db_path}  -  "
            f"size: {size_kb:.1f} KB  -  "
            f"rows: {len(rows)}  -  "
            f"refreshed at {time.strftime('%H:%M:%S')}"
        )

        if not rows:
            return dbc.Alert("Empty database.", color="secondary"), status

        for r in rows:
            for k in ("submit_meta", "extra"):
                if r.get(k):
                    try:
                        r[k] = json.dumps(json.loads(r[k]), separators=(",", ":"))
                    except (TypeError, json.JSONDecodeError):
                        pass
            for k in ("created_at", "updated_at"):
                if r.get(k):
                    r[k] = time.strftime("%Y-%m-%d %H:%M:%S",
                                          time.localtime(r[k]))

        columns = [{"name": c, "id": c} for c in rows[0].keys()]
        table = dash_table.DataTable(
            data=rows,
            columns=columns,
            page_size=25,
            sort_action="native",
            filter_action="native",
            style_table={"overflowX": "auto"},
            style_cell={
                "fontFamily": "monospace",
                "fontSize": "0.78rem",
                "padding": "4px 8px",
                "backgroundColor": "#1c1f24",
                "color": "#c9d1d9",
                "border": "1px solid #30363d",
                "maxWidth": "260px",
                "overflow": "hidden",
                "textOverflow": "ellipsis",
            },
            style_header={
                "fontWeight": "bold",
                "backgroundColor": "#2d333b",
                "color": "#fff",
                "border": "1px solid #444c56",
            },
            style_data_conditional=[
                {"if": {"filter_query": "{state} = COMPLETED"},
                 "color": "#7ee787"},
                {"if": {"filter_query": "{state} = FAILED"},
                 "color": "#ff7b72"},
                {"if": {"filter_query": "{state} = RUNNING"},
                 "color": "#79c0ff"},
                {"if": {"filter_query": "{state} = PENDING"},
                 "color": "#ffd866"},
            ],
            tooltip_data=[
                {col: {"value": str(row[col]) if row[col] is not None else "",
                       "type": "markdown"}
                 for col in row}
                for row in rows
            ],
            tooltip_duration=None,
        )
        return table, status

    @app.callback(
        Output("last-action", "data", allow_duplicate=True),
        Input("btn-db-open-external", "n_clicks"),
        prevent_initial_call=True,
    )
    def _open_external_browser(n):
        if not n:
            return no_update
        db_path = get_db_path()
        if db_path is None:
            return {"ok": False, "msg": "DB not initialized.", "ts": time.time()}
        result = open_in_sqlite_browser(db_path)
        return {"ok": result.ok, "msg": result.msg, "ts": time.time()}

    @app.callback(
        Output("last-action", "data", allow_duplicate=True),
        Input("btn-db-vacuum", "n_clicks"),
        prevent_initial_call=True,
    )
    def _vacuum(n):
        if not n:
            return no_update
        try:
            r = vacuum_db()
            if not r.get("ok"):
                return {"ok": False, "msg": r.get("msg", "vacuum failed"),
                        "ts": time.time()}
            saved = r["saved"]
            msg = (f"VACUUM done. Size: {r['before']/1024:.1f} KB -> "
                   f"{r['after']/1024:.1f} KB  "
                   f"({'saved' if saved >= 0 else 'grew by'} {abs(saved)} bytes)")
            return {"ok": True, "msg": msg, "ts": time.time()}
        except Exception as exc:
            log.exception("VACUUM failed")
            return {"ok": False, "msg": f"VACUUM failed: {exc}", "ts": time.time()}
