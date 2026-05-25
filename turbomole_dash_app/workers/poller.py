"""
On-demand SLURM status refresh.

No background thread. The UI calls `refresh_active_jobs()` whenever the
user clicks the "Refresh status" button (or before performing an action
that depends on up-to-date state). State is persisted in SQLite as before.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from backend.config import AppConfig
from backend.db import list_active_jobs, update_job
from remote.slurm_remote import query_state


log = logging.getLogger("poller")


@dataclass
class RefreshReport:
    """Summary of what a refresh pass did. Useful for UI toasts."""
    checked: int = 0          # how many active jobs we queried
    updated: int = 0          # how many changed state
    errors: int = 0           # how many failed (cluster unreachable, etc.)
    details: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.checked == 0:
            return "No active jobs to check."
        bits = [f"Checked {self.checked} job(s)"]
        if self.updated:
            bits.append(f"{self.updated} updated")
        if self.errors:
            bits.append(f"{self.errors} error(s)")
        return " — ".join(bits) + "."


def refresh_active_jobs(cfg: AppConfig) -> RefreshReport:
    """
    Query SLURM for every active job and update the DB.

    Active = UPLOADED / SUBMITTED / PENDING / RUNNING.
    Terminal states are skipped (they need a user action to change).
    """
    report = RefreshReport()
    active = list_active_jobs()
    report.checked = len(active)

    for job in active:
        slurm_id = job.get("slurm_id")
        if not slurm_id:
            continue

        cluster = cfg.clusters.get(job["cluster"])
        if cluster is None:
            report.errors += 1
            report.details.append(
                f"job {job['id']}: cluster '{job['cluster']}' is no longer configured"
            )
            continue

        try:
            new_state = query_state(cluster, slurm_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not query %s on %s: %s",
                        slurm_id, cluster.name, exc)
            report.errors += 1
            report.details.append(
                f"job {job['id']} ({cluster.name}): {exc}"
            )
            continue

        if new_state in ("UNKNOWN", job["state"]):
            continue

        update_job(job["id"], state=new_state)
        report.updated += 1
        report.details.append(
            f"job {job['id']} ({job['name']}): {job['state']} → {new_state}"
        )
        log.info("Job %s (%s): %s → %s",
                 job["id"], slurm_id, job["state"], new_state)

    return report


def refresh_single_job(cfg: AppConfig, job_id: int) -> str | None:
    """
    Refresh state for a single job. Used right before an action like
    Download to avoid acting on stale state.
    """
    from backend.db import get_job

    job = get_job(job_id)
    if not job or not job.get("slurm_id"):
        return None
    if job["state"] in ("COMPLETED", "FAILED", "DOWNLOADED", "CLEANED"):
        return job["state"]

    cluster = cfg.clusters.get(job["cluster"])
    if cluster is None:
        return None

    try:
        new_state = query_state(cluster, job["slurm_id"])
    except Exception:  # noqa: BLE001
        return None

    if new_state not in ("UNKNOWN", job["state"]):
        update_job(job_id, state=new_state)
        return new_state
    return job["state"]
