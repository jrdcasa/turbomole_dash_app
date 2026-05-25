"""
SLURM-side operations performed over SSH.
"""

from __future__ import annotations

import re
import shlex

from backend.config import RemoteCluster
from remote.ssh_client import run


_SBATCH_RE = re.compile(r"Submitted batch job (\d+)")

_SQUEUE_STATE_MAP = {
    "PD": "PENDING",
    "R":  "RUNNING",
    "CG": "RUNNING",
    "CD": "COMPLETED",
    "F":  "FAILED",
    "NF": "FAILED",
    "TO": "FAILED",
    "CA": "FAILED",
    "BF": "FAILED",
    "DL": "FAILED",
    "OOM": "FAILED",
    "PENDING":   "PENDING",
    "RUNNING":   "RUNNING",
    "COMPLETED": "COMPLETED",
    "COMPLETING":"RUNNING",
    "FAILED":    "FAILED",
    "CANCELLED": "FAILED",
    "TIMEOUT":   "FAILED",
    "NODE_FAIL": "FAILED",
    "OUT_OF_MEMORY": "FAILED",
    "BOOT_FAIL": "FAILED",
}


def submit(cluster: RemoteCluster, remote_workdir: str, script_name: str = "submit.slurm") -> str:
    """Submit `script_name` from `remote_workdir`. Returns the SLURM job id."""
    cmd = f"cd {shlex.quote(remote_workdir)} && sbatch {shlex.quote(script_name)}"
    rc, out, err = run(cluster, cmd, timeout=60)
    if rc != 0:
        raise RuntimeError(f"sbatch failed: {err.strip() or out.strip()}")
    m = _SBATCH_RE.search(out)
    if not m:
        raise RuntimeError(f"Could not parse sbatch output: {out!r}")
    return m.group(1)


def query_state(cluster: RemoteCluster, slurm_id: str) -> str:
    """
    Return one of: PENDING, RUNNING, COMPLETED, FAILED, UNKNOWN.
    Tries `squeue` first; falls back to `sacct` for terminal states.
    """
    rc, out, _ = run(
        cluster,
        f"squeue --noheader -j {shlex.quote(slurm_id)} -o '%T'",
        timeout=30,
    )
    state = out.strip().split("\n")[0].strip() if out.strip() else ""
    if state:
        return _SQUEUE_STATE_MAP.get(state, "UNKNOWN")

    rc, out, _ = run(
        cluster,
        f"sacct -j {shlex.quote(slurm_id)} -X --noheader -o State -P",
        timeout=30,
    )
    if out.strip():
        st = out.strip().split("\n")[0].split()[0].strip()
        return _SQUEUE_STATE_MAP.get(st, "UNKNOWN")
    return "UNKNOWN"


def cancel(cluster: RemoteCluster, slurm_id: str) -> None:
    run(cluster, f"scancel {shlex.quote(slurm_id)}", timeout=30)
