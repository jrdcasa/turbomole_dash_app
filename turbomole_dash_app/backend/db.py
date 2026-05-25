"""
Persistent job store (SQLite).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_LOCK = threading.RLock()
_DB_PATH: Path | None = None


JOB_STATES = (
    "DRAFT", "UPLOADED", "SUBMITTED", "PENDING", "RUNNING",
    "COMPLETED", "FAILED", "DOWNLOADED", "CLEANED",
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    cluster         TEXT NOT NULL,
    state           TEXT NOT NULL,
    slurm_id        TEXT,
    task_type       TEXT NOT NULL,
    functional      TEXT,
    basis_set       TEXT,
    charge          INTEGER DEFAULT 0,
    multiplicity    INTEGER DEFAULT 1,
    local_dir       TEXT NOT NULL,
    remote_dir      TEXT NOT NULL,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    submit_meta     TEXT,
    extra           TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);
CREATE INDEX IF NOT EXISTS idx_jobs_cluster ON jobs(cluster);
"""


@dataclass
class JobRecord:
    name: str
    cluster: str
    state: str
    task_type: str
    local_dir: str
    remote_dir: str
    slurm_id: str | None = None
    functional: str | None = None
    basis_set: str | None = None
    charge: int = 0
    multiplicity: int = 1
    submit_meta: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    id: int | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


def init_db(path: Path) -> None:
    global _DB_PATH
    _DB_PATH = Path(path)
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as cx:
        cx.executescript(SCHEMA)


def get_db_path() -> Path | None:
    """Return the SQLite file path (or None if not initialized yet)."""
    return _DB_PATH


def _connect() -> sqlite3.Connection:
    global _DB_PATH
    if _DB_PATH is None:
        from backend.config import load_config
        cfg = load_config()
        _DB_PATH = Path(cfg.db_path)
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        cx_init = sqlite3.connect(_DB_PATH, timeout=10, isolation_level=None)
        cx_init.executescript(SCHEMA)
        cx_init.close()

    cx = sqlite3.connect(_DB_PATH, timeout=10, isolation_level=None)
    cx.row_factory = sqlite3.Row
    cx.execute("PRAGMA journal_mode=WAL")
    cx.execute("PRAGMA foreign_keys=ON")
    return cx


def insert_job(job: JobRecord) -> int:
    with _LOCK, _connect() as cx:
        cur = cx.execute(
            """INSERT INTO jobs
               (name, cluster, state, slurm_id, task_type, functional, basis_set,
                charge, multiplicity, local_dir, remote_dir,
                created_at, updated_at, submit_meta, extra)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                job.name, job.cluster, job.state, job.slurm_id, job.task_type,
                job.functional, job.basis_set, job.charge, job.multiplicity,
                job.local_dir, job.remote_dir,
                job.created_at, job.updated_at,
                json.dumps(job.submit_meta), json.dumps(job.extra),
            ),
        )
        return int(cur.lastrowid)


def update_job(job_id: int, **fields) -> None:
    if not fields:
        return
    fields["updated_at"] = time.time()
    for k in ("submit_meta", "extra"):
        if k in fields and isinstance(fields[k], dict):
            fields[k] = json.dumps(fields[k])
    cols = ", ".join(f"{k}=?" for k in fields)
    with _LOCK, _connect() as cx:
        cx.execute(f"UPDATE jobs SET {cols} WHERE id=?",
                   (*fields.values(), job_id))


def delete_job(job_id: int) -> bool:
    """Delete a job row from the DB. Returns True if a row was deleted."""
    with _LOCK, _connect() as cx:
        cur = cx.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        return cur.rowcount > 0


def get_job(job_id: int) -> dict | None:
    with _connect() as cx:
        row = cx.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return _row_to_dict(row) if row else None


def list_jobs(states: list[str] | None = None) -> list[dict]:
    q = "SELECT * FROM jobs"
    args: tuple = ()
    if states:
        placeholders = ",".join("?" * len(states))
        q += f" WHERE state IN ({placeholders})"
        args = tuple(states)
    q += " ORDER BY created_at DESC"
    with _connect() as cx:
        rows = cx.execute(q, args).fetchall()
    return [_row_to_dict(r) for r in rows]


def list_active_jobs() -> list[dict]:
    return list_jobs(["UPLOADED", "SUBMITTED", "PENDING", "RUNNING"])


def list_jobs_raw() -> list[dict]:
    """Like list_jobs() but keeps submit_meta/extra as JSON strings.

    Used by the in-app DB inspector so the values render flat in a table
    instead of as nested Python dicts.
    """
    with _connect() as cx:
        rows = cx.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def vacuum_db() -> dict:
    """Run VACUUM to reclaim space after deletions. Returns size before/after."""
    if _DB_PATH is None or not _DB_PATH.exists():
        return {"ok": False, "msg": "DB not initialized"}
    size_before = _DB_PATH.stat().st_size
    with _LOCK:
        # VACUUM needs autocommit; close and reopen without WAL pragma
        cx = sqlite3.connect(_DB_PATH, isolation_level=None)
        try:
            cx.execute("VACUUM")
        finally:
            cx.close()
    size_after = _DB_PATH.stat().st_size
    return {
        "ok": True,
        "before": size_before,
        "after": size_after,
        "saved": size_before - size_after,
    }


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for k in ("submit_meta", "extra"):
        if d.get(k):
            try:
                d[k] = json.loads(d[k])
            except (TypeError, json.JSONDecodeError):
                d[k] = {}
        else:
            d[k] = {}
    return d
