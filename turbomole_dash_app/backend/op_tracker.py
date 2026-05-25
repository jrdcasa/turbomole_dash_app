"""
Thread-safe registry of in-flight upload / download operations.

Used by the UI to show a "UPLOADING.../DOWNLOADING..." badge and spinner
on jobs whose SFTP transfer is still running in the background executor.

This lives in process memory (not SQLite) because the only reader is the
local UI within the same process. If the app restarts, any in-flight
operation is gone anyway (the thread dies with the process), so there is
nothing meaningful to persist.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class _Op:
    job_id: int
    kind: str           # "uploading" | "downloading"
    started_at: float


_LOCK = threading.RLock()
_OPS: dict[int, _Op] = {}      # job_id -> latest op (one per job is enough)


def mark_started(job_id: int, kind: str) -> None:
    """Register that an operation has begun for this job."""
    assert kind in ("uploading", "downloading"), f"bad kind: {kind!r}"
    with _LOCK:
        _OPS[job_id] = _Op(job_id=job_id, kind=kind, started_at=time.time())


def mark_finished(job_id: int) -> None:
    """Register that any pending operation on this job has finished."""
    with _LOCK:
        _OPS.pop(job_id, None)


def active() -> dict[int, dict]:
    """Return a snapshot of all in-flight ops, keyed by job_id.

    Returns plain dicts (not _Op instances) so they survive being placed
    in a dcc.Store / serialized to JSON.
    """
    with _LOCK:
        return {
            jid: {"kind": op.kind, "started_at": op.started_at}
            for jid, op in _OPS.items()
        }


def is_active(job_id: int) -> str | None:
    """Return the kind ('uploading'/'downloading') if active, else None."""
    with _LOCK:
        op = _OPS.get(job_id)
        return op.kind if op else None


def any_active() -> bool:
    with _LOCK:
        return bool(_OPS)
