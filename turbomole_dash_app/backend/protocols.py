"""
Save / load reusable submission "protocols" as JSON.

A protocol captures everything in the New job tab *except* the molecular
structure: method, basis, RI, grid, task type, charge, multiplicity, AIMD
params (incl. distance constraints), cluster, partition, walltime, nodes,
ntasks, mem, reservation.

The idea: configure once for a workflow (e.g. BP86-D3 optimization on
drago with 48 cores), save it under a memorable name, then reuse it for
many different molecules.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path


PROTOCOL_VERSION = 1


# Hard defaults — these match the initial values in app_ui/layout.py.
# Used both by the "Reset to defaults" button and as a fallback when a
# protocol file is missing keys.
DEFAULTS: dict = {
    "method": {
        "functional": "BP86",
        "basis_set": "def2-SVP",
        "use_ri": True,
        "grid": "m4",
        "dispersion": "none",
    },
    "task": {
        "type": "single_point",
        "charge": 0,
        "multiplicity": 1,
        # AIMD defaults follow the user's preferred mdprep settings:
        # 256 steps, 80.0 a.u. timestep (~1.935 fs), 300 K, no constraints.
        "aimd_steps": 256,
        "aimd_timestep_au": 80.0,
        "aimd_temperature_K": 300,
        "aimd_use_constraints": False,
        "aimd_constraint_algorithm": "shake",
        "aimd_constraints_text": "",
    },
    "submission": {
        "cluster": None,        # filled at load-time with the first available
        "partition": "",        # blank → cluster defaults are used by callback
        "walltime": "",
        "nodes": None,
        "ntasks": None,
        "mem": "",
        "reservation": "",
    },
}


@dataclass
class Protocol:
    name: str
    description: str
    method: dict
    task: dict
    submission: dict
    created_at: str            # ISO-8601
    version: int = PROTOCOL_VERSION

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "method": self.method,
            "task": self.task,
            "submission": self.submission,
        }


# ---------------------------------------------------------------------------
# File-system helpers
# ---------------------------------------------------------------------------

_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(name: str) -> str:
    """Map a user-provided protocol name to a safe .json filename.

    Spaces and exotic characters become underscores so the file can be
    listed and opened reliably across filesystems.
    """
    base = _NAME_RE.sub("_", name.strip()) or "protocol"
    return base + ".json"


def list_protocols(protocols_dir: Path) -> list[dict]:
    """Return a list of dicts with at least {'name', 'filename', 'path'}.

    Sorted by name. Skips files that aren't valid JSON or look corrupted.
    """
    out: list[dict] = []
    if not protocols_dir.exists():
        return out
    for p in sorted(protocols_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        out.append({
            "name": data.get("name") or p.stem,
            "description": data.get("description", ""),
            "filename": p.name,
            "path": str(p),
        })
    return sorted(out, key=lambda d: d["name"].lower())


def save_protocol(protocols_dir: Path, name: str, description: str,
                  method: dict, task: dict, submission: dict,
                  overwrite: bool = True) -> Path:
    """Persist a protocol as JSON. Returns the target path.

    If `overwrite=False` and the target file already exists, raises
    FileExistsError.
    """
    protocols_dir.mkdir(parents=True, exist_ok=True)
    proto = Protocol(
        name=name.strip(),
        description=description.strip(),
        method=method,
        task=task,
        submission=submission,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    path = protocols_dir / _safe_filename(name)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Protocol already exists: {path.name}")
    path.write_text(json.dumps(proto.to_dict(), indent=2))
    return path


def load_protocol(path: Path) -> dict:
    """Load a protocol from disk. Missing keys are filled from DEFAULTS,
    so older protocols stay compatible if we add fields later."""
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid protocol file: {path}")
    return _merge_with_defaults(raw)


def delete_protocol(path: Path) -> bool:
    """Delete a protocol file. Returns True if it was deleted."""
    p = Path(path)
    if not p.exists():
        return False
    try:
        p.unlink()
        return True
    except OSError:
        return False


def _merge_with_defaults(raw: dict) -> dict:
    """Fill in any missing top-level / nested keys from DEFAULTS so the
    UI can always read every field without KeyError."""
    out = {
        "version":     raw.get("version", PROTOCOL_VERSION),
        "name":        raw.get("name", ""),
        "description": raw.get("description", ""),
        "created_at":  raw.get("created_at", ""),
        "method":      {**DEFAULTS["method"],     **(raw.get("method") or {})},
        "task":        {**DEFAULTS["task"],       **(raw.get("task") or {})},
        "submission":  {**DEFAULTS["submission"], **(raw.get("submission") or {})},
    }
    return out


def defaults_payload(first_cluster: str | None = None) -> dict:
    """Return a 'fresh' payload (same shape as load_protocol output) used
    by the Reset button. `first_cluster` populates submission.cluster."""
    payload = {
        "version":     PROTOCOL_VERSION,
        "name":        "",
        "description": "",
        "created_at":  "",
        "method":      dict(DEFAULTS["method"]),
        "task":        dict(DEFAULTS["task"]),
        "submission":  dict(DEFAULTS["submission"]),
    }
    if first_cluster:
        payload["submission"]["cluster"] = first_cluster
    return payload