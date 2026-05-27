"""
Configuration loader. Reads config.yaml from:
  1. The project root (next to app.py) — recommended
  2. The current working directory
  3. ~/.turbomole_orchestrator/config.yaml
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# Project root = the directory that contains the backend/, app_ui/, ... packages.
# This file lives in <project_root>/backend/config.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG_PATHS = [
    _PROJECT_ROOT / "config.yaml",                          # 1. next to app.py
    Path.cwd() / "config.yaml",                              # 2. cwd
    Path.home() / ".turbomole_orchestrator" / "config.yaml", # 3. user-global
]


@dataclass
class RemoteCluster:
    name: str
    host: str
    port: int = 22
    user: str = ""
    key_filename: str | None = None       # path to private key
    use_agent: bool = True                # try ssh-agent first
    remote_workdir: str = "/scratch/$USER/turbomole_jobs"
    module_load: list[str] = field(default_factory=list)
    env_setup: list[str] = field(default_factory=list)   # arbitrary shell lines
    turbomole_version: str = "7.8"        # "7.8" | "8.0"
    default_partition: str = "compute"
    default_time: str = "24:00:00"
    default_nodes: int = 1
    default_ntasks: int = 16
    default_mem: str = "32G"


@dataclass
class AppConfig:
    db_path: Path = Path.home() / ".turbomole_orchestrator" / "jobs.sqlite"
    local_workdir: Path = Path.home() / ".turbomole_orchestrator" / "workspace"
    download_dir: Path = Path.home() / ".turbomole_orchestrator" / "downloads"
    # Reusable JSON submission protocols saved from the New job tab.
    protocols_dir: Path = Path.home() / ".turbomole_orchestrator" / "protocols"
    # App-wide settings (external tool paths, etc.).
    settings_dir: Path = Path.home() / ".turbomole_orchestrator"
    poll_interval_s: int = 30
    clusters: dict[str, RemoteCluster] = field(default_factory=dict)


def _ensure_dirs(cfg: AppConfig) -> None:
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.local_workdir.mkdir(parents=True, exist_ok=True)
    cfg.download_dir.mkdir(parents=True, exist_ok=True)
    cfg.protocols_dir.mkdir(parents=True, exist_ok=True)


def load_config(path: str | os.PathLike | None = None) -> AppConfig:
    candidates = [Path(path)] if path else DEFAULT_CONFIG_PATHS
    raw: dict[str, Any] = {}
    for p in candidates:
        if p.exists():
            print(f"config.yaml: {p}")
            with open(p, "r") as fh:
                raw = yaml.safe_load(fh) or {}
            break

    cfg = AppConfig()
    if "db_path" in raw:
        cfg.db_path = Path(raw["db_path"]).expanduser()
    if "local_workdir" in raw:
        cfg.local_workdir = Path(raw["local_workdir"]).expanduser()
    if "download_dir" in raw:
        cfg.download_dir = Path(raw["download_dir"]).expanduser()
    if "protocols_dir" in raw:
        cfg.protocols_dir = Path(raw["protocols_dir"]).expanduser()
    if "poll_interval_s" in raw:
        cfg.poll_interval_s = int(raw["poll_interval_s"])

    for name, c in (raw.get("clusters") or {}).items():
        cfg.clusters[name] = RemoteCluster(name=name, **c)

    # Provide a placeholder cluster so the UI is usable on first launch
    if not cfg.clusters:
        cfg.clusters["example"] = RemoteCluster(
            name="example",
            host="hpc.example.org",
            user=os.environ.get("USER", "user"),
        )

    _ensure_dirs(cfg)
    return cfg