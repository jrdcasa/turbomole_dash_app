"""
Persistent app-wide settings (paths to external tools, etc.).

Lives in ~/.turbomole_orchestrator/app_settings.json so it survives app
restarts and is separate from cluster config (which is YAML-managed).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


SETTINGS_FILENAME = "app_settings.json"


@dataclass
class ExternalTools:
    """Absolute paths to optional desktop helpers. Empty string = use PATH."""
    vmd: str = ""
    cosmobuild: str = ""
    cosmoquick: str = ""


@dataclass
class AppSettings:
    external_tools: ExternalTools = field(default_factory=ExternalTools)

    def to_dict(self) -> dict:
        return {"external_tools": asdict(self.external_tools)}

    @classmethod
    def from_dict(cls, raw: dict) -> "AppSettings":
        et = (raw or {}).get("external_tools", {}) or {}
        return cls(external_tools=ExternalTools(
            vmd=str(et.get("vmd", "") or ""),
            cosmobuild=str(et.get("cosmobuild", "") or ""),
            cosmoquick=str(et.get("cosmoquick", "") or ""),
        ))


def _settings_path(settings_dir: Path) -> Path:
    return settings_dir / SETTINGS_FILENAME


def load_settings(settings_dir: Path) -> AppSettings:
    """Load settings; return defaults if the file is missing or unreadable."""
    p = _settings_path(settings_dir)
    if not p.exists():
        return AppSettings()
    try:
        raw = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return AppSettings()
    if not isinstance(raw, dict):
        return AppSettings()
    return AppSettings.from_dict(raw)


def save_settings(settings_dir: Path, settings: AppSettings) -> Path:
    settings_dir.mkdir(parents=True, exist_ok=True)
    p = _settings_path(settings_dir)
    p.write_text(json.dumps(settings.to_dict(), indent=2))
    return p