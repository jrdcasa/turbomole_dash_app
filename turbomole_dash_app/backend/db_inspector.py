"""
Helpers for the in-app DB inspector + launcher for SQLite Browser.

Why a separate module: keeps subprocess and platform-specific code out of
db.py (which stays a pure data-access layer).
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LaunchResult:
    ok: bool
    msg: str
    binary: str | None = None


# Order matters: we prefer the GUI app and fall back to the CLI.
# Each entry is (display_name, executable, supports_readonly_flag).
_CANDIDATES_LINUX = [
    ("DB Browser for SQLite", "sqlitebrowser", True),
    ("DB Browser for SQLite (alt)", "DB Browser for SQLite", True),
    ("SQLite CLI",            "sqlite3",       False),
]
_CANDIDATES_MAC = [
    ("DB Browser for SQLite", "sqlitebrowser", True),
    # macOS bundles often install as an .app — open it via `open -a`
    # We handle this case explicitly in _launch_macos_bundle below.
    ("SQLite CLI",            "sqlite3",       False),
]
_CANDIDATES_WINDOWS = [
    ("DB Browser for SQLite", "sqlitebrowser.exe", True),
    ("DB Browser for SQLite", "sqlitebrowser",     True),
    ("SQLite CLI",            "sqlite3.exe",       False),
    ("SQLite CLI",            "sqlite3",           False),
]


def _candidates() -> list[tuple[str, str, bool]]:
    sysname = platform.system()
    if sysname == "Linux":
        return _CANDIDATES_LINUX
    if sysname == "Darwin":
        return _CANDIDATES_MAC
    if sysname == "Windows":
        return _CANDIDATES_WINDOWS
    return _CANDIDATES_LINUX


def find_sqlite_browser() -> tuple[str, str, bool] | None:
    """Return (display_name, executable, supports_readonly) for the first
    candidate that is on PATH, or None."""
    for display, exe, ro in _candidates():
        if shutil.which(exe):
            return display, exe, ro
    return None


def _launch_macos_bundle(db_path: Path) -> LaunchResult:
    """On macOS the user may have the .app bundle installed instead of
    the CLI symlink. Try `open -a` as a last resort."""
    app_name = "DB Browser for SQLite"
    try:
        subprocess.Popen(
            ["open", "-a", app_name, "--args", "--read-only", str(db_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return LaunchResult(
            ok=True,
            msg=f"Opened DB Browser for SQLite (read-only): {db_path}",
            binary=app_name,
        )
    except (FileNotFoundError, OSError) as exc:
        return LaunchResult(
            ok=False,
            msg=f"Could not open '{app_name}.app': {exc}",
        )


def open_in_sqlite_browser(db_path: Path) -> LaunchResult:
    """Launch an external SQLite browser in read-only mode.

    Read-only matters because the GUI tool keeps an exclusive lock on the
    file when opened for writing, which would block this app's own writes.
    The `-R` / `--read-only` flag prevents that.
    """
    if not db_path.exists():
        return LaunchResult(ok=False, msg=f"DB file not found: {db_path}")

    found = find_sqlite_browser()
    if found is None:
        if platform.system() == "Darwin":
            return _launch_macos_bundle(db_path)
        return LaunchResult(
            ok=False,
            msg=(
                "No SQLite browser found in PATH.\n"
                "Install one of:\n"
                "  - DB Browser for SQLite (recommended): "
                "https://sqlitebrowser.org/dl/\n"
                "  - On Debian/Ubuntu:  sudo apt install sqlitebrowser\n"
                "  - On macOS:          brew install --cask db-browser-for-sqlite\n"
                "  - On Windows:        download installer from sqlitebrowser.org"
            ),
        )

    display, exe, supports_ro = found

    # Build argument list with read-only when supported
    if exe.startswith("sqlite3"):
        # sqlite3 CLI is read-only by default unless you write SQL.
        # Open it in a terminal so the user can interact.
        args = _wrap_in_terminal([exe, str(db_path)])
    else:
        args = [exe]
        if supports_ro:
            # DB Browser for SQLite supports both -R and --read-only
            args.append("--read-only")
        args.append(str(db_path))

    try:
        subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            # Detach so closing the app doesn't kill the browser
            start_new_session=(platform.system() != "Windows"),
        )
        return LaunchResult(
            ok=True,
            msg=f"Opened {display} (read-only): {db_path}",
            binary=exe,
        )
    except (FileNotFoundError, OSError) as exc:
        return LaunchResult(
            ok=False,
            msg=f"Could not launch '{exe}': {exc}",
        )


def _wrap_in_terminal(cmd: list[str]) -> list[str]:
    """Open an interactive command in a terminal window.

    Only used for the sqlite3 CLI fallback. Best-effort: tries a few
    common terminals; if none works, the command runs detached without
    a window (less useful, but won't crash).
    """
    cmd_str = " ".join(_shell_escape(a) for a in cmd)
    sysname = platform.system()

    if sysname == "Linux":
        for term in ("x-terminal-emulator", "gnome-terminal",
                     "konsole", "xterm"):
            if shutil.which(term):
                if term == "gnome-terminal":
                    return [term, "--", "bash", "-c",
                            f"{cmd_str}; exec bash"]
                if term == "konsole":
                    return [term, "-e", "bash", "-c",
                            f"{cmd_str}; exec bash"]
                return [term, "-e", f"bash -c '{cmd_str}; exec bash'"]
    elif sysname == "Darwin":
        return ["open", "-a", "Terminal", "--args"] + cmd
    elif sysname == "Windows":
        return ["cmd.exe", "/c", "start", "cmd.exe", "/k"] + cmd

    return cmd  # last resort: no terminal


def _shell_escape(s: str) -> str:
    if any(c in s for c in " \t\"'\\"):
        return "'" + s.replace("'", "'\\''") + "'"
    return s
