"""
SSH / SFTP client wrapper around paramiko.
"""

from __future__ import annotations

import os
import shlex
import stat
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import paramiko

from backend.config import RemoteCluster


_CLIENTS: dict[str, paramiko.SSHClient] = {}
_LOCK = threading.RLock()


def _connect(cluster: RemoteCluster) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    kwargs = {
        "hostname": cluster.host,
        "port": cluster.port,
        "username": cluster.user,
        "timeout": 20,
        "auth_timeout": 20,
        "banner_timeout": 20,
    }
    if cluster.key_filename:
        kwargs["key_filename"] = os.path.expanduser(cluster.key_filename)
    if cluster.use_agent:
        kwargs["allow_agent"] = True
        kwargs["look_for_keys"] = True

    client.connect(**kwargs)
    transport = client.get_transport()
    if transport is not None:
        transport.set_keepalive(30)
    return client


def get_client(cluster: RemoteCluster) -> paramiko.SSHClient:
    with _LOCK:
        client = _CLIENTS.get(cluster.name)
        if client is not None:
            t = client.get_transport()
            if t is not None and t.is_active():
                return client
        client = _connect(cluster)
        _CLIENTS[cluster.name] = client
        return client


def close_all() -> None:
    with _LOCK:
        for c in _CLIENTS.values():
            try:
                c.close()
            except Exception:
                pass
        _CLIENTS.clear()


def run(cluster: RemoteCluster, command: str, timeout: int = 60) -> tuple[int, str, str]:
    client = get_client(cluster)
    _stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    rc = stdout.channel.recv_exit_status()
    return rc, out, err


def ensure_dir(cluster: RemoteCluster, path: str) -> None:
    rc, _, err = run(cluster, f"mkdir -p {shlex.quote(path)}")
    if rc != 0:
        raise RuntimeError(f"Could not create remote dir {path}: {err}")


def expand_remote(cluster: RemoteCluster, path: str) -> str:
    rc, out, err = run(cluster, f"echo {path}")
    if rc != 0:
        raise RuntimeError(f"Remote path expansion failed: {err}")
    return out.strip()


@contextmanager
def sftp(cluster: RemoteCluster) -> Iterator[paramiko.SFTPClient]:
    client = get_client(cluster)
    s = client.open_sftp()
    try:
        yield s
    finally:
        s.close()


def upload_dir(cluster: RemoteCluster, local_dir: Path, remote_dir: str) -> None:
    ensure_dir(cluster, remote_dir)
    with sftp(cluster) as s:
        for root, _dirs, files in os.walk(local_dir):
            rel = os.path.relpath(root, local_dir)
            r_root = remote_dir if rel == "." else f"{remote_dir}/{rel.replace(os.sep, '/')}"
            try:
                s.stat(r_root)
            except FileNotFoundError:
                s.mkdir(r_root)
            for fname in files:
                lpath = os.path.join(root, fname)
                rpath = f"{r_root}/{fname}"
                s.put(lpath, rpath)


def upload_text(cluster: RemoteCluster, remote_path: str, content: str, mode: int = 0o755) -> None:
    with sftp(cluster) as s:
        with s.file(remote_path, "w") as fh:
            fh.write(content)
        s.chmod(remote_path, mode)


def download_dir(cluster: RemoteCluster, remote_dir: str, local_dir: Path,
                 partial: bool = False) -> list[Path]:
    local_dir.mkdir(parents=True, exist_ok=True)
    fetched: list[Path] = []
    with sftp(cluster) as s:
        _sftp_walk_get(s, remote_dir, local_dir, fetched, partial=partial)
    return fetched


def _sftp_walk_get(s: paramiko.SFTPClient, rdir: str, ldir: Path,
                   fetched: list[Path], partial: bool) -> None:
    ldir.mkdir(parents=True, exist_ok=True)
    try:
        entries = s.listdir_attr(rdir)
    except FileNotFoundError:
        return
    for entry in entries:
        rpath = f"{rdir}/{entry.filename}"
        lpath = ldir / entry.filename
        if stat.S_ISDIR(entry.st_mode):
            _sftp_walk_get(s, rpath, lpath, fetched, partial)
        else:
            try:
                s.get(rpath, str(lpath))
                fetched.append(lpath)
            except (OSError, FileNotFoundError):
                if not partial:
                    raise


def remove_remote_dir(cluster: RemoteCluster, remote_dir: str) -> None:
    rd = remote_dir.strip()
    if not rd or rd in ("/", "~", "."):
        raise ValueError(f"Refusing to remove suspicious remote path: {remote_dir!r}")
    rc, _, err = run(cluster, f"rm -rf -- {shlex.quote(rd)}")
    if rc != 0:
        raise RuntimeError(f"Failed to remove {remote_dir}: {err}")


# ---------------------------------------------------------------------------
# Lightweight remote read helpers (for the job-detail modal)
# ---------------------------------------------------------------------------

def list_remote_files(cluster: RemoteCluster, remote_dir: str,
                      timeout: int = 20) -> list[dict]:
    """List files in remote_dir. Returns list of {name, size, mtime}."""
    cmd = f"ls -la --time-style=+%s {shlex.quote(remote_dir)} 2>/dev/null"
    rc, out, _ = run(cluster, cmd, timeout=timeout)
    if rc != 0 or not out.strip():
        return []
    files = []
    for line in out.splitlines():
        # Skip "total N" header and entries that are not regular files
        parts = line.split(None, 6)
        if len(parts) < 7:
            continue
        if not parts[0].startswith("-"):    # only regular files (no dirs/links)
            continue
        try:
            size = int(parts[4])
            mtime = int(parts[5])
            name = parts[6]
        except (ValueError, IndexError):
            continue
        files.append({"name": name, "size": size, "mtime": mtime})
    return files


def read_remote_file_tail(cluster: RemoteCluster, remote_path: str,
                          n_lines: int = 200, timeout: int = 20) -> str:
    """Read the last n_lines of a remote file. Empty string if file missing."""
    cmd = f"tail -n {int(n_lines)} {shlex.quote(remote_path)} 2>/dev/null"
    rc, out, _ = run(cluster, cmd, timeout=timeout)
    return out if rc == 0 else ""


def read_remote_file(cluster: RemoteCluster, remote_path: str,
                     max_bytes: int = 200_000, timeout: int = 30) -> str:
    """Read up to max_bytes of a remote file via SFTP."""
    with sftp(cluster) as s:
        try:
            with s.file(remote_path, "r") as fh:
                data = fh.read(max_bytes)
        except (OSError, FileNotFoundError):
            return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data
