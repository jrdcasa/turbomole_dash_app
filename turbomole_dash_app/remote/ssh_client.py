"""
SSH / SFTP client wrapper around paramiko.
"""

from __future__ import annotations

import os
import shlex
import stat
import tarfile
import tempfile
import threading
import uuid
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


# ---------------------------------------------------------------------------
# Tar-based bulk transfer (preferred)
# ---------------------------------------------------------------------------
#
# Rationale: SFTP has per-file round-trip overhead. Turbomole jobs (jobex,
# aoforce) easily produce hundreds of small files (mos, alpha, beta, energy,
# gradient, restart.cc.*, etc.). Transferring a single tarball is one or two
# orders of magnitude faster on high-latency links and avoids partial-state
# races in the remote directory while a job is still writing.
#
# We use the stdlib `tarfile` module + gzip; `tar` is universally available
# on Linux HPCs (unlike `zip`/`unzip`) and preserves POSIX permissions.

def _make_local_tarball(local_dir: Path, tar_path: Path) -> None:
    """Create a gzipped tar of `local_dir`'s contents (no top-level dir).
    Entries are stored with paths relative to local_dir so that extracting
    on the remote side into a freshly-created dir reproduces the layout."""
    with tarfile.open(tar_path, "w:gz") as tf:
        for entry in sorted(local_dir.rglob("*")):
            arcname = entry.relative_to(local_dir).as_posix()
            tf.add(entry, arcname=arcname, recursive=False)


def _extract_tarball(tar_path: Path, dest_dir: Path) -> None:
    """Safely extract a tarball into dest_dir, rejecting absolute paths
    and path-traversal attempts (CVE-2007-4559)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_abs = dest_dir.resolve()
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            member_path = (dest_abs / member.name).resolve()
            if not str(member_path).startswith(str(dest_abs)):
                raise RuntimeError(
                    f"Unsafe tar member outside dest: {member.name!r}"
                )
        tf.extractall(dest_dir, filter="data")


def upload_dir_tar(cluster: RemoteCluster, local_dir: Path, remote_dir: str) -> None:
    """Upload a directory as a single tarball, extract it remotely.

    Steps:
      1. tar.gz `local_dir` locally to a temp file
      2. ensure remote parent exists and remote_dir is fresh
      3. SFTP the tarball to <remote_dir>/.payload.tar.gz
      4. ssh `tar -xzf` into remote_dir, then remove the tarball
    """
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise FileNotFoundError(f"Local dir does not exist: {local_dir}")

    with tempfile.TemporaryDirectory() as td:
        local_tar = Path(td) / "payload.tar.gz"
        _make_local_tarball(local_dir, local_tar)

        ensure_dir(cluster, remote_dir)
        remote_tar = f"{remote_dir.rstrip('/')}/.payload.tar.gz"

        with sftp(cluster) as s:
            s.put(str(local_tar), remote_tar)

        # Extract on the cluster. -m avoids touching mtimes on lustre/gpfs
        # which sometimes complains; the trailing rm cleans up the archive.
        cmd = (
            f"cd {shlex.quote(remote_dir)} && "
            f"tar -xzf {shlex.quote(remote_tar)} && "
            f"rm -f {shlex.quote(remote_tar)}"
        )
        rc, out, err = run(cluster, cmd, timeout=300)
        if rc != 0:
            raise RuntimeError(
                f"Remote extraction failed (rc={rc}): {err.strip() or out.strip()}"
            )


def download_dir_tar(cluster: RemoteCluster, remote_dir: str, local_dir: Path,
                     partial: bool = False) -> Path:
    """Download a remote directory as a single tarball, extract locally.

    GNU tar exit codes:
      0 = success
      1 = "some files differ" or "file changed as we read it" — the archive
          is still written completely; this is NOT a fatal error and can
          happen even on supposedly-quiescent dirs (Lustre metadata races,
          SLURM still flushing slurm-*.out, files with odd permissions).
      2 = fatal error (cannot open, disk full, etc.)

    We always tolerate rc=1, log the diagnostic to the job's `extra` via
    the returned tarball, and only escalate rc>=2. The archive is verified
    non-empty before extraction.
    """
    local_dir = Path(local_dir)
    tar_name = f".download_{uuid.uuid4().hex[:8]}.tar.gz"
    remote_tar = f"{remote_dir.rstrip('/')}/{tar_name}"

    # `--ignore-failed-read` keeps tar going on unreadable files;
    # `--warning=no-file-changed` silences the noisy stderr line; useful
    # in both partial AND full mode because Lustre/GPFS can still trigger
    # it spuriously even when no process is writing.
    tar_opts = "--ignore-failed-read --warning=no-file-changed"

    # Capture stderr SEPARATELY (not merged into stdout) so we can include
    # the real tar diagnostic in the error message if rc>=2.
    pack_cmd = (
        f"cd {shlex.quote(remote_dir)} && "
        f"tar {tar_opts} -czf {shlex.quote(remote_tar)} "
        f"--exclude={shlex.quote(tar_name)} ."
    )
    rc, out, err = run(cluster, pack_cmd, timeout=600)

    if rc >= 2:
        run(cluster, f"rm -f {shlex.quote(remote_tar)}", timeout=30)
        diag = err.strip() or out.strip() or "(tar produced no message)"
        raise RuntimeError(f"Remote tar failed (rc={rc}): {diag}")

    # rc may be 0 or 1; in either case the tarball should exist. Verify.
    rc_check, size_out, _ = run(
        cluster,
        f"stat -c %s {shlex.quote(remote_tar)} 2>/dev/null",
        timeout=20,
    )
    try:
        tar_size = int(size_out.strip()) if rc_check == 0 else 0
    except ValueError:
        tar_size = 0
    if tar_size == 0:
        run(cluster, f"rm -f {shlex.quote(remote_tar)}", timeout=30)
        diag = err.strip() or out.strip() or "(no output)"
        raise RuntimeError(
            f"Remote tar produced no archive (rc={rc}): {diag}"
        )

    local_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        local_tar = Path(td) / "payload.tar.gz"
        try:
            with sftp(cluster) as s:
                s.get(remote_tar, str(local_tar))
        finally:
            run(cluster, f"rm -f {shlex.quote(remote_tar)}", timeout=30)

        _extract_tarball(local_tar, local_dir)

    return local_dir


# ---------------------------------------------------------------------------
# Legacy per-file SFTP transfer (kept as fallback / for tests)
# ---------------------------------------------------------------------------

def upload_dir(cluster: RemoteCluster, local_dir: Path, remote_dir: str) -> None:
    """Legacy per-file SFTP upload. Prefer `upload_dir_tar` for performance."""
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
    """Legacy per-file SFTP download. Prefer `download_dir_tar` for performance."""
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
        parts = line.split(None, 6)
        if len(parts) < 7:
            continue
        if not parts[0].startswith("-"):
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
