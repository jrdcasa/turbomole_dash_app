"""Tests for tar-based bulk transfer helpers (no real SSH required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from remote import ssh_client


def _make_tree(root: Path) -> None:
    """Create a small directory tree mimicking a Turbomole job."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "coord").write_text("$coord\n  0.0 0.0 0.0 h\n$end\n")
    (root / "define.inp").write_text("\na coord\n*\nno\nb\nall def2-SVP\n*\n")
    (root / "submit.slurm").write_text("#!/bin/bash -l\n#SBATCH --job-name=t\n")
    sub = root / "subdir"
    sub.mkdir()
    (sub / "nested.txt").write_text("nested content\n")


def test_make_and_extract_tarball_roundtrip(tmp_path):
    src = tmp_path / "src"
    _make_tree(src)
    tar_path = tmp_path / "payload.tar.gz"
    ssh_client._make_local_tarball(src, tar_path)
    assert tar_path.exists()
    assert tar_path.stat().st_size > 0

    dest = tmp_path / "dest"
    ssh_client._extract_tarball(tar_path, dest)

    assert (dest / "coord").read_text().startswith("$coord")
    assert (dest / "define.inp").exists()
    assert (dest / "submit.slurm").exists()
    assert (dest / "subdir" / "nested.txt").read_text() == "nested content\n"


def test_tarball_has_no_top_level_dir(tmp_path):
    """The archive must store relative paths (so extract-into-dir reproduces
    the original layout without a wrapping directory)."""
    import tarfile

    src = tmp_path / "src"
    _make_tree(src)
    tar_path = tmp_path / "p.tar.gz"
    ssh_client._make_local_tarball(src, tar_path)

    with tarfile.open(tar_path, "r:gz") as tf:
        names = tf.getnames()
    # No entry should start with the source directory's name
    assert not any(n.startswith("src/") or n == "src" for n in names)
    assert "coord" in names
    assert "subdir/nested.txt" in names


def test_extract_rejects_path_traversal(tmp_path):
    """The safe extractor must refuse archives containing ../ entries."""
    import tarfile, io

    tar_path = tmp_path / "evil.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        data = b"pwned"
        info = tarfile.TarInfo(name="../escape.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))

    dest = tmp_path / "dest"
    with pytest.raises(RuntimeError, match="Unsafe tar member"):
        ssh_client._extract_tarball(tar_path, dest)
    assert not (tmp_path / "escape.txt").exists()


def test_extract_rejects_absolute_paths(tmp_path):
    """Absolute member paths must be refused as well."""
    import tarfile, io

    tar_path = tmp_path / "abs.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        data = b"x"
        info = tarfile.TarInfo(name="/etc/passwd")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))

    dest = tmp_path / "dest"
    with pytest.raises(RuntimeError, match="Unsafe tar member"):
        ssh_client._extract_tarball(tar_path, dest)


def test_empty_dir_roundtrip(tmp_path):
    src = tmp_path / "empty"
    src.mkdir()
    tar_path = tmp_path / "e.tar.gz"
    ssh_client._make_local_tarball(src, tar_path)

    dest = tmp_path / "out"
    ssh_client._extract_tarball(tar_path, dest)
    assert dest.is_dir()
    assert list(dest.iterdir()) == []

def _fake_tar_setup(tmp_path):
    """Build a real tarball that the mocked SFTP can return."""
    from remote import ssh_client as sc
    fake_remote = tmp_path / "fake_remote"
    _make_tree(fake_remote)
    fake_tar = tmp_path / "fake.tar.gz"
    sc._make_local_tarball(fake_remote, fake_tar)
    return fake_tar, fake_tar.stat().st_size


def _install_fake_run(monkeypatch, tar_rc: int, tar_bytes: int, tar_err: str = ""):
    """Patch ssh_client.run + sftp to simulate a remote tar workflow."""
    from remote import ssh_client as sc

    def fake_run(cluster, cmd, timeout=60):
        if cmd.startswith("cd ") and "tar " in cmd:
            return tar_rc, "", tar_err
        if cmd.startswith("stat -c %s"):
            # Return zero bytes if tar was supposed to fail catastrophically
            return 0, f"{tar_bytes}\n", ""
        if cmd.startswith("rm -f"):
            return 0, "", ""
        return 0, "", ""

    monkeypatch.setattr(sc, "run", fake_run)


def test_download_tar_rc1_tolerated_on_partial(tmp_path, monkeypatch):
    """rc=1 must NOT raise on partial download."""
    from remote import ssh_client as sc
    fake_tar, sz = _fake_tar_setup(tmp_path)
    _install_fake_run(monkeypatch, tar_rc=1, tar_bytes=sz,
                      tar_err="tar: file changed as we read it")

    class FakeSFTP:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, remote, local):
            Path(local).write_bytes(fake_tar.read_bytes())

    monkeypatch.setattr(sc, "sftp", lambda cluster: FakeSFTP())

    dest = tmp_path / "out_partial"
    sc.download_dir_tar(cluster=None, remote_dir="/remote/job",
                        local_dir=dest, partial=True)
    assert (dest / "coord").exists()


def test_download_tar_rc1_tolerated_on_full(tmp_path, monkeypatch):
    """rc=1 must also NOT raise on full download — tar still wrote the
    archive correctly. This is the regression that bit the user on a
    COMPLETED job."""
    from remote import ssh_client as sc
    fake_tar, sz = _fake_tar_setup(tmp_path)
    _install_fake_run(monkeypatch, tar_rc=1, tar_bytes=sz,
                      tar_err="tar: file changed as we read it")

    class FakeSFTP:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, remote, local):
            Path(local).write_bytes(fake_tar.read_bytes())

    monkeypatch.setattr(sc, "sftp", lambda cluster: FakeSFTP())

    dest = tmp_path / "out_full"
    sc.download_dir_tar(cluster=None, remote_dir="/remote/job",
                        local_dir=dest, partial=False)
    assert (dest / "coord").exists()
    assert (dest / "subdir" / "nested.txt").exists()


def test_download_tar_rc2_always_fatal(tmp_path, monkeypatch):
    """rc>=2 is always a real error, regardless of partial flag."""
    from remote import ssh_client as sc
    _install_fake_run(monkeypatch, tar_rc=2, tar_bytes=0,
                      tar_err="tar: Cannot open: No such file or directory")

    monkeypatch.setattr(
        sc, "sftp",
        lambda cluster: (_ for _ in ()).throw(
            AssertionError("SFTP must not be reached on rc>=2"))
    )

    for partial in (False, True):
        with pytest.raises(RuntimeError, match="Remote tar failed"):
            sc.download_dir_tar(cluster=None, remote_dir="/missing",
                                local_dir=tmp_path / f"out_{partial}",
                                partial=partial)


def test_download_tar_empty_archive_raises(tmp_path, monkeypatch):
    """If tar reports rc<=1 but the archive is missing/empty (disk full,
    quota, etc.) we must escalate rather than silently produce an empty
    download dir."""
    from remote import ssh_client as sc
    _install_fake_run(monkeypatch, tar_rc=0, tar_bytes=0,
                      tar_err="")

    monkeypatch.setattr(
        sc, "sftp",
        lambda cluster: (_ for _ in ()).throw(
            AssertionError("SFTP must not be reached on empty archive"))
    )

    with pytest.raises(RuntimeError, match="produced no archive"):
        sc.download_dir_tar(cluster=None, remote_dir="/x",
                            local_dir=tmp_path / "out", partial=False)