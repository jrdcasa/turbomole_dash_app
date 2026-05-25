"""Config loading tests."""

from pathlib import Path

import pytest

from backend.config import AppConfig, RemoteCluster, load_config


def test_load_config_returns_appconfig():
    cfg = load_config()
    assert isinstance(cfg, AppConfig)


def test_at_least_one_cluster_defined():
    cfg = load_config()
    assert cfg.clusters, "No clusters found — did you create config.yaml?"


def test_clusters_have_required_fields():
    cfg = load_config()
    for name, c in cfg.clusters.items():
        assert isinstance(c, RemoteCluster)
        assert c.host,           f"cluster '{name}': host is empty"
        assert c.user,           f"cluster '{name}': user is empty"
        assert c.remote_workdir, f"cluster '{name}': remote_workdir is empty"


def test_local_dirs_exist_after_load():
    cfg = load_config()
    assert cfg.db_path.parent.is_dir()
    assert cfg.local_workdir.is_dir()
    assert cfg.download_dir.is_dir()


def test_clusters_have_turbomole_version():
    cfg = load_config()
    for name, c in cfg.clusters.items():
        assert c.turbomole_version, f"cluster '{name}': turbomole_version is empty"
