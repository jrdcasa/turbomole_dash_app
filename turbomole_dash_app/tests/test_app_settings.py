"""Tests for backend.app_settings."""

from backend import app_settings


def test_load_defaults_when_no_file(tmp_path):
    s = app_settings.load_settings(tmp_path)
    assert s.external_tools.vmd == ""
    assert s.external_tools.cosmobuild == ""
    assert s.external_tools.cosmoquick == ""


def test_roundtrip(tmp_path):
    s = app_settings.AppSettings(
        external_tools=app_settings.ExternalTools(
            vmd="/opt/vmd/bin/vmd",
            cosmobuild="/opt/cosmobuild",
            cosmoquick="",
        )
    )
    app_settings.save_settings(tmp_path, s)
    loaded = app_settings.load_settings(tmp_path)
    assert loaded.external_tools.vmd == "/opt/vmd/bin/vmd"
    assert loaded.external_tools.cosmobuild == "/opt/cosmobuild"
    assert loaded.external_tools.cosmoquick == ""


def test_load_ignores_garbage_file(tmp_path):
    (tmp_path / "app_settings.json").write_text("not valid json {{{")
    s = app_settings.load_settings(tmp_path)
    assert s.external_tools.vmd == ""


def test_load_tolerates_partial_file(tmp_path):
    """Missing keys should fall back to empty strings, not crash."""
    (tmp_path / "app_settings.json").write_text(
        '{"external_tools": {"vmd": "/path/to/vmd"}}'
    )
    s = app_settings.load_settings(tmp_path)
    assert s.external_tools.vmd == "/path/to/vmd"
    assert s.external_tools.cosmobuild == ""