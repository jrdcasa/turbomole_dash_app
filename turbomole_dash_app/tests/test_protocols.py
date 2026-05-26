"""Tests for backend.protocols."""

import json
from pathlib import Path

import pytest

from backend import protocols as proto_mod


@pytest.fixture
def empty_dir(tmp_path):
    d = tmp_path / "protocols"
    d.mkdir()
    return d


def _sample_payload():
    return {
        "method": {
            "functional": "PBE0",
            "basis_set": "def2-TZVP",
            "use_ri": True,
            "grid": "m5",
        },
        "task": {
            "type": "optimization",
            "charge": 0,
            "multiplicity": 1,
            "aimd_steps": 500,
            "aimd_timestep_fs": 0.5,
            "aimd_temperature_K": 300,
        },
        "submission": {
            "cluster": "drago",
            "partition": "generic",
            "walltime": "24:00:00",
            "nodes": 1,
            "ntasks": 48,
            "mem": "190G",
            "reservation": "",
        },
    }


def test_list_protocols_on_empty_dir(empty_dir):
    assert proto_mod.list_protocols(empty_dir) == []


def test_save_and_load_roundtrip(empty_dir):
    p = _sample_payload()
    path = proto_mod.save_protocol(
        empty_dir, "DFT opt organom.", "desc text",
        p["method"], p["task"], p["submission"],
    )
    assert path.exists()
    loaded = proto_mod.load_protocol(path)
    assert loaded["name"] == "DFT opt organom."
    assert loaded["description"] == "desc text"
    assert loaded["method"]["functional"] == "PBE0"
    assert loaded["submission"]["ntasks"] == 48


def test_safe_filename_special_chars(empty_dir):
    p = _sample_payload()
    path = proto_mod.save_protocol(
        empty_dir, "BP86 + D3 / TZVP", "",
        p["method"], p["task"], p["submission"],
    )
    # No slashes, spaces or plus signs in the filename
    assert "/" not in path.name
    assert " " not in path.name
    assert "+" not in path.name
    assert path.name.endswith(".json")


def test_overwrite_default(empty_dir):
    p = _sample_payload()
    proto_mod.save_protocol(empty_dir, "test", "v1",
                            p["method"], p["task"], p["submission"])
    p["method"]["functional"] = "B3LYP"
    proto_mod.save_protocol(empty_dir, "test", "v2",
                            p["method"], p["task"], p["submission"])
    files = list(empty_dir.glob("*.json"))
    assert len(files) == 1
    loaded = proto_mod.load_protocol(files[0])
    assert loaded["method"]["functional"] == "B3LYP"
    assert loaded["description"] == "v2"


def test_overwrite_false_raises(empty_dir):
    p = _sample_payload()
    proto_mod.save_protocol(empty_dir, "test", "",
                            p["method"], p["task"], p["submission"])
    with pytest.raises(FileExistsError):
        proto_mod.save_protocol(empty_dir, "test", "",
                                p["method"], p["task"], p["submission"],
                                overwrite=False)


def test_list_sorted_by_name(empty_dir):
    p = _sample_payload()
    for n in ("zebra", "alpha", "mid"):
        proto_mod.save_protocol(empty_dir, n, "",
                                p["method"], p["task"], p["submission"])
    names = [x["name"] for x in proto_mod.list_protocols(empty_dir)]
    assert names == ["alpha", "mid", "zebra"]


def test_list_skips_invalid_json(empty_dir):
    (empty_dir / "garbage.json").write_text("this is not json {{{")
    (empty_dir / "valid.json").write_text(json.dumps({
        "name": "good", "method": {}, "task": {}, "submission": {}
    }))
    items = proto_mod.list_protocols(empty_dir)
    assert len(items) == 1
    assert items[0]["name"] == "good"


def test_load_fills_missing_keys_from_defaults(empty_dir):
    # Old-style file missing whole sections
    p = empty_dir / "old.json"
    p.write_text(json.dumps({"name": "minimal", "method": {"functional": "BP86"}}))
    loaded = proto_mod.load_protocol(p)
    # Defaults must have been merged in
    assert loaded["method"]["basis_set"] == "def2-SVP"
    assert loaded["task"]["type"] == "single_point"
    assert "submission" in loaded


def test_delete_protocol(empty_dir):
    p = _sample_payload()
    path = proto_mod.save_protocol(empty_dir, "todelete", "",
                                   p["method"], p["task"], p["submission"])
    assert path.exists()
    assert proto_mod.delete_protocol(path) is True
    assert not path.exists()
    assert proto_mod.delete_protocol(path) is False   # already gone


def test_defaults_payload_with_cluster():
    payload = proto_mod.defaults_payload(first_cluster="my_cluster")
    assert payload["submission"]["cluster"] == "my_cluster"
    assert payload["method"]["functional"] == "BP86"
    assert payload["task"]["type"] == "single_point"


def test_defaults_payload_no_cluster():
    payload = proto_mod.defaults_payload(first_cluster=None)
    assert payload["submission"]["cluster"] is None


def test_dispersion_in_defaults():
    payload = proto_mod.defaults_payload()
    assert payload["method"]["dispersion"] == "none"


def test_dispersion_roundtrip(empty_dir):
    p = _sample_payload()
    p["method"]["dispersion"] = "d3bj"
    path = proto_mod.save_protocol(empty_dir, "with_d3", "",
                                   p["method"], p["task"], p["submission"])
    loaded = proto_mod.load_protocol(path)
    assert loaded["method"]["dispersion"] == "d3bj"


def test_old_protocol_without_dispersion_gets_default(empty_dir):
    """Old protocols saved before dispersion existed must load cleanly
    with dispersion='none'."""
    import json as _json
    p = empty_dir / "old.json"
    p.write_text(_json.dumps({
        "name": "legacy",
        "method": {"functional": "BP86", "basis_set": "def2-SVP",
                   "use_ri": True, "grid": "m4"},
        "task": {"type": "single_point"},
        "submission": {},
    }))
    loaded = proto_mod.load_protocol(p)
    assert loaded["method"]["dispersion"] == "none"
