"""Tests for the mdprep.inp generator and the AIMD wiring."""

from __future__ import annotations

import math

import pytest
from ase.build import molecule

from backend.opt_trajectory import BOHR_TO_ANGSTROM
from backend.turbomole_io import (
    AU_TIME_TO_FS,
    CONSTRAINT_ALGORITHMS,
    CalcSpec,
    build_control_file,
    build_mdprep_input,
    parse_constraints,
    turbomole_driver_commands,
)


# ---------------------------------------------------------------------------
# parse_constraints
# ---------------------------------------------------------------------------

def test_parse_constraints_empty():
    assert parse_constraints("") == []
    assert parse_constraints("   ;  ;   ") == []


def test_parse_constraints_single():
    assert parse_constraints("1 2 1.09") == [(1, 2, 1.09)]


def test_parse_constraints_multiple():
    out = parse_constraints("1 2 1.09; 3 4 1.54 ;  5 6 0.96")
    assert out == [(1, 2, 1.09), (3, 4, 1.54), (5, 6, 0.96)]


def test_parse_constraints_rejects_same_atom():
    with pytest.raises(ValueError, match="must differ"):
        parse_constraints("1 1 1.0")


def test_parse_constraints_rejects_negative_distance():
    with pytest.raises(ValueError, match="positive"):
        parse_constraints("1 2 -1.0")


def test_parse_constraints_rejects_zero_index():
    with pytest.raises(ValueError, match="1-based"):
        parse_constraints("0 1 1.0")


def test_parse_constraints_rejects_malformed():
    with pytest.raises(ValueError, match="Invalid constraint"):
        parse_constraints("not even close")


def test_parse_constraints_accepts_scientific_notation():
    out = parse_constraints("1 2 1.09e0")
    assert out == [(1, 2, 1.09)]


# ---------------------------------------------------------------------------
# build_mdprep_input — basic structure
# ---------------------------------------------------------------------------

def _spec(**overrides) -> CalcSpec:
    base = dict(
        functional="BP86", basis_set="def2-SVP", task_type="aimd",
        aimd_steps=256, aimd_timestep_au=80.0, aimd_temperature_K=300.0,
    )
    base.update(overrides)
    return CalcSpec(**base)


def test_mdprep_default_has_seven_steps_in_order():
    """No constraints: the sequence should be q, q, q, q, i T q, i dt q q q,
    i N q (the first 4 `q`s correspond to menu items 1-4, the rest are the
    customised steps 5-7)."""
    text = build_mdprep_input(_spec())
    lines = [l for l in text.splitlines() if l]   # strip blank padding

    # Step 1-3: three plain `q`s (num atoms, coord, cavity barrier)
    assert lines[:3] == ["q", "q", "q"]
    # Step 4 (no constraints): one more `q`
    assert lines[3] == "q"
    # Step 5: initial velocity / temperature
    assert lines[4] == "i"
    assert lines[5] == "300"
    assert lines[6] == "q"
    # Step 6: timestep in a.u.
    assert lines[7] == "i"
    assert lines[8] == "80"
    assert lines[9:12] == ["q", "q", "q"]
    # Step 7: number of MD steps
    assert lines[12] == "i"
    assert lines[13] == "256"
    assert lines[14] == "q"


def test_mdprep_includes_safety_padding():
    text = build_mdprep_input(_spec())
    # The renderer appends a few trailing blank lines so any unexpected
    # extra prompt picks the default. Make sure they're there.
    assert text.endswith("\n\n\n\n")


def test_mdprep_temperature_default_300_K_in_velocity_block():
    text = build_mdprep_input(_spec(aimd_temperature_K=350.0))
    lines = [l for l in text.splitlines() if l]
    # After the 4 leading `q`s and `i` we should see the temperature.
    i_idx = lines.index("i")
    assert lines[i_idx + 1] == "350"


def test_mdprep_custom_timestep_and_steps():
    text = build_mdprep_input(_spec(aimd_timestep_au=40.0, aimd_steps=1000))
    assert "40" in text.splitlines()
    assert "1000" in text.splitlines()


def test_mdprep_rejects_non_aimd_task():
    spec = CalcSpec(functional="BP86", basis_set="def2-SVP",
                    task_type="single_point")
    with pytest.raises(ValueError, match="task_type='aimd'"):
        build_mdprep_input(spec)


def test_mdprep_validate_rejects_zero_temperature():
    with pytest.raises(ValueError, match="aimd_temperature_K"):
        build_mdprep_input(_spec(aimd_temperature_K=0.0))


def test_mdprep_validate_rejects_zero_timestep():
    with pytest.raises(ValueError, match="aimd_timestep_au"):
        build_mdprep_input(_spec(aimd_timestep_au=0.0))


def test_mdprep_validate_rejects_zero_steps():
    with pytest.raises(ValueError, match="aimd_steps"):
        build_mdprep_input(_spec(aimd_steps=0))


# ---------------------------------------------------------------------------
# Distance constraints branch
# ---------------------------------------------------------------------------

def test_mdprep_with_single_constraint_inserts_full_submenu():
    spec = _spec(
        aimd_use_constraints=True,
        aimd_constraint_algorithm="shake",
        aimd_constraints=[(1, 2, 1.09)],   # 1.09 Å (~ C-H)
    )
    text = build_mdprep_input(spec)
    lines = text.splitlines()
    # Submenu starts with a/g/<alg>/e/<i j d_bohr>/q
    seq = ["a", "g", "shake", "e"]
    for needle in seq:
        assert needle in lines

    # Find the line with the distance in Bohr; must match Å/0.5291...
    expected_bohr = 1.09 / BOHR_TO_ANGSTROM
    dist_line = next(l for l in lines if l.lstrip().startswith("1 2 "))
    parts = dist_line.split()
    assert int(parts[0]) == 1
    assert int(parts[1]) == 2
    assert math.isclose(float(parts[2]), expected_bohr, rel_tol=1e-9)


def test_mdprep_with_multiple_constraints_reuses_e_for_each():
    spec = _spec(
        aimd_use_constraints=True,
        aimd_constraint_algorithm="maltshake",
        aimd_constraints=[(1, 2, 1.09), (3, 4, 1.54)],
    )
    text = build_mdprep_input(spec)
    lines = text.splitlines()

    # 'maltshake' is the algorithm token
    assert "maltshake" in lines
    # Two 'e' inputs, one per constraint
    assert lines.count("e") == 2

    # Constraint distances appear in Bohr (1.09 Å and 1.54 Å)
    bohr_109 = 1.09 / BOHR_TO_ANGSTROM
    bohr_154 = 1.54 / BOHR_TO_ANGSTROM
    line_1 = next(l for l in lines if l.lstrip().startswith("1 2 "))
    line_2 = next(l for l in lines if l.lstrip().startswith("3 4 "))
    assert math.isclose(float(line_1.split()[2]), bohr_109, rel_tol=1e-9)
    assert math.isclose(float(line_2.split()[2]), bohr_154, rel_tol=1e-9)


def test_mdprep_constraint_submenu_terminates_with_q():
    spec = _spec(
        aimd_use_constraints=True,
        aimd_constraint_algorithm="shake",
        aimd_constraints=[(1, 2, 1.09)],
    )
    text = build_mdprep_input(spec)
    lines = text.splitlines()
    # Last `e <i j d>` line must be followed by a `q` (close submenu)
    e_idx = max(i for i, l in enumerate(lines) if l == "e")
    # next non-empty line after the distance is q
    after = [l for l in lines[e_idx + 2:] if l]
    assert after and after[0] == "q"


def test_mdprep_without_constraints_skips_submenu():
    text = build_mdprep_input(_spec(
        aimd_use_constraints=False,
        aimd_constraint_algorithm="shake",
        aimd_constraints=[],
    ))
    # No 'a' / 'g' / 'e' tokens when constraints are off
    lines = text.splitlines()
    assert "a" not in lines
    assert "g" not in lines
    assert "e" not in lines


def test_mdprep_validates_use_without_constraints():
    spec = CalcSpec(
        functional="BP86", basis_set="def2-SVP", task_type="aimd",
        aimd_use_constraints=True, aimd_constraints=[],
    )
    with pytest.raises(ValueError, match="no constraints were provided"):
        build_mdprep_input(spec)


def test_mdprep_validates_unknown_algorithm():
    spec = CalcSpec(
        functional="BP86", basis_set="def2-SVP", task_type="aimd",
        aimd_constraint_algorithm="wrong",
    )
    with pytest.raises(ValueError, match="Unknown constraint algorithm"):
        spec.validate()


def test_supported_constraint_algorithms_exposed():
    """The UI uses available_constraint_algorithms() to populate the
    dropdown. Make sure both expected entries are present."""
    assert set(CONSTRAINT_ALGORITHMS) == {"shake", "maltshake"}


# ---------------------------------------------------------------------------
# build_control_file emits mdprep.inp only for AIMD
# ---------------------------------------------------------------------------

def test_build_control_writes_mdprep_for_aimd(tmp_path):
    atoms = molecule("H2O")
    out = tmp_path / "j"
    build_control_file(
        atoms, out,
        functional="BP86", basis_set="def2-SVP", task_type="aimd",
        aimd_steps=10, aimd_timestep_au=80.0, aimd_temperature_K=300.0,
        turbomole_version="7.8",
    )
    mdprep = out / "mdprep.inp"
    assert mdprep.exists()
    text = mdprep.read_text()
    # Sanity: contains the timestep we passed and the step count
    assert "80" in text.splitlines()
    assert "10" in text.splitlines()


def test_build_control_no_mdprep_for_single_point(tmp_path):
    atoms = molecule("H2O")
    out = tmp_path / "j"
    build_control_file(
        atoms, out,
        functional="BP86", basis_set="def2-SVP", task_type="single_point",
        turbomole_version="7.8",
    )
    assert not (out / "mdprep.inp").exists()


def test_build_control_aimd_with_constraints_roundtrip(tmp_path):
    atoms = molecule("H2O")
    out = tmp_path / "j"
    build_control_file(
        atoms, out,
        functional="BP86", basis_set="def2-SVP", task_type="aimd",
        aimd_use_constraints=True,
        aimd_constraint_algorithm="shake",
        aimd_constraints=[(1, 2, 0.96)],
        turbomole_version="7.8",
    )
    text = (out / "mdprep.inp").read_text()
    assert "shake" in text.splitlines()
    # Distance was given in Å, must be persisted in Bohr
    assert any(l.startswith("1 2 ") for l in text.splitlines())


# ---------------------------------------------------------------------------
# AIMD driver commands wiring
# ---------------------------------------------------------------------------

def test_aimd_driver_uses_mdprep_inp():
    """The shell script must feed mdprep.inp into mdprep, not run
    `mdprep -default` blindly."""
    cmds = turbomole_driver_commands("aimd", turbomole_version="7.8")
    joined = "\n".join(cmds)
    assert "mdprep < mdprep.inp" in joined
    assert "mdprep -default" not in joined
    # frog still runs after mdprep
    assert "frog" in joined


def test_aimd_driver_order_define_ridft_mdprep_frog():
    cmds = turbomole_driver_commands("aimd", turbomole_version="7.8")
    idx_define = next(i for i, l in enumerate(cmds) if "define <" in l)
    idx_ridft  = next(i for i, l in enumerate(cmds) if "ridft >" in l)
    idx_mdprep = next(i for i, l in enumerate(cmds) if "mdprep <" in l)
    idx_frog   = next(i for i, l in enumerate(cmds) if "frog >" in l)
    assert idx_define < idx_ridft < idx_mdprep < idx_frog


# ---------------------------------------------------------------------------
# Sanity check on the AU→fs conversion exported for the UI label
# ---------------------------------------------------------------------------

def test_au_time_to_fs_constant():
    # 80 a.u. of time ≈ 1.935 fs (well-known value used as default timestep)
    assert math.isclose(80.0 * AU_TIME_TO_FS, 1.9351, abs_tol=1e-3)