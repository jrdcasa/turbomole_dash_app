"""Tests for backend.opt_trajectory."""

from pathlib import Path

import pytest

from backend.opt_trajectory import (
    BOHR_TO_ANGSTROM, DEFAULT_THRESHOLDS,
    parse_energy, parse_gradient, parse_thresholds_from_control,
    load_trajectory, OptTrajectory,
)


_ENERGY_FILE = """$energy
   1   -76.420000000000   76.0   -152.4
   2   -76.421000000000   76.1   -152.5
   3   -76.423521234560   76.2   -152.6
$end
"""

# Two-cycle H2O gradient block (Bohr).
_GRADIENT_FILE = """$grad   cartesian gradients
  cycle =  1  SCF energy = -76.420000000000  |dE/dxyz| = 1.234E-02
    0.000000000   0.000000000   0.000000000  o
    1.808845000   0.000000000   0.000000000  h
    0.000000000   1.808845000   0.000000000  h
    1.000000E-02   0.000000E+00   0.000000E+00
   -5.000000E-03   0.000000E+00   0.000000E+00
    0.000000E+00  -5.000000E-03   0.000000E+00
  cycle =  2  SCF energy = -76.423521234560  |dE/dxyz| = 1.234E-04
    0.000000000   0.000000000   0.000000000  o
    1.810000000   0.000000000   0.000000000  h
    0.000000000   1.810000000   0.000000000  h
    1.000000E-04   0.000000E+00   0.000000E+00
   -5.000000E-05   0.000000E+00   0.000000E+00
    0.000000E+00  -5.000000E-05   0.000000E+00
$end
"""

_CONTROL_FILE = """$title
$symmetry c1
$coord file=coord
$jobex
    energy=7
    gcart=4
$end
$dft
   functional b-p
   gridsize m4
$end
$scfconv 7
$end
"""


# ---------------------------------------------------------------------------
# energy file
# ---------------------------------------------------------------------------

def test_parse_energy_three_cycles():
    rows = parse_energy(_ENERGY_FILE)
    assert len(rows) == 3
    assert rows[0][0] == 1
    assert abs(rows[2][1] - (-76.423521234560)) < 1e-12


def test_parse_energy_handles_short_lines():
    rows = parse_energy("$energy\n   1   -76.42\n$end\n")
    assert rows == [(1, -76.42, None, None)]


def test_parse_energy_from_path(tmp_path):
    p = tmp_path / "energy"
    p.write_text(_ENERGY_FILE)
    rows = parse_energy(p)
    assert len(rows) == 3


def test_parse_energy_empty():
    assert parse_energy("") == []


# ---------------------------------------------------------------------------
# gradient file
# ---------------------------------------------------------------------------

def test_parse_gradient_two_cycles():
    cycles = parse_gradient(_GRADIENT_FILE)
    assert len(cycles) == 2
    assert cycles[0].cycle == 1
    assert abs(cycles[0].scf_energy - (-76.420000000000)) < 1e-12
    assert abs(cycles[1].grad_norm - 1.234e-04) < 1e-9


def test_parse_gradient_geometry_in_bohr():
    cycles = parse_gradient(_GRADIENT_FILE)
    last = cycles[-1]
    assert len(last.coords_bohr) == 3
    sym, x, _, _ = last.coords_bohr[1]
    assert sym == "h"
    assert abs(x - 1.810000000) < 1e-9


def test_parse_gradient_grad_max():
    cycles = parse_gradient(_GRADIENT_FILE)
    assert abs(cycles[0].grad_max - 1.0e-2) < 1e-9
    assert abs(cycles[1].grad_max - 1.0e-4) < 1e-9


def test_parse_gradient_handles_fortran_d_notation():
    text = """$grad
  cycle =  1  SCF energy = -1.0  |dE/dxyz| = 1.0E-03
    0.0   0.0   0.0  h
    0.12345D-03   0.0   0.0
$end
"""
    cycles = parse_gradient(text)
    assert len(cycles) == 1
    assert abs(cycles[0].grad_max - 0.12345e-3) < 1e-9


# ---------------------------------------------------------------------------
# control thresholds
# ---------------------------------------------------------------------------

def test_parse_thresholds_defaults_when_no_jobex_block():
    thr = parse_thresholds_from_control("$end\n")
    assert thr == DEFAULT_THRESHOLDS


def test_parse_thresholds_from_jobex_block():
    thr = parse_thresholds_from_control(_CONTROL_FILE)
    assert thr["energy_change"] == pytest.approx(1e-7)
    assert thr["gradient_max"] == pytest.approx(1e-4)


# ---------------------------------------------------------------------------
# load_trajectory + OptTrajectory API
# ---------------------------------------------------------------------------

def _setup_job_dir(tmp_path: Path) -> Path:
    d = tmp_path / "job_dir"
    d.mkdir()
    (d / "energy").write_text(_ENERGY_FILE)
    (d / "gradient").write_text(_GRADIENT_FILE)
    (d / "control").write_text(_CONTROL_FILE)
    return d


def test_load_trajectory_merges_energy_and_gradient(tmp_path):
    traj = load_trajectory(_setup_job_dir(tmp_path))
    assert traj.n_cycles == 2
    # SCFKIN from `energy` is merged in by matching cycle number;
    # `_GRADIENT_FILE` has cycles 1 and 2, so we read rows 1 and 2 of energy.
    assert traj.cycles[0].scf_kinetic == pytest.approx(76.0)
    assert traj.cycles[1].scf_kinetic == pytest.approx(76.1)


def test_load_trajectory_thresholds_from_control(tmp_path):
    traj = load_trajectory(_setup_job_dir(tmp_path))
    assert traj.thresholds["energy_change"] == pytest.approx(1e-7)


def test_load_trajectory_last_geometry_in_angstrom(tmp_path):
    traj = load_trajectory(_setup_job_dir(tmp_path))
    assert len(traj.last_geometry) == 3
    sym, x, y, z = traj.last_geometry[1]
    assert sym == "H"
    assert abs(x - 1.810000000 * BOHR_TO_ANGSTROM) < 1e-9


def test_xyz_rendering(tmp_path):
    traj = load_trajectory(_setup_job_dir(tmp_path))
    xyz = traj.to_xyz()
    lines = xyz.splitlines()
    assert lines[0] == "3"
    assert lines[2].split()[0] == "O"
    assert len(lines) == 5      # natoms + comment + 3 atoms


def test_convergence_evaluation(tmp_path):
    """With energy=7 and gcart=4 in $jobex, the last cycle (|ΔE|≈3.5e-3,
    |grad|max=1e-4) should fail the energy criterion. The gradient_max
    threshold is 1e-4 too, so |1e-4 < 1e-4| is False → not converged."""
    traj = load_trajectory(_setup_job_dir(tmp_path))
    st = traj.convergence_status()
    assert st["energy_change"]["ok"] is False
    assert st["gradient_max"]["ok"] is False
    assert traj.converged() is False


def test_load_trajectory_missing_gradient_fallback(tmp_path):
    """If only `energy` is available (job mid-flight), build cycles from
    it and leave last_geometry empty."""
    d = tmp_path / "j"
    d.mkdir()
    (d / "energy").write_text(_ENERGY_FILE)
    traj = load_trajectory(d)
    assert traj.n_cycles == 3
    assert traj.last_geometry == []
    assert traj.to_xyz() == ""


def test_empty_dir():
    traj = load_trajectory(Path("/nonexistent/dir/here"))
    assert traj.n_cycles == 0
    assert traj.last_geometry == []

def test_trajectory_xyz_multiframe(tmp_path):
    from backend.opt_trajectory import trajectory_xyz, BOHR_TO_ANGSTROM
    d = _setup_job_dir(tmp_path)
    xyz = trajectory_xyz(d, name="h2o")
    frames = xyz.strip().split("\n")
    # 2 cycles * (1 header + 1 comment + 3 atoms) = 10 lines
    assert len(frames) == 10
    assert frames[0] == "3"
    # Cycle 2 layout (0-indexed):
    #   frames[5] = header "3"
    #   frames[6] = comment
    #   frames[7] = O at (0,0,0)
    #   frames[8] = H at (1.81,0,0) Bohr
    #   frames[9] = H at (0,1.81,0) Bohr
    assert frames[5] == "3"
    parts = frames[8].split()
    assert parts[0] == "H"
    assert abs(float(parts[1]) - 1.81 * BOHR_TO_ANGSTROM) < 1e-6


def test_trajectory_xyz_no_gradient(tmp_path):
    from backend.opt_trajectory import trajectory_xyz
    d = tmp_path / "empty"
    d.mkdir()
    assert trajectory_xyz(d) == ""