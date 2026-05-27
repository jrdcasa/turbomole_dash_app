"""Tests for CSV and gnuplot script generators."""

import csv
import io

import pytest

from backend.analysis_exports import (
    JobMeta,
    gnuplot_optimization, gnuplot_spectrum, gnuplot_trajectory,
    single_point_to_csv, spectrum_to_csv, trajectory_to_csv,
)
from backend.opt_trajectory import OptCycle, OptTrajectory
from backend.result_parser import AoforceSummary, RidftSummary


def _sample_traj() -> OptTrajectory:
    return OptTrajectory(
        cycles=[
            OptCycle(cycle=1, scf_energy=-76.42, grad_norm=1e-2, grad_max=2e-2),
            OptCycle(cycle=2, scf_energy=-76.43, grad_norm=1e-3, grad_max=2e-3),
            OptCycle(cycle=3, scf_energy=-76.435, grad_norm=1e-4, grad_max=2e-4),
        ],
        last_geometry=[("O", 0.0, 0.0, 0.0)],
    )


def test_trajectory_csv_header_and_rows():
    csv_text = trajectory_to_csv(_sample_traj())
    rows = list(csv.reader(io.StringIO(csv_text)))
    assert rows[0] == ["cycle", "scf_energy_Ha", "dE_vs_first_Ha",
                       "grad_norm", "grad_max"]
    assert len(rows) == 4
    assert rows[1][0] == "1"
    # dE for first row must be zero (or near-zero)
    assert abs(float(rows[1][2])) < 1e-15
    # Third row: ΔE = -0.015
    assert abs(float(rows[3][2]) - (-0.015)) < 1e-9


def test_trajectory_csv_empty():
    csv_text = trajectory_to_csv(OptTrajectory())
    rows = list(csv.reader(io.StringIO(csv_text)))
    assert rows == [["cycle", "scf_energy_Ha", "dE_vs_first_Ha",
                     "grad_norm", "grad_max"]]


def test_spectrum_csv_marks_imaginary():
    s = AoforceSummary(frequencies_cm1=[-512.3, 0.0, 1592.0, 3700.0])
    csv_text = spectrum_to_csv(s)
    rows = list(csv.reader(io.StringIO(csv_text)))
    assert rows[0] == ["mode", "frequency_cm-1", "kind"]
    kinds = [r[2] for r in rows[1:]]
    assert kinds == ["imaginary", "near_zero", "real", "real"]


def test_single_point_csv_keyvalue():
    s = RidftSummary(total_energy_ha=-76.42352, scf_iterations=8,
                     converged=True)
    csv_text = single_point_to_csv(s)
    rows = list(csv.reader(io.StringIO(csv_text)))
    assert rows[0] == ["key", "value"]
    keys = [r[0] for r in rows[1:]]
    assert "Total energy" in keys
    assert "SCF iterations" in keys


# ---------------------------------------------------------------------------
# gnuplot scripts: structural checks (we don't actually run gnuplot here)
# ---------------------------------------------------------------------------

def test_gnuplot_optimization_includes_template_and_filename():
    s = gnuplot_optimization("traj.csv",
                             JobMeta(name="h2o", functional="BP86",
                                     basis_set="def2-SVP"))
    assert "set style line 1" in s            # style block kept verbatim
    assert "set encoding utf8" in s
    assert 'f1="traj.csv"' in s
    assert "conv=627.5092" in s
    assert "kcal/mol" in s
    assert "(BP86/def2-SVP)" in s


def test_gnuplot_trajectory_includes_log_axis():
    s = gnuplot_trajectory("traj.csv",
                           JobMeta(name="h2o", functional="BP86",
                                   basis_set="def2-SVP"))
    assert "set logscale y" in s
    assert "Gradient" in s


def test_gnuplot_spectrum_uses_impulses_and_reversed_axis():
    s = gnuplot_spectrum("spec.csv",
                         JobMeta(name="h2o", functional="BP86",
                                 basis_set="def2-SVP"))
    assert "impulses" in s
    assert "reverse" in s
    assert "cm^{-1}" in s


def test_gnuplot_title_handles_missing_method():
    s = gnuplot_optimization("traj.csv",
                             JobMeta(name="h2o", functional=None,
                                     basis_set=None))
    # No method in parentheses
    assert "()" not in s
    assert "h2o" in s


def test_gnuplot_escapes_quotes_in_title():
    s = gnuplot_optimization('traj.csv',
                             JobMeta(name='weird "name"',
                                     functional="HF", basis_set="def2-SVP"))
    assert r'weird \"name\"' in s          # escaped properly

# ---------------------------------------------------------------------------
# SCF convergence exporters
# ---------------------------------------------------------------------------

from backend.analysis_exports import (
    gnuplot_scf_convergence, scf_convergence_to_csv,
)


def test_scf_convergence_csv_header_and_diff():
    csv_text = scf_convergence_to_csv([(1, -76.40), (2, -76.42), (3, -76.421)])
    rows = list(csv.reader(io.StringIO(csv_text)))
    assert rows[0] == ["iteration", "scf_energy_Ha", "dE_vs_prev_Ha"]
    assert rows[1][2] == ""                            # first row: no diff
    assert abs(float(rows[2][2]) - (-0.02)) < 1e-9
    assert abs(float(rows[3][2]) - (-0.001)) < 1e-9


def test_scf_convergence_csv_empty():
    csv_text = scf_convergence_to_csv([])
    rows = list(csv.reader(io.StringIO(csv_text)))
    assert rows == [["iteration", "scf_energy_Ha", "dE_vs_prev_Ha"]]


def test_gnuplot_scf_convergence_references_csv():
    s = gnuplot_scf_convergence("foo_scf.csv",
                                JobMeta(name="h2o", functional="BP86",
                                        basis_set="def2-SVP"))
    assert 'f1="foo_scf.csv"' in s
    assert "logscale y" in s
    assert "kcal/mol" in s
    assert "SCF iteration" in s