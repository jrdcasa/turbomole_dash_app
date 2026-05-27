"""Tests for backend.analysis_writer."""

from pathlib import Path

import pytest

from backend.analysis_writer import ANALYSIS_SUBDIR, write_artifacts


_ENERGY = """$energy
   1   -76.420   76.0   -152.4
   2   -76.423   76.1   -152.5
$end
"""

_GRADIENT = """$grad
  cycle =  1  SCF energy = -76.420  |dE/dxyz| = 1.0E-02
    0.0   0.0   0.0  o
    1.8   0.0   0.0  h
    0.0   1.8   0.0  h
    1.0E-2  0.0  0.0
    0.0     0.0  0.0
    0.0     0.0  0.0
  cycle =  2  SCF energy = -76.423  |dE/dxyz| = 1.0E-04
    0.0   0.0   0.0  o
    1.81  0.0   0.0  h
    0.0   1.81  0.0  h
    1.0E-4  0.0  0.0
    0.0     0.0  0.0
    0.0     0.0  0.0
$end
"""

_RIDFT = """
   ITERATION  ENERGY
       1      -76.0
       2      -76.4
  convergence criteria satisfied after  2 iterations
       total energy      =   -76.42352123456 |
"""

_AOFORCE = """
   mode               1        2        3        4        5        6
   frequency        -0.00     0.00     0.00     0.00     0.00     0.00
   mode               7        8        9
   frequency      1592.34  3742.56  3851.12
"""

_RIDFT_FULL = """\
 ITERATION  ENERGY          1e-ENERGY        2e-ENERGY     NORM[dD(SAO)]  TOL
   1  -76.4000000000    -123.4500000000     46.0000000000    0.000D+00 0.296D-10
                            Exc = -10.0
 ITERATION  ENERGY          1e-ENERGY        2e-ENERGY     NORM[dD(SAO)]  TOL
   2  -76.4200000000    -123.4600000000     46.0100000000    0.259D-03 0.188D-10
 ITERATION  ENERGY          1e-ENERGY        2e-ENERGY     NORM[dD(SAO)]  TOL
   3  -76.4235212345    -123.4650000000     46.0150000000    0.331D-04 0.176D-10
  convergence criteria satisfied after  3 iterations
       total energy      =   -76.42352123450 |
"""

def _job(task: str, name: str = "h2o") -> dict:
    return {"id": 1, "name": name, "task_type": task,
            "functional": "BP86", "basis_set": "def2-SVP"}


def test_optimization_writes_expected_files(tmp_path):
    (tmp_path / "energy").write_text(_ENERGY)
    (tmp_path / "gradient").write_text(_GRADIENT)
    result = write_artifacts(_job("optimization"), tmp_path)

    assert result.ok
    out = tmp_path / ANALYSIS_SUBDIR
    names = sorted(p.name for p in result.files)
    assert names == sorted([
        "h2o_last.xyz",
        "h2o_trajectory.xyz",
        "h2o_trajectory.csv",
        "h2o_energy.gp",
        "h2o_gradient.gp",
    ])
    # CSV header is the first line of the trajectory.csv
    assert (out / "h2o_trajectory.csv").read_text().splitlines()[0] \
        .startswith("cycle,")
    # gnuplot scripts must reference the CSV file by its bare name
    assert 'f1="h2o_trajectory.csv"' in (out / "h2o_energy.gp").read_text()


def test_optimization_without_data_warns_only(tmp_path):
    result = write_artifacts(_job("optimization"), tmp_path)
    assert not result.ok
    assert result.warnings
    # The analysis directory may still have been created, but no files
    assert all(not f.exists() for f in result.files)


def test_single_point_writes_summary(tmp_path):
    (tmp_path / "ridft.out").write_text(_RIDFT)
    result = write_artifacts(_job("single_point"), tmp_path)
    assert result.ok
    assert any(p.name == "h2o_singlepoint.csv" for p in result.files)


def test_single_point_missing_ridft_warns(tmp_path):
    result = write_artifacts(_job("single_point"), tmp_path)
    assert not result.ok
    assert any("ridft" in w for w in result.warnings)


def test_frequencies_writes_csv_and_gnuplot(tmp_path):
    (tmp_path / "aoforce.out").write_text(_AOFORCE)
    result = write_artifacts(_job("frequencies"), tmp_path)
    names = sorted(p.name for p in result.files)
    assert names == ["h2o_spectrum.csv", "h2o_spectrum.gp"]
    # CSV must contain at least one real mode
    csv_text = (tmp_path / ANALYSIS_SUBDIR / "h2o_spectrum.csv").read_text()
    assert "1592" in csv_text


def test_aimd_writes_trajectory_when_gradient_present(tmp_path):
    (tmp_path / "gradient").write_text(_GRADIENT)
    result = write_artifacts(_job("aimd"), tmp_path)
    assert result.ok
    names = sorted(p.name for p in result.files)
    assert "h2o_trajectory.xyz" in names


def test_writer_overwrites_previous_run(tmp_path):
    (tmp_path / "energy").write_text(_ENERGY)
    (tmp_path / "gradient").write_text(_GRADIENT)
    r1 = write_artifacts(_job("optimization"), tmp_path)
    csv_path = next(p for p in r1.files if p.name.endswith(".csv"))
    csv_path.write_text("STALE")
    r2 = write_artifacts(_job("optimization"), tmp_path)
    # The second run must have rewritten the CSV with the real content
    text = csv_path.read_text()
    assert text != "STALE"
    assert text.startswith("cycle,")

def test_single_point_writes_scf_convergence(tmp_path):
    (tmp_path / "ridft.out").write_text(_RIDFT_FULL)
    result = write_artifacts(_job("single_point"), tmp_path)
    names = sorted(p.name for p in result.files)
    assert "h2o_singlepoint.csv" in names
    assert "h2o_scf_convergence.csv" in names
    assert "h2o_scf.gp" in names


def test_optimization_also_writes_singlepoint_and_scf(tmp_path):
    (tmp_path / "energy").write_text(_ENERGY)
    (tmp_path / "gradient").write_text(_GRADIENT)
    (tmp_path / "ridft.out").write_text(_RIDFT_FULL)
    result = write_artifacts(_job("optimization"), tmp_path)
    names = sorted(p.name for p in result.files)
    assert "h2o_singlepoint.csv" in names
    assert "h2o_scf_convergence.csv" in names
    assert "h2o_trajectory.csv" in names


def test_frequencies_also_writes_singlepoint_and_scf(tmp_path):
    (tmp_path / "aoforce.out").write_text(_AOFORCE)
    (tmp_path / "ridft.out").write_text(_RIDFT_FULL)
    result = write_artifacts(_job("frequencies"), tmp_path)
    names = sorted(p.name for p in result.files)
    assert "h2o_spectrum.csv" in names
    assert "h2o_singlepoint.csv" in names
    assert "h2o_scf_convergence.csv" in names