"""Tests for backend.result_parser."""

from backend.result_parser import parse_ridft, parse_energy_file


_RIDFT_EXAMPLE = """
            =====================================
            *       ridft - SCF calculation     *
            =====================================

   ITERATION  ENERGY               1e-ENERGY
       1      -76.123456789012     ...
       2      -76.234567890123     ...
       3      -76.345678901234     ...
       4      -76.420000000000     ...
       5      -76.421000000000     ...

  convergence criteria satisfied after  5 iterations

       total energy      =   -76.42352123456 |
                ******************************
       |dipole moment| =     0.7345 a.u. =      1.867 debye
       HOMO-LUMO gap         :    0.12345 a.u. = 3.358 eV

       total cpu-time   :   00:00:02
"""


def test_total_energy_parsed():
    s = parse_ridft(_RIDFT_EXAMPLE)
    assert s.total_energy_ha is not None
    assert abs(s.total_energy_ha - (-76.42352123456)) < 1e-10


def test_iteration_count():
    s = parse_ridft(_RIDFT_EXAMPLE)
    assert s.scf_iterations == 5


def test_converged_flag():
    s = parse_ridft(_RIDFT_EXAMPLE)
    assert s.converged is True


def test_not_converged_when_no_marker():
    s = parse_ridft("ITERATION 1 -76.0\nITERATION 2 -76.1\n")
    assert s.converged is False


def test_cpu_time_hms():
    s = parse_ridft(_RIDFT_EXAMPLE)
    assert s.cpu_time_s == 2.0
    assert "00:00:02" in s.cpu_time_str


def test_homo_lumo():
    s = parse_ridft(_RIDFT_EXAMPLE)
    assert s.homo_lumo_ha is not None
    assert abs(s.homo_lumo_ha - 0.12345) < 1e-6
    assert s.homo_lumo_ev is not None
    assert abs(s.homo_lumo_ev - 3.358) < 1e-3


def test_dipole():
    s = parse_ridft(_RIDFT_EXAMPLE)
    assert s.dipole_au is not None
    assert abs(s.dipole_au - 0.7345) < 1e-4
    assert abs(s.dipole_debye - 1.867) < 1e-3


def test_display_dict_keys():
    s = parse_ridft(_RIDFT_EXAMPLE)
    d = s.as_display_dict()
    assert "Total energy" in d
    assert "SCF iterations" in d
    assert "Converged" in d
    assert "HOMO-LUMO gap" in d
    assert "|dipole|" in d


def test_parse_energy_file():
    content = """$energy
       1   -76.420000000000   -76.420000000000   0.000
       2   -76.421000000000   -76.421000000000   0.000
       3   -76.423521234560   -76.423521234560   0.000
$end
"""
    e = parse_energy_file(content)
    assert e is not None
    assert abs(e - (-76.423521234560)) < 1e-10


def test_path_input(tmp_path):
    p = tmp_path / "ridft.out"
    p.write_text(_RIDFT_EXAMPLE)
    s = parse_ridft(p)
    assert s.total_energy_ha is not None
    assert s.converged is True


def test_empty_input():
    s = parse_ridft("")
    assert s.total_energy_ha is None
    assert s.converged is False
    assert s.as_display_dict() == {"Converged": "no"}
