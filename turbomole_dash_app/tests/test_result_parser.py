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


# ===========================================================================
# aoforce parser
# ===========================================================================

from backend.result_parser import parse_aoforce, AoforceSummary


# Synthetic but realistic aoforce.out excerpt for H2O at BP86/def2-SVP.
# Real Turbomole output is longer; we keep only the lines we parse.
_AOFORCE_H2O = """
   *** vibrational analysis ***

   mode               1        2        3        4        5        6
   frequency        -0.00     0.00     0.00     0.00     0.00     0.00
   mode               7        8        9
   frequency      1592.34  3742.56  3851.12

  zero point VIBRATIONAL energy  :        0.0210550   Hartree

   thermodynamic functions at T =     298.15 K
   enthalpy           :   -76.3987654   Hartree
   chem. potential    :   -76.4203456   Hartree
   entropy            :     45.123     cal/(mol*K)
"""

# Same molecule but with one imaginary mode (transition state)
_AOFORCE_TS = """
   mode               1        2        3        4        5        6
   frequency        -0.00     0.00     0.00     0.00     0.00     0.00
   mode               7        8        9
   frequency      -512.34  1450.12  3700.45

  zero point VIBRATIONAL energy  :        0.0150000   Hartree

   T =     298.15 K
   enthalpy           :   -76.300   Hartree
   chem. potential    :   -76.320   Hartree
   entropy            :     43.0   cal/(mol*K)
"""


def test_aoforce_parses_frequencies():
    s = parse_aoforce(_AOFORCE_H2O)
    # 6 zero-modes + 3 real vibrations = 9 freqs total
    assert s.n_modes == 9
    # First non-zero vibration should be around 1592
    assert 1500 < max(s.frequencies_cm1) < 4000
    assert 1592.34 in s.frequencies_cm1


def test_aoforce_no_imaginary_for_minimum():
    s = parse_aoforce(_AOFORCE_H2O)
    assert s.n_imaginary == 0


def test_aoforce_detects_transition_state():
    s = parse_aoforce(_AOFORCE_TS)
    assert s.n_imaginary == 1
    # The imaginary mode is at -512.34
    assert -512.34 in s.frequencies_cm1


def test_aoforce_zpe():
    s = parse_aoforce(_AOFORCE_H2O)
    assert s.zpe_ha is not None
    assert abs(s.zpe_ha - 0.0210550) < 1e-6


def test_aoforce_temperature():
    s = parse_aoforce(_AOFORCE_H2O)
    assert s.temperature_K is not None
    assert abs(s.temperature_K - 298.15) < 0.01


def test_aoforce_enthalpy_and_gibbs():
    s = parse_aoforce(_AOFORCE_H2O)
    assert s.enthalpy_ha is not None
    assert abs(s.enthalpy_ha - (-76.3987654)) < 1e-6
    assert s.gibbs_ha is not None
    assert abs(s.gibbs_ha - (-76.4203456)) < 1e-6


def test_aoforce_entropy_cal():
    s = parse_aoforce(_AOFORCE_H2O)
    assert s.entropy_cal_mol_K is not None
    assert abs(s.entropy_cal_mol_K - 45.123) < 0.01


def test_aoforce_entropy_J_converted_to_cal():
    """If entropy is reported in J/(mol·K), it must be converted."""
    text = """
   T =     298.15 K
   enthalpy           :   -76.0   Hartree
   chem. potential    :   -76.1   Hartree
   entropy            :     188.0   J/(mol*K)
"""
    s = parse_aoforce(text)
    assert s.entropy_cal_mol_K is not None
    # 188 J / 4.184 ≈ 44.93 cal
    assert abs(s.entropy_cal_mol_K - 44.93) < 0.1


def test_aoforce_lowest_frequencies():
    s = parse_aoforce(_AOFORCE_TS)
    lows = s.lowest_frequencies(3)
    assert lows[0] == -512.34
    # Next two are the zero-modes near 0
    assert all(abs(f) < 1 for f in lows[1:])


def test_aoforce_display_dict_marks_minimum():
    s = parse_aoforce(_AOFORCE_H2O)
    d = s.as_display_dict()
    assert "minimum" in d.get("Imaginary modes", "").lower()


def test_aoforce_display_dict_marks_ts():
    s = parse_aoforce(_AOFORCE_TS)
    d = s.as_display_dict()
    assert "transition state" in d.get("Imaginary modes", "").lower()


def test_aoforce_empty_input():
    s = parse_aoforce("")
    assert s.frequencies_cm1 == []
    assert s.n_modes is None
    assert s.zpe_ha is None
