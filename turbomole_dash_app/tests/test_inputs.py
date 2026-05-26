"""Smoke tests for the local input-generation pipeline (no SSH/SLURM)."""

from pathlib import Path

import pytest
from ase.build import molecule

from backend.turbomole_io import (
    build_control_file,
    turbomole_driver_commands,
    available_functionals,
    available_basis_sets,
    get_driver,
    CalcSpec,
    TASK_TYPES,
    SUPPORTED_TM_VERSIONS,
)
from backend.slurm import SlurmParams, build_slurm_script


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

def test_functionals_basis_nonempty():
    assert available_functionals()
    assert available_basis_sets()


def test_supported_versions():
    assert "7.8" in SUPPORTED_TM_VERSIONS
    assert "8.0" in SUPPORTED_TM_VERSIONS


# ---------------------------------------------------------------------------
# Driver resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("version", ["7.8", "7.8.1", "8.0"])
def test_driver_resolution(version):
    drv = get_driver(version)
    assert drv is not None


def test_driver_unknown_version():
    with pytest.raises(ValueError):
        get_driver("999.0")


# ---------------------------------------------------------------------------
# define.inp generation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", TASK_TYPES)
def test_build_define_inp(tmp_path, task):
    atoms = molecule("H2O")
    workdir = tmp_path / f"job_{task}"
    define_inp = build_control_file(
        atoms, workdir,
        functional="BP86", basis_set="def2-SVP", task_type=task,
        charge=0, multiplicity=1,
        turbomole_version="7.8",
    )
    assert (workdir / "coord").exists()
    assert define_inp.exists()
    assert define_inp.name == "define.inp"
    assert (workdir / ".tm_params").exists()

    text = define_inp.read_text()
    # Core sequence: geometry, basis, eht, scf, quit
    assert "a coord" in text
    assert "def2-SVP" in text
    assert "eht" in text
    assert "scf" in text


def test_define_inp_dft_block_has_functional_and_grid(tmp_path):
    atoms = molecule("H2O")
    define_inp = build_control_file(
        atoms, tmp_path / "j", functional="B3LYP", basis_set="def2-TZVP",
        task_type="single_point", grid="m5", use_ri=True,
        turbomole_version="7.8",
    )
    text = define_inp.read_text()
    assert "dft" in text
    assert "func b3-lyp" in text
    assert "grid m5" in text
    assert "ri" in text


def test_define_inp_open_shell(tmp_path):
    """Doublet should generate the unpaired-electron branch."""
    atoms = molecule("CH3")  # methyl radical, doublet
    define_inp = build_control_file(
        atoms, tmp_path / "j", functional="BP86", basis_set="def2-SVP",
        task_type="single_point", charge=0, multiplicity=2,
        turbomole_version="7.8",
    )
    text = define_inp.read_text()
    assert "u 1" in text          # 1 unpaired electron


def test_define_inp_hf_no_dft_block(tmp_path):
    atoms = molecule("H2O")
    define_inp = build_control_file(
        atoms, tmp_path / "j", functional="HF", basis_set="def2-SVP",
        task_type="single_point",
        turbomole_version="7.8",
    )
    text = define_inp.read_text()
    # HF should not enter the dft submenu
    assert "func" not in text


@pytest.mark.parametrize("version", ["7.8", "8.0"])
def test_define_inp_both_versions(tmp_path, version):
    """8.0 currently inherits from 7.8; both should produce valid output."""
    atoms = molecule("H2O")
    define_inp = build_control_file(
        atoms, tmp_path / f"j_{version}", functional="BP86",
        basis_set="def2-SVP", task_type="single_point",
        turbomole_version=version,
    )
    assert define_inp.exists()


def test_unknown_functional_rejected(tmp_path):
    atoms = molecule("H2O")
    with pytest.raises(ValueError):
        build_control_file(
            atoms, tmp_path / "x",
            functional="NOT_A_FUNCTIONAL", basis_set="def2-SVP",
            task_type="single_point",
        )


# ---------------------------------------------------------------------------
# Driver commands
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", TASK_TYPES)
def test_driver_commands(task):
    cmds = turbomole_driver_commands(task, turbomole_version="7.8")
    assert cmds
    # Every task must first run define
    joined = "\n".join(cmds)
    assert "define < define.inp" in joined
    # And produce a control file check
    assert "control" in joined


def test_driver_commands_singlepoint_runs_ridft():
    cmds = turbomole_driver_commands("single_point", turbomole_version="7.8")
    assert any("ridft" in c for c in cmds)


def test_driver_commands_opt_runs_jobex():
    cmds = turbomole_driver_commands("optimization", turbomole_version="7.8")
    assert any("jobex" in c for c in cmds)


def test_driver_commands_aimd_runs_frog():
    cmds = turbomole_driver_commands("aimd", turbomole_version="7.8")
    assert any("frog" in c for c in cmds)


# ---------------------------------------------------------------------------
# SLURM script integration
# ---------------------------------------------------------------------------

def test_slurm_script_renders():
    sl = SlurmParams(job_name="t", partition="cpu", nodes=2, ntasks=32, mem="64G")
    cmds = turbomole_driver_commands("single_point", turbomole_version="7.8")
    txt = build_slurm_script(
        sl, "/scratch/me/job1",
        ["turbomole/7.8"], cmds,
    )
    assert txt.startswith("#!/bin/bash")
    assert "#SBATCH --job-name=t" in txt
    assert "cd /scratch/me/job1" in txt
    assert "module load turbomole/7.8" in txt
    assert "define < define.inp" in txt
    assert "ridft" in txt


def test_slurm_script_omits_reservation_by_default():
    sl = SlurmParams(job_name="t", partition="cpu", nodes=1, ntasks=4, mem="4G")
    cmds = turbomole_driver_commands("single_point", turbomole_version="7.8")
    txt = build_slurm_script(sl, "/scratch/me/job1", [], cmds)
    assert "--reservation" not in txt


def test_slurm_script_includes_reservation_when_set():
    sl = SlurmParams(
        job_name="t", partition="cpu", nodes=1, ntasks=4, mem="4G",
        reservation="my_dedicated_block",
    )
    cmds = turbomole_driver_commands("single_point", turbomole_version="7.8")
    txt = build_slurm_script(sl, "/scratch/me/job1", [], cmds)
    assert "#SBATCH --reservation=my_dedicated_block" in txt


def test_slurm_script_omits_reservation_when_empty_or_none():
    """None / empty / whitespace-only should all omit the directive."""
    cmds = turbomole_driver_commands("single_point", turbomole_version="7.8")
    for value in (None, "", "   "):
        sl = SlurmParams(
            job_name="t", partition="cpu", nodes=1, ntasks=4, mem="4G",
            reservation=value,
        )
        txt = build_slurm_script(sl, "/scratch/me/job1", [], cmds)
        assert "--reservation" not in txt, f"failed for value={value!r}"


def test_slurm_script_strips_reservation_whitespace():
    """Surrounding whitespace must be stripped in the emitted directive."""
    sl = SlurmParams(
        job_name="t", partition="cpu", nodes=1, ntasks=4, mem="4G",
        reservation="  my_block  ",
    )
    cmds = turbomole_driver_commands("single_point", turbomole_version="7.8")
    txt = build_slurm_script(sl, "/scratch/me/job1", [], cmds)
    assert "#SBATCH --reservation=my_block" in txt
    assert "--reservation=  " not in txt


# ---------------------------------------------------------------------------
# Dispersion corrections
# ---------------------------------------------------------------------------

def test_no_dispersion_by_default():
    """By default no $disp keyword is inserted."""
    cmds = turbomole_driver_commands("single_point", turbomole_version="7.8")
    joined = "\n".join(cmds)
    assert "$disp" not in joined
    assert "sed" not in joined


def test_dispersion_d3_inserts_keyword():
    cmds = turbomole_driver_commands("single_point", turbomole_version="7.8",
                                     dispersion="d3")
    joined = "\n".join(cmds)
    assert "$disp3" in joined
    assert "sed" in joined
    # The injection must run BEFORE ridft
    disp_idx = next(i for i, l in enumerate(cmds) if "$disp3" in l)
    ridft_idx = next(i for i, l in enumerate(cmds) if "ridft" in l)
    assert disp_idx < ridft_idx


def test_dispersion_d3bj_uses_bj():
    cmds = turbomole_driver_commands("single_point", turbomole_version="7.8",
                                     dispersion="d3bj")
    joined = "\n".join(cmds)
    assert "$disp3 bj" in joined


def test_dispersion_d4():
    cmds = turbomole_driver_commands("single_point", turbomole_version="7.8",
                                     dispersion="d4")
    joined = "\n".join(cmds)
    assert "$disp4" in joined


def test_dispersion_idempotent_guard():
    """The inserted shell snippet must be idempotent: grep -q guards
    against running it twice."""
    cmds = turbomole_driver_commands("single_point", turbomole_version="7.8",
                                     dispersion="d3bj")
    joined = "\n".join(cmds)
    assert "grep -q" in joined
    assert "fi" in joined


def test_unknown_dispersion_raises():
    import pytest
    from backend.turbomole_io import CalcSpec
    with pytest.raises(ValueError, match="Unknown dispersion"):
        CalcSpec(functional="BP86", basis_set="def2-SVP",
                 task_type="single_point", dispersion="d99").validate()


def test_dispersion_applies_to_optimization_too():
    cmds = turbomole_driver_commands("optimization", turbomole_version="7.8",
                                     dispersion="d3bj")
    joined = "\n".join(cmds)
    assert "$disp3 bj" in joined
    assert "jobex" in joined
    # And ordering: disp insertion runs before jobex
    disp_idx = next(i for i, l in enumerate(cmds) if "$disp3 bj" in l)
    jobex_idx = next(i for i, l in enumerate(cmds) if "jobex" in l)
    assert disp_idx < jobex_idx


# ---------------------------------------------------------------------------
# Frequencies task (aoforce)
# ---------------------------------------------------------------------------

def test_frequencies_task_runs_ridft_then_aoforce():
    cmds = turbomole_driver_commands("frequencies", turbomole_version="7.8")
    joined = "\n".join(cmds)
    assert "define < define.inp" in joined
    assert "ridft" in joined
    assert "aoforce" in joined
    # ridft must come before aoforce so MOs exist when aoforce runs
    ridft_idx = next(i for i, l in enumerate(cmds) if "ridft >" in l)
    aoforce_idx = next(i for i, l in enumerate(cmds) if "aoforce >" in l)
    assert ridft_idx < aoforce_idx


def test_frequencies_with_dispersion():
    """Frequencies task + dispersion should still inject $disp* before ridft."""
    cmds = turbomole_driver_commands("frequencies", turbomole_version="7.8",
                                     dispersion="d3bj")
    joined = "\n".join(cmds)
    assert "$disp3 bj" in joined
    assert "aoforce" in joined
    # Order: disp_injection → ridft → aoforce
    disp_idx = next(i for i, l in enumerate(cmds) if "$disp3 bj" in l)
    ridft_idx = next(i for i, l in enumerate(cmds) if "ridft >" in l)
    aoforce_idx = next(i for i, l in enumerate(cmds) if "aoforce >" in l)
    assert disp_idx < ridft_idx < aoforce_idx


def test_frequencies_in_full_slurm_script():
    """End-to-end check that aoforce appears in a real submit.slurm."""
    sl = SlurmParams(job_name="freqtest", partition="cpu",
                     nodes=1, ntasks=8, mem="16G")
    cmds = turbomole_driver_commands("frequencies", turbomole_version="7.8")
    script = build_slurm_script(sl, "/scratch/me/freqtest", [], cmds)
    assert "aoforce > aoforce.out" in script
    assert "ridft > ridft.out" in script
