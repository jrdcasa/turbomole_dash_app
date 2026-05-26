"""
Turbomole input generation using ASE.

Design notes:
- ASE writes the geometry as a `coord` file (Turbomole's native format).
  This format is stable across Turbomole versions.
- The actual `control` file is *not* written locally. We generate a
  `define.inp` (a script of responses for Turbomole's interactive
  `define`) and let it build the control on the cluster.
- `define`'s prompt sequence differs between Turbomole versions, so we
  factor that out into per-version driver classes.

Supported versions: 7.8, 8.0 (placeholder until 8.0 is released; the
driver inherits from 7.8 and overrides where needed).
"""

from __future__ import annotations

import io
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from ase import Atoms
from ase.io import read as ase_read, write as ase_write


SUPPORTED_INPUT_FORMATS = (
    "xyz", "coord", "mol", "mol2", "pdb", "sdf", "cif", "gen", "vasp", "cml",
)

TASK_TYPES = ("single_point", "optimization", "aimd", "frequencies")

SUPPORTED_TM_VERSIONS = ("7.8", "8.0")


# ---------------------------------------------------------------------------
# Method dictionaries
# ---------------------------------------------------------------------------

# UI name → (define keyword for "func", needs RI auxbasis)
_RIDFT_FUNCTIONALS = {
    "BP86":    ("b-p",      True),
    "BLYP":    ("b-lyp",    True),
    "B3LYP":   ("b3-lyp",   True),
    "PBE":     ("pbe",      True),
    "PBE0":    ("pbe0",     True),
    "TPSS":    ("tpss",     True),
    "TPSSh":   ("tpssh",    True),
    "M06-L":   ("m06-l",    True),
    "M06-2X":  ("m06-2x",   True),
    "wB97X-D": ("wb97x-d",  True),
    "HF":      (None,       False),
}

_BASIS_SETS = (
    "def2-SV(P)", "def2-SVP", "def2-TZVP", "def2-TZVPP", "def2-QZVP",
    "def-SV(P)",  "def-SVP",  "def-TZVP",
    "dhf-SVP",    "dhf-TZVP", "dhf-TZVPP",
    "cc-pVDZ",    "cc-pVTZ",  "cc-pVQZ",
    "aug-cc-pVDZ","aug-cc-pVTZ",
)

# Dispersion corrections.
# Each entry maps a UI value to (control_keyword, display_label).
# control_keyword is the exact line we add to the `control` file before
# $end. None means "no correction" (we skip the insertion entirely).
DISPERSION_OPTIONS = {
    "none":  (None,        "None"),
    "d3":    ("$disp3",    "D3 (zero damping)"),
    "d3bj":  ("$disp3 bj", "D3(BJ) — Becke-Johnson damping"),
    "d4":    ("$disp4",    "D4"),
}


def available_functionals() -> list[str]:
    return list(_RIDFT_FUNCTIONALS.keys())


def available_basis_sets() -> list[str]:
    return list(_BASIS_SETS)


def available_dispersions() -> list[tuple[str, str]]:
    """List of (value, label) tuples for the UI dropdown."""
    return [(k, v[1]) for k, v in DISPERSION_OPTIONS.items()]


# ---------------------------------------------------------------------------
# Structure I/O
# ---------------------------------------------------------------------------

def load_structure(path: str | Path) -> Atoms:
    return ase_read(str(path))


def load_structure_from_bytes(data: bytes, fmt: str) -> Atoms:
    if fmt.lower() not in SUPPORTED_INPUT_FORMATS:
        raise ValueError(f"Unsupported format: {fmt}")
    buf = io.StringIO(data.decode("utf-8", errors="replace"))
    return ase_read(buf, format=fmt.lower())


def write_coord(atoms: Atoms, workdir: Path) -> Path:
    """Write Turbomole 'coord' file (Bohr units, $coord block)."""
    workdir.mkdir(parents=True, exist_ok=True)
    coord_path = workdir / "coord"
    ase_write(str(coord_path), atoms, format="turbomole")
    return coord_path


# ---------------------------------------------------------------------------
# Calculation spec — version-independent description of the job
# ---------------------------------------------------------------------------

@dataclass
class CalcSpec:
    """All physical/chemical parameters needed to drive `define`."""
    functional: str          # UI name, e.g. "BP86"
    basis_set: str
    task_type: str
    charge: int = 0
    multiplicity: int = 1
    scf_conv: int = 7
    scf_iter: int = 200
    grid: str = "m4"         # m3 | m4 | m5
    use_ri: bool = True
    dispersion: str = "none" # none | d3 | d3bj | d4
    # AIMD-specific
    aimd_steps: int = 500
    aimd_timestep_fs: float = 0.5
    aimd_temperature_K: float = 300.0

    def validate(self) -> None:
        if self.task_type not in TASK_TYPES:
            raise ValueError(f"Unknown task_type {self.task_type!r}")
        if self.functional not in _RIDFT_FUNCTIONALS:
            raise ValueError(f"Unknown functional {self.functional!r}")
        if self.basis_set not in _BASIS_SETS:
            raise ValueError(f"Unknown basis set {self.basis_set!r}")
        if self.multiplicity < 1:
            raise ValueError(f"multiplicity must be >= 1, got {self.multiplicity}")
        if self.dispersion not in DISPERSION_OPTIONS:
            raise ValueError(
                f"Unknown dispersion {self.dispersion!r}. "
                f"Allowed: {list(DISPERSION_OPTIONS.keys())}"
            )


# ---------------------------------------------------------------------------
# Version-specific drivers
# ---------------------------------------------------------------------------

class DefineDriver(ABC):
    """Abstract base. Subclass per Turbomole version."""
    version: str = "abstract"

    @abstractmethod
    def build_define_input(self, spec: CalcSpec) -> str:
        """Return the text of define.inp for this calculation."""
        ...

    @abstractmethod
    def driver_commands(self, spec: CalcSpec) -> list[str]:
        """Shell commands to run inside the SLURM script."""
        ...

    @staticmethod
    def _xc_keyword(functional: str) -> str | None:
        xc, _ = _RIDFT_FUNCTIONALS[functional]
        return xc

    @staticmethod
    def _safety_pad(lines: list[str], n: int = 4) -> list[str]:
        """Add a few blank lines at the end so unexpected extra prompts
        get accepted with the default (pressing Enter)."""
        return lines + [""] * n


def _disp_injection_commands(spec: CalcSpec) -> list[str]:
    """Return shell lines that add the dispersion keyword to `control`.

    `define` does not write $disp* entries on its own (the keywords are
    handled at runtime by ridft via DFT-D3/D4 libraries). We insert them
    right before $end so they take effect on the next ridft/jobex run.
    """
    keyword = DISPERSION_OPTIONS[spec.dispersion][0]
    if keyword is None:
        return []
    return [
        f'# 1b) Add dispersion correction ({spec.dispersion}) to control',
        # Idempotent: skip if already present
        f'if ! grep -q "^{keyword}\\b" control; then',
        # Insert keyword on its own line just before the $end line
        f'    sed -i "/^\\$end$/i {keyword}" control',
        'fi',
        '',
    ]


class DefineDriver_7_8(DefineDriver):
    """
    Driver for Turbomole 7.8.x.

    Menu sequence we use:
        title  →  (blank)
        geom   →  a coord  /  *  /  no
        basis  →  b / all <BASIS> / *
        eht    →  eht / y / <charge> / [singlet|u N]
        dft    →  dft / on / func <XC> / grid <G> / (blank back) [+ ri / on / blank]
        scf    →  scf / conv N / iter N / (blank back)
        quit   →  *
    """
    version = "7.8"

    def build_define_input(self, spec: CalcSpec) -> str:
        spec.validate()
        unpaired = spec.multiplicity - 1
        xc = self._xc_keyword(spec.functional)

        lines: list[str] = []

        # --- title menu ---
        lines += [
            "",
            "",                          # title: just press Enter
            "a coord",                   # add geometry from coord file
            "*",                         # exit geometry submenu
            "no",                        # no internal coordinates
        ]

        # --- basis menu ---
        lines += [
            "b",
            f"all {spec.basis_set}",
            "*",
        ]

        # --- occupation / start MOs (eht) ---
        lines += [
            "eht",
            "y",                         # accept defaults
            str(spec.charge),
        ]
        if unpaired == 0:
            lines += ["y"]               # confirm closed-shell singlet
        else:
            lines += [
                "n",                     # not closed-shell singlet
                f"u {unpaired}",         # unpaired electrons
                "*",
            ]

        # --- method (DFT or HF) ---
        if xc is not None:
            lines += [
                "dft",
                "on",
                f"func {xc}",
                f"grid {spec.grid}",
                "",                      # back to main
            ]
            if spec.use_ri:
                lines += [
                    "ri",
                    "on",
                    "",                  # back to main
                ]

        # --- SCF ---
        lines += [
            "scf",
            f"conv {spec.scf_conv}",
            f"iter {spec.scf_iter}",
            "",                          # back to main
        ]

        # --- quit, write control ---
        lines += ["*"]

        lines = self._safety_pad(lines, n=4)
        return "\n".join(lines) + "\n"

    def driver_commands(self, spec: CalcSpec) -> list[str]:
        common_pre = [
            "# 1) Build control via define ---------------------------",
            "define < define.inp > define.out 2>&1",
            "if [ ! -s control ]; then",
            '    echo "ERROR: define did not produce a control file" >&2',
            "    tail -80 define.out >&2",
            "    exit 1",
            "fi",
            "",
        ]
        common_pre += _disp_injection_commands(spec)

        if spec.task_type == "single_point":
            return common_pre + [
                "# 2) Single-point energy --------------------------------",
                "ridft > ridft.out 2>&1",
            ]
        if spec.task_type == "optimization":
            return common_pre + [
                "# 2) Geometry optimization ------------------------------",
                "jobex -ri -c 200 > jobex.out 2>&1",
            ]
        if spec.task_type == "frequencies":
            # aoforce needs converged MOs (`mos` file). Run ridft first
            # so we have a guaranteed SCF point at the input geometry.
            return common_pre + [
                "# 2) SCF first (so we have converged MOs) ---------------",
                "ridft > ridft.out 2>&1",
                "# 3) Harmonic frequencies via aoforce -------------------",
                "aoforce > aoforce.out 2>&1",
            ]
        if spec.task_type == "aimd":
            return common_pre + [
                "# 2) AIMD --------------------------------------------",
                "ridft > ridft.out 2>&1",
                "mdprep -default > mdprep.out 2>&1 || true",
                "frog > frog.out 2>&1",
            ]
        raise ValueError(f"Unknown task_type {spec.task_type!r}")


class DefineDriver_8_0(DefineDriver_7_8):
    """
    Driver for Turbomole 8.0 (anticipated).
    """
    version = "8.0"

    def build_define_input(self, spec: CalcSpec) -> str:
        return super().build_define_input(spec)


_DRIVERS: dict[str, type[DefineDriver]] = {
    "7.8": DefineDriver_7_8,
    "8.0": DefineDriver_8_0,
}


def get_driver(version: str) -> DefineDriver:
    if version in _DRIVERS:
        return _DRIVERS[version]()
    major_minor = ".".join(version.split(".")[:2])
    if major_minor in _DRIVERS:
        return _DRIVERS[major_minor]()
    raise ValueError(
        f"Unsupported Turbomole version: {version!r}. "
        f"Supported: {sorted(_DRIVERS.keys())}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_control_file(
    atoms: Atoms,
    workdir: Path,
    *,
    functional: str,
    basis_set: str,
    task_type: str,
    charge: int = 0,
    multiplicity: int = 1,
    scf_conv: int = 7,
    scf_iter: int = 200,
    grid: str = "m4",
    use_ri: bool = True,
    dispersion: str = "none",
    aimd_steps: int = 500,
    aimd_timestep_fs: float = 0.5,
    aimd_temperature_K: float = 300.0,
    turbomole_version: str = "7.8",
) -> Path:
    """
    Prepare a Turbomole job directory.

    Writes:
      - coord       : geometry in Turbomole format
      - define.inp  : stdin script for `define` (version-specific)
      - .tm_params  : JSON sidecar with the spec (useful for debugging)

    The actual `control` file is generated by `define` on the cluster
    at submit time.

    Returns the path to define.inp.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    write_coord(atoms, workdir)

    spec = CalcSpec(
        functional=functional, basis_set=basis_set, task_type=task_type,
        charge=charge, multiplicity=multiplicity,
        scf_conv=scf_conv, scf_iter=scf_iter, grid=grid, use_ri=use_ri,
        dispersion=dispersion,
        aimd_steps=aimd_steps,
        aimd_timestep_fs=aimd_timestep_fs,
        aimd_temperature_K=aimd_temperature_K,
    )
    spec.validate()

    driver = get_driver(turbomole_version)
    define_inp = workdir / "define.inp"
    define_inp.write_text(driver.build_define_input(spec))

    sidecar = {
        "turbomole_version": turbomole_version,
        "driver": driver.version,
        "spec": spec.__dict__,
    }
    (workdir / ".tm_params").write_text(json.dumps(sidecar, indent=2))

    return define_inp


def turbomole_driver_commands(
    task_type: str,
    *,
    turbomole_version: str = "7.8",
    functional: str = "BP86",
    basis_set: str = "def2-SVP",
    use_ri: bool = True,
    dispersion: str = "none",
) -> list[str]:
    """Shell commands to run inside the SLURM script body."""
    spec = CalcSpec(
        functional=functional, basis_set=basis_set, task_type=task_type,
        use_ri=use_ri, dispersion=dispersion,
    )
    driver = get_driver(turbomole_version)
    return driver.driver_commands(spec)
