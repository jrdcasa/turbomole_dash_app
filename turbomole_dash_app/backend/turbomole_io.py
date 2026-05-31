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
- For AIMD jobs we also generate `mdprep.inp` locally so that the
  cluster-side `mdprep` step can be fully scripted (constraints, T,
  timestep, number of steps). The actual `mdmaster` is produced by
  `mdprep` at submit time on the cluster.

Supported versions: 7.8, 8.0 (placeholder until 8.0 is released; the
driver inherits from 7.8 and overrides where needed).
"""

from __future__ import annotations

import io
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from ase import Atoms
from ase.io import read as ase_read, write as ase_write

from backend.opt_trajectory import BOHR_TO_ANGSTROM


SUPPORTED_INPUT_FORMATS = (
    "xyz", "coord", "mol", "mol2", "pdb", "sdf", "cif", "gen", "vasp", "cml",
)

TASK_TYPES = ("single_point", "optimization", "aimd", "frequencies")

SUPPORTED_TM_VERSIONS = ("7.8", "8.0")

# Atomic-unit timestep <-> femtoseconds (1 a.u. of time = 0.024188843265857 fs)
AU_TIME_TO_FS = 0.024188843265857

# Supported constraint algorithms in `mdprep`'s constraints submenu.
CONSTRAINT_ALGORITHMS = ("shake", "maltshake")


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


def available_constraint_algorithms() -> tuple[str, ...]:
    return CONSTRAINT_ALGORITHMS


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
# AIMD: distance-constraint parsing
# ---------------------------------------------------------------------------

# Each constraint is (atom_i, atom_j, distance_Angstrom). Atom indices are
# 1-based, exactly as written in Turbomole's `$coord` section, and the
# distance is supplied in Angstrom by the user (the renderer converts to
# Bohr before writing mdprep.inp because mdprep's internal unit is Bohr).
ConstraintTriple = tuple[int, int, float]


_CONSTRAINT_TOKEN_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$"
)


def parse_constraints(text: str) -> list[ConstraintTriple]:
    """Parse the constraints textarea: 'at1 at2 d_A; at1 at2 d_A; ...'.

    Empty input yields []. Whitespace-tolerant. Raises ValueError with a
    descriptive message for the first malformed entry so the UI can
    surface it as feedback.
    """
    if not text or not text.strip():
        return []
    out: list[ConstraintTriple] = []
    for raw in text.split(";"):
        token = raw.strip()
        if not token:
            continue
        m = _CONSTRAINT_TOKEN_RE.match(token)
        if not m:
            raise ValueError(
                f"Invalid constraint entry: {token!r}. "
                "Expected 'at1 at2 distance_A' (e.g. '1 2 1.09')."
            )
        i, j, d = int(m.group(1)), int(m.group(2)), float(m.group(3))
        if i == j:
            raise ValueError(
                f"Invalid constraint {token!r}: atom indices must differ."
            )
        if i < 1 or j < 1:
            raise ValueError(
                f"Invalid constraint {token!r}: atom indices are 1-based."
            )
        if d <= 0.0:
            raise ValueError(
                f"Invalid constraint {token!r}: distance must be positive."
            )
        out.append((i, j, d))
    return out


# ---------------------------------------------------------------------------
# Calculation spec — version-independent description of the job
# ---------------------------------------------------------------------------

@dataclass
class CalcSpec:
    """All physical/chemical parameters needed to drive `define` (+ mdprep)."""
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
    aimd_steps: int = 256
    aimd_timestep_au: float = 80.0
    aimd_temperature_K: float = 300.0
    aimd_use_constraints: bool = False
    aimd_constraint_algorithm: str = "shake"     # shake | maltshake
    aimd_constraints: list[ConstraintTriple] = field(default_factory=list)

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
        if self.task_type == "aimd":
            if self.aimd_steps < 1:
                raise ValueError("aimd_steps must be >= 1")
            if self.aimd_timestep_au <= 0:
                raise ValueError("aimd_timestep_au must be > 0")
            if self.aimd_temperature_K <= 0:
                raise ValueError("aimd_temperature_K must be > 0")
            if self.aimd_constraint_algorithm not in CONSTRAINT_ALGORITHMS:
                raise ValueError(
                    f"Unknown constraint algorithm "
                    f"{self.aimd_constraint_algorithm!r}. "
                    f"Allowed: {CONSTRAINT_ALGORITHMS}"
                )
            if self.aimd_use_constraints and not self.aimd_constraints:
                raise ValueError(
                    "aimd_use_constraints is True but no constraints "
                    "were provided."
                )


# ---------------------------------------------------------------------------
# mdprep.inp generator
# ---------------------------------------------------------------------------
#
# mdprep is interactive. Each main menu entry below corresponds to one of
# the seven steps described by the user; for the steps we don't customise
# we simply press `q` to accept the defaults already produced by the
# previous `ridft` run and the `coord` / `control` files.
#
# Menu summary as implemented here (Turbomole 7.x/8.x):
#   1) Number of atoms        — accept (q)
#   2) Initial positions       — accept ($coord, q)
#   3) Cavity barrier          — accept defaults (q)
#   4) Distance constraints    — optional; a → g → <alg> → e <i j d_Bohr>... → q
#   5) Initial velocities (T)  — i → <T> → q
#   6) Timestep                — i → <dt_au> → q → q → q
#   7) Number of MD steps      — i → <n_steps> → q

def build_mdprep_input(spec: CalcSpec) -> str:
    """Render mdprep.inp from a CalcSpec (task_type must be 'aimd')."""
    if spec.task_type != "aimd":
        raise ValueError("build_mdprep_input is only valid for task_type='aimd'")
    spec.validate()

    lines: list[str] = []

    # 1) Number of atoms — take from control
    lines.append("q")

    # 2) Initial positions — take from $coord
    lines.append("q")

    # 3) Cavity barrier attributes — skip
    lines.append("q")

    # 4) Distance constraints (optional)
    if spec.aimd_use_constraints and spec.aimd_constraints:
        lines.append("a")
        lines.append("g")
        lines.append(spec.aimd_constraint_algorithm)
        for i, j, d_ang in spec.aimd_constraints:
            d_bohr = d_ang / BOHR_TO_ANGSTROM
            lines.append("e")
            # mdprep tolerates whitespace; use plain spaces.
            lines.append(f"{i} {j} {d_bohr:.10f}")
        lines.append("q")
    else:
        # No constraints menu interaction needed — accept defaults.
        lines.append("q")

    # 5) Initial velocity / temperature
    lines.append("i")
    lines.append(f"{spec.aimd_temperature_K:g}")
    lines.append("q")

    # 6) Timestep (a.u.)
    lines.append("i")
    lines.append(f"{spec.aimd_timestep_au:g}")
    lines.append("q")
    lines.append("q")
    lines.append("q")

    # 7) Number of MD steps
    lines.append("i")
    lines.append(str(int(spec.aimd_steps)))
    lines.append("q")

    # A few safety blank lines so any extra final prompt gets the default.
    lines += [""] * 3

    return "\n".join(lines) + "\n"


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
            "conv",
            f"{spec.scf_conv}",
            "iter",
            f"{spec.scf_iter}",
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
            # mdprep.inp is generated locally by build_control_file and
            # uploaded along with coord/define.inp/submit.slurm.
            return common_pre + [
                "# 2) AIMD --------------------------------------------",
                "ridft > ridft.out 2>&1",
                "# 2b) Build mdmaster from the user-provided mdprep.inp",
                "if [ ! -f mdprep.inp ]; then",
                '    echo "ERROR: mdprep.inp not found in job dir" >&2',
                "    exit 1",
                "fi",
                "mdprep < mdprep.inp > mdprep.out 2>&1",
                "# 2c) Propagate trajectory",
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
    aimd_steps: int = 256,
    aimd_timestep_au: float = 80.0,
    aimd_temperature_K: float = 300.0,
    aimd_use_constraints: bool = False,
    aimd_constraint_algorithm: str = "shake",
    aimd_constraints: list[ConstraintTriple] | None = None,
    turbomole_version: str = "7.8",
) -> Path:
    """
    Prepare a Turbomole job directory.

    Writes:
      - coord       : geometry in Turbomole format
      - define.inp  : stdin script for `define` (version-specific)
      - mdprep.inp  : stdin script for `mdprep` (only when task_type=='aimd')
      - .tm_params  : JSON sidecar with the spec (useful for debugging)

    The actual `control` file is generated by `define` on the cluster
    at submit time; `mdmaster` is generated by `mdprep` at submit time.

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
        aimd_timestep_au=aimd_timestep_au,
        aimd_temperature_K=aimd_temperature_K,
        aimd_use_constraints=aimd_use_constraints,
        aimd_constraint_algorithm=aimd_constraint_algorithm,
        aimd_constraints=list(aimd_constraints or []),
    )
    spec.validate()

    driver = get_driver(turbomole_version)
    define_inp = workdir / "define.inp"
    define_inp.write_text(driver.build_define_input(spec))

    if spec.task_type == "aimd":
        (workdir / "mdprep.inp").write_text(build_mdprep_input(spec))

    sidecar = {
        "turbomole_version": turbomole_version,
        "driver": driver.version,
        "spec": {**spec.__dict__,
                 "aimd_constraints": list(spec.aimd_constraints)},
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