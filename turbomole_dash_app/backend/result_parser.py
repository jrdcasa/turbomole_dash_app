"""
Parsers for Turbomole output files (ridft.out, slurm logs, etc.).

All functions accept either:
  - a file path (Path or str), OR
  - the file content as a string,
so they work both for locally downloaded files and for
remote `tail`-style snippets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Regexes for Turbomole ridft / dscf / escf output
# ---------------------------------------------------------------------------

# Final SCF total energy line. Examples:
#   |  total energy      =    -76.42352123456 |
#   total energy   =  -76.42352123456
_RE_TOTAL_ENERGY = re.compile(
    r"total\s+energy\s*=\s*(-?\d+\.\d+)", re.IGNORECASE
)

# Per-iteration energy line that appears in ridft output:
#   current damping :  0.700
#         ITERATION  ENERGY          1e-ENERGY        ...
#   1  -76.123...
# We count "convergence criteria satisfied" / "convergence reached" instead.
_RE_SCF_CONVERGED = re.compile(
    r"(convergence criteria satisfied|convergence reached|SCF converged)",
    re.IGNORECASE,
)

# Number of iterations: ridft prints "ITERATION N" for each cycle.
# We pick the largest N seen.
_RE_ITERATION = re.compile(r"^\s*(\d+)\s+-?\d+\.\d{6,}", re.MULTILINE)

# CPU time line. Different binaries phrase it differently:
#   total cpu-time :   1 second
#   total  cpu-time   :   00:00:02
#   total cpu-time    =  1.23 seconds
_RE_CPU_TIME = re.compile(
    r"total\s+cpu[- ]time\s*[:=]\s*([0-9:.]+)\s*(seconds?|s|min|hour)?",
    re.IGNORECASE,
)

# WALL time line. Different binaries phrase it differently:
#   total wall-time :   1 second
#   total wall-time   :   00:00:02
#   total wall-time    =  1.23 seconds
_RE_WALL_TIME = re.compile(
    r"total\s+wall[- ]time\s*[:=]\s*([0-9:.]+)\s*(seconds?|s|min|hour)?",
    re.IGNORECASE,
)

# HOMO-LUMO gap: in ridft output we get the orbital energies block:
#   number of occupied orbitals ...
#   orbital energies:
#     ...
#     5a       -0.3xxxxx        (occupied)
#     6a        0.0xxxxx        (virtual)
# Easiest: look for "HOMO-LUMO gap" or compute from orbital energies block.
# Turbomole 7.x prints a summary like:
#   "HOMO-LUMO gap         :    0.12345 a.u. = 3.358 eV"
_RE_HOMO_LUMO = re.compile(
    r"HOMO[- ]LUMO\s+(?:gap|GAP)\s*[:=]?\s*"
    r"(-?\d+\.\d+)\s*(?:a\.?u\.?|hartree)?\s*"
    r"(?:=\s*(-?\d+\.\d+)\s*eV)?",
    re.IGNORECASE,
)

# Dipole moment magnitude. Turbomole prints:
#       | dipole moment | =     0.7345 a.u. =      1.867 debye
_RE_DIPOLE = re.compile(
    r"\|\s*dipole moment\s*\|\s*=\s*(-?\d+\.\d+)\s*a\.?u\.?"
    r"\s*(?:=\s*(-?\d+\.\d+)\s*debye)?",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class RidftSummary:
    """All fields are optional — set only those that we could parse."""
    total_energy_ha: float | None = None
    scf_iterations: int | None = None
    converged: bool | None = None
    cpu_time_s: float | None = None
    cpu_time_str: str | None = None        # raw string, in case it was "00:01:23"
    wall_time_s: float | None = None
    wall_time_str: str | None = None       # raw string, in case it was "00:01:23"
    homo_lumo_ha: float | None = None
    homo_lumo_ev: float | None = None
    dipole_au: float | None = None
    dipole_debye: float | None = None
    warnings: list[str] = field(default_factory=list)

    def as_display_dict(self) -> dict[str, str]:
        """Return a dict of pretty-formatted key/value strings for the UI."""
        d: dict[str, str] = {}
        if self.total_energy_ha is not None:
            d["Total energy"] = f"{self.total_energy_ha:.8f} Ha"
        if self.scf_iterations is not None:
            d["SCF iterations"] = str(self.scf_iterations)
        if self.converged is not None:
            d["Converged"] = "yes" if self.converged else "no"
        if self.cpu_time_str:
            d["CPU time"] = self.cpu_time_str
        if self.wall_time_str:
            d["Wall time"] = self.wall_time_str
        if self.homo_lumo_ha is not None:
            ev = f" ({self.homo_lumo_ev:.3f} eV)" if self.homo_lumo_ev is not None else ""
            d["HOMO-LUMO gap"] = f"{self.homo_lumo_ha:.5f} Ha{ev}"
        if self.dipole_au is not None:
            deb = f" ({self.dipole_debye:.3f} D)" if self.dipole_debye is not None else ""
            d["|dipole|"] = f"{self.dipole_au:.4f} a.u.{deb}"
        return d


# ---------------------------------------------------------------------------
# Parsing entry points
# ---------------------------------------------------------------------------
def parse_ridft(source: str | Path) -> RidftSummary:
    """Parse ridft output. `source` is a path OR the file content as string."""
    text = _read_text(source)
    summary = RidftSummary()

    # --- total energy: take the LAST occurrence (last SCF cycle wins) ---
    energies = _RE_TOTAL_ENERGY.findall(text)
    if energies:
        try:
            summary.total_energy_ha = float(energies[-1])
        except ValueError:
            summary.warnings.append("could not parse total energy as float")

    # --- iteration count: highest number seen at start of a line ---
    iters = _RE_ITERATION.findall(text)
    if iters:
        try:
            summary.scf_iterations = max(int(n) for n in iters)
        except ValueError:
            pass

    # --- convergence ---
    summary.converged = bool(_RE_SCF_CONVERGED.search(text))

    # --- CPU time ---
    m = _RE_CPU_TIME.search(text)
    if m:
        raw = m.group(1)
        unit = (m.group(2) or "s").lower()
        summary.cpu_time_str = f"{raw} {unit}".strip()
        summary.cpu_time_s = _to_seconds(raw, unit)

    # --- WALL time ---
    m = _RE_WALL_TIME.search(text)
    if m:
        raw = m.group(1)
        unit = (m.group(2) or "s").lower()
        summary.wall_time_str = f"{raw} {unit}".strip()
        summary.wall_time_s = _to_seconds(raw, unit)

    # --- HOMO-LUMO ---
    m = _RE_HOMO_LUMO.search(text)
    if m:
        try:
            summary.homo_lumo_ha = float(m.group(1))
            if m.group(2):
                summary.homo_lumo_ev = float(m.group(2))
        except (TypeError, ValueError):
            pass

    # --- Dipole ---
    m = _RE_DIPOLE.search(text)
    if m:
        try:
            summary.dipole_au = float(m.group(1))
            if m.group(2):
                summary.dipole_debye = float(m.group(2))
        except (TypeError, ValueError):
            pass

    return summary


def parse_energy_file(source: str | Path) -> float | None:
    """Read Turbomole's $energy file. Last line, second column = total E."""
    text = _read_text(source)
    last_e: float | None = None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].lstrip("-").isdigit():
            try:
                last_e = float(parts[1])
            except ValueError:
                continue
    return last_e


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_text(source: str | Path) -> str:
    """Accept a Path/str path OR a raw string. Heuristic: if `source` is a Path
    or a short string with no newlines and the file exists, treat as path."""
    if isinstance(source, Path):
        try:
            return source.read_text(errors="replace")
        except (OSError, FileNotFoundError):
            return ""
    # string: could be a path or content
    if isinstance(source, str):
        if "\n" not in source and len(source) < 1024:
            p = Path(source)
            if p.exists():
                try:
                    return p.read_text(errors="replace")
                except OSError:
                    return ""
        return source
    return ""


def _to_seconds(raw: str, unit: str) -> float | None:
    """Convert e.g. '1.23', 'seconds' → 1.23; '00:01:23', 's' → 83.0."""
    raw = raw.strip()
    if ":" in raw:
        # HH:MM:SS or MM:SS
        parts = raw.split(":")
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            return None
        if len(nums) == 3:
            return nums[0] * 3600 + nums[1] * 60 + nums[2]
        if len(nums) == 2:
            return nums[0] * 60 + nums[1]
        return nums[0]
    try:
        val = float(raw)
    except ValueError:
        return None
    unit = unit.lower()
    if unit.startswith("min"):
        return val * 60
    if unit.startswith("hour"):
        return val * 3600
    return val


# ===========================================================================
# aoforce output parsing (harmonic vibrational analysis)
# ===========================================================================

@dataclass
class AoforceSummary:
    """Parsed output of `aoforce`. All fields optional."""
    frequencies_cm1: list[float] = field(default_factory=list)
    n_modes: int | None = None
    n_imaginary: int | None = None
    zpe_ha: float | None = None
    enthalpy_ha: float | None = None
    gibbs_ha: float | None = None
    entropy_cal_mol_K: float | None = None
    temperature_K: float | None = None
    warnings: list[str] = field(default_factory=list)

    def lowest_frequencies(self, n: int = 5) -> list[float]:
        """Return the n smallest frequencies (most-imaginary first)."""
        return sorted(self.frequencies_cm1)[:n]

    def as_display_dict(self) -> dict[str, str]:
        d: dict[str, str] = {}
        if self.n_modes is not None:
            d["Vibrational modes"] = str(self.n_modes)
        if self.n_imaginary is not None:
            tag = ""
            if self.n_imaginary == 0:
                tag = "  (minimum)"
            elif self.n_imaginary == 1:
                tag = "  (possible transition state)"
            elif self.n_imaginary > 1:
                tag = "  (higher-order saddle point)"
            d["Imaginary modes"] = f"{self.n_imaginary}{tag}"
        if self.frequencies_cm1:
            lows = self.lowest_frequencies(5)
            d["Lowest 5 frequencies"] = ", ".join(
                f"{f:+.1f}" for f in lows
            ) + " cm⁻¹"
        if self.zpe_ha is not None:
            d["Zero-point energy"] = f"{self.zpe_ha:.6f} Ha"
        if self.temperature_K is not None:
            d["Thermo temperature"] = f"{self.temperature_K:.2f} K"
        if self.enthalpy_ha is not None:
            d["Enthalpy H(T)"] = f"{self.enthalpy_ha:.6f} Ha"
        if self.gibbs_ha is not None:
            d["Gibbs free energy G(T)"] = f"{self.gibbs_ha:.6f} Ha"
        if self.entropy_cal_mol_K is not None:
            d["Total entropy S"] = f"{self.entropy_cal_mol_K:.3f} cal/(mol·K)"
        return d


# Regex collection for aoforce.out. Turbomole's output format has been
# stable across 7.x/8.x for the lines we care about.

# Frequency lines look like:
#   mode               1        2        3        4        5        6
#   frequency        -0.00     0.00     0.00     0.00     0.00     0.00
# We capture *every* frequency value on lines starting with "frequency".
# Negative values denote imaginary modes (the binary prints them as
# negative numbers; some versions print them with an "i" suffix as well).
_RE_AOFORCE_FREQ_LINE = re.compile(r"^\s*frequency\s+(.+)$",
                                    re.IGNORECASE | re.MULTILINE)

# A single frequency token: optional sign, digits.digits, optional 'i'
_RE_FREQ_TOKEN = re.compile(r"(-?\d+\.\d+)i?")

# Zero-point vibrational energy:
#   zero point VIBRATIONAL energy  :        0.0234567   Hartree
_RE_ZPE = re.compile(
    r"zero\s*point\s+VIBRATIONAL\s+energy\s*:\s*(-?\d+\.\d+)\s*Hartree",
    re.IGNORECASE,
)

# Thermo block. Different Turbomole versions phrase it differently;
# we accept the common patterns. Example:
#   T =       298.15 K
#   enthalpy            =     -76.42352   Hartree
#   chem. potential     =     -76.45123   Hartree    <- this is G
#   entropy             =      45.123     J/(mol K)
_RE_THERMO_T = re.compile(
    r"\bT\s*=\s*(-?\d+\.\d+)\s*K", re.IGNORECASE,
)
_RE_THERMO_H = re.compile(
    r"\b(?:enthalpy|H\(T\))\s*[:=]\s*(-?\d+\.\d+)\s*Hartree",
    re.IGNORECASE,
)
_RE_THERMO_G = re.compile(
    r"\b(?:chem\.\s*potential|G\(T\)|Gibbs\s*free\s*energy)\s*"
    r"[:=]\s*(-?\d+\.\d+)\s*Hartree",
    re.IGNORECASE,
)
# Entropy: keep the cal/(mol·K) form because that's the common chemistry
# unit. Turbomole sometimes prints in J/(mol·K); we accept either and
# convert to cal if needed.
_RE_THERMO_S_CAL = re.compile(
    r"\b(?:entropy|S\(T\))\s*[:=]\s*(-?\d+\.\d+)\s*cal/\(mol\*K\)",
    re.IGNORECASE,
)
_RE_THERMO_S_J = re.compile(
    r"\b(?:entropy|S\(T\))\s*[:=]\s*(-?\d+\.\d+)\s*J/\(mol\*?K\)",
    re.IGNORECASE,
)


def parse_aoforce(source: str | Path) -> AoforceSummary:
    """Parse aoforce.out. `source` is a path OR the file content."""
    text = _read_text(source)
    s = AoforceSummary()

    # --- Frequencies ---
    freqs: list[float] = []
    for line in _RE_AOFORCE_FREQ_LINE.findall(text):
        for tok in _RE_FREQ_TOKEN.findall(line):
            try:
                freqs.append(float(tok))
            except ValueError:
                continue
    # The first 6 modes (3 translations + 3 rotations) come out near zero
    # for non-linear molecules; we keep them in the list but report the
    # imaginary count using a small tolerance to ignore numerical noise.
    if freqs:
        s.frequencies_cm1 = freqs
        s.n_modes = len(freqs)
        # Modes are imaginary if frequency < -threshold; tolerance 1 cm-1
        # excludes the near-zero translational/rotational ones.
        s.n_imaginary = sum(1 for f in freqs if f < -1.0)

    # --- ZPE ---
    m = _RE_ZPE.search(text)
    if m:
        try:
            s.zpe_ha = float(m.group(1))
        except ValueError:
            pass

    # --- Temperature ---
    m = _RE_THERMO_T.search(text)
    if m:
        try:
            s.temperature_K = float(m.group(1))
        except ValueError:
            pass

    # --- Enthalpy ---
    m = _RE_THERMO_H.search(text)
    if m:
        try:
            s.enthalpy_ha = float(m.group(1))
        except ValueError:
            pass

    # --- Gibbs ---
    m = _RE_THERMO_G.search(text)
    if m:
        try:
            s.gibbs_ha = float(m.group(1))
        except ValueError:
            pass

    # --- Entropy ---
    m = _RE_THERMO_S_CAL.search(text)
    if m:
        try:
            s.entropy_cal_mol_K = float(m.group(1))
        except ValueError:
            pass
    else:
        m = _RE_THERMO_S_J.search(text)
        if m:
            try:
                # 1 cal = 4.184 J
                s.entropy_cal_mol_K = float(m.group(1)) / 4.184
            except ValueError:
                pass

    return s

# ===========================================================================
# SCF iteration parser (per-cycle energies from ridft.out)
# ===========================================================================

# Matches the per-iteration energy line that follows the "ITERATION ENERGY"
# header in ridft.out. Example:
#    1  -7401.0407959107    -40945.416506     18159.243113    0.000D+00 0.296D-10
# We anchor on: optional leading spaces, integer iteration index, then a
# float (the total SCF energy in Hartree). Subsequent columns are ignored.
_RE_SCF_ITER = re.compile(
    r"^\s*(\d+)\s+(-?\d+\.\d+)\s+-?\d+\.\d+\s+-?\d+\.\d+\s+"
    r"-?\d+\.\d+[DdEe][+-]?\d+",
    re.MULTILINE,
)


def parse_scf_iterations(source: str | Path) -> list[tuple[int, float]]:
    """Extract (iteration, total_energy_Ha) pairs from a ridft.out file.

    Returns a chronologically ordered list. Duplicate iteration numbers
    (e.g. SCF restarted after grid refinement) are kept as-is so the
    caller can see the full trajectory.
    """
    text = _read_text(source)
    out: list[tuple[int, float]] = []
    for m in _RE_SCF_ITER.finditer(text):
        try:
            out.append((int(m.group(1)), float(m.group(2))))
        except ValueError:
            continue
    return out
