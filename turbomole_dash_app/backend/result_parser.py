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
