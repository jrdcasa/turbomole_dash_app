"""
Parsers for Turbomole optimization trajectory files.

Reads:
  - `energy`   : per-cycle SCF/SCFKIN/SCFPOT (Hartree)
  - `gradient` : per-cycle geometry + |dE/dxyz| (Bohr / Hartree·Bohr^-1)
  - `control`  : optional convergence thresholds ($gdiis, jobex defaults)

All functions accept either a Path/str path or the raw file content,
mirroring the convention in result_parser.py so the same code works for
locally downloaded files and remote `tail`-style snippets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Bohr -> Angstrom (CODATA 2018)
BOHR_TO_ANGSTROM = 0.529177210903

# jobex default convergence thresholds (Turbomole 7.x / 8.x).
# These are the values jobex compares against when deciding to stop;
# we use them as fallback when the `control` file doesn't override them.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "energy_change":   1.0e-6,   # Hartree
    "gradient_max":    1.0e-3,   # Hartree / Bohr
    "gradient_rms":    5.0e-4,   # Hartree / Bohr
    "displacement_max":1.0e-3,   # Bohr
    "displacement_rms":5.0e-4,   # Bohr
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OptCycle:
    """One geometry-optimization cycle."""
    cycle: int
    scf_energy: float                  # total SCF energy (Hartree)
    scf_kinetic: float | None = None
    scf_potential: float | None = None
    grad_norm: float | None = None     # |dE/dxyz|
    grad_max: float | None = None      # max |grad_i|


@dataclass
class OptTrajectory:
    """Full optimization trajectory parsed from `energy` + `gradient`."""
    cycles: list[OptCycle] = field(default_factory=list)
    # last_geometry is stored in Angstrom (converted from Bohr at parse time)
    last_geometry: list[tuple[str, float, float, float]] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))

    @property
    def n_cycles(self) -> int:
        return len(self.cycles)

    @property
    def total_energy_drop(self) -> float | None:
        if len(self.cycles) < 2:
            return None
        return self.cycles[-1].scf_energy - self.cycles[0].scf_energy

    @property
    def last_delta_e(self) -> float | None:
        if len(self.cycles) < 2:
            return None
        return self.cycles[-1].scf_energy - self.cycles[-2].scf_energy

    @property
    def last_grad_max(self) -> float | None:
        if not self.cycles:
            return None
        return self.cycles[-1].grad_max

    @property
    def last_grad_norm(self) -> float | None:
        if not self.cycles:
            return None
        return self.cycles[-1].grad_norm

    def convergence_status(self) -> dict[str, dict]:
        """Compare last-cycle |ΔE| and |grad|max to thresholds.

        Returns a mapping criterion -> {value, threshold, ok}. Missing
        values are reported with ok=None (not evaluated).
        """
        out: dict[str, dict] = {}

        de = self.last_delta_e
        out["energy_change"] = {
            "value": abs(de) if de is not None else None,
            "threshold": self.thresholds["energy_change"],
            "ok": (abs(de) < self.thresholds["energy_change"]) if de is not None else None,
        }
        gmax = self.last_grad_max
        out["gradient_max"] = {
            "value": gmax,
            "threshold": self.thresholds["gradient_max"],
            "ok": (gmax < self.thresholds["gradient_max"]) if gmax is not None else None,
        }
        gnorm = self.last_grad_norm
        out["gradient_norm"] = {
            "value": gnorm,
            "threshold": self.thresholds["gradient_rms"],
            "ok": (gnorm < self.thresholds["gradient_rms"]) if gnorm is not None else None,
        }
        return out

    def converged(self) -> bool:
        """True iff every evaluated criterion is satisfied."""
        status = self.convergence_status()
        evaluated = [v["ok"] for v in status.values() if v["ok"] is not None]
        return bool(evaluated) and all(evaluated)

    def to_xyz(self, comment: str | None = None) -> str:
        """Render the last geometry as a standard .xyz file."""
        if not self.last_geometry:
            return ""
        if comment is None:
            e = self.cycles[-1].scf_energy if self.cycles else None
            comment = (f"cycle={self.n_cycles} E={e:.10f} Ha"
                       if e is not None else f"cycle={self.n_cycles}")
        lines = [str(len(self.last_geometry)), comment]
        for sym, x, y, z in self.last_geometry:
            lines.append(f"{sym:<3s} {x:>15.8f} {y:>15.8f} {z:>15.8f}")
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_text(source: str | Path) -> str:
    """Accept Path/str path OR raw content. Heuristic identical to
    result_parser._read_text so behaviour stays consistent."""
    if isinstance(source, Path):
        try:
            return source.read_text(errors="replace")
        except (OSError, FileNotFoundError):
            return ""
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


# ---------------------------------------------------------------------------
# `energy` file parser
# ---------------------------------------------------------------------------
#
# Format (one cycle per line, after $energy header):
#     cycle   SCF              SCFKIN          SCFPOT
#       1   -76.42352123     76.0         -152.4
#       2   -76.42452123     76.0         -152.4
#     ...
#   $end

def parse_energy(source: str | Path) -> list[tuple[int, float, float | None, float | None]]:
    """Return [(cycle, scf, kin, pot), ...]."""
    text = _read_text(source)
    out: list[tuple[int, float, float | None, float | None]] = []
    in_block = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("$energy"):
            in_block = True
            continue
        if not in_block:
            continue
        if s.startswith("$"):                # next section / $end
            break
        parts = s.split()
        if len(parts) < 2:
            continue
        try:
            cyc = int(parts[0])
        except ValueError:
            continue
        try:
            scf = float(parts[1])
        except ValueError:
            continue
        kin = _safe_float(parts[2]) if len(parts) > 2 else None
        pot = _safe_float(parts[3]) if len(parts) > 3 else None
        out.append((cyc, scf, kin, pot))
    return out


# ---------------------------------------------------------------------------
# `gradient` file parser
# ---------------------------------------------------------------------------
#
# Format (one block per cycle):
#     $grad   cartesian gradients
#     cycle =  1  SCF energy = -76.42... |dE/dxyz| = 1.234E-02
#       0.000  0.000  0.000   o
#       1.234  0.000  0.000   h
#       0.000  1.234  0.000   h
#       <gx>   <gy>   <gz>
#       <gx>   <gy>   <gz>
#       <gx>   <gy>   <gz>
#     cycle =  2  ...
#       ...
#     $end
#
# Coordinates are in Bohr; gradient components in Hartree/Bohr.

_RE_CYCLE_HEADER = re.compile(
    r"cycle\s*=\s*(\d+)\s+"
    r"SCF\s+energy\s*=\s*(-?\d+\.\d+(?:[eE][+-]?\d+)?)\s+"
    r"\|dE/dxyz\|\s*=\s*(-?\d+\.\d+(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)

# A coord/gradient line is 3 floats + optional element symbol. Element
# present -> coord line; element absent -> gradient line.
_RE_COORD_LINE = re.compile(
    r"^\s*(-?\d+\.\d+(?:[eE][+-]?\d+)?)\s+"
    r"(-?\d+\.\d+(?:[eE][+-]?\d+)?)\s+"
    r"(-?\d+\.\d+(?:[eE][+-]?\d+)?)\s+"
    r"([A-Za-z]{1,3})\s*$"
)

# Turbomole writes gradient components in Fortran 'D' notation
# (e.g. '0.12345D-03'); allow both E and D.
_RE_GRAD_LINE = re.compile(
    r"^\s*(-?\d+\.\d+(?:[eEdD][+-]?\d+)?)\s+"
    r"(-?\d+\.\d+(?:[eEdD][+-]?\d+)?)\s+"
    r"(-?\d+\.\d+(?:[eEdD][+-]?\d+)?)\s*$"
)


@dataclass
class _GradCycle:
    cycle: int
    scf_energy: float
    grad_norm: float
    coords_bohr: list[tuple[str, float, float, float]]
    grad_max: float | None


def parse_gradient(source: str | Path) -> list[_GradCycle]:
    """Parse all cycles from a Turbomole `gradient` file."""
    text = _read_text(source)
    cycles: list[_GradCycle] = []
    cur_header: re.Match | None = None
    cur_coords: list[tuple[str, float, float, float]] = []
    cur_grads: list[tuple[float, float, float]] = []

    def _flush():
        if cur_header is None:
            return
        gmax = None
        if cur_grads:
            gmax = max(max(abs(g) for g in tup) for tup in cur_grads)
        cycles.append(_GradCycle(
            cycle=int(cur_header.group(1)),
            scf_energy=float(cur_header.group(2)),
            grad_norm=float(cur_header.group(3)),
            coords_bohr=list(cur_coords),
            grad_max=gmax,
        ))

    in_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("$grad"):
            in_block = True
            continue
        if not in_block:
            continue
        if stripped.startswith("$"):
            _flush()
            break

        m = _RE_CYCLE_HEADER.search(line)
        if m:
            _flush()
            cur_header = m
            cur_coords = []
            cur_grads = []
            continue

        m = _RE_COORD_LINE.match(line)
        if m:
            x, y, z, sym = m.group(1), m.group(2), m.group(3), m.group(4)
            cur_coords.append((sym.lower(), float(x), float(y), float(z)))
            continue
        m = _RE_GRAD_LINE.match(line)
        if m:
            # Fortran 'D' exponent -> Python 'E'
            vals = tuple(float(g.replace("D", "E").replace("d", "e"))
                         for g in m.groups())
            cur_grads.append(vals)
            continue
    else:
        # File ended without explicit $end
        _flush()

    return cycles


# ---------------------------------------------------------------------------
# `control` file: optional override of jobex thresholds
# ---------------------------------------------------------------------------

# $gdiis section may override convergence; jobex also uses these defaults
# without an explicit block. Examples seen in the wild:
#   $jobex
#       energy=6
#       gcart=3
#   $end
# meaning 1e-6 Hartree for energy and 1e-3 for cartesian gradient.

_RE_JOBEX_BLOCK = re.compile(
    r"\$jobex\b(.*?)(?=^\$|\Z)", re.IGNORECASE | re.DOTALL | re.MULTILINE,
)
_RE_JOBEX_ENERGY = re.compile(r"\benergy\s*=\s*(\d+)", re.IGNORECASE)
_RE_JOBEX_GCART  = re.compile(r"\bgcart\s*=\s*(\d+)",  re.IGNORECASE)


def parse_thresholds_from_control(source: str | Path) -> dict[str, float]:
    """Return effective convergence thresholds for jobex.

    Recognises `energy=N` (-> 10^-N Hartree) and `gcart=N` (-> 10^-N a.u.)
    inside `$jobex`. Missing keys fall back to DEFAULT_THRESHOLDS.
    """
    text = _read_text(source)
    thr = dict(DEFAULT_THRESHOLDS)
    if not text:
        return thr
    m = _RE_JOBEX_BLOCK.search(text)
    if not m:
        return thr
    body = m.group(1)
    me = _RE_JOBEX_ENERGY.search(body)
    if me:
        thr["energy_change"] = 10.0 ** (-int(me.group(1)))
    mg = _RE_JOBEX_GCART.search(body)
    if mg:
        thr["gradient_max"] = 10.0 ** (-int(mg.group(1)))
        thr["gradient_rms"] = 10.0 ** (-int(mg.group(1))) / 2.0
    return thr


# ---------------------------------------------------------------------------
# High-level: load a full trajectory from a downloaded job directory
# ---------------------------------------------------------------------------

def load_trajectory(job_dir: Path) -> OptTrajectory:
    """Build an OptTrajectory from the files inside a downloaded job dir.

    The directory is expected to contain at least `energy` and `gradient`
    (produced by ridft + jobex). `control` is optional and only used to
    refine the convergence thresholds. Missing or empty files yield an
    empty trajectory (n_cycles == 0).
    """
    energy_path = job_dir / "energy"
    gradient_path = job_dir / "gradient"
    control_path = job_dir / "control"

    thresholds = (parse_thresholds_from_control(control_path)
                  if control_path.exists() else dict(DEFAULT_THRESHOLDS))

    energies = parse_energy(energy_path) if energy_path.exists() else []
    grad_cycles = parse_gradient(gradient_path) if gradient_path.exists() else []

    # Merge: gradient gives geometry + |grad|; energy gives SCFKIN/SCFPOT.
    # gradient cycles are authoritative for the cycle count because the
    # `energy` file may contain one extra row from the final SP.
    cycles: list[OptCycle] = []
    energies_by_cycle = {c: (scf, kin, pot) for c, scf, kin, pot in energies}

    for g in grad_cycles:
        kin = pot = None
        if g.cycle in energies_by_cycle:
            _, kin, pot = energies_by_cycle[g.cycle]
        cycles.append(OptCycle(
            cycle=g.cycle,
            scf_energy=g.scf_energy,
            scf_kinetic=kin,
            scf_potential=pot,
            grad_norm=g.grad_norm,
            grad_max=g.grad_max,
        ))

    # Fallback: no gradient file (e.g. a still-running job that hasn't
    # written gradient yet). Use energies only, no geometry.
    if not cycles and energies:
        for c, scf, kin, pot in energies:
            cycles.append(OptCycle(cycle=c, scf_energy=scf,
                                   scf_kinetic=kin, scf_potential=pot))

    last_geom_ang: list[tuple[str, float, float, float]] = []
    if grad_cycles:
        last = grad_cycles[-1]
        last_geom_ang = [
            (sym.capitalize(),
             x * BOHR_TO_ANGSTROM,
             y * BOHR_TO_ANGSTROM,
             z * BOHR_TO_ANGSTROM)
            for sym, x, y, z in last.coords_bohr
        ]

    return OptTrajectory(
        cycles=cycles,
        last_geometry=last_geom_ang,
        thresholds=thresholds,
    )


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def _safe_float(s: str) -> float | None:
    try:
        return float(s.replace("D", "E").replace("d", "e"))
    except ValueError:
        return None

# ---------------------------------------------------------------------------
# Multi-frame trajectory (VMD-compatible .xyz)
# ---------------------------------------------------------------------------

def trajectory_xyz(job_dir: Path, name: str = "frame") -> str:
    """Render every gradient cycle as a multi-frame .xyz (Angstrom).

    Output format (one frame per cycle, concatenated):
        N
        cycle=<i> E=<scf_energy> Ha
        <sym>  x y z
        ...

    Returns an empty string if no gradient file is present.
    """
    gradient_path = job_dir / "gradient"
    if not gradient_path.exists():
        return ""
    cycles = parse_gradient(gradient_path)
    if not cycles:
        return ""

    out: list[str] = []
    for g in cycles:
        out.append(str(len(g.coords_bohr)))
        out.append(f"{name} cycle={g.cycle} E={g.scf_energy:.10f} Ha "
                   f"|grad|={g.grad_norm:.3e}")
        for sym, xb, yb, zb in g.coords_bohr:
            out.append(
                f"{sym.capitalize():<3s} "
                f"{xb * BOHR_TO_ANGSTROM:>15.8f} "
                f"{yb * BOHR_TO_ANGSTROM:>15.8f} "
                f"{zb * BOHR_TO_ANGSTROM:>15.8f}"
            )
    return "\n".join(out) + "\n"