"""
CSV exports and gnuplot script generation for the Analysis tab.

All functions return strings (never write to disk). The Dash callbacks
hand the result to dcc.Download, which streams the bytes directly to
the browser. This keeps the app stateless on disk.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

from backend.opt_trajectory import OptTrajectory
from backend.result_parser import AoforceSummary, RidftSummary


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def trajectory_to_csv(traj: OptTrajectory) -> str:
    """One row per optimization cycle.

    Columns are chosen to be directly plottable by the gnuplot template:
      cycle, scf_energy_Ha, dE_Ha, grad_norm, grad_max
    where dE is the energy relative to cycle 1 (so plots can show the
    descent towards the minimum without subtracting on the fly).
    """
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["cycle", "scf_energy_Ha", "dE_vs_first_Ha",
                "grad_norm", "grad_max"])
    if not traj.cycles:
        return buf.getvalue()
    e0 = traj.cycles[0].scf_energy
    for c in traj.cycles:
        w.writerow([
            c.cycle,
            f"{c.scf_energy:.10f}",
            f"{c.scf_energy - e0:.10f}",
            "" if c.grad_norm is None else f"{c.grad_norm:.6e}",
            "" if c.grad_max  is None else f"{c.grad_max:.6e}",
        ])
    return buf.getvalue()


def spectrum_to_csv(summary: AoforceSummary) -> str:
    """One row per vibrational mode: index, freq, real_or_imaginary."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["mode", "frequency_cm-1", "kind"])
    for i, f in enumerate(summary.frequencies_cm1, start=1):
        kind = "imaginary" if f < -1.0 else (
            "near_zero" if abs(f) < 1.0 else "real"
        )
        w.writerow([i, f"{f:.4f}", kind])
    return buf.getvalue()


def single_point_to_csv(summary: RidftSummary) -> str:
    """key,value table from a parsed ridft.out."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["key", "value"])
    for k, v in summary.as_display_dict().items():
        w.writerow([k, v])
    return buf.getvalue()


# ---------------------------------------------------------------------------
# gnuplot scripts
# ---------------------------------------------------------------------------

_STYLE_BLOCK = """\
reset
set style line 1 lt 1 ps 1.2 lc rgb "black"  pt 6 lw 2.0
set style line 2 lt 1 ps 0.4 lc rgb "red"    pt 4 lw 2.0
set style line 3 lt 2 ps 0.4 lc rgb "blue"   pt 4 lw 2.0
set style line 4 lt 1 ps 0.4 lc rgb "green"  pt 4 lw 2.0
set style line 5 lt 2 ps 0.4 lc rgb "yellow" pt 4 lw 2.0
set style line 6 lt 2 ps 0.4 lc rgb "orange" pt 4 lw 2.0
set encoding utf8
###############################################################################
"""


def _gnuplot_escape(s: str) -> str:
    """Escape a string for use inside a double-quoted gnuplot literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


@dataclass
class JobMeta:
    """Minimal job metadata used for plot titles."""
    name: str
    functional: str | None
    basis_set: str | None

    def title(self, extra: str) -> str:
        method = "/".join(x for x in (self.functional, self.basis_set) if x)
        if method:
            return f"{extra} — {self.name} ({method})"
        return f"{extra} — {self.name}"


def gnuplot_optimization(csv_filename: str, meta: JobMeta) -> str:
    """Gnuplot script: SCF energy per cycle, ΔE in kcal/mol.

    Uses the same style/conversion idiom as the user's template:
      - reads E0 from the first data row,
      - prints (E - E0) * conv,
      - conv = 627.5092 (Ha -> kcal/mol).
    """
    title = _gnuplot_escape(meta.title("Optimization"))
    fname = _gnuplot_escape(csv_filename)
    return _STYLE_BLOCK + f"""\
set term wxt 1 enhanced dashed size 600,400 font "DejaVu Sans,12"
set multiplot layout 1,1
set title "{title}" nonenhanced
f1="{fname}"
set datafile separator ","
# Skip header line when computing E0 (column 2 = scf_energy_Ha)
E0=real(word(system(sprintf("sed -n '2p' %s | tr ',' ' '",f1)),2))
print E0
conv=627.5092
# Plot settings
#set xrange [1 : *]
#set yrange [* : *]
set format x "%.0f"
set format y "%.1f"
set xlabel "Optimization cycle" font "Arial, 12"
set ylabel "{{/Symbol D}}E (kcal/mol)" font "Arial, 12"
set grid
set key top right
p f1 every ::1 u 1:(($2-E0)*conv) w p ls 1 notitle, \\
  f1 every ::1 u 1:(($2-E0)*conv) w l ls 1 notitle
unset multiplot
"""


def gnuplot_trajectory(csv_filename: str, meta: JobMeta) -> str:
    """Gnuplot script: |grad|, |grad|max vs cycle (log y).

    Useful companion to the energy plot for diagnosing slow convergence.
    """
    title = _gnuplot_escape(meta.title("Gradient norm"))
    fname = _gnuplot_escape(csv_filename)
    return _STYLE_BLOCK + f"""\
set term wxt 2 enhanced dashed size 600,400 font "DejaVu Sans,12"
set multiplot layout 1,1
set title "{title}" nonenhanced
f1="{fname}"
set datafile separator ","
set logscale y
set format y "10^{{%T}}"
set xlabel "Optimization cycle" font "Arial, 12"
set ylabel "Gradient (Ha/Bohr)" font "Arial, 12"
set grid
set key top right
# Column layout: cycle, scf_energy_Ha, dE_vs_first_Ha, grad_norm, grad_max
p f1 every ::1 u 1:4 w lp ls 1 title "|grad|", \\
  f1 every ::1 u 1:5 w lp ls 2 title "|grad|_{{max}}"
unset multiplot
"""


def gnuplot_spectrum(csv_filename: str, meta: JobMeta) -> str:
    """Gnuplot stick plot of vibrational frequencies (cm^-1)."""
    title = _gnuplot_escape(meta.title("Vibrational spectrum"))
    fname = _gnuplot_escape(csv_filename)
    return _STYLE_BLOCK + f"""\
set term wxt 1 enhanced dashed size 700,400 font "DejaVu Sans,12"
set multiplot layout 1,1
set title "{title}" nonenhanced
f1="{fname}"
set datafile separator ","
# Filter out near-zero translational/rotational modes (column 3 != near_zero)
set xlabel "Wavenumber (cm^{{-1}})" font "Arial, 12"
set ylabel "Intensity (a.u.)" font "Arial, 12"
set xrange [*:*] reverse
set yrange [0:1.2]
set grid
unset key
# One vertical impulse per mode (real + imaginary); colours match the UI.
p f1 every ::1 u 2:(1.0):(stringcolumn(3) eq "imaginary" ? 0xff7b72 : 0x58a6ff) \\
       w impulses lc rgb variable lw 2 notitle
unset multiplot
"""