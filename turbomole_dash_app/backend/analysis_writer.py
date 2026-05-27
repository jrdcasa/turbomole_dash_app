"""
Materialize analysis artifacts (xyz, csv, gnuplot scripts) on disk.

Files are written inside `<job_local_dir>/analysis/` so they live next
to the raw Turbomole output and follow the job's lifecycle (a "Clean
job" action removes them automatically).

This module is intentionally side-effect heavy: it writes to disk and
returns the paths it created. The Dash callback then renders those
paths in a copy-to-clipboard panel.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from backend.analysis_exports import (
    JobMeta,
    gnuplot_optimization, gnuplot_scf_convergence, gnuplot_spectrum,
    gnuplot_trajectory,
    scf_convergence_to_csv,
    single_point_to_csv, spectrum_to_csv, trajectory_to_csv,
)
from backend.opt_trajectory import load_trajectory, trajectory_xyz
from backend.result_parser import (
    parse_aoforce, parse_ridft, parse_scf_iterations,
)

log = logging.getLogger("analysis_writer")


ANALYSIS_SUBDIR = "analysis"


@dataclass
class ArtifactSet:
    """Outcome of writing analysis files for one job."""
    out_dir: Path
    files: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.files)


def write_artifacts(job: dict, local_dir: Path) -> ArtifactSet:
    """Generate all analysis artifacts for `job` under
    `<local_dir>/analysis/` and return the list of files written.

    The set of files depends on the job's task_type:
      - optimization: last.xyz, trajectory.xyz, trajectory.csv,
                      energy.gp, gradient.gp
      - single_point: summary.csv
      - frequencies:  spectrum.csv, spectrum.gp
      - aimd:         last.xyz, trajectory.xyz (when available)
    """
    out_dir = local_dir / ANALYSIS_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    result = ArtifactSet(out_dir=out_dir)

    name = job["name"]
    meta = JobMeta(name=name,
                   functional=job.get("functional"),
                   basis_set=job.get("basis_set"))
    task = job["task_type"]

    if task == "optimization":
        _write_optimization(out_dir, name, meta, local_dir, result)
    elif task == "single_point":
        _write_single_point(out_dir, name, meta, local_dir, result)
    elif task == "frequencies":
        _write_frequencies(out_dir, name, meta, local_dir, result)
    elif task == "aimd":
        _write_aimd(out_dir, name, local_dir, result)
    else:
        result.warnings.append(
            f"Analysis artifacts for task '{task}' are not implemented."
        )

    return result


# ---------------------------------------------------------------------------
# Per-task writers
# ---------------------------------------------------------------------------

def _write_optimization(out_dir: Path, name: str, meta: JobMeta,
                        local_dir: Path, result: ArtifactSet) -> None:
    traj = load_trajectory(local_dir)
    if traj.n_cycles == 0:
        result.warnings.append(
            "No `energy` / `gradient` data found; "
            "no optimization artifacts written."
        )
    else:
        last_xyz = traj.to_xyz(
            comment=(f"{name} cycle={traj.n_cycles} "
                     f"E={traj.cycles[-1].scf_energy:.10f} Ha")
        )
        if last_xyz:
            _safe_write(out_dir / f"{name}_last.xyz", last_xyz, result)

        traj_xyz_text = trajectory_xyz(local_dir, name=name)
        if traj_xyz_text:
            _safe_write(out_dir / f"{name}_trajectory.xyz",
                        traj_xyz_text, result)

        csv_filename = f"{name}_trajectory.csv"
        _safe_write(out_dir / csv_filename,
                    trajectory_to_csv(traj), result)
        _safe_write(out_dir / f"{name}_energy.gp",
                    gnuplot_optimization(csv_filename, meta), result)
        _safe_write(out_dir / f"{name}_gradient.gp",
                    gnuplot_trajectory(csv_filename, meta), result)

    # Single-point summary + SCF convergence of the *final* ridft pass
    _write_scf_block(out_dir, name, meta, local_dir, result)


def _write_single_point(out_dir: Path, name: str, meta: JobMeta,
                        local_dir: Path, result: ArtifactSet) -> None:
    if not (local_dir / "ridft.out").exists():
        result.warnings.append("No ridft.out; cannot write summary.csv.")
        return
    _write_scf_block(out_dir, name, meta, local_dir, result)


def _write_frequencies(out_dir: Path, name: str, meta: JobMeta,
                       local_dir: Path, result: ArtifactSet) -> None:
    aoforce = local_dir / "aoforce.out"
    if aoforce.exists():
        summary = parse_aoforce(aoforce)
        if summary.frequencies_cm1:
            csv_filename = f"{name}_spectrum.csv"
            _safe_write(out_dir / csv_filename,
                        spectrum_to_csv(summary), result)
            _safe_write(out_dir / f"{name}_spectrum.gp",
                        gnuplot_spectrum(csv_filename, meta), result)
        else:
            result.warnings.append(
                "aoforce.out present but no frequencies were parsed."
            )
    else:
        result.warnings.append("No aoforce.out; cannot write spectrum.csv.")

    # The SCF that backs the frequencies is in ridft.out
    _write_scf_block(out_dir, name, meta, local_dir, result)


def _write_aimd(out_dir: Path, name: str, local_dir: Path,
                result: ArtifactSet) -> None:
    """For aimd we currently reuse the `gradient`-based trajectory.
    `frog`'s native mdlog.x format is not yet supported."""
    traj_xyz_text = trajectory_xyz(local_dir, name=name)
    if not traj_xyz_text:
        result.warnings.append(
            "No trajectory data found for this AIMD job."
        )
        return
    _safe_write(out_dir / f"{name}_trajectory.xyz",
                traj_xyz_text, result)
    # Last frame as a separate file for convenience.
    traj = load_trajectory(local_dir)
    last = traj.to_xyz()
    if last:
        _safe_write(out_dir / f"{name}_last.xyz", last, result)


# ---------------------------------------------------------------------------
# I/O helper
# ---------------------------------------------------------------------------

def _safe_write(path: Path, content: str, result: ArtifactSet) -> None:
    """Write `content` to `path`, recording the success or the error."""
    try:
        path.write_text(content)
    except OSError as exc:
        log.exception("Could not write %s", path)
        result.warnings.append(f"Could not write {path.name}: {exc}")
        return
    result.files.append(path)

def _write_scf_block(out_dir: Path, name: str, meta: JobMeta,
                     local_dir: Path, result: ArtifactSet) -> None:
    """Write summary.csv + SCF convergence artifacts from ridft.out.

    Shared by single_point, optimization (final ridft of jobex) and
    frequencies (ridft is run before aoforce). Silently skips if
    ridft.out is missing — the caller already warned about it.
    """
    ridft = local_dir / "ridft.out"
    if not ridft.exists():
        return
    summary = parse_ridft(ridft)
    _safe_write(out_dir / f"{name}_singlepoint.csv",
                single_point_to_csv(summary), result)

    iters = parse_scf_iterations(ridft)
    if len(iters) >= 2:
        csv_filename = f"{name}_scf_convergence.csv"
        _safe_write(out_dir / csv_filename,
                    scf_convergence_to_csv(iters), result)
        _safe_write(out_dir / f"{name}_scf.gp",
                    gnuplot_scf_convergence(csv_filename, meta), result)