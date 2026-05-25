"""
SLURM script generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class SlurmParams:
    job_name: str
    partition: str = "compute"
    time: str = "24:00:00"
    nodes: int = 1
    ntasks: int = 16
    cpus_per_task: int = 1
    mem: str = "32G"
    account: str | None = None
    qos: str | None = None
    constraint: str | None = None
    reservation: str | None = None      # NEW: optional SLURM reservation
    extra_sbatch: tuple[str, ...] = ()


def build_slurm_script(
    params: SlurmParams,
    remote_workdir: str,
    module_load: Iterable[str],
    driver_commands: Iterable[str],
    env_setup: Iterable[str] = (),
) -> str:
    """Return the text of a SLURM submission script.

    env_setup runs before module_load. Use it for arbitrary shell
    setup (source files, export VAR=value, ...) when the cluster
    doesn't expose your software via the module system.
    """
    header = [
        "#!/bin/bash -l",
        f"#SBATCH --job-name={params.job_name}",
        f"#SBATCH --partition={params.partition}",
        f"#SBATCH --time={params.time}",
        f"#SBATCH --nodes={params.nodes}",
        f"#SBATCH --ntasks={params.ntasks}",
        f"#SBATCH --cpus-per-task={params.cpus_per_task}",
        f"#SBATCH --mem={params.mem}",
        "#SBATCH --output=slurm-%j.out",
        "#SBATCH --error=slurm-%j.err",
    ]
    if params.account:
        header.append(f"#SBATCH --account={params.account}")
    if params.qos:
        header.append(f"#SBATCH --qos={params.qos}")
    if params.constraint:
        header.append(f"#SBATCH --constraint={params.constraint}")
    if params.reservation and params.reservation.strip():
        header.append(f"#SBATCH --reservation={params.reservation.strip()}")
    header.extend(params.extra_sbatch)

    body = [
        "",
        "set -euo pipefail",
        "",
        "# --- environment ---------------------------------------------------",
    ]

    for line in env_setup:
        body.append(line)

    for m in module_load:
        body.append(f"module load {m}")

    body += [
        "",
        "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}",
        f"export PARNODES={params.ntasks}",
        '# PARA_ARCH defaults to SMP; override in env_setup if MPI',
        ': "${PARA_ARCH:=SMP}"',
        "export PARA_ARCH",
        "",
        f"cd {remote_workdir}",
        "",
        "# --- run -----------------------------------------------------------",
    ]
    body.extend(driver_commands)
    body += [
        "",
        "echo TURBOMOLE_JOB_DONE > .job_done",
    ]

    return "\n".join(header + body) + "\n"
