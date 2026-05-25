# Turbomole Orchestrator (Dash)

A local Dash application that builds Turbomole inputs with ASE and
dispatches the heavy quantum-chemistry jobs to one or more HPC clusters
running SLURM. Designed for multiscale workflows (organometallics,
polymers, membrane models) where local pre/post-processing must coexist
with remote DFT/AIMD runs.

## Architecture

```
┌──────────────────────────── local machine ────────────────────────────┐
│                                                                       │
│  Dash UI (app.py + app_ui/layout.py + app_ui/callbacks.py)            │
│     │                                                                 │
│     ├── ASE input builder (backend/turbomole_io.py)                   │
│     │       └── per-Turbomole-version drivers (7.8, 8.0)              │
│     ├── SLURM script builder (backend/slurm.py)                       │
│     ├── SQLite job store (backend/db.py)        ← survives shutdown   │
│     └── On-demand refresh (workers/poller.py)   ← user-triggered      │
│              │                                                        │
│              │  ssh / sftp (paramiko)                                 │
│              ▼                                                        │
└──────────────┼────────────────────────────────────────────────────────┘
               │
┌──────────────▼────────────────────── HPC cluster ─────────────────────┐
│  $REMOTE_WORKDIR/<job>_<ts>/                                          │
│      coord, define.inp, submit.slurm                                  │
│           │                                                           │
│           │  sbatch → bash -l → define < define.inp → control         │
│           ▼                                                           │
│      Turbomole binaries: ridft / jobex / frog                         │
└───────────────────────────────────────────────────────────────────────┘
```

## Key design points

| Requirement | Implementation |
|---|---|
| Local Dash app talking to remote HPC | `remote/ssh_client.py` + `remote/slurm_remote.py` |
| ASE input generation | `backend/turbomole_io.py`: writes `coord` + `define.inp` |
| Control file built on cluster by `define` | per-version driver classes generate a stdin script |
| Multi-version Turbomole support | `turbomole_version` per cluster; 7.8 + 8.0 driver placeholders |
| Structure formats | xyz, coord, mol/mol2, pdb, sdf, cif, etc. (ASE-supported) |
| SLURM submission with custom env | `env_setup:` for arbitrary shell lines; `module_load:` for modules |
| Survives local shutdown | SQLite + on-demand refresh; no background daemon |
| Download partial / final / clean | per-job action buttons in Job Manager |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml   # then edit
python app.py
```

Open <http://127.0.0.1:8050>.

## Configuration

`config.yaml` is searched in this order:

1. Next to `app.py` (recommended)
2. Current working directory
3. `~/.turbomole_orchestrator/config.yaml`

Each cluster declares:

- **`host`**, **`user`**, **`key_filename` / `use_agent`**: SSH access
- **`remote_workdir`**: base directory on the cluster for job folders
- **`module_load`**: list of modules to `module load` in every job
- **`env_setup`**: arbitrary shell lines (e.g. `source /path/to/setup.sh`)
- **`turbomole_version`**: `"7.8"` or `"8.0"`. Drives which `define.inp` flavor is generated.
- **`default_*`**: SLURM defaults pre-filled in the UI

Authentication uses `ssh-agent` by default; passwords are never stored.

## Workflow

1. **New job tab** — drop an xyz/pdb/mol structure, pick functional/basis/task,
   pick a cluster, hit *Build & submit*.
2. The builder writes `coord`, `define.inp`, and `submit.slurm` locally to
   `~/.turbomole_orchestrator/workspace/<jobname>_<ts>/`.
3. The directory is SFTP-uploaded; `sbatch submit.slurm` is run on the cluster.
4. The SLURM script runs `define < define.inp` first to build a proper `control`,
   then runs `ridft` / `jobex -ri` / `frog` according to the task.
5. **Job Manager tab** offers per-job actions:
   - *Partial* — pull current contents of remote dir into a timestamped subfolder.
     Safe on running jobs. Useful for checking convergence mid-flight.
   - *Download* — full pull on COMPLETED/FAILED. State becomes `DOWNLOADED`.
   - *DL + clean* — full pull then `rm -rf` of remote dir. State becomes `CLEANED`.
   - *Cancel* — `scancel`.
6. **No auto-refresh.** Click *Refresh status* to query SLURM on demand.

## Migrating to Turbomole 8.0

When TM 8.0 is installed on a cluster:

1. SSH into the cluster, run `define` interactively on a small case.
2. Diff its prompt sequence with the 7.8 sequence in `DefineDriver_7_8.build_define_input`.
3. Override the relevant methods in `DefineDriver_8_0` in `backend/turbomole_io.py`.
4. In `config.yaml`, change that cluster's `turbomole_version` to `"8.0"`.

Clusters with `"7.8"` keep using the 7.8 driver — no regression risk.

## Tests

```bash
pytest tests/ -v
```

Covers config loading, DB roundtrip, define.inp generation for both
versions, driver commands, and SLURM script rendering.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `paramiko.AuthenticationException` | ssh-agent not running | `eval $(ssh-agent) && ssh-add ~/.ssh/id_ed25519` |
| Job stuck in `DRAFT` | Upload failed (SSH error) | Check app.py terminal traceback |
| Job goes to `FAILED` instantly | `define` failed or bad SLURM script | `ssh user@cluster "cat <remote_dir>/define.out"` |
| `define` complains about format | Version mismatch | Confirm `turbomole_version` in `config.yaml` matches cluster's TM |
| `KeyError: 'callback not found'` | Browser cache after code change | `Ctrl+Shift+R` in browser |
| `_DB_PATH=None` errors | Should not happen (lazy init) | Run `pytest tests/test_db.py` to verify |

For clusters that load Turbomole via `~/.bashrc`, the SLURM script uses
`#!/bin/bash -l` so login configuration is sourced automatically.
