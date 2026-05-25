"""
Manual sanity check for backend/config.py.

Run from the project root:
    python check_config.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from backend.config import load_config


def main() -> int:
    print("=" * 60)
    print("Turbomole Orchestrator — config check")
    print("=" * 60)

    try:
        cfg = load_config()
    except Exception as exc:                          # noqa: BLE001
        print(f"\n[FAIL] Could not load config: {exc}")
        return 1

    print(f"\nLocal paths")
    print(f"  db_path:        {cfg.db_path}")
    print(f"  local_workdir:  {cfg.local_workdir}")
    print(f"  download_dir:   {cfg.download_dir}")
    print(f"  poll_interval:  {cfg.poll_interval_s} s")

    if not cfg.clusters:
        print("\n[WARN] No clusters defined — load_config() returned the default placeholder.")
        return 1

    print(f"\nClusters ({len(cfg.clusters)} found)")
    print("-" * 60)
    for name, c in cfg.clusters.items():
        print(f"  [{name}]")
        print(f"    host             : {c.user}@{c.host}:{c.port}")
        print(f"    key_filename     : {c.key_filename or '(ssh-agent only)'}")
        print(f"    use_agent        : {c.use_agent}")
        print(f"    remote_workdir   : {c.remote_workdir}")
        print(f"    module_load      : {c.module_load}")
        print(f"    env_setup        : {len(c.env_setup)} line(s)")
        print(f"    turbomole_version: {c.turbomole_version}")
        print(f"    default_partition: {c.default_partition}")
        print(f"    default_time     : {c.default_time}")
        print(f"    default_ntasks   : {c.default_ntasks}")
        print(f"    default_mem      : {c.default_mem}")
        print()

    issues: list[str] = []
    for name, c in cfg.clusters.items():
        if not c.host or c.host == "hpc.example.org":
            issues.append(f"  - cluster '{name}': host looks like a placeholder")
        if not c.user:
            issues.append(f"  - cluster '{name}': user is empty")
        if c.key_filename and not Path(c.key_filename).expanduser().exists():
            issues.append(f"  - cluster '{name}': key_filename does not exist: {c.key_filename}")

    if issues:
        print("[WARN] Possible issues:")
        for line in issues:
            print(line)
        return 1

    print("[OK] Config looks valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
