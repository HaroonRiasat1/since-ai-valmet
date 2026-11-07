#!/usr/bin/env python3
"""Run the object-extraction stages in order (1 view scene ... 7 export app objects)."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent
STAGES = [
    ("view_scene", "scripts/01_view_scene.py"),
    ("render_views", "scripts/02_render_views.py"),
    ("detect_segment", "scripts/03_detect_segment.py"),
    ("lift_masks", "scripts/04_lift_masks.py"),
    ("build_instances", "scripts/05_build_instances.py"),
    ("export_objects", "scripts/06_export_objects.py"),
    ("export_app_objects", "scripts/07_export_app_objects.py"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=_REPO / "config.yaml")
    ap.add_argument("--from-stage", type=int, default=1, help="first stage to run (1-7)")
    ap.add_argument("--to-stage", type=int, default=len(STAGES), help="last stage to run (1-7)")
    ap.add_argument("--force", action="store_true", help="re-render the view set instead of reusing it")
    args = ap.parse_args()

    env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1", "TOKENIZERS_PARALLELISM": "false"}
    timings = []
    t_all = time.time()
    for i, (name, rel) in enumerate(STAGES, start=1):
        if not args.from_stage <= i <= args.to_stage:
            continue
        cmd = [sys.executable, str(_REPO / rel), "--config", str(args.config)]
        if name == "render_views" and args.force:
            cmd.append("--force")
        print(f"\n=== [{i}/{len(STAGES)}] {name} ===", flush=True)
        t0 = time.time()
        subprocess.run(cmd, cwd=_REPO, env=env, check=True)
        timings.append((name, time.time() - t0))
    print("\n=== timings ===")
    for name, dt in timings:
        print(f"{name:<16} {dt:7.1f}s")
    print(f"{'total':<16} {time.time() - t_all:7.1f}s")


if __name__ == "__main__":
    main()
