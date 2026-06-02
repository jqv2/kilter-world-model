"""Launch parallel per-route RL training jobs.

Reads a text file of route names (one per line) and spawns an
independent ``train_rl.py`` process for each route, with isolated
output directories namespaced by route name.

Usage::

    python -m scripts.train_rl_parallel --routes data/rl_eval_routes.txt
    python -m scripts.train_rl_parallel --routes data/rl_eval_routes.txt --total-frames 500000 --device cuda
    python -m scripts.train_rl_parallel --routes data/rl_eval_routes.txt --resume
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import config


def _parse_route_names(path: Path) -> list[str]:
    """Read non-empty, non-comment lines from a route file."""
    names = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line)
    return names


def main():
    parser = argparse.ArgumentParser(
        description="Launch parallel per-route RL training jobs",
    )
    parser.add_argument(
        "--routes", required=True,
        help="Text file with one climb name per line",
    )
    parser.add_argument("--dataset", default=str(config.DATA_DIR / "dataset.npz"))
    parser.add_argument("--total-frames", type=int, default=config.RL_TOTAL_FRAMES)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume each route from its latest checkpoint (skips routes with no checkpoint)",
    )
    args = parser.parse_args()

    route_names = _parse_route_names(Path(args.routes))
    if not route_names:
        print("No routes found in", args.routes)
        sys.exit(1)

    print(f"Launching {len(route_names)} independent training jobs:")
    for name in route_names:
        print(f"  - {name}")

    # Write per-route temp files and build commands
    tmp_files = []
    procs = []
    for name in route_names:
        tf = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix=f"rl_route_{name}_", delete=False,
        )
        tf.write(name + "\n")
        tf.close()
        tmp_files.append(tf.name)

        # Sanitize route name for use as directory name
        run_name = name.replace(" ", "_").replace("/", "_")

        cmd = [
            sys.executable, "-m", "scripts.train_rl",
            "--dataset", args.dataset,
            "--total-frames", str(args.total_frames),
            "--seed", str(args.seed),
            "--train-routes", tf.name,
            "--ref-routes", tf.name,
            "--run-name", run_name,
            "--device", args.device or "cpu",
        ]

        if args.resume:
            ckpt_dir = config.RL_CHECKPOINT_DIR / run_name
            if ckpt_dir.exists():
                # Pick the most recently written checkpoint (step, milestone, or final)
                all_ckpts = list(ckpt_dir.glob("*.pt"))
                if all_ckpts:
                    ckpt_path = max(all_ckpts, key=lambda p: p.stat().st_mtime)
                    print(f"  Resuming {run_name} from {ckpt_path.name}")
                    cmd += ["--resume", str(ckpt_path)]
                else:
                    print(f"  No checkpoint found for {run_name}, starting fresh")
            else:
                print(f"  No checkpoint dir for {run_name}, starting fresh")

        if args.dry_run:
            print(f"  [dry-run] {' '.join(cmd)}")
        else:
            print(f"  Starting: {run_name}")
            procs.append((run_name, subprocess.Popen(cmd)))

    if args.dry_run:
        return

    # Wait for all to finish
    for run_name, proc in procs:
        rc = proc.wait()
        status = "OK" if rc == 0 else f"FAILED (exit {rc})"
        print(f"  {run_name}: {status}")

    # Clean up temp files
    for tf_path in tmp_files:
        Path(tf_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()