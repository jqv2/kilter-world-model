"""
Build training dataset from pose extractions + calibrations + route definitions.

Runs the full pipeline: clean poses -> transform to board space -> look up routes -> split.

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --train-fraction 0.85
    python scripts/build_dataset.py --output data/dataset.npz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from pipeline.dataset import build_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Build training dataset from poses + calibrations + routes"
    )
    parser.add_argument(
        "--output", type=Path,
        default=config.DATA_DIR / "dataset.npz",
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Building dataset...")
    data = build_dataset(
        train_fraction=args.train_fraction,
        seed=args.seed,
    )

    meta = data["metadata"]
    print(f"\nDataset built:")
    print(f"  Train: {meta['n_train_videos']} videos, {meta['total_train_frames']} frames")
    print(f"  Test:  {meta['n_test_videos']} videos, {meta['total_test_frames']} frames")
    print(f"  Skipped: {meta['n_skipped']} (insufficient data)")
    if meta["skipped_stems"]:
        print(f"    Skipped: {meta['skipped_stems']}")
    print(f"  No route found: {meta['n_no_route']}")
    if meta["no_route_stems"]:
        print(f"    Missing: {meta['no_route_stems']}")
    print(f"  Train videos: {data['train_stems']}")
    print(f"  Test videos:  {data['test_stems']}")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        train_sequences=np.array(data["train_sequences"], dtype=object),
        train_scores=np.array(data["train_scores"], dtype=object),
        train_holds=np.array(data["train_holds"], dtype=object),
        train_roles=np.array(data["train_roles"], dtype=object),
        train_route_holds=np.array(data["train_route_holds"], dtype=object),
        test_route_holds=np.array(data["test_route_holds"], dtype=object),
        test_sequences=np.array(data["test_sequences"], dtype=object),
        test_scores=np.array(data["test_scores"], dtype=object),
        test_holds=np.array(data["test_holds"], dtype=object),
        test_roles=np.array(data["test_roles"], dtype=object),
        train_stems=np.array(data["train_stems"]),
        test_stems=np.array(data["test_stems"]),
        fps=np.array(data["fps"]),
        metadata=np.array(json.dumps(meta)),
    )
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()