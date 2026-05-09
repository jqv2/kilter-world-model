"""
Evaluate the static pose baseline on extracted pose data.

Very naive static baseline: predict that every frame's pose
equals the first detected frame's pose (i.e. the climber doesn't move).

Usage:
    python scripts/run_eval_baselines.py
    python scripts/run_eval_baselines.py data/poses/climb_001.json
    python scripts/run_eval_baselines.py --all

Prints per-video and aggregate metrics.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from evaluation.metrics import (
    per_frame_errors,
    summarize_teacher_forcing,
    summarize_autoregressive
)


def load_pose_sequence(json_path: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Load ground truth poses from a pose extraction JSON file.

    Filters to frames where a person was detected.

    Args:
        json_path: Path to the pose JSON file.

    Returns:
        Tuple of (keypoints_list, scores_list), each a list of arrays.
        keypoints_list[i] has shape (17, 2), scores_list[i] has shape (17,).
    """
    with open(json_path) as f:
        data = json.load(f)

    keypoints_list = []
    scores_list = []
    for frame in data["frames"]:
        if frame["keypoints"] is not None:
            keypoints_list.append(np.array(frame["keypoints"]))
            scores_list.append(np.array(frame["scores"]))

    return keypoints_list, scores_list


def static_baseline_predictions(
    gt_frames: list[np.ndarray],
) -> list[np.ndarray]:
    """
    Static baseline: predict the first frame's pose for all subsequent frames.

    Args:
        gt_frames: List of N ground truth poses, each (17, 2).

    Returns:
        List of N-1 predicted poses (all identical to gt_frames[0]).
    """
    first_pose = gt_frames[0]
    return [first_pose.copy() for _ in range(len(gt_frames) - 1)]


def evaluate_video(json_path: Path) -> dict | None:
    """
    Run static baseline evaluation on a single video's pose data.

    Returns:
        Dict with teacher_forcing and autoregressive summary stats,
        or None if insufficient data.
    """
    keypoints, scores = load_pose_sequence(json_path)
    if len(keypoints) < 2:
        print(f"  Skipping {json_path.name}: fewer than 2 detected frames")
        return None

    predictions = static_baseline_predictions(keypoints)

    # For the static baseline, teacher forcing and autoregressive are identical:
    # the prediction is always the first frame regardless of input.
    tf_errors = per_frame_errors(
        predictions, keypoints, scores, config.KEYPOINT_CONFIDENCE_THRESHOLD
    )
    ar_errors = per_frame_errors(
        predictions, keypoints, scores, config.KEYPOINT_CONFIDENCE_THRESHOLD
    )

    return {
        "video": json_path.stem,
        "n_frames": len(keypoints),
        "teacher_forcing": summarize_teacher_forcing(tf_errors),
        "autoregressive": summarize_autoregressive(ar_errors),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate static pose baseline on pose extraction outputs"
    )
    parser.add_argument(
        "json_file", type=Path, nargs="?", default=None,
        help="Path to a single pose JSON file",
    )
    parser.add_argument(
        "--all", action="store_true",
        help=f"Evaluate all pose JSONs in {config.POSES_DIR}",
    )
    args = parser.parse_args()

    if args.json_file and args.all:
        parser.error("Provide either a JSON path or --all, not both.")
    if not args.json_file and not args.all:
        parser.error("Provide a JSON path or use --all.")

    if args.all:
        json_files = sorted(config.POSES_DIR.rglob("*.json"))
        if not json_files:
            print(f"No pose JSONs found in {config.POSES_DIR}")
            return
    else:
        if not args.json_file.exists():
            print(f"Error: File not found: {args.json_file}")
            sys.exit(1)
        json_files = [args.json_file]

    print(f"=== Static Pose Baseline ===")
    print(f"Baseline: predict first frame's pose for all subsequent frames\n")

    for json_path in json_files:
        result = evaluate_video(json_path)
        if result is None:
            continue

        tf = result["teacher_forcing"]
        ar = result["autoregressive"]
        print(f"{result['video']} ({result['n_frames']} frames):")
        print(f"  Teacher forcing >>> mean: {tf['mean']:.2f}, median: {tf['median']:.2f}, max: {tf['max']:.2f}")
        print(f"  Autoregressive  >>> error at 25%: {ar['p25']:.2f}, 50%: {ar['p50']:.2f}, 75%: {ar['p75']:.2f}, 100%: {ar['p100']:.2f}")

    if len(json_files) > 1:
        print(f"\n--- Aggregate ({len(json_files)} videos) ---")
        # Re-run to collect raw errors for aggregate stats
        all_errors = []
        for json_path in json_files:
            keypoints, scores = load_pose_sequence(json_path)
            if len(keypoints) < 2:
                continue
            preds = static_baseline_predictions(keypoints)
            tf = per_frame_errors(preds, keypoints, scores, config.KEYPOINT_CONFIDENCE_THRESHOLD)
            all_errors.append(tf)
        if all_errors:
            combined_tf = np.concatenate(all_errors)
            agg_tf = summarize_teacher_forcing(combined_tf)
            print(f"  Teacher forcing >>> mean: {agg_tf['mean']:.2f}, median: {agg_tf['median']:.2f}")
            print(f"  Total frames evaluated: {agg_tf['n_valid']}")


if __name__ == "__main__":
    main()