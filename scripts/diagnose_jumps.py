"""
Find frames with large keypoint jumps in the dataset.

Reports the video, frame index, and displacement magnitude for outliers,
so you can inspect them in the overlay videos.

Usage:
    python scripts/diagnose_jumps.py
    python scripts/diagnose_jumps.py --threshold 20
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from pipeline.pose_cleaning import clean_pose_sequence
from pipeline.calibration import (
    load_calibration,
    compute_homographies,
    kick_threshold_px,
    transform_keypoints,
)


def analyze_video(pose_json_path: Path, video_stem: str, threshold: float) -> list[dict]:
    """Find frames with displacement above threshold in a single video."""
    with open(pose_json_path) as f:
        data = json.load(f)

    frames = clean_pose_sequence(data["frames"])

    # Transform to board space
    try:
        cal = load_calibration(video_stem)
        H_main, H_kick = compute_homographies(cal)
        thresh_px = kick_threshold_px(cal)
    except (FileNotFoundError, ValueError):
        return []

    # Extract valid keypoints in board space
    board_kps = []
    frame_indices = []
    for f in frames:
        if f["keypoints"] is not None:
            kp = np.array(f["keypoints"])
            kp_board = transform_keypoints(kp, H_main, H_kick, thresh_px)
            board_kps.append(kp_board)
            frame_indices.append(f["frame_idx"])

    if len(board_kps) < 2:
        return []

    # Compute per-frame displacements
    outliers = []
    for i in range(1, len(board_kps)):
        delta = board_kps[i] - board_kps[i - 1]
        per_kp_dist = np.linalg.norm(delta, axis=1)
        mean_dist = float(per_kp_dist.mean())
        max_dist = float(per_kp_dist.max())
        worst_kp = int(per_kp_dist.argmax())

        if mean_dist > threshold:
            outliers.append({
                "video": video_stem,
                "frame_idx": frame_indices[i],
                "prev_frame_idx": frame_indices[i - 1],
                "mean_displacement": mean_dist,
                "max_displacement": max_dist,
                "worst_keypoint": config.COCO_KEYPOINT_NAMES[worst_kp],
                "worst_kp_dist": float(per_kp_dist[worst_kp]),
                "filled": frames[frame_indices[i]].get("filled", False) if frame_indices[i] < len(frames) else False,
            })

    return outliers


def main():
    parser = argparse.ArgumentParser(description="Find large keypoint jumps")
    parser.add_argument(
        "--threshold", type=float, default=15.0,
        help="Mean displacement threshold in board units (default: 15)",
    )
    args = parser.parse_args()

    pose_jsons = sorted(
        p for p in config.POSES_DIR.rglob("*.json")
        if not p.stem.endswith("_overlay")
    )

    print(f"Scanning {len(pose_jsons)} videos for jumps > {args.threshold} board units...\n")

    all_outliers = []
    for pj in pose_jsons:
        stem = pj.stem
        outliers = analyze_video(pj, stem, args.threshold)
        if outliers:
            all_outliers.extend(outliers)

    if not all_outliers:
        print("No outliers found.")
        return

    # Sort by displacement
    all_outliers.sort(key=lambda x: x["mean_displacement"], reverse=True)

    # Summary by video
    videos = {}
    for o in all_outliers:
        videos.setdefault(o["video"], []).append(o)

    print(f"Found {len(all_outliers)} outlier frames across {len(videos)} videos:\n")

    for video, outliers in sorted(videos.items(), key=lambda x: -len(x[1])):
        worst = max(outliers, key=lambda x: x["mean_displacement"])
        print(f"  {video}: {len(outliers)} jumps, "
              f"worst={worst['mean_displacement']:.1f} at frame {worst['frame_idx']} "
              f"({worst['worst_keypoint']} moved {worst['worst_kp_dist']:.1f})")
        if len(outliers) <= 5:
            for o in outliers:
                filled = " [interpolated]" if o["filled"] else ""
                print(f"    frame {o['prev_frame_idx']}->{o['frame_idx']}: "
                      f"mean={o['mean_displacement']:.1f}, "
                      f"max={o['max_displacement']:.1f} ({o['worst_keypoint']})"
                      f"{filled}")


if __name__ == "__main__":
    main()