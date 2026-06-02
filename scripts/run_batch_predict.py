"""
Batch autoregressive rollout: generate prediction videos for all dataset
videos and rank them by mean keypoint error.

Usage:
    python scripts/run_batch_predict.py --checkpoint data/checkpoints/best.pt

    # Only test set
    python scripts/run_batch_predict.py --checkpoint data/checkpoints/best.pt --split test

    # Skip video generation (just print rankings)
    python scripts/run_batch_predict.py --checkpoint data/checkpoints/best.pt --no-video
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from models.world_model import (
    PoseTransformer, StructuredPoseTransformer,
    enforce_bone_lengths, compute_reference_bone_lengths,
    resolve_hold_sequence_and_targets,
    extract_target_hands,
)
from evaluation.visualize import (
    lookup_route,
    render_pose_video,
    render_pose_video_with_targets,
    autoregressive_rollout,
    autoregressive_rollout_structured,
)
from pipeline.routes import holds_to_array
from scripts.run_visualize import load_gt_poses, load_model, hold_orders_applied_for


def compute_rollout_error(
    predicted: list[np.ndarray],
    gt_poses: list[np.ndarray],
    stride: int,
    context_frames: int,
) -> dict:
    """
    Compute per-frame keypoint errors between rollout and GT.

    Compares at stride-spaced GT indices starting after the seed.

    Args:
        predicted: Full rollout pose list (including seed + interpolated).
        gt_poses: Ground truth poses for the full video.
        stride: Rollout stride used.
        context_frames: Number of seed frames (context_len * stride).

    Returns:
        Dict with mean, median, max error and per-frame error array.
    """
    errors = []
    # Compare at stride-spaced points after seed
    for i, t in enumerate(range(context_frames, len(gt_poses), stride)):
        # Map GT index t to the predicted frame index
        pred_idx = context_frames + (i * stride)
        if pred_idx >= len(predicted):
            break
        pred = predicted[pred_idx]
        gt = gt_poses[t][config.CLIMBING_KEYPOINT_INDICES]
        err = np.linalg.norm(pred - gt, axis=1).mean()
        errors.append(err)

    if not errors:
        return {"mean": float("nan"), "median": float("nan"), "max": float("nan"), "errors": []}

    errors = np.array(errors)
    return {
        "mean": float(np.nanmean(errors)),
        "median": float(np.nanmedian(errors)),
        "max": float(np.nanmax(errors)),
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch rollout: predict + rank all dataset videos"
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--dataset", type=Path,
        default=config.DATA_DIR / "dataset.npz",
    )
    parser.add_argument(
        "--split", choices=["all", "train", "test"], default="all",
        help="Which split to evaluate",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Optional output dir. Defaults to data/viz/batch_<checkpoint_name>",
    )
    parser.add_argument(
        "--no-video", action="store_true",
        help="Skip video generation, just compute and rank errors",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Playback speed multiplier for output videos",
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = config.DATA_DIR / "viz" / f"batch_{args.checkpoint.stem}"

    device = config.get_device(args.device)

    # Load model
    model = load_model(args.checkpoint, device)
    is_structured = isinstance(model, StructuredPoseTransformer)
    apply_hold_orders = hold_orders_applied_for(args.checkpoint)

    # Load dataset to get stems and route holds
    raw = np.load(args.dataset, allow_pickle=True)
    train_stems = list(raw["train_stems"])
    test_stems = list(raw["test_stems"])
    train_route_holds = list(raw["train_route_holds"])
    test_route_holds = list(raw["test_route_holds"])

    # Build list of (stem, route_holds, split_label)
    videos = []
    if args.split in ("all", "train"):
        for stem, rh in zip(train_stems, train_route_holds):
            videos.append((stem, rh, "train"))
    if args.split in ("all", "test"):
        for stem, rh in zip(test_stems, test_route_holds):
            videos.append((stem, rh, "test"))

    print(f"Evaluating {len(videos)} videos ({args.split} split)...")
    print(f"Model: {'structured' if is_structured else 'direct'}")
    if is_structured:
        print(f"Hold orders: {'applied' if apply_hold_orders else 'ignored'}")
    print(f"Rollout stride: {config.ROLLOUT_STRIDE}\n")

    stride = config.ROLLOUT_STRIDE
    results = []

    for stem, route_holds, split_label in videos:
        print(f"  {stem} ({split_label})...", end=" ", flush=True)

        try:
            gt_poses, fps = load_gt_poses(stem)
        except FileNotFoundError:
            print("SKIP (no pose data)")
            continue

        n_seed = config.CONTEXT_WINDOW * stride
        if len(gt_poses) <= n_seed:
            print(f"SKIP (only {len(gt_poses)} frames, need >{n_seed})")
            continue

        hold_positions, hold_roles = holds_to_array(route_holds, normalize=True)
        seed = np.array(gt_poses[:n_seed])
        ref_bones = compute_reference_bone_lengths([np.array(gt_poses)])
        n_frames = len(gt_poses)

        # Rollout
        if is_structured:
            gt_filtered = np.array(gt_poses)[:, config.CLIMBING_KEYPOINT_INDICES, :]
            hold_seq, _ = resolve_hold_sequence_and_targets(
                gt_filtered, route_holds, stem if apply_hold_orders else None
            )
            if len(hold_seq) == 0:
                print("SKIP (no hold sequence)")
                continue
            poses, target_positions = autoregressive_rollout_structured(
                model, seed, n_frames, hold_positions, hold_roles,
                hold_seq, device, max_bone_lengths=ref_bones,
                gt_poses=np.array(gt_poses),
            )
            target_hands = extract_target_hands(hold_seq, target_positions)
        else:
            poses = autoregressive_rollout(
                model, seed, n_frames, hold_positions, hold_roles, device,
                max_bone_lengths=ref_bones,
            )

        # Compute error
        err = compute_rollout_error(poses, gt_poses, stride, n_seed)
        print(f"mean={err['mean']:.2f}, median={err['median']:.2f}, max={err['max']:.2f}")

        results.append({
            "stem": stem,
            "split": split_label,
            "mean_error": err["mean"],
            "median_error": err["median"],
            "max_error": err["max"],
            "n_frames": len(gt_poses),
            "poses": poses,
            "route_holds": route_holds,
            "fps": fps,
            "gt_poses_climbing": [p[config.CLIMBING_KEYPOINT_INDICES] for p in gt_poses],
            "target_positions": target_positions if is_structured else None,
            "target_hands": target_hands if is_structured else None,
        })

    if not results:
        print("No videos evaluated.")
        return

    # Rank by mean error
    results.sort(key=lambda r: r["mean_error"])

    print(f"\n{'='*60}")
    print(f"Rankings (best to worst by mean error):")
    print(f"{'='*60}")
    for i, r in enumerate(results, 1):
        marker = "*" if r["split"] == "test" else " "
        print(f"  {i:2d}. {marker}{r['stem']:30s} [{r['split']:5s}] "
              f"mean={r['mean_error']:6.2f}  median={r['median_error']:6.2f}  "
              f"max={r['max_error']:6.2f}  ({r['n_frames']} frames)")
    print(f"\n  * = test set")

    # Generate videos
    if not args.no_video:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nGenerating videos to {args.output_dir}...")

        for i, r in enumerate(results, 1):
            out_path = args.output_dir / f"{i:02d}_{r['stem']}_{r['split']}.mp4"
            title = f"#{i} {r['stem']} ({r['split']}) - mean err {r['mean_error']:.1f}"
            if r["target_positions"] is not None:
                render_pose_video_with_targets(
                    poses=r["poses"],
                    target_positions=r["target_positions"],
                    output_path=out_path,
                    route_holds=r["route_holds"],
                    fps=round(r["fps"] * args.speed),
                    title=title,
                    gt_poses=r["gt_poses_climbing"],
                    target_hands=r["target_hands"],
                )
            else:
                render_pose_video(
                    poses=r["poses"],
                    output_path=out_path,
                    route_holds=r["route_holds"],
                    fps=round(r["fps"] * args.speed),
                    title=title,
                    gt_poses=r["gt_poses_climbing"],
                )

    # Save summary JSON
    summary = [
        {
            "rank": i,
            "stem": r["stem"],
            "split": r["split"],
            "mean_error": r["mean_error"],
            "median_error": r["median_error"],
            "max_error": r["max_error"],
            "n_frames": r["n_frames"],
        }
        for i, r in enumerate(results, 1)
    ]
    summary_path = args.output_dir
    summary_path.mkdir(parents=True, exist_ok=True)
    with open(summary_path / "rankings.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Rankings saved to {summary_path / 'rankings.json'}")


if __name__ == "__main__":
    main()