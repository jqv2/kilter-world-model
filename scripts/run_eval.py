"""
Unified evaluation: compare baselines and trained models side-by-side.

Evaluates on the test split of the built dataset and prints a comparison
table with teacher-forcing and autoregressive metrics for each method.

Usage:
    # Static baseline only
    python scripts/run_eval.py

    # Add greedy IK baseline
    python scripts/run_eval.py --greedy

    # Add one or more model checkpoints
    python scripts/run_eval.py --greedy --checkpoints data/checkpoints/best_structured.pt data/checkpoints/best_unstructured.pt

    # Save results to JSON
    python scripts/run_eval.py --greedy --checkpoints data/checkpoints/best.pt --output eval_results.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from evaluation.metrics import (
    per_frame_errors,
    summarize_teacher_forcing,
    summarize_autoregressive,
)
from pipeline.dataset import load_dataset
from pipeline.routes import pad_holds, normalize_board_coords
from evaluation.baselines import greedy_ik_baseline_predictions
from models.world_model import (
    PoseTransformer, StructuredPoseTransformer,
    compute_reference_bone_lengths, enforce_bone_lengths,
)
from scripts.run_visualize import load_model
from scripts.train import (
    evaluate_teacher_forcing,
    evaluate_autoregressive,
    evaluate_teacher_forcing_structured,
    evaluate_autoregressive_structured,
)


def static_baseline_predictions(gt_frames):
    """Predict the first frame's pose for all subsequent frames."""
    first = gt_frames[0]
    return [first.copy() for _ in range(len(gt_frames) - 1)]


def eval_baseline(name, predict_fn, sequences, scores, stems, route_holds=None):
    """
    Evaluate a baseline that produces predictions from GT frames.

    Args:
        name: Display name for this method.
        predict_fn: Callable(gt_frames, route_holds?) -> list of predicted poses.
            If route_holds is needed, it must accept it as a second arg.
        sequences: Test set pose sequences.
        scores: Test set confidence scores.
        stems: Test set video stems.
        route_holds: Test set route holds (needed for greedy IK).

    Returns:
        Dict with per-video and aggregate metrics.
    """
    all_errors = []
    per_video = []

    for i, (seq, sc, stem) in enumerate(zip(sequences, scores, stems)):
        gt_frames = list(seq)
        if len(gt_frames) < 2:
            continue

        if route_holds is not None:
            preds, _ = predict_fn(gt_frames, route_holds[i], verbose=False)
        else:
            preds = predict_fn(gt_frames)

        errors = per_frame_errors(
            preds, gt_frames, list(sc), config.KEYPOINT_CONFIDENCE_THRESHOLD
        )
        all_errors.append(errors)

        tf = summarize_teacher_forcing(errors)
        ar = summarize_autoregressive(errors)
        per_video.append({
            "stem": stem,
            "n_frames": len(gt_frames),
            "tf_mean": tf["mean"],
            "tf_median": tf["median"],
            "ar_p50": ar["p50"],
            "ar_p100": ar["p100"],
        })

    if not all_errors:
        return None

    combined = np.concatenate(all_errors)
    agg_tf = summarize_teacher_forcing(combined)
    agg_ar = summarize_autoregressive(combined)

    return {
        "name": name,
        "tf_mean": agg_tf["mean"],
        "tf_median": agg_tf["median"],
        "ar_p25": agg_ar["p25"],
        "ar_p50": agg_ar["p50"],
        "ar_p75": agg_ar["p75"],
        "ar_p100": agg_ar["p100"],
        "n_videos": len(all_errors),
        "n_frames": agg_tf["n_valid"],
        "per_video": per_video,
    }


def eval_model_checkpoint(
    checkpoint_path, data, device, ref_bones,
):
    """
    Evaluate a trained model checkpoint under teacher forcing and
    autoregressive rollout.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        data: Dataset dict from load_dataset.
        device: Torch device.
        ref_bones: Reference bone lengths from training set.

    Returns:
        Dict with aggregate metrics, or None on failure.
    """
    model = load_model(checkpoint_path, device)
    is_structured = isinstance(model, StructuredPoseTransformer)
    label = f"{checkpoint_path.stem} ({'structured' if is_structured else 'direct'})"

    if is_structured:
        tf = evaluate_teacher_forcing_structured(
            model,
            data["test_sequences"], data["test_scores"],
            data["test_holds"], data["test_roles"],
            data["test_route_holds"],
            device,
        )
        ar = evaluate_autoregressive_structured(
            model,
            data["test_sequences"], data["test_scores"],
            data["test_holds"], data["test_roles"],
            data["test_route_holds"],
            device,
            max_bone_lengths=ref_bones,
        )
    else:
        tf = evaluate_teacher_forcing(
            model,
            data["test_sequences"], data["test_scores"],
            data["test_holds"], data["test_roles"],
            device,
        )
        ar = evaluate_autoregressive(
            model,
            data["test_sequences"], data["test_scores"],
            data["test_holds"], data["test_roles"],
            device,
            max_bone_lengths=ref_bones,
        )

    return {
        "name": label,
        "checkpoint": str(checkpoint_path),
        "tf_mean": tf["mean"],
        "tf_median": tf["median"],
        "ar_p25": ar["p25"],
        "ar_p50": ar["p50"],
        "ar_p75": ar["p75"],
        "ar_p100": ar["p100"],
        "n_videos": len(data["test_sequences"]),
    }


def print_comparison_table(results):
    """Print a formatted comparison table of all evaluated methods."""
    print(f"\n{'='*90}")
    print(f"{'Method':<35} {'TF Mean':>8} {'TF Med':>8} {'AR p25':>8} {'AR p50':>8} {'AR p75':>8} {'AR p100':>8}")
    print(f"{'-'*90}")
    for r in results:
        print(
            f"{r['name']:<35} "
            f"{r['tf_mean']:>8.2f} {r['tf_median']:>8.2f} "
            f"{r['ar_p25']:>8.2f} {r['ar_p50']:>8.2f} "
            f"{r['ar_p75']:>8.2f} {r['ar_p100']:>8.2f}"
        )
    print(f"{'='*90}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluation: compare baselines and models"
    )
    parser.add_argument(
        "--dataset", type=Path, default=config.DATA_DIR / "dataset.npz",
    )
    parser.add_argument(
        "--greedy", action="store_true",
        help="Include greedy IK baseline",
    )
    parser.add_argument(
        "--checkpoints", type=Path, nargs="+", default=[],
        help="One or more .pt checkpoint paths to evaluate",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Save results to JSON file",
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = config.get_device(args.device)

    data = load_dataset(args.dataset)
    sequences = data["test_sequences"]
    scores = data["test_scores"]
    stems = data["test_stems"]
    route_holds = data["test_route_holds"]

    print(f"Test set: {len(sequences)} videos")
    results = []

    # --- Static baseline ---
    print("\nEvaluating static baseline...")
    static = eval_baseline(
        "Static (repeat frame 0)", static_baseline_predictions,
        sequences, scores, stems,
    )
    if static:
        results.append(static)

    # --- Greedy IK baseline ---
    if args.greedy:
        print("Evaluating greedy IK baseline...")
        greedy = eval_baseline(
            "Greedy IK", greedy_ik_baseline_predictions,
            sequences, scores, stems, route_holds,
        )
        if greedy:
            results.append(greedy)

    # --- Model checkpoints ---
    if args.checkpoints:
        ref_bones = compute_reference_bone_lengths(data["train_sequences"])
        for ckpt in args.checkpoints:
            print(f"Evaluating checkpoint: {ckpt.name}...")
            model_result = eval_model_checkpoint(ckpt, data, device, ref_bones)
            if model_result:
                results.append(model_result)

    print_comparison_table(results)

    if args.output:
        # Strip non-serializable fields
        serializable = []
        for r in results:
            entry = {k: v for k, v in r.items() if k != "per_video"}
            serializable.append(entry)
        with open(args.output, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()