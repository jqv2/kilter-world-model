"""
Batch evaluation for writeup: aggregate metrics across test set.

Generates:
    1. Teacher-forcing vs autoregressive error table
    2. Full test set table (world model + hands-only, 18 climbs)
    3. RL subset table (all 3 methods, configurable climbs)
    4. Per-climb NN pose plausibility bar chart (RL subset)

Usage:
    # Full test set (world model + hands-only only):
    python scripts/run_writeup_eval.py \
        --checkpoint data/checkpoints/best.pt \
        --output-dir data/writeup_eval

    # Include RL for specific stems (provide stem:checkpoint pairs):
    python scripts/run_writeup_eval.py \
        --checkpoint data/checkpoints/best.pt \
        --rl-evals stem1:path/to/rl1.pt stem2:path/to/rl2.pt \
        --output-dir data/writeup_eval

    # Or provide RL .npz files instead of checkpoints:
    python scripts/run_writeup_eval.py \
        --checkpoint data/checkpoints/best.pt \
        --rl-npz-evals stem1:path/to/rl1.npz stem2:path/to/rl2.npz \
        --output-dir data/writeup_eval
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from pipeline.dataset import load_dataset, load_climb_log, resolve_route
from pipeline.routes import holds_to_array, apply_route_edits
from evaluation.metrics import (
    mean_centroid_distance,
    path_aligned_keypoint_error,
    nearest_neighbor_pose_distance,
    build_pose_bank,
    summarize_nn_distances,
)
from evaluation.baselines import hanging_baseline_predictions, compute_rl_bone_lengths
from models.world_model import (
    StructuredPoseTransformer,
    compute_reference_bone_lengths,
    resolve_hold_sequence_and_targets,
    extract_target_hands,
    check_hand_arrival,
)
from evaluation.visualize import (
    autoregressive_rollout,
    autoregressive_rollout_structured,
)
from scripts.run_visualize import load_gt_poses, load_model, hold_orders_applied_for
from scripts.run_comparison import load_rl_poses, rollout_rl_deterministic


# ---------------------------------------------------------------------------
# HVR helpers
# ---------------------------------------------------------------------------

def compute_wm_hvr(
    poses: list[np.ndarray],
    targets: np.ndarray,
    hold_seq: list[dict],
    threshold: float = config.ROLLOUT_ARRIVAL_THRESHOLD_HAND,
) -> dict:
    """
    Compute hold visit rate for the world model from its rollout output.

    Counts target transitions where the predicted wrist was within
    threshold of the old target at the transition frame (i.e. actual
    arrivals, not timeout skips).

    Args:
        poses: List of (12, 2) predicted poses.
        targets: (T, 2) per-frame target positions from the rollout.
        hold_seq: Ordered list of hold dicts (the full sequence).
        threshold: Distance threshold for counting an arrival.

    Returns:
        Dict with 'arrivals', 'timeouts', 'total', 'hvr'.
    """
    if targets is None or len(targets) < 2:
        return {"arrivals": 0, "timeouts": 0, "total": len(hold_seq), "hvr": 0.0}

    arrivals = 0
    timeouts = 0

    for t in range(1, min(len(targets), len(poses))):
        # Detect target transition
        if not np.allclose(targets[t], targets[t - 1], atol=1e-3):
            # Check if wrist was near the OLD target at the previous frame
            old_target = targets[t - 1]
            pose = poses[t - 1]
            near = check_hand_arrival(pose, old_target, threshold=threshold)
            if near is not None:
                arrivals += 1
            else:
                timeouts += 1

    total = len(hold_seq)
    return {
        "arrivals": arrivals,
        "timeouts": timeouts,
        "total": total,
        "hvr": arrivals / total if total > 0 else 0.0,
    }


def compute_rl_hvr_from_info(info: dict) -> float:
    """Extract HVR from RL rollout info dict."""
    visited = info.get("holds_visited", 0)
    total = info.get("total_holds", 1)
    return visited / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-climb evaluation
# ---------------------------------------------------------------------------

def evaluate_single_climb(
    video_stem: str,
    climb_name: str,
    wm_checkpoint: Path,
    dataset_data: dict,
    bank: np.ndarray,
    bank_norm: np.ndarray,
    device: torch.device,
    rl_npz_path: Path | None = None,
    rl_checkpoint_path: Path | None = None,
    dataset_path: Path = config.DATA_DIR / "dataset.npz",
) -> dict:
    """
    Run all methods on a single climb and return per-method metrics.

    Returns dict with keys per method, each containing:
        traj, path_procrustes, nn_procrustes_mean, hvr
    """
    print(f"\n--- {video_stem} ({climb_name}) ---")

    # Load GT
    gt_poses_17kp, fps = load_gt_poses(video_stem)
    idx = config.CLIMBING_KEYPOINT_INDICES
    gt_climbing = [p[idx] for p in gt_poses_17kp]

    # Route holds (with edits matching checkpoint)
    from evaluation.visualize import lookup_route
    route = lookup_route(climb_name)
    route_holds = route["holds"]

    ckpt = torch.load(wm_checkpoint, map_location="cpu", weights_only=False)
    ds_meta = ckpt.get("dataset_metadata", {})
    if ds_meta.get("route_edits_applied", True):
        route_holds = apply_route_edits(route_holds, video_stem)

    bl = compute_rl_bone_lengths(dataset_data["train_sequences"])

    # Derive hold sequence
    apply_ho = hold_orders_applied_for(wm_checkpoint)
    gt_filtered = np.array(gt_poses_17kp)[:, idx, :]
    hold_seq, targets_gt = resolve_hold_sequence_and_targets(
        gt_filtered, route_holds, video_stem if apply_ho else None,
    )
    total_holds = len(hold_seq)

    # --- Hands-only baseline ---
    print("  Hands-only baseline...")
    hanging_preds, hang_tgt_pos, initial_pose, hang_tgt_hands = hanging_baseline_predictions(
        gt_poses_17kp, route_holds, video_stem, bone_lengths=bl,
    )
    hanging_seq = [initial_pose] + hanging_preds

    # --- World model ---
    print("  World model rollout...")
    model = load_model(wm_checkpoint, device)
    hold_positions, hold_roles = holds_to_array(route_holds, normalize=True)
    stride = config.ROLLOUT_STRIDE
    seed = np.array(gt_poses_17kp[:config.CONTEXT_WINDOW * stride])
    ref_bones = compute_reference_bone_lengths([np.array(gt_poses_17kp)])
    n_frames = len(gt_poses_17kp)

    is_structured = isinstance(model, StructuredPoseTransformer)
    wm_targets = None
    if is_structured:
        wm_poses, wm_targets = autoregressive_rollout_structured(
            model, seed, n_frames, hold_positions, hold_roles,
            hold_seq, device, max_bone_lengths=ref_bones,
            gt_poses=np.array(gt_poses_17kp),
        )
    else:
        wm_poses = autoregressive_rollout(
            model, seed, n_frames, hold_positions, hold_roles,
            device, max_bone_lengths=ref_bones,
        )

    # --- Compute metrics ---
    results = {}

    # Hands-only
    traj_ho = mean_centroid_distance(hanging_seq, gt_climbing)
    path_ho = path_aligned_keypoint_error(hanging_seq, gt_climbing, align="procrustes")
    nn_ho = summarize_nn_distances(nearest_neighbor_pose_distance(hanging_seq, bank, bank_norm))
    results["Hands-Only"] = {
        "traj": traj_ho,
        "path_procrustes": path_ho,
        "nn_procrustes_mean": nn_ho["procrustes"]["mean"],
        "hvr": 1.0,  # hands-only always visits all holds by construction
        "holds_visited": total_holds,
        "total_holds": total_holds,
    }

    # World model
    traj_wm = mean_centroid_distance(wm_poses, gt_climbing)
    path_wm = path_aligned_keypoint_error(wm_poses, gt_climbing, align="procrustes")
    nn_wm = summarize_nn_distances(nearest_neighbor_pose_distance(wm_poses, bank, bank_norm))
    wm_hvr_info = compute_wm_hvr(wm_poses, wm_targets, hold_seq)
    results["World Model"] = {
        "traj": traj_wm,
        "path_procrustes": path_wm,
        "nn_procrustes_mean": nn_wm["procrustes"]["mean"],
        "hvr": wm_hvr_info["hvr"],
        "holds_visited": wm_hvr_info["arrivals"],
        "total_holds": total_holds,
    }

    # Print per-climb summary
    for label, m in results.items():
        print(f"  {label:15s}  traj={m['traj']:.3f}  pose={m['path_procrustes']:.2f}"
              f"  nn={m['nn_procrustes_mean']:.4f}  hvr={m['hvr']:.2f}")

    # --- RL (optional) ---
    if rl_npz_path is not None:
        print(f"  Loading RL from {rl_npz_path}...")
        rl_poses = load_rl_poses(rl_npz_path)
        # No info dict from npz, count HVR as N/A or estimate from poses
        nn_rl = summarize_nn_distances(nearest_neighbor_pose_distance(rl_poses, bank, bank_norm))
        results["RL"] = {
            "traj": None,  # typically N/A for stuck agents
            "path_procrustes": None,
            "nn_procrustes_mean": nn_rl["procrustes"]["mean"],
            "hvr": None,  # fill in manually or from separate info
            "holds_visited": None,
            "total_holds": total_holds,
        }
        print(f"  {'RL':15s}  nn={nn_rl['procrustes']['mean']:.4f}  (traj/pose/hvr from rollout info)")

    elif rl_checkpoint_path is not None:
        print(f"  RL deterministic rollout from {rl_checkpoint_path}...")
        rl_poses, rl_targets, rl_target_hands, rl_info = rollout_rl_deterministic(
            rl_checkpoint_path, video_stem, dataset_path, device,
        )
        nn_rl = summarize_nn_distances(nearest_neighbor_pose_distance(rl_poses, bank, bank_norm))

        # Try to compute trajectory metrics (may be meaningless if stuck)
        traj_rl = mean_centroid_distance(rl_poses, gt_climbing)
        path_rl = path_aligned_keypoint_error(rl_poses, gt_climbing, align="procrustes")

        rl_visited = rl_info.get("holds_visited", 0)
        rl_total = rl_info.get("total_holds", total_holds)
        rl_hvr = rl_visited / rl_total if rl_total > 0 else 0.0

        results["RL"] = {
            "traj": traj_rl,
            "path_procrustes": path_rl,
            "nn_procrustes_mean": nn_rl["procrustes"]["mean"],
            "hvr": rl_hvr,
            "holds_visited": rl_visited,
            "total_holds": rl_total,
        }
        print(f"  {'RL':15s}  traj={traj_rl:.3f}  pose={path_rl:.2f}"
              f"  nn={nn_rl['procrustes']['mean']:.4f}  hvr={rl_hvr:.2f}"
              f" ({rl_visited}/{rl_total})")

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_metrics(per_climb: dict[str, dict], methods: list[str]) -> dict:
    """
    Aggregate per-climb metrics into mean +/- std.

    Args:
        per_climb: Dict mapping stem -> method -> metric dict.
        methods: List of method names to aggregate.

    Returns:
        Dict mapping method -> metric -> {"mean": ..., "std": ...}
    """
    agg = {}
    for method in methods:
        vals = {"traj": [], "path_procrustes": [], "nn_procrustes_mean": [], "hvr": []}
        for stem, climb_results in per_climb.items():
            if method not in climb_results:
                continue
            m = climb_results[method]
            for k in vals:
                if m.get(k) is not None:
                    vals[k].append(m[k])

        agg[method] = {}
        for k, v in vals.items():
            if v:
                agg[method][k] = {"mean": np.mean(v), "std": np.std(v), "n": len(v)}
            else:
                agg[method][k] = {"mean": None, "std": None, "n": 0}
    return agg


def aggregate_metrics_robust(per_climb: dict[str, dict], methods: list[str]) -> dict:
    """
    Aggregate per-climb metrics using median and IQR (robust to outliers).
    """
    agg = {}
    for method in methods:
        vals = {"traj": [], "path_procrustes": [], "nn_procrustes_mean": [], "hvr": []}
        for stem, climb_results in per_climb.items():
            if method not in climb_results:
                continue
            m = climb_results[method]
            for k in vals:
                if m.get(k) is not None:
                    vals[k].append(m[k])

        agg[method] = {}
        for k, v in vals.items():
            if v:
                arr = np.array(v)
                agg[method][k] = {
                    "median": float(np.median(arr)),
                    "q1": float(np.percentile(arr, 25)),
                    "q3": float(np.percentile(arr, 75)),
                    "mean": float(np.mean(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "n": len(v),
                }
            else:
                agg[method][k] = None
    return agg


def print_robust_table(agg: dict, title: str, methods: list[str]) -> None:
    """Print aggregate table with median [Q1, Q3]."""
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")
    header = (f"{'Method':<15} {'Trajectory':>22} {'Pose Correct':>22}"
              f" {'NN Plausibility':>22} {'HVR':>22}")
    print(header)
    print("-" * 100)
    for method in methods:
        m = agg[method]
        parts = []
        for k in ["traj", "path_procrustes", "nn_procrustes_mean", "hvr"]:
            info = m.get(k)
            if info is not None:
                parts.append(f"{info['median']:.3f} [{info['q1']:.3f}, {info['q3']:.3f}]")
            else:
                parts.append("N/A")
        print(f"{method:<15} {parts[0]:>22} {parts[1]:>22} {parts[2]:>22} {parts[3]:>22}")
    print("=" * 100)


def print_aggregate_table(agg: dict, title: str, methods: list[str]) -> None:
    """Print a formatted aggregate table."""
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")
    header = (f"{'Method':<15} {'Trajectory':>18} {'Pose Correct':>18}"
              f" {'NN Plausibility':>18} {'HVR':>18}")
    print(header)
    print("-" * 90)
    for method in methods:
        m = agg[method]
        parts = []
        for k in ["traj", "path_procrustes", "nn_procrustes_mean", "hvr"]:
            info = m[k]
            if info["mean"] is not None:
                parts.append(f"{info['mean']:.3f}+/-{info['std']:.3f}")
            else:
                parts.append("N/A")
        print(f"{method:<15} {parts[0]:>18} {parts[1]:>18} {parts[2]:>18} {parts[3]:>18}")
    print("=" * 90)


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------

def plot_nn_bar_chart(
    per_climb: dict[str, dict],
    methods: list[str],
    climb_names: dict[str, str],
    output_path: Path,
) -> None:
    """
    Generate a grouped bar chart of NN pose plausibility per climb.

    Args:
        per_climb: Dict mapping stem -> method -> metric dict.
        methods: List of method names to include.
        climb_names: Dict mapping stem -> human-readable climb name.
        output_path: Path to save the chart.
    """
    stems = [s for s in per_climb if all(m in per_climb[s] for m in methods)]
    if not stems:
        print("No climbs with all methods present; skipping bar chart.")
        return

    n_climbs = len(stems)
    n_methods = len(methods)
    x = np.arange(n_climbs)
    width = 0.8 / n_methods
    colors = ["#E8913A", "#4A90D9", "#50B050"]  # hands-only, world model, RL

    fig, ax = plt.subplots(figsize=(max(8, n_climbs * 1.5), 5))

    for i, method in enumerate(methods):
        vals = [per_climb[s][method]["nn_procrustes_mean"] for s in stems]
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=method,
               color=colors[i] if i < len(colors) else None)

    labels = [climb_names.get(s, s) for s in stems]
    # Truncate long names
    labels = [l[:20] + "..." if len(l) > 23 else l for l in labels]

    ax.set_xlabel("Climb")
    ax.set_ylabel("NN Pose Distance (lower = more plausible)")
    ax.set_title("Per-Climb Pose Plausibility")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Bar chart saved to {output_path}")
    
    
def plot_metric_distributions(
    per_climb: dict[str, dict],
    method: str,
    climb_names: dict[str, str],
    output_path: Path,
) -> None:
    """
    Strip plot of all four metrics for one method across climbs.

    Useful for spotting bimodal behavior in the world model.

    Args:
        per_climb: Dict mapping stem -> method -> metric dict.
        method: Which method to plot (e.g. "World Model").
        climb_names: Dict mapping stem -> human-readable name.
        output_path: Path to save the figure.
    """
    stems = [s for s in per_climb if method in per_climb[s]]
    if not stems:
        print(f"No climbs with {method}; skipping distribution plot.")
        return

    metrics = [
        ("traj", "Trajectory Error\n(frac of path length)"),
        ("path_procrustes", "Pose Correctness Error\n(board units)"),
        ("nn_procrustes_mean", "NN Pose Distance\n(0-1)"),
        ("hvr", "HVR"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(f"{method}: Per-Climb Metric Distributions (n={len(stems)})", fontsize=13)

    for ax, (key, label) in zip(axes, metrics):
        vals = []
        names = []
        for s in stems:
            v = per_climb[s][method].get(key)
            if v is not None:
                vals.append(v)
                names.append(climb_names.get(s, s))

        if not vals:
            ax.set_title(label)
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            continue

        vals = np.array(vals)
        # Strip plot: all points at x=0 with jitter
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(jitter, vals, alpha=0.7, s=40, color="#4A90D9", edgecolors="white", linewidths=0.5)

        # Add mean line
        ax.axhline(np.mean(vals), color="#E8913A", linewidth=2, linestyle="--",
                    label=f"mean={np.mean(vals):.3f}")
        # Add median line
        ax.axhline(np.median(vals), color="#50B050", linewidth=2, linestyle=":",
                    label=f"median={np.median(vals):.3f}")

        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        if key == "traj":
            ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Distribution plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch writeup evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="World model checkpoint")
    parser.add_argument("--dataset", type=Path,
                        default=config.DATA_DIR / "dataset.npz")
    parser.add_argument("--output-dir", type=Path,
                        default=config.DATA_DIR / "writeup_eval")
    parser.add_argument("--device", default=None)

    # RL evaluation: provide stem:path pairs
    parser.add_argument("--rl-evals", nargs="*", default=[],
                        help="stem:checkpoint pairs for RL evaluation, e.g. "
                             "stem1:path/to/rl1.pt stem2:path/to/rl2.pt")
    parser.add_argument("--rl-npz-evals", nargs="*", default=[],
                        help="stem:npz_path pairs for pre-saved RL predictions")

    # Optional: only evaluate specific test stems
    parser.add_argument("--stems", nargs="*", default=None,
                        help="Evaluate only these stems (default: all test stems)")

    args = parser.parse_args()
    device = config.get_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Parse RL mappings
    rl_checkpoints = {}  # stem -> Path
    for entry in args.rl_evals:
        stem, path = entry.split(":", 1)
        rl_checkpoints[stem] = Path(path)

    rl_npzs = {}  # stem -> Path
    for entry in args.rl_npz_evals:
        stem, path = entry.split(":", 1)
        rl_npzs[stem] = Path(path)

    # Load dataset
    print("Loading dataset...")
    data = load_dataset(args.dataset)
    test_stems = list(data["test_stems"])
    if args.stems:
        test_stems = [s for s in test_stems if s in args.stems]
    print(f"  Test stems: {len(test_stems)}")

    # Build pose bank from training data
    print("Building pose bank...")
    bank, bank_norm = build_pose_bank(data["train_sequences"])
    print(f"  Bank size: {len(bank)} poses")

    # Stem -> climb name mapping
    climb_log = load_climb_log()
    stem_to_name = {}
    for stem in test_stems:
        entry = climb_log.get(stem, {})
        name = entry.get("route_name", "").strip()
        if not name:
            print(f"  WARNING: no route_name for {stem}, skipping")
        stem_to_name[stem] = name

    # Evaluate each climb
    per_climb = {}
    for stem in test_stems:
        name = stem_to_name.get(stem)
        if not name:
            continue

        try:
            result = evaluate_single_climb(
                video_stem=stem,
                climb_name=name,
                wm_checkpoint=args.checkpoint,
                dataset_data=data,
                bank=bank,
                bank_norm=bank_norm,
                device=device,
                rl_npz_path=rl_npzs.get(stem),
                rl_checkpoint_path=rl_checkpoints.get(stem),
                dataset_path=args.dataset,
            )
            per_climb[stem] = result
        except Exception as e:
            print(f"  ERROR on {stem}: {e}")
            continue

    if not per_climb:
        print("No climbs evaluated successfully.")
        return

    # --- Table 1: Full test set (world model + hands-only) ---
    full_methods = ["Hands-Only", "World Model"]
    agg_full = aggregate_metrics(per_climb, full_methods)
    print_aggregate_table(agg_full, "Full Test Set (World Model vs Baseline)", full_methods)
    agg_robust = aggregate_metrics_robust(per_climb, full_methods)
    print_robust_table(agg_robust, "Full Test Set — Median [Q1, Q3]", full_methods)

    # --- Table 2: RL subset ---
    rl_stems = {s for s in per_climb if "RL" in per_climb[s]}
    if rl_stems:
        rl_subset = {s: per_climb[s] for s in rl_stems}
        rl_methods = ["Hands-Only", "World Model", "RL"]
        agg_rl = aggregate_metrics(rl_subset, rl_methods)
        print_aggregate_table(agg_rl, "RL Subset", rl_methods)
        agg_rl_robust = aggregate_metrics_robust(rl_subset, rl_methods)
        print_robust_table(agg_rl_robust, "RL Subset — Median [Q1, Q3]", rl_methods)

        # --- Bar chart ---
        plot_nn_bar_chart(
            rl_subset, rl_methods, stem_to_name,
            args.output_dir / "nn_plausibility_bar_chart.png",
        )
        
    # --- Distribution plots ---
    plot_metric_distributions(
        per_climb, "World Model", stem_to_name,
        args.output_dir / "world_model_distributions.png",
    )

    # --- Save raw results to JSON ---
    json_out = {}
    for stem, methods in per_climb.items():
        json_out[stem] = {
            "climb_name": stem_to_name.get(stem, ""),
        }
        for method, m in methods.items():
            json_out[stem][method] = {
                k: float(v) if v is not None else None
                for k, v in m.items()
            }

    json_path = args.output_dir / "writeup_metrics.json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nRaw metrics saved to {json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()