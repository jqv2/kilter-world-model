"""
Single-climb comparison: GT vs Hands-Only vs World Model vs RL.

Generates:
    1. Trajectory overlay image (board + 4 centroid paths)
    2. path-aligned comparison video (4 panels, Procrustes-aligned)
    3. Metrics table printed to console

Usage:
    python scripts/run_comparison.py \
        --video static_controller \
        --climb "Static controller" \
        --checkpoint data/checkpoints/best_DONE.pt \
        --rl-npz data/rl_viz/deterministic/step_2752512_static_controller_keypoints.npz \
        --output-dir data/viz/comparison
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from pipeline.dataset import load_dataset
from pipeline.routes import holds_to_array
from evaluation.metrics import (
    discrete_frechet_distance,
    mean_centroid_distance,
    path_aligned_keypoint_error,
    nearest_neighbor_pose_distance,
    build_pose_bank,
    summarize_nn_distances,
)
from evaluation.visualize import (
    lookup_route,
    render_trajectory_comparison_image,
    render_path_aligned_comparison_video,
    TRAJECTORY_COLORS, render_2x2_comparison_video,
    interpolate_pose_sequence,
)
from evaluation.baselines import hanging_baseline_predictions, compute_rl_bone_lengths
from models.rl_baseline import prepare_routes_for_rl, ClimbingEnv, rollout_episode
from scripts.train_rl import PolicyNetwork, _make_eval_action_fn
from models.world_model import (
    StructuredPoseTransformer,
    compute_reference_bone_lengths,
    resolve_hold_sequence_and_targets,
    extract_target_hands,
)
from scripts.run_visualize import (
    load_gt_poses,
    load_model,
    hold_orders_applied_for,
)
from evaluation.visualize import (
    autoregressive_rollout,
    autoregressive_rollout_structured,
)
from pipeline.routes import apply_route_edits


def load_rl_poses(npz_path: Path) -> list[np.ndarray]:
    """
    Load RL prediction poses from a saved .npz file.

    Expects the file to contain a single array of (T, 12, 2) or
    an object array of (12, 2) arrays.

    Args:
        npz_path: Path to the .npz file.

    Returns:
        List of (12, 2) climbing-keypoint poses.
    """
    data = np.load(npz_path, allow_pickle=True)
    # Handle both np.savez (keyed) and np.save (single array)
    if hasattr(data, 'files'):
        arr = data[data.files[0]]
    else:
        arr = data

    if arr.dtype == object:
        return [np.array(p, dtype=np.float32) for p in arr]
    # (T, 12, 2) array
    return [arr[t] for t in range(len(arr))]


def rollout_rl_deterministic(
    rl_checkpoint_path: Path,
    video_stem: str,
    dataset_path: Path,
    device: torch.device,
) -> tuple[list[np.ndarray], np.ndarray | None, list | None, dict]:
    """
    Run a deterministic RL rollout for a specific route.

    Loads the policy from a checkpoint, finds the route matching
    video_stem, and runs a single deterministic episode.

    Args:
        rl_checkpoint_path: Path to RL .pt checkpoint.
        video_stem: Video stem to match against route stems.
        dataset_path: Path to dataset.npz (for route preparation).
        device: Torch device.

    Returns:
        Tuple of (poses, targets, target_hands, info) where poses is a
        list of (12, 2) climbing-keypoint arrays, targets is (T, 2) or
        None, target_hands is a list or None, and info is the episode
        info dict containing 'holds_visited' and 'total_holds'.
    """
    data = load_dataset(dataset_path)
    routes, bone_lengths = prepare_routes_for_rl(data)
    env = ClimbingEnv(routes, bone_lengths)

    obs_dim = env.observation_space.shape[0]
    policy = PolicyNetwork(obs_dim).to(device)
    ckpt = torch.load(rl_checkpoint_path, weights_only=False, map_location=device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    stem_to_idx = {r.stem: i for i, r in enumerate(routes)}
    if video_stem not in stem_to_idx:
        available = ", ".join(sorted(stem_to_idx.keys()))
        raise ValueError(
            f"Route stem '{video_stem}' not found. Available: {available}"
        )

    action_fn = _make_eval_action_fn(policy, device)
    episode = rollout_episode(
        env, action_fn, route_index=stem_to_idx[video_stem],
        max_steps=config.RL_STEP_LIMIT,
    )
    print(f"  RL rollout: {episode['info']['holds_visited']}/"
          f"{episode['info']['total_holds']} holds, {episode['outcome']}")
    targets = episode.get("targets")
    target_hands = episode.get("target_hands")
    # Convert target list to (T, 2) array if present
    if targets is not None:
        targets = np.array(targets, dtype=np.float32)
    return episode["poses"], targets, target_hands, episode["info"]


def generate_predictions(
    video_stem: str,
    climb_name: str,
    checkpoint_path: Path,
    device: torch.device,
    dataset_path: Path,
    rl_npz_path: Path | None = None,
    rl_checkpoint_path: Path | None = None,
) -> dict:
    """
    Generate or load predictions for all four methods.

    Returns:
        Dict with keys 'gt', 'Hands-Only', 'World Model', 'RL', each
        mapping to a list of (12, 2) climbing-keypoint poses. Also
        includes 'route_holds', 'gt_17kp', and 'train_sequences'.
    """
    gt_poses_17kp, fps = load_gt_poses(video_stem)
    idx = config.CLIMBING_KEYPOINT_INDICES
    gt_climbing = [p[idx] for p in gt_poses_17kp]

    route = lookup_route(climb_name)
    route_holds = route["holds"]

    # Match route edit behavior to checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ds_meta = ckpt.get("dataset_metadata", {})
    if ds_meta.get("route_edits_applied", True):
        route_holds = apply_route_edits(route_holds, video_stem)

    # Load dataset for bone lengths and pose bank
    data = load_dataset(dataset_path)
    bl = compute_rl_bone_lengths(data["train_sequences"])
    
    # --- Derive hold sequence (for GT targets + structured model) ---
    apply_ho = hold_orders_applied_for(checkpoint_path)
    gt_filtered = np.array(gt_poses_17kp)[:, idx, :]
    hold_seq, targets_gt = resolve_hold_sequence_and_targets(
        gt_filtered, route_holds, video_stem if apply_ho else None,
    )
    target_hands_gt = extract_target_hands(hold_seq, targets_gt)

    # --- Hanging baseline ---
    print("Generating hands-only baseline...")
    hanging_preds, hang_tgt_pos, initial_pose, hang_tgt_hands = hanging_baseline_predictions(
        gt_poses_17kp, route_holds, video_stem, bone_lengths=bl,
    )
    hanging_seq = [initial_pose] + hanging_preds
    hanging_targets = np.vstack([[[np.nan, np.nan]], hang_tgt_pos])
    hanging_target_hands = [None] + hang_tgt_hands

    # --- World model ---
    print("Generating world model predictions...")
    model = load_model(checkpoint_path, device)
    apply_ho = hold_orders_applied_for(checkpoint_path)
    hold_positions, hold_roles = holds_to_array(route_holds, normalize=True)
    stride = config.ROLLOUT_STRIDE
    seed = np.array(gt_poses_17kp[:config.CONTEXT_WINDOW * stride])
    ref_bones = compute_reference_bone_lengths([np.array(gt_poses_17kp)])
    n_frames = len(gt_poses_17kp)

    is_structured = isinstance(model, StructuredPoseTransformer)
    wm_targets, wm_target_hands = None, None
    if is_structured:
        wm_poses, wm_targets = autoregressive_rollout_structured(
            model, seed, n_frames, hold_positions, hold_roles,
            hold_seq, device, max_bone_lengths=ref_bones,
            gt_poses=np.array(gt_poses_17kp),
        )
        wm_target_hands = extract_target_hands(hold_seq, wm_targets)
    else:
        wm_poses = autoregressive_rollout(
            model, seed, n_frames, hold_positions, hold_roles,
            device, max_bone_lengths=ref_bones,
        )

    # --- RL ---
    rl_poses = None
    rl_targets, rl_target_hands = None, None
    if rl_npz_path is not None:
        print(f"Loading RL predictions from {rl_npz_path}...")
        rl_poses = load_rl_poses(rl_npz_path)
    elif rl_checkpoint_path is not None:
        print(f"Running deterministic RL rollout from {rl_checkpoint_path}...")
        rl_poses, rl_targets, rl_target_hands, _rl_info = rollout_rl_deterministic(
            rl_checkpoint_path, video_stem, dataset_path, device,
        )

    return {
        "Ground Truth": gt_climbing,
        "Hands-Only": hanging_seq,
        "World Model": wm_poses,
        "RL": rl_poses,
        "targets": {
            "Ground Truth": (targets_gt, target_hands_gt),
            "Hands-Only": (hanging_targets, hanging_target_hands),
            "World Model": (wm_targets, wm_target_hands),
            "RL": (rl_targets, rl_target_hands),
        },
        "route_holds": route_holds,
        "train_sequences": data["train_sequences"],
        "fps": fps,
    }


def compute_metrics(
    gt_seq: list[np.ndarray],
    pred_seq: list[np.ndarray],
    bank: np.ndarray,
    bank_norm: np.ndarray,
) -> dict:
    """
    Compute all comparison metrics for one method vs GT.

    Args:
        gt_seq: List of (12, 2) GT climbing-keypoint poses.
        pred_seq: List of (12, 2) predicted poses.
        bank: Pose bank from build_pose_bank.
        bank_norm: Normalized pose bank.

    Returns:
        Dict with traj, path_procrustes, nn_procrustes_mean.
    """
    traj = mean_centroid_distance(pred_seq, gt_seq)
    path_err = path_aligned_keypoint_error(
        pred_seq, gt_seq, align="procrustes",
    )
    nn_dists = nearest_neighbor_pose_distance(pred_seq, bank, bank_norm)
    nn_summary = summarize_nn_distances(nn_dists)

    return {
        "traj": traj,
        "path_procrustes": path_err,
        "nn_procrustes_mean": nn_summary["procrustes"]["mean"],
    }


def print_metrics_table(all_metrics: dict[str, dict]) -> None:
    """Print a formatted comparison table."""
    header = (
        f"{'Method':<20} {'Traj':>10} {'Path 20Proc':>10} "
        f"{'NN Proc':>10} {'Hold Visits':>12}"
    )
    print(f"\n{'=' * len(header)}")
    print(header)
    print(f"{'-' * len(header)}")
    for label, m in all_metrics.items():
        print(
            f"{label:<20} "
            f"{m['traj']:>10.2f} {m['path_procrustes']:>10.2f} "
            f"{m['nn_procrustes_mean']:>10.4f} {'(manual)':>12}"
        )
    print(f"{'=' * len(header)}")


def main():
    parser = argparse.ArgumentParser(
        description="Single-climb comparison: GT vs Hands-Only vs World Model vs RL"
    )
    parser.add_argument("--video", required=True, help="Video stem")
    parser.add_argument("--climb", required=True, help="Climb name in Kilter DB")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="World model checkpoint")
    rl_group = parser.add_mutually_exclusive_group(required=False)
    rl_group.add_argument("--rl-npz", type=Path,
                          help="Path to pre-saved RL predictions .npz")
    rl_group.add_argument("--rl-checkpoint", type=Path,
                          help="Path to RL .pt checkpoint (deterministic rollout)")
    parser.add_argument("--dataset", type=Path,
                        default=config.DATA_DIR / "dataset.npz")
    parser.add_argument("--output-dir", type=Path,
                        default=config.DATA_DIR / "viz" / "comparison")
    parser.add_argument("--device", default=None)
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier for output videos")
    args = parser.parse_args()

    device = config.get_device(args.device)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # --- Generate all predictions ---
    results = generate_predictions(
        args.video, args.climb, args.checkpoint,
        device, args.dataset,
        rl_npz_path=args.rl_npz,
        rl_checkpoint_path=args.rl_checkpoint,
    )

    gt_seq = results["Ground Truth"]
    methods = {k: results[k] for k in ("Hands-Only", "World Model")}
    if results["RL"] is not None:
        methods["RL"] = results["RL"]

    # --- Build pose bank ---
    print("Building pose bank...")
    bank, bank_norm = build_pose_bank(results["train_sequences"])
    print(f"  Bank size: {len(bank)} poses")

    # --- Compute metrics ---
    print("Computing metrics...")
    all_metrics = {}
    for label, pred_seq in methods.items():
        all_metrics[label] = compute_metrics(gt_seq, pred_seq, bank, bank_norm)

    print_metrics_table(all_metrics)

    # --- Trajectory overlay image ---
    print("Rendering trajectory comparison image...")
    render_trajectory_comparison_image(
        method_sequences={"Ground Truth": gt_seq, **methods},
        route_holds=results["route_holds"],
        output_path=out / "trajectories.png",
    )

    # --- path-aligned comparison video ---
    print("Rendering path-aligned comparison video...")
    render_path_aligned_comparison_video(
        method_sequences={"Ground Truth": gt_seq, **methods},
        output_path=out / "path_aligned_comparison.mp4",
        fps=10.0,
        align="procrustes",
    )
    
    # --- 2x2 grid video ---
    print("Rendering 2x2 comparison video...")
    
    # Interpolate RL poses for smooth playback (RL runs at control_hz, not video fps)
    rl_interp_factor = config.RL_PHYSICS_HZ // config.RL_CONTROL_HZ
    rl_tgt, rl_tgt_hands = results["targets"]["RL"]
    rl_poses_smooth, rl_tgt_smooth, rl_hands_smooth = interpolate_pose_sequence(
        results["RL"], rl_interp_factor, rl_tgt, rl_tgt_hands,
    )
    results["RL"] = rl_poses_smooth
    results["targets"]["RL"] = (rl_tgt_smooth, rl_hands_smooth)
    
    panel_order = ["Ground Truth", "Hands-Only", "World Model", "RL"]
    panels = []
    for label in panel_order:
        tgt, tgt_hands = results["targets"][label]
        panels.append({
            "label": label,
            "poses": results[label],
            "targets": tgt,
            "target_hands": tgt_hands,
        })
    render_2x2_comparison_video(
        panels=panels,
        route_holds=results["route_holds"],
        output_path=out / "grid_comparison.mp4",
        fps=round(results["fps"] * args.speed),
    )

    print(f"\nAll outputs saved to {out}/")


if __name__ == "__main__":
    main()