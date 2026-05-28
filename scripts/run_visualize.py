"""
Visualize model predictions on a Kilter Board climb.

Modes:
    Ground truth: render actual poses from a video's extracted data
    Prediction:   autoregressive rollout from a video's first frames

Usage:
    # Ground truth overlay for a specific video
    python scripts/run_visualize.py --video bump_it --mode gt
    
    # Greedy IK baseline for a specific climb
    python scripts/run_visualize.py --video bump_it --mode greedy --climb "Bump It"

    # Model prediction from a video's starting pose
    python scripts/run_visualize.py --video bump_it --mode predict --checkpoint data/checkpoints/best.pt

    # Model prediction with route holds highlighted
    python scripts/run_visualize.py --video bump_it --mode predict --checkpoint data/checkpoints/best.pt --climb "Bump It"

    # Specify output path and frame count
    python scripts/run_visualize.py --video bump_it --mode predict --checkpoint data/checkpoints/best.pt --n-frames 300 --output my_viz.mp4
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from pipeline.routes import apply_route_edits, holds_to_array
from pipeline.pose_cleaning import clean_pose_sequence, clean_board_space_poses
from pipeline.calibration import load_calibration, compute_homographies, kick_threshold_px, transform_keypoints
from evaluation.visualize import (
    lookup_route,
    render_pose_video,
    render_pose_video_with_targets,
    autoregressive_rollout,
    autoregressive_rollout_structured,
)
from models.world_model import (
    PoseTransformer,
    enforce_bone_lengths,
    compute_reference_bone_lengths,
    StructuredPoseTransformer,
    resolve_hold_sequence_and_targets,
)
from evaluation.baselines import greedy_ik_baseline_predictions


def load_gt_poses(video_stem: str) -> tuple[list[np.ndarray], float]:
    """
    Load ground truth board-space poses for a video.

    Returns:
        (poses, fps) where poses is a list of (17, 2) arrays.
    """
    # Find pose JSON
    matches = list(config.POSES_DIR.rglob(f"{video_stem}.json"))
    if not matches:
        raise FileNotFoundError(f"No pose JSON found for '{video_stem}'")

    with open(matches[0]) as f:
        data = json.load(f)

    fps = data["fps"]
    frames = clean_pose_sequence(data["frames"])

    # Transform to board space
    cal = load_calibration(video_stem)
    H_main, H_kick = compute_homographies(cal)
    threshold = kick_threshold_px(cal)

    poses = []
    for frame in frames:
        if frame["keypoints"] is not None:
            kp = np.array(frame["keypoints"])
            kp_board = transform_keypoints(kp, H_main, H_kick, threshold)
            poses.append(kp_board)
            
    poses = clean_board_space_poses(poses)

    return poses, fps


def load_model(checkpoint_path: Path, device: torch.device) -> PoseTransformer | StructuredPoseTransformer:
    """Load a trained model from a checkpoint. Auto-detects structured vs direct."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("config", {})
    is_structured = model_config.get("structured", False)

    ModelClass = StructuredPoseTransformer if is_structured else PoseTransformer
    model = ModelClass(
        pose_dim=model_config.get("pose_dim", config.NUM_CLIMBING_KEYPOINTS * 2),
        hidden_dim=model_config.get("hidden_dim", config.MODEL_HIDDEN_DIM),
        n_layers=model_config.get("n_layers", config.MODEL_LAYERS),
        n_heads=model_config.get("n_heads", config.MODEL_HEADS),
        context_len=model_config.get("context_window", config.CONTEXT_WINDOW),
        dropout=model_config.get("dropout", config.MODEL_DROPOUT),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    variant = "structured" if is_structured else "direct"
    epoch = checkpoint.get("epoch", "?")
    val_loss = checkpoint.get("val_loss", "?")
    print(f"Loaded {variant} checkpoint: epoch {epoch}, val_loss {val_loss}")

    return model


def hold_orders_applied_for(checkpoint_path: Path | None) -> bool:
    """
    Read the hold_orders_applied training setting from a checkpoint.

    Returns True (apply data/hold_orders/ overrides) when no checkpoint is
    given or the field is absent, matching the default training behavior.
    Lets evaluation and visualization honor overrides exactly as the model
    was trained, mirroring route_edits_applied for route edits.
    """
    if checkpoint_path is None:
        return True
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return ckpt.get("config", {}).get("hold_orders_applied", True)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize model predictions on a Kilter Board climb"
    )
    parser.add_argument(
        "--video", type=str, required=True,
        help="Video stem (e.g. 'bump_it') to use as source for GT or seed poses",
    )
    parser.add_argument(
        "--mode", choices=["gt", "gt_moves", "predict", "greedy"], default="gt",
        help="'gt' renders ground truth, 'gt_moves' shows detected move targets, 'predict' runs model rollout",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Path to model checkpoint (required for predict mode)",
    )
    parser.add_argument(
        "--climb", type=str, default=None,
        help="Climb name to look up in Kilter DB (required for predict mode, optional for gt)",
    )
    parser.add_argument(
        "--n-frames", type=int, default=None,
        help="Number of frames to generate in predict mode (default: match GT length)",
    )
    parser.add_argument(
        "--enforce-bones", action="store_true",
        help="Apply additional bone-length projection as post-processing after rollout (rollout always clamps internally)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output video path (default: data/viz/<video>_<mode>.mp4)",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--speed", type=float, default=2.0,
                    help="Rollout speed multiplier: generates speed×GT frames, plays speed× faster")
    args = parser.parse_args()

    if args.mode == "predict" and args.checkpoint is None:
        parser.error("--checkpoint is required for predict mode")

    device = config.get_device(args.device)
    apply_hold_orders = hold_orders_applied_for(args.checkpoint)

    # Load GT poses (needed for both modes)
    print(f"Loading ground truth poses for '{args.video}'...")
    gt_poses, fps = load_gt_poses(args.video)
    print(f"  {len(gt_poses)} frames @ {fps:.1f} fps")

    # Route lookup (optional)
    route_holds = None
    title_parts = [args.video]
    if args.climb:
        try:
            route = lookup_route(args.climb)
            route_holds = route["holds"]
            grade = route["grade"] or "?"
            title_parts = [f"{route['name']} ({grade})"]
            print(f"  Route: {route['name']} - {grade}, {len(route_holds)} holds")

            # Match route edit behavior to what the model was trained on
            apply_edits = True
            if args.checkpoint:
                ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
                ds_meta = ckpt.get("dataset_metadata", {})
                apply_edits = ds_meta.get("route_edits_applied", True)
            if apply_edits:
                route_holds = apply_route_edits(route_holds, args.video)
                n_excluded = len(route["holds"]) - len(route_holds)
                if n_excluded:
                    print(f"  Route edits: {n_excluded} holds excluded")
            else:
                print(f"  Route edits: ignored (matches dataset)")
        except ValueError as e:
            print(f"  Warning: {e}")
            
    is_structured = False
    target_positions = None

    # Generate poses
    if args.mode == "gt":
        poses = gt_poses
        title_parts.append("- ground truth")
    elif args.mode == "gt_moves":
        if route_holds is None:
            print("Error: --climb is required for gt_moves mode (need holds for move detection)")
            sys.exit(1)
        gt_filtered = np.array(gt_poses)[:, config.CLIMBING_KEYPOINT_INDICES, :]
        hold_seq, targets_board = resolve_hold_sequence_and_targets(
            gt_filtered, route_holds, args.video if apply_hold_orders else None
        )
        print(f"  Derived hold sequence: {len(hold_seq)} holds visited"
              f"{'' if apply_hold_orders else ' (overrides ignored)'}")
        title_parts.append("- move targets")

        if args.output:
            output_path = args.output
        else:
            viz_dir = config.DATA_DIR / "viz"
            output_path = viz_dir / f"{args.video}_{args.mode}.mp4"

        render_pose_video_with_targets(
            poses=gt_poses,
            target_positions=targets_board,
            output_path=output_path,
            route_holds=route_holds,
            fps=fps,
            title=" ".join(title_parts),
        )
        return
    elif args.mode == "greedy":
        if route_holds is None:
            print("Error: --climb is required for greedy mode (need holds for interpolation)")
            sys.exit(1)
        preds, target_positions = greedy_ik_baseline_predictions(gt_poses, route_holds)
        poses = [gt_poses[0]] + preds
        # Prepend a NaN row so target_positions aligns with poses (T frames)
        target_positions = np.vstack([[[np.nan, np.nan]], target_positions])
        title_parts.append("- greedy IK baseline")
    else:
        model = load_model(args.checkpoint, device)
        n_frames = args.n_frames or int(len(gt_poses) * args.speed)

        stride = config.ROLLOUT_STRIDE
        seed = np.array(gt_poses[:config.CONTEXT_WINDOW * stride])

        # Get normalized hold arrays for the route
        if route_holds is None:
            print("Error: --climb is required for predict mode (model needs route context)")
            sys.exit(1)
        hold_positions, hold_roles = holds_to_array(route_holds, normalize=True)

        # Derive hold sequence for structured model
        is_structured = isinstance(model, StructuredPoseTransformer)
        hold_seq = None
        if is_structured:
            gt_filtered = np.array(gt_poses)[:, config.CLIMBING_KEYPOINT_INDICES, :]
            hold_seq, _ = resolve_hold_sequence_and_targets(
                gt_filtered, route_holds, args.video if apply_hold_orders else None
            )
            print(f"  Derived hold sequence: {len(hold_seq)} holds visited"
                  f"{'' if apply_hold_orders else ' (overrides ignored)'}")

        ref_bones = compute_reference_bone_lengths([np.array(gt_poses)])

        print(f"Rolling out {n_frames} frames from {config.CONTEXT_WINDOW}-frame seed "
              f"({'structured' if is_structured else 'direct'} model)...")
        if is_structured:
            poses, target_positions = autoregressive_rollout_structured(
                model, seed, n_frames, hold_positions, hold_roles,
                hold_seq, device, max_bone_lengths=ref_bones,
                gt_poses=np.array(gt_poses),
            )
        else:
            poses = autoregressive_rollout(
                model, seed, n_frames, hold_positions, hold_roles, device,
                max_bone_lengths=ref_bones,
            )
        if args.enforce_bones:
            # Additional post-hoc pass (cosmetic, since rollout already clamps)
            poses = [
                enforce_bone_lengths(p, ref_bones) if i >= config.CONTEXT_WINDOW else p
                for i, p in enumerate(poses)
            ]
            title_parts.append("+ bone projection")
        title_parts.append("- model prediction")

    title = " ".join(title_parts)

    # Output path
    if args.output:
        output_path = args.output
    else:
        viz_dir = config.DATA_DIR / "viz"
        output_path = viz_dir / f"{args.video}_{args.mode}.mp4"

    # Render
    if target_positions is not None:
        render_pose_video_with_targets(
            poses=poses,
            target_positions=target_positions,
            output_path=output_path,
            route_holds=route_holds,
            fps=round(fps * args.speed),
            title=title,
            gt_poses=([p[config.CLIMBING_KEYPOINT_INDICES] for p in gt_poses]
                      if args.mode == "predict" else None),
        )
    else:
        render_pose_video(
            poses=poses,
            output_path=output_path,
            route_holds=route_holds,
            fps=round(fps * args.speed),
            title=title,
            gt_poses=([p[config.CLIMBING_KEYPOINT_INDICES] for p in gt_poses]
                      if args.mode == "predict" else None),
        )


if __name__ == "__main__":
    main()