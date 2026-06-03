"""
Train the pose prediction transformer with route context.

Usage:
    python scripts/train.py
    python scripts/train.py --dataset data/dataset.npz --epochs 100
    python scripts/train.py --device cuda --batch-size 128

Saves checkpoints to data/checkpoints/ and prints train/val metrics each epoch.
"""

import argparse
import json
import sys
import time
from pathlib import Path
import statistics

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from models.world_model import (
    PoseTransformer, PoseDataset, weighted_mse_loss, bone_length_loss,
    compute_reference_bone_lengths, enforce_bone_lengths,
    hold_proximity_loss, check_hand_arrival,
    StructuredPoseTransformer, StructuredPoseDataset,
    resolve_hold_sequence_and_targets,
)
from evaluation.metrics import (
    mean_keypoint_error,
    summarize_teacher_forcing,
    summarize_autoregressive,
)
from pipeline.dataset import load_dataset
from pipeline.routes import pad_holds, normalize_board_coords, prepare_holds_for_model


def _make_checkpoint_dict(model, optimizer, epoch, val_loss, args, apply_hold_orders, raw_meta):
    """Build the dict saved at best/periodic/final checkpoints."""
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "config": {
            "hidden_dim": config.MODEL_HIDDEN_DIM,
            "n_layers": config.MODEL_LAYERS,
            "n_heads": config.MODEL_HEADS,
            "context_window": config.CONTEXT_WINDOW,
            "max_holds": config.MAX_ROUTE_HOLDS,
            "noise_std": args.noise_std,
            "bone_weight": args.bone_weight,
            "scheduled_sampling_max": args.scheduled_sampling_max,
            "structured": args.structured,
            "hold_orders_applied": apply_hold_orders,
            "hold_weight": args.hold_weight,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "pose_dim": config.NUM_CLIMBING_KEYPOINTS * 2,
        },
        "dataset_metadata": raw_meta,
    }


def evaluate_teacher_forcing(
    model: PoseTransformer,
    sequences: list[np.ndarray],
    scores: list[np.ndarray],
    holds: list[np.ndarray],
    roles: list[np.ndarray],
    device: torch.device,
) -> dict:
    """
    Evaluate the model under teacher forcing on a set of sequences.

    For each frame t, feeds the ground truth context window and predicts
    frame t+1. Reports mean keypoint error across all frames and sequences.
    """
    model.eval()
    all_errors = []

    with torch.no_grad():
        for seq, sc, h_pos, h_roles in zip(sequences, scores, holds, roles):
            seq_f = seq[:, config.CLIMBING_KEYPOINT_INDICES, :]
            sc_f = sc[:, config.CLIMBING_KEYPOINT_INDICES] if sc.ndim == 2 else sc
            T = seq_f.shape[0]
            if T <= config.CONTEXT_WINDOW * config.ROLLOUT_STRIDE:
                continue

            flat = torch.from_numpy(
                seq_f.reshape(T, -1).astype("float32")
            ).to(device)

            h_pos_t, h_roles_t, mask_t = prepare_holds_for_model(h_pos, h_roles, device)

            stride = config.ROLLOUT_STRIDE
            n_seed = config.CONTEXT_WINDOW * stride
            predicted_frames = []
            gt_indices = []
            for t in range(n_seed, T, stride):
                ctx_indices = list(range(t - config.CONTEXT_WINDOW * stride, t, stride))
                context = flat[ctx_indices].unsqueeze(0)
                pred_abs = model.predict_absolute(
                    context, h_pos_t, h_roles_t, mask_t
                ).squeeze(0).cpu().numpy()
                predicted_frames.append(pred_abs.reshape(config.NUM_CLIMBING_KEYPOINTS, 2))
                gt_indices.append(t)

            gt_slice = [seq_f[t] for t in gt_indices]
            conf_slice = [sc_f[t] for t in gt_indices]

            errors = np.array([
                mean_keypoint_error(predicted_frames[i], gt_slice[i], conf_slice[i])
                for i in range(len(predicted_frames))
            ])
            all_errors.append(errors)

    if not all_errors:
        return summarize_teacher_forcing(np.array([np.nan]))
    return summarize_teacher_forcing(np.concatenate(all_errors))


def evaluate_autoregressive(
    model: PoseTransformer,
    sequences: list[np.ndarray],
    scores: list[np.ndarray],
    holds: list[np.ndarray],
    roles: list[np.ndarray],
    device: torch.device,
    max_bone_lengths: np.ndarray | None = None,
) -> dict:
    """
    Evaluate the model under autoregressive rollout.

    Feeds the first context_window * ROLLOUT_STRIDE frames as ground truth,
    then predicts every ROLLOUT_STRIDE-th frame using its own predictions
    as input. Compares against GT at the corresponding stride-spaced indices.
    """
    model.eval()
    all_errors = []

    with torch.no_grad():
        for seq, sc, h_pos, h_roles in zip(sequences, scores, holds, roles):
            seq_f = seq[:, config.CLIMBING_KEYPOINT_INDICES, :]
            sc_f = sc[:, config.CLIMBING_KEYPOINT_INDICES] if sc.ndim == 2 else sc
            T = seq_f.shape[0]
            if T <= config.CONTEXT_WINDOW * config.ROLLOUT_STRIDE + config.ROLLOUT_STRIDE:
                continue

            flat = seq_f.reshape(T, -1).astype("float32")

            h_pos_t, h_roles_t, mask_t = prepare_holds_for_model(h_pos, h_roles, device)

            stride = config.ROLLOUT_STRIDE
            n_seed = min(config.CONTEXT_WINDOW * stride, T)
            history = list(flat[:n_seed])
            predicted_frames = []
            gt_indices = []

            for t in range(n_seed, T, stride):
                indices = list(range(len(history) - config.CONTEXT_WINDOW * stride, len(history), stride))
                indices = [max(0, i) for i in indices]
                context = torch.from_numpy(
                    np.array([history[i] for i in indices])
                ).unsqueeze(0).to(device)

                pred_abs = model.predict_absolute(
                    context, h_pos_t, h_roles_t, mask_t
                ).squeeze(0).cpu().numpy()
                pred_pose = pred_abs.reshape(config.NUM_CLIMBING_KEYPOINTS, 2)
                if max_bone_lengths is not None:
                    pred_pose = enforce_bone_lengths(pred_pose, max_bone_lengths)
                    pred_abs = pred_pose.reshape(-1)
                predicted_frames.append(pred_pose)
                history.append(pred_abs)
                gt_indices.append(t)

            gt_slice = [seq_f[t] for t in gt_indices]
            conf_slice = [sc_f[t] for t in gt_indices]

            errors = np.array([
                mean_keypoint_error(predicted_frames[i], gt_slice[i], conf_slice[i])
                for i in range(len(predicted_frames))
            ])
            all_errors.append(errors)

    if not all_errors:
        return summarize_autoregressive(np.array([np.nan]))
    return summarize_autoregressive(np.concatenate(all_errors))


def train_epoch(
    model: PoseTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    noise_std: float = 0.0,
    bone_weight: float = 0.5,
    sampling_prob: float = 0.0,
) -> float:
    """Train for one epoch. Returns mean loss.

    Args:
        model: The pose transformer.
        loader: Training DataLoader.
        optimizer: Optimizer.
        device: Torch device.
        noise_std: Gaussian noise std added to context (except last frame).
        bone_weight: Weight of bone-length loss relative to MSE.
        sampling_prob: Probability of replacing the last context frame with
            the model's own prediction before computing the final loss.
            Ramps from 0 to SCHEDULED_SAMPLING_MAX over training to bridge
            the gap between teacher-forced training and autoregressive rollout.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for context, target_abs, h_pos, h_roles, h_mask, displacement in loader:
        context = context.to(device)
        target_abs = target_abs.to(device)
        h_pos = h_pos.to(device)
        h_roles = h_roles.to(device)
        h_mask = h_mask.to(device)
        displacement = displacement.to(device)

        if noise_std > 0.0:
            noise = torch.randn_like(context) * noise_std
            # Don't corrupt the last frame
            noise[:, -1, :] = 0.0
            context = context + noise

        # Scheduled sampling: replace last context frame with model's own prediction
        if sampling_prob > 0.0 and torch.rand(1).item() < sampling_prob:
            with torch.no_grad():
                self_pred = model.predict_absolute(context, h_pos, h_roles, h_mask)
            context = context.clone()
            context[:, -1, :] = self_pred.detach()

        pred_abs = model.predict_absolute(context, h_pos, h_roles, h_mask)
        mse = weighted_mse_loss(pred_abs, target_abs, displacement)
        bone = bone_length_loss(pred_abs, target_abs)
        loss = mse + bone_weight * bone

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train_epoch_structured(
    model: StructuredPoseTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    noise_std: float = 0.0,
    bone_weight: float = 0.5,
    sampling_prob: float = 0.0,
    hold_weight: float = 0.0,
) -> float:
    """Train structured model for one epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for (context, target_abs, h_pos, h_roles, h_mask, displacement,
         tgt_pos) in loader:
        context = context.to(device)
        target_abs = target_abs.to(device)
        h_pos = h_pos.to(device)
        h_roles = h_roles.to(device)
        h_mask = h_mask.to(device)
        displacement = displacement.to(device)
        tgt_pos = tgt_pos.to(device)

        if noise_std > 0.0:
            noise = torch.randn_like(context) * noise_std
            noise[:, -1, :] = 0.0
            context = context + noise

        if sampling_prob > 0.0 and torch.rand(1).item() < sampling_prob:
            with torch.no_grad():
                self_pred = model.predict_absolute(
                    context, h_pos, h_roles, tgt_pos, h_mask
                )
            context = context.clone()
            context[:, -1, :] = self_pred.detach()

        pred_abs = model.predict_absolute(
            context, h_pos, h_roles, tgt_pos, h_mask
        )
        mse = weighted_mse_loss(pred_abs, target_abs, displacement)
        bone = bone_length_loss(pred_abs, target_abs)
        loss = mse + bone_weight * bone
        if hold_weight > 0.0:
            loss = loss + hold_weight * hold_proximity_loss(pred_abs, tgt_pos)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate_teacher_forcing_structured(
    model: StructuredPoseTransformer,
    sequences: list[np.ndarray],
    scores: list[np.ndarray],
    holds: list[np.ndarray],
    roles: list[np.ndarray],
    route_holds: list[list[dict]],
    device: torch.device,
    stems: list[str] | None = None,
) -> dict:
    """Evaluate structured model under teacher forcing.

    stems (parallel to sequences) enables per-video hold order/timing
    overrides; None uses automatic detection.
    """
    model.eval()
    all_errors = []

    with torch.no_grad():
        if stems is None:
            stems = [None] * len(sequences)
        for seq, sc, h_pos, h_roles, r_holds, stem in zip(
            sequences, scores, holds, roles, route_holds, stems
        ):
            seq_f = seq[:, config.CLIMBING_KEYPOINT_INDICES, :]
            sc_f = sc[:, config.CLIMBING_KEYPOINT_INDICES] if sc.ndim == 2 else sc
            T = seq_f.shape[0]
            if T <= config.CONTEXT_WINDOW * config.ROLLOUT_STRIDE:
                continue

            flat = torch.from_numpy(
                seq_f.reshape(T, -1).astype("float32")
            ).to(device)

            h_pos_t, h_roles_t, mask_t = prepare_holds_for_model(h_pos, h_roles, device)

            # Targets from GT (or manual hold-order override for this video)
            hold_seq, targets_board = resolve_hold_sequence_and_targets(
                seq_f, r_holds, stem
            )
            if len(hold_seq) == 0:
                continue
            targets_norm = normalize_board_coords(targets_board)

            stride = config.ROLLOUT_STRIDE
            n_seed = config.CONTEXT_WINDOW * stride
            predicted_frames = []
            gt_indices = []
            for t in range(n_seed, T, stride):
                ctx_indices = list(range(t - config.CONTEXT_WINDOW * stride, t, stride))
                context = flat[ctx_indices].unsqueeze(0)
                tgt = torch.from_numpy(targets_norm[t]).unsqueeze(0).to(device)

                pred_abs = model.predict_absolute(
                    context, h_pos_t, h_roles_t, tgt, mask_t
                ).squeeze(0).cpu().numpy()
                predicted_frames.append(pred_abs.reshape(config.NUM_CLIMBING_KEYPOINTS, 2))
                gt_indices.append(t)

            gt_slice = [seq_f[t] for t in gt_indices]
            conf_slice = [sc_f[t] for t in gt_indices]

            errors = np.array([
                mean_keypoint_error(predicted_frames[i], gt_slice[i], conf_slice[i])
                for i in range(len(predicted_frames))
            ])
            all_errors.append(errors)

    if not all_errors:
        return summarize_teacher_forcing(np.array([np.nan]))
    return summarize_teacher_forcing(np.concatenate(all_errors))


def evaluate_autoregressive_structured(
    model: StructuredPoseTransformer,
    sequences: list[np.ndarray],
    scores: list[np.ndarray],
    holds: list[np.ndarray],
    roles: list[np.ndarray],
    route_holds: list[list[dict]],
    device: torch.device,
    max_bone_lengths: np.ndarray | None = None,
    stems: list[str] | None = None,
) -> dict:
    """
    Evaluate structured model under autoregressive rollout.

    Derives the hold sequence from GT poses, then predicts every
    ROLLOUT_STRIDE-th frame using the model's own predicted poses
    for both input and arrival detection.
    """
    model.eval()
    all_errors = []

    with torch.no_grad():
        if stems is None:
            stems = [None] * len(sequences)
        for seq, sc, h_pos, h_roles, r_holds, stem in zip(
            sequences, scores, holds, roles, route_holds, stems
        ):
            seq_f = seq[:, config.CLIMBING_KEYPOINT_INDICES, :]
            sc_f = sc[:, config.CLIMBING_KEYPOINT_INDICES] if sc.ndim == 2 else sc
            T = seq_f.shape[0]
            stride = config.ROLLOUT_STRIDE
            if T <= config.CONTEXT_WINDOW * stride + stride:
                continue

            flat = seq_f.reshape(T, -1).astype("float32")

            h_pos_t, h_roles_t, mask_t = prepare_holds_for_model(h_pos, h_roles, device)

            # Hold sequence "plan" from GT (or manual override for this video)
            hold_seq, _ = resolve_hold_sequence_and_targets(seq_f, r_holds, stem)
            if len(hold_seq) == 0:
                continue

            stride = config.ROLLOUT_STRIDE
            n_seed = min(config.CONTEXT_WINDOW * stride, T)
            history = list(flat[:n_seed])
            predicted_frames = []
            gt_indices = []

            seq_idx = 0
            consecutive_near = 0
            frames_on_current = 0
            arrival_needed = max(1, config.HOLD_ARRIVAL_FRAMES // stride)

            for t in range(n_seed, T, stride):
                indices = list(range(len(history) - config.CONTEXT_WINDOW * stride, len(history), stride))
                indices = [max(0, i) for i in indices]
                context = torch.from_numpy(
                    np.array([history[i] for i in indices])
                ).unsqueeze(0).to(device)

                target_hold = hold_seq[min(seq_idx, len(hold_seq) - 1)]
                tgt_board = np.array([target_hold["x"], target_hold["y"]], dtype=np.float32)
                tgt_norm = normalize_board_coords(tgt_board.reshape(1, 2)).reshape(2)
                tgt_t = torch.from_numpy(tgt_norm).unsqueeze(0).to(device)

                pred_abs = model.predict_absolute(
                    context, h_pos_t, h_roles_t, tgt_t, mask_t
                ).squeeze(0).cpu().numpy()
                pred_pose = pred_abs.reshape(config.NUM_CLIMBING_KEYPOINTS, 2)
                if max_bone_lengths is not None:
                    pred_pose = enforce_bone_lengths(pred_pose, max_bone_lengths)
                    pred_abs = pred_pose.reshape(-1)
                predicted_frames.append(pred_pose)
                history.append(pred_abs)
                gt_indices.append(t)

                # Advance hold sequence
                if seq_idx < len(hold_seq):
                    frames_on_current += 1
                    if check_hand_arrival(pred_pose, tgt_board,
                                      threshold=config.ROLLOUT_ARRIVAL_THRESHOLD_HAND) is not None:
                        consecutive_near += 1
                        if consecutive_near >= arrival_needed:
                            seq_idx += 1
                            consecutive_near = 0
                            frames_on_current = 0
                    elif frames_on_current >= config.ROLLOUT_HOLD_TIMEOUT:
                        seq_idx += 1
                        consecutive_near = 0
                        frames_on_current = 0
                    else:
                        consecutive_near = 0

            gt_slice = [seq_f[t] for t in gt_indices]
            conf_slice = [sc_f[t] for t in gt_indices]

            errors = np.array([
                mean_keypoint_error(predicted_frames[i], gt_slice[i], conf_slice[i])
                for i in range(len(predicted_frames))
            ])
            all_errors.append(errors)

    if not all_errors:
        return summarize_autoregressive(np.array([np.nan]))
    return summarize_autoregressive(np.concatenate(all_errors))


def main():
    parser = argparse.ArgumentParser(description="Train pose prediction transformer")
    parser.add_argument(
        "--dataset", type=Path,
        default=config.DATA_DIR / "dataset.npz",
    )
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--noise-std", type=float, default=config.NOISE)
    parser.add_argument("--bone-weight", type=float, default=config.BONE_LOSS_WEIGHT)
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=config.DATA_DIR / "checkpoints",
    )
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--scheduled-sampling-max", type=float, default=config.SCHEDULED_SAMPLING_MAX,
                        help="Max probability for scheduled sampling ramp (0 to disable)")
    parser.add_argument("--strides", type=int, nargs="+", default=config.TRAINING_STRIDES,
                        help="Temporal strides for dataset construction (default: from config)")
    parser.add_argument("--structured", action="store_true",
                        help="Use structured model variant (target hold as input)")
    parser.add_argument("--hold-weight", type=float, default=0.0,
                        help="Weight for hold proximity loss (structured model only, 0 to disable)")
    parser.add_argument("--dropout", type=float, default=config.MODEL_DROPOUT,
                        help="Dropout rate for transformer layers")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="AdamW weight decay")
    parser.add_argument("--no-hold-orders", action="store_true",
                        help="Ignore data/hold_orders/ overrides (structured model only); "
                             "use automatic hold-sequence detection")
    args = parser.parse_args()

    device = config.get_device(args.device)
    print(f"Device: {device}")

    # Load data
    print(f"Loading dataset from {args.dataset}...")
    data = load_dataset(args.dataset)
    apply_hold_orders = not args.no_hold_orders
    train_stems_arg = data["train_stems"] if apply_hold_orders else None
    test_stems_arg = data["test_stems"] if apply_hold_orders else None
    print(f"  Train: {len(data['train_sequences'])} videos, "
          f"{sum(s.shape[0] for s in data['train_sequences'])} frames")
    print(f"  Test:  {len(data['test_sequences'])} videos, "
          f"{sum(s.shape[0] for s in data['test_sequences'])} frames")
    raw_meta = json.loads(str(np.load(args.dataset, allow_pickle=True)["metadata"]))
    print(f"  Route edits: {'applied' if raw_meta.get('route_edits_applied', True) else 'ignored'}")
    if args.structured:
        print(f"  Hold orders: {'applied' if apply_hold_orders else 'ignored'}")

    # Build datasets
    if args.structured:
        train_ds = StructuredPoseDataset(
            data["train_sequences"], data["train_scores"],
            data["train_holds"], data["train_roles"],
            data["train_route_holds"],
            stems=train_stems_arg,
            strides=args.strides,
        )
    else:
        train_ds = PoseDataset(
            data["train_sequences"], data["train_scores"],
            data["train_holds"], data["train_roles"],
            strides=args.strides,
        )
    
    disp_idx = -2 if args.structured else -1  # displacement field position in sample tuple
    disps = [s[disp_idx] for s in train_ds.samples]
    print(f"  Displacement stats: min={min(disps):.3f}, median={statistics.median(disps):.3f}, "
        f"mean={statistics.mean(disps):.3f}, max={max(disps):.3f}")
    
    if args.structured:
        val_ds = StructuredPoseDataset(
            data["test_sequences"], data["test_scores"],
            data["test_holds"], data["test_roles"],
            data["test_route_holds"],
            stems=test_stems_arg,
            strides=args.strides,
        )
    else:
        val_ds = PoseDataset(
            data["test_sequences"], data["test_scores"],
            data["test_holds"], data["test_roles"],
            strides=args.strides,
        )
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Val samples:   {len(val_ds)}")
    
    # Compute max bone lengths for autoregressive eval
    ref_bones = compute_reference_bone_lengths(data["train_sequences"])

    if len(train_ds) == 0:
        print("Error: No training samples. Check context window vs sequence lengths.")
        sys.exit(1)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # Model
    ModelClass = StructuredPoseTransformer if args.structured else PoseTransformer
    model = ModelClass(dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")
    print(f"  Hidden dim: {config.MODEL_HIDDEN_DIM}, "
          f"Layers: {config.MODEL_LAYERS}, Heads: {config.MODEL_HEADS}")
    print(f"  Context window: {config.CONTEXT_WINDOW}, "
          f"Max holds: {config.MAX_ROUTE_HOLDS}")
    print(f"  Noise std: {args.noise_std}, Bone weight: {args.bone_weight}, "
          f"Scheduled sampling max: {args.scheduled_sampling_max}, "
          f"Strides: {args.strides}")
    print(f"  Model variant: {'structured' if args.structured else 'direct'}")
    if args.hold_weight > 0.0:
        print(f"  Hold proximity weight: {args.hold_weight}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Squashed ramp: reaches max at N/2 epochs, then plateaus for the rest of training
        ramp_epochs = max(args.epochs // 2, 1)
        sampling_prob = args.scheduled_sampling_max * min(1.0, (epoch - 1) / ramp_epochs)

        epoch_fn = train_epoch_structured if args.structured else train_epoch
        if args.structured:
            train_loss = epoch_fn(model, train_loader, optimizer, device, args.noise_std, args.bone_weight, sampling_prob, args.hold_weight)
        else:
            train_loss = epoch_fn(model, train_loader, optimizer, device, args.noise_std, args.bone_weight, sampling_prob)

        # Validation loss
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                if args.structured:
                    (context, target_abs, h_pos, h_roles, h_mask, displacement,
                     tgt_pos) = batch
                    tgt_pos = tgt_pos.to(device)
                else:
                    context, target_abs, h_pos, h_roles, h_mask, displacement = batch

                context = context.to(device)
                target_abs = target_abs.to(device)
                h_pos = h_pos.to(device)
                h_roles = h_roles.to(device)
                h_mask = h_mask.to(device)
                displacement = displacement.to(device)
                
                if args.structured:
                    pred_abs = model.predict_absolute(
                        context, h_pos, h_roles, tgt_pos, h_mask
                    )
                else:
                    pred_abs = model.predict_absolute(context, h_pos, h_roles, h_mask)
                mse = weighted_mse_loss(pred_abs, target_abs, displacement)
                bone = bone_length_loss(pred_abs, target_abs)
                loss = mse + args.bone_weight * bone
                if args.structured and args.hold_weight > 0.0:
                    loss = loss + args.hold_weight * hold_proximity_loss(pred_abs, tgt_pos)
                val_loss += loss.item()
                n_val += 1
        val_loss = val_loss / max(n_val, 1)

        scheduler.step()
        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"train={train_loss:.6f} | val={val_loss:.6f} | "
              f"lr={lr:.2e} | ss={sampling_prob:.2f} | {dt:.1f}s")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                _make_checkpoint_dict(model, optimizer, epoch, val_loss, args, apply_hold_orders, raw_meta),
                args.checkpoint_dir / "best.pt",
            )
        
        # Save periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(
                _make_checkpoint_dict(model, optimizer, epoch, val_loss, args, apply_hold_orders, raw_meta),
                args.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt",
            )

        # Periodic evaluation
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            print(f"\n  --- Evaluation at epoch {epoch} ---")

            if args.structured:
                tf = evaluate_teacher_forcing_structured(
                    model,
                    data["test_sequences"], data["test_scores"],
                    data["test_holds"], data["test_roles"],
                    data["test_route_holds"],
                    device,
                    test_stems_arg,
                )
                ar = evaluate_autoregressive_structured(
                    model,
                    data["test_sequences"], data["test_scores"],
                    data["test_holds"], data["test_roles"],
                    data["test_route_holds"],
                    device,
                    max_bone_lengths=ref_bones,
                    stems=test_stems_arg,
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

            print(f"  Teacher forcing: mean={tf['mean']:.3f}, "
                  f"median={tf['median']:.3f}, max={tf['max']:.3f}")
            print(f"  Autoregressive:  p25={ar['p25']:.3f}, p50={ar['p50']:.3f}, "
                  f"p75={ar['p75']:.3f}, p100={ar['p100']:.3f}")
            print()

    # Save final
    torch.save(
        _make_checkpoint_dict(model, optimizer, args.epochs, val_loss, args, apply_hold_orders, raw_meta),
        args.checkpoint_dir / "final.pt",
    )

    print(f"\nDone. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to {args.checkpoint_dir}")


if __name__ == "__main__":
    main()