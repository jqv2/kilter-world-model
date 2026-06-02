"""
Evaluation metrics:
    - Per-frame mean keypoint error (teacher forcing): Given ground truth frame t,
      how far is the predicted frame t+1 from actual frame t+1?
    - Per-problem accumulated error (autoregressive): Given only the first frame,
      predict the full sequence and calculate divergence from ground truth over time.
"""

import numpy as np


def mean_keypoint_error(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    confidence: np.ndarray | None = None,
    confidence_threshold: float = 0.3,
) -> float:
    """
    Mean Euclidean distance between predicted and ground truth keypoints,
    averaged over all valid keypoints.

    Args:
        predicted: (17, 2) predicted keypoint positions.
        ground_truth: (17, 2) ground truth keypoint positions.
        confidence: (17,) confidence scores for ground truth keypoints.
            If provided, only keypoints above confidence_threshold are included.
        confidence_threshold: Minimum confidence to include a keypoint.

    Returns:
        Mean Euclidean distance (in coordinate units) across valid keypoints,
        or np.nan if no valid keypoints exist.
    """
    if confidence is not None:
        mask = confidence >= confidence_threshold
    else:
        mask = np.ones(len(predicted), dtype=bool)

    if not mask.any():
        return np.nan

    distances = np.linalg.norm(predicted[mask] - ground_truth[mask], axis=1)
    return float(distances.mean())


def per_frame_errors(
    predicted_frames: list[np.ndarray],
    gt_frames: list[np.ndarray],
    gt_confidences: list[np.ndarray] | None = None,
    confidence_threshold: float = 0.3,
) -> np.ndarray:
    """
    Per-frame mean keypoint error between predicted and ground truth poses.

    predicted_frames[i] is compared against gt_frames[i+1]. The caller is
    responsible for generating predictions under the desired conditions:
        - Teacher forcing: each prediction uses ground truth as input.
        - Autoregressive: each prediction uses the model's own prior output.

    Args:
        predicted_frames: List of N-1 predicted poses, each (17, 2).
        gt_frames: List of N ground truth poses, each (17, 2).
        gt_confidences: List of N confidence arrays, each (17,). Optional.
        confidence_threshold: Minimum confidence to include a keypoint.

    Returns:
        Array of N-1 per-frame errors.
    """
    n_predictions = len(predicted_frames)
    assert n_predictions == len(gt_frames) - 1, (
        f"Expected {len(gt_frames) - 1} predictions for {len(gt_frames)} GT frames, "
        f"got {n_predictions}"
    )

    errors = np.empty(n_predictions)
    for i in range(n_predictions):
        conf = gt_confidences[i + 1] if gt_confidences is not None else None
        errors[i] = mean_keypoint_error(
            predicted_frames[i], gt_frames[i + 1], conf, confidence_threshold
        )
    return errors


def summarize_teacher_forcing(errors: np.ndarray) -> dict:
    """
    Compute summary statistics for a sequence of per-frame errors.

    Args:
        errors: Array of per-frame errors (may contain NaN for skipped frames).

    Returns:
        Dict with mean, median, std, max, and count of valid frames.
    """
    valid = errors[~np.isnan(errors)]
    if len(valid) == 0:
        return {"mean": np.nan, "median": np.nan, "std": np.nan, "max": np.nan, "n_valid": 0}
    return {
        "mean": float(valid.mean()),
        "median": float(np.median(valid)),
        "std": float(valid.std()),
        "max": float(valid.max()),
        "n_valid": len(valid),
    }
    
def summarize_autoregressive(errors: np.ndarray) -> dict:
    """
    Summarize autoregressive errors by reporting error at progression milestones.

    Shows how error evolves as the model predicts further into the sequence,
    which reveals whether predictions diverge over time.

    Args:
        errors: Array of per-frame errors in sequence order (may contain NaN).

    Returns:
        Dict with error at 25%, 50%, 75%, and 100% through the sequence.
    """
    valid = errors[~np.isnan(errors)]
    if len(valid) == 0:
        return {"p25": np.nan, "p50": np.nan, "p75": np.nan, "p100": np.nan}
    indices = [len(valid) // 4, len(valid) // 2, 3 * len(valid) // 4, len(valid) - 1]
    return {
        "p25": float(valid[indices[0]]),
        "p50": float(valid[indices[1]]),
        "p75": float(valid[indices[2]]),
        "p100": float(valid[indices[3]]),
    }
    

# --- Climbing keypoint indices for hip/shoulder in climbing space ---
_HIP_L, _HIP_R = 6, 7
_SHOULDER_L, _SHOULDER_R = 0, 1


def build_pose_bank(
    train_sequences: list[np.ndarray],
    dedup_threshold: float | None = None,
    skip_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a deduplicated, hip-centered reference pose bank.

    Extracts climbing keypoints from training sequences, centers each pose
    on the hip midpoint, and filters near-static duplicates so the bank
    represents the diversity of configurations rather than the frequency
    of standing still.

    Returns two versions of the bank: the raw (N, 12, 2) hip-centered poses
    for raw Euclidean lookup, and a unit-Frobenius-normalized copy for
    vectorized Procrustes distance. No KD-tree is needed — the 2D
    closed-form Procrustes is fast enough to evaluate the full bank.

    Args:
        train_sequences: List of (T, 17, 2) training pose arrays.
        dedup_threshold: Minimum mean per-keypoint displacement (board units)
            between consecutive included poses from the same sequence.
            Defaults to config.NN_POSE_DEDUP_THRESHOLD.
        skip_frames: Number of frames to skip at the start of each sequence
            (setup/establishment phase). Defaults to
            CONTEXT_WINDOW * ROLLOUT_STRIDE (the seed length).

    Returns:
        (bank, bank_norm) where bank is (N, 12, 2) hip-centered poses and
        bank_norm is (N, 12, 2) unit-Frobenius-normalized for Procrustes.
    """
    import config

    if dedup_threshold is None:
        dedup_threshold = config.NN_POSE_DEDUP_THRESHOLD
    if skip_frames is None:
        skip_frames = config.CONTEXT_WINDOW * config.ROLLOUT_STRIDE

    idx = config.CLIMBING_KEYPOINT_INDICES
    collected = []

    for seq in train_sequences:
        climbing = seq[skip_frames:, idx, :]  # (T - skip, 12, 2)
        hips = climbing[:, [_HIP_L, _HIP_R], :].mean(axis=1, keepdims=True)
        centered = climbing - hips

        last_kept = None
        for t in range(len(centered)):
            if last_kept is None:
                collected.append(centered[t])
                last_kept = centered[t]
            else:
                mean_disp = np.linalg.norm(centered[t] - last_kept, axis=1).mean()
                if mean_disp >= dedup_threshold:
                    collected.append(centered[t])
                    last_kept = centered[t]

    bank = np.array(collected)  # (N, 12, 2)

    # Pre-normalize for Procrustes
    norms = np.linalg.norm(bank.reshape(len(bank), -1), axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    bank_norm = bank / norms.reshape(len(bank), 1, 1)

    return bank, bank_norm


def pose_bank_summary(
    bank: np.ndarray,
    train_sequences: list[np.ndarray],
) -> dict:
    """
    Diagnostic info about the pose bank for sanity checking.

    Args:
        bank: (N, 12, 2) hip-centered bank from build_pose_bank.
        train_sequences: Original training sequences (for raw frame count).

    Returns:
        Dict with bank_size, raw_frame_count, and compression_ratio.
    """
    import config

    raw = sum(len(s) for s in train_sequences)
    return {
        "bank_size": len(bank),
        "raw_frame_count": raw,
        "compression_ratio": f"{raw / len(bank):.1f}x",
    }


def _batch_procrustes_distances(
    query: np.ndarray,
    bank_norm: np.ndarray,
) -> np.ndarray:
    """
    Vectorized Procrustes distance from one query to the entire bank.

    Uses the closed-form 2D Procrustes distance: for unit-normalized
    shapes q and r, d² = 2 - 2·sqrt(trace(MᵀM) + 2·det(M)) where
    M = rᵀq is the 2×2 cross-covariance. No SVD or per-candidate
    loop needed.

    Args:
        query: (12, 2) hip-centered pose, NOT yet normalized.
        bank_norm: (N, 12, 2) unit-Frobenius-normalized bank.

    Returns:
        (N,) Procrustes distances to every bank entry.
    """
    q_norm = np.linalg.norm(query)
    if q_norm < 1e-8:
        return np.full(len(bank_norm), float("inf"))
    q = query / q_norm

    # M_all[i] = bank_norm[i].T @ q, shape (N, 2, 2)
    M_all = np.einsum("nkj,kl->njl", bank_norm, q)

    # trace(M.T @ M) = sum of squared entries
    tr = (M_all ** 2).sum(axis=(1, 2))

    # det(M) for 2x2
    det = M_all[:, 0, 0] * M_all[:, 1, 1] - M_all[:, 0, 1] * M_all[:, 1, 0]

    d_sq = 2.0 - 2.0 * np.sqrt(np.clip(tr + 2.0 * det, 0, None))
    return np.sqrt(np.clip(d_sq, 0, None))


def nearest_neighbor_pose_distance(
    predicted_poses: list[np.ndarray],
    bank: np.ndarray,
    bank_norm: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Per-frame nearest-neighbor pose distance, with and without Procrustes.

    Computes exact Procrustes distance to every bank entry using a
    closed-form 2D formula (no approximation, no KD-tree). Also returns
    the raw hip-centered Euclidean 1-NN distance as a diagnostic.

    Args:
        predicted_poses: List of (12, 2) predicted climbing-keypoint poses.
        bank: (N, 12, 2) hip-centered reference bank from build_pose_bank.
        bank_norm: (N, 12, 2) unit-normalized bank from build_pose_bank.

    Returns:
        Dict with:
            'procrustes': (T,) per-frame exact Procrustes NN distances.
            'raw': (T,) per-frame hip-centered Euclidean NN distances.
    """
    bank_flat = bank.reshape(len(bank), -1)  # (N, 24) for raw Euclidean

    proc_dists = np.empty(len(predicted_poses))
    raw_dists = np.empty(len(predicted_poses))

    for t, pose in enumerate(predicted_poses):
        hip = pose[[_HIP_L, _HIP_R]].mean(axis=0)
        centered = pose - hip

        # Raw: Euclidean 1-NN
        raw_dists[t] = np.linalg.norm(bank_flat - centered.ravel(), axis=1).min()

        # Procrustes: exact, vectorized over full bank
        proc_dists[t] = _batch_procrustes_distances(centered, bank_norm).min()

    return {"procrustes": proc_dists, "raw": raw_dists}


def summarize_nn_distances(
    distances: dict[str, np.ndarray],
    poses: list[np.ndarray],
) -> dict:
    """
    Aggregate per-frame NN pose distances into per-climb summary statistics.

    Uses displacement-weighted mean as the primary aggregate so that
    transition frames (where methods actually differ) contribute more
    than static frames.

    Args:
        distances: Dict with 'procrustes' and 'raw' arrays from
            nearest_neighbor_pose_distance, each shape (T,).
        poses: List of T (12, 2) predicted poses, used to compute
            per-frame displacement weights.

    Returns:
        Dict with weighted_mean, median, and p95 for each variant.
    """
    # Displacement weights: mean keypoint movement from previous frame
    displacements = np.zeros(len(poses))
    for t in range(1, len(poses)):
        displacements[t] = np.linalg.norm(poses[t] - poses[t - 1], axis=1).mean()
    # Ensure nonzero total weight
    weights = displacements / displacements.sum() if displacements.sum() > 0 else np.ones(len(poses)) / len(poses)

    summary = {}
    for key in ("procrustes", "raw"):
        d = distances[key]
        summary[key] = {
            "weighted_mean": float(np.average(d, weights=weights)),
            "median": float(np.median(d)),
            "p95": float(np.percentile(d, 95)),
        }
    return summary