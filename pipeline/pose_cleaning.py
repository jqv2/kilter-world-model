"""
Pose cleaning: fill missing frames in pose extraction output.

Operates in pixel space (before homography transformation) so that
linear interpolation is geometrically correct.

Strategies:
    - Gaps between two valid frames: linear interpolation of keypoints and scores.
    - Trailing gaps (no valid frame after): forward-fill from last valid frame.
    - Gaps exceeding MAX_INTERPOLATION_GAP: left as-is (unrecoverable).
"""

import copy

import numpy as np

import config


def clean_pose_sequence(frames: list[dict]) -> list[dict]:
    """
    Fill missing frames in a pose sequence via interpolation or forward-fill.

    Handles frames where no person was detected. Does NOT handle keypoint
    jumps; those are caught by clean_board_space_poses after the
    homography transform.

    Args:
        frames: List of frame dicts from pose extraction JSON. Each has
            'frame_idx', 'keypoints' (list or None), 'scores' (list or None),
            'box' (list or None).

    Returns:
        New list of frame dicts with missing frames filled where possible.
        Filled frames have 'filled': True added. Original frames are not
        modified.
    """
    out = [copy.deepcopy(f) for f in frames]
    n = len(out)

    gaps = _find_missing_gaps(out)

    for start, end in gaps:
        prev_idx = start - 1 if start > 0 else None
        next_idx = end if end < n else None

        has_prev = prev_idx is not None and out[prev_idx]["keypoints"] is not None
        has_next = next_idx is not None and out[next_idx]["keypoints"] is not None

        if has_prev and has_next:
            _interpolate_gap(out, prev_idx, next_idx)
        elif has_prev:
            _forward_fill_gap(out, prev_idx, start, end)

    return out


def clean_board_space_poses(
    poses: list[np.ndarray],
    max_displacement_per_frame: float | None = None,
) -> list[np.ndarray]:
    """
    Per-keypoint cleaning on board-space poses. Detects individual keypoints
    that teleport between frames and interpolates only those keypoints,
    leaving the rest of the skeleton untouched.

    Uses a scaled threshold: a keypoint is marked invalid if it moves more
    than max_displacement_per_frame * elapsed_frames since that keypoint
    was last valid.

    Args:
        poses: List of (17, 2) board-space keypoint arrays.
        max_displacement_per_frame: Per-keypoint per-frame threshold in
            board units. Defaults to config.BOARD_SPACE_MAX_DISPLACEMENT.

    Returns:
        New list of (17, 2) arrays with per-keypoint jumps smoothed.
    """
    if max_displacement_per_frame is None:
        max_displacement_per_frame = config.BOARD_SPACE_MAX_DISPLACEMENT

    n = len(poses)
    if n < 2:
        return [p.copy() for p in poses]

    n_kp = poses[0].shape[0]
    out = [p.copy() for p in poses]

    # Track validity per keypoint
    valid = [[True] * n_kp for _ in range(n)]

    # Pass 1: mark invalid keypoints
    prev_valid_idx = [0] * n_kp  # last valid frame index per keypoint

    for i in range(1, n):
        for k in range(n_kp):
            prev = prev_valid_idx[k]
            displacement = float(np.linalg.norm(poses[i][k] - poses[prev][k]))
            elapsed = i - prev
            if displacement > max_displacement_per_frame * elapsed:
                valid[i][k] = False
            else:
                prev_valid_idx[k] = i

    # Pass 2: interpolate invalid keypoints independently
    for k in range(n_kp):
        i = 0
        while i < n:
            if not valid[i][k]:
                start = i
                while i < n and not valid[i][k]:
                    i += 1
                prev = start - 1 if start > 0 else None
                nxt = i if i < n else None

                if prev is not None and nxt is not None:
                    total = nxt - prev
                    for j in range(start, i):
                        t = (j - prev) / total
                        out[j][k] = out[prev][k] + t * (out[nxt][k] - out[prev][k])
                elif prev is not None:
                    for j in range(start, i):
                        out[j][k] = out[prev][k].copy()
            else:
                i += 1

    return out


def _find_missing_gaps(frames: list[dict]) -> list[tuple[int, int]]:
    """
    Find contiguous runs of frames with no keypoints.

    Returns:
        List of (start, end) tuples where frames[start:end] are all missing.
    """
    gaps = []
    i = 0
    n = len(frames)
    while i < n:
        if frames[i]["keypoints"] is None:
            start = i
            while i < n and frames[i]["keypoints"] is None:
                i += 1
            gaps.append((start, i))
        else:
            i += 1
    return gaps


def _interpolate_gap(
    frames: list[dict],
    prev_idx: int,
    next_idx: int,
) -> None:
    """Linearly interpolate keypoints and scores between two valid frames."""
    prev_kp = np.array(frames[prev_idx]["keypoints"])
    next_kp = np.array(frames[next_idx]["keypoints"])
    prev_scores = np.array(frames[prev_idx]["scores"])
    next_scores = np.array(frames[next_idx]["scores"])
    prev_box = np.array(frames[prev_idx]["box"])
    next_box = np.array(frames[next_idx]["box"])

    total_steps = next_idx - prev_idx
    for i in range(prev_idx + 1, next_idx):
        t = (i - prev_idx) / total_steps
        frames[i]["keypoints"] = (prev_kp + t * (next_kp - prev_kp)).tolist()
        frames[i]["scores"] = (prev_scores + t * (next_scores - prev_scores)).tolist()
        frames[i]["box"] = (prev_box + t * (next_box - prev_box)).tolist()
        frames[i]["filled"] = True


def _forward_fill_gap(
    frames: list[dict],
    source_idx: int,
    gap_start: int,
    gap_end: int,
) -> None:
    """Copy keypoints from source frame into trailing missing frames."""
    for i in range(gap_start, gap_end):
        frames[i]["keypoints"] = copy.deepcopy(frames[source_idx]["keypoints"])
        frames[i]["scores"] = copy.deepcopy(frames[source_idx]["scores"])
        frames[i]["box"] = copy.deepcopy(frames[source_idx]["box"])
        frames[i]["filled"] = True