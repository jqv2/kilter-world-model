"""
Naive hanging baseline for climb pose prediction.

Builds key poses at each hand hold transition — shoulders a fixed distance
below the hand midpoint, torso and legs vertical — then linearly
interpolates between them using timing from the hold order file.

Bone lengths are computed identically to the RL baseline: left/right pairs
pooled, percentile taken, producing a single symmetric value per bone type.
"""

import numpy as np

import config
from pipeline.routes import load_hold_order_edit

# --- Symmetric bone-length computation (shared with RL baseline) ---------

# COCO 17-keypoint index pairs: left and right pooled before percentile
BONE_PAIRS_COCO = {
    "upper_arm": [(5, 7), (6, 8)],
    "forearm":   [(7, 9), (8, 10)],
    "thigh":     [(11, 13), (12, 14)],
    "shin":      [(13, 15), (14, 16)],
    "torso":     [(5, 11), (6, 12)],
}
WIDTH_PAIRS_COCO = {
    "half_shoulder_width": (5, 6),
    "half_hip_width":      (11, 12),
}


def compute_rl_bone_lengths(
    sequences: list[np.ndarray],
    percentile: float = config.RL_BONE_LENGTH_PERCENTILE,
) -> dict[str, float]:
    """Compute symmetric bone lengths from pose sequences.

    For each bone, computes the Euclidean distance per frame across all
    sequences, averages left and right sides, and takes the given
    percentile.  Returns 5 bone lengths and 2 half-widths, all in
    board units.

    Args:
        sequences: List of (T_i, 17, 2) arrays in board space
            (full COCO keypoints).
        percentile: Percentile to use (e.g. 97 for 97th-percentile).

    Returns:
        Dict with keys upper_arm, forearm, thigh, shin,
        torso, half_shoulder_width, half_hip_width.
    """
    result: dict[str, float] = {}

    for name, pairs in BONE_PAIRS_COCO.items():
        all_lengths: list[np.ndarray] = []
        for seq in sequences:
            for i, j in pairs:
                all_lengths.append(np.linalg.norm(seq[:, i] - seq[:, j], axis=1))
        result[name] = float(np.percentile(np.concatenate(all_lengths), percentile))

    for name, (i, j) in WIDTH_PAIRS_COCO.items():
        all_widths = [np.linalg.norm(seq[:, i] - seq[:, j], axis=1) for seq in sequences]
        result[name] = float(np.percentile(np.concatenate(all_widths), percentile)) / 2.0

    return result


# --- Geometry helpers ----------------------------------------------------

def _solve_two_bone_ik(
    root: np.ndarray,
    target: np.ndarray,
    len_upper: float,
    len_lower: float,
    bend_sign: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve 2-bone IK in 2D (returns mid-joint and end-effector positions)."""
    to_target = target - root
    dist = np.linalg.norm(to_target)

    if dist < 1e-6:
        return root + np.array([0.0, -len_upper]), target.copy()

    max_reach = len_upper + len_lower
    if dist >= max_reach:
        direction = to_target / dist
        return root + direction * len_upper, root + direction * max_reach

    cos_angle = np.clip(
        (len_upper**2 + dist**2 - len_lower**2) / (2 * len_upper * dist),
        -1.0, 1.0,
    )
    base_angle = np.arctan2(to_target[1], to_target[0])
    mid_angle = base_angle + bend_sign * np.arccos(cos_angle)
    mid = root + len_upper * np.array([np.cos(mid_angle), np.sin(mid_angle)])

    return mid, target.copy()


# --- Baseline ------------------------------------------------------------

_HEAD_RADIUS_BU = config.RL_HEAD_RADIUS / config.RL_BOARD_UNIT_TO_METERS


def _build_hanging_pose(
    l_hand_pos: np.ndarray,
    r_hand_pos: np.ndarray,
    bl: dict[str, float],
    shoulder_offset: float,
) -> np.ndarray:
    """
    Build a climbing-keypoint pose hanging from two hand positions.

    Shoulders sit a fixed distance below the hand midpoint. Torso and
    legs hang vertically.  Arms are solved via 2-bone IK with elbows
    bending outward.  The skeleton is fully symmetric.

    Args:
        l_hand_pos: (2,) left hand hold position in board units.
        r_hand_pos: (2,) right hand hold position in board units.
        bl: Bone lengths dict from compute_rl_bone_lengths.
        shoulder_offset: Vertical drop from hand midpoint to shoulders.

    Returns:
        (NUM_CLIMBING_KEYPOINTS, 2) pose in board units.
    """
    pose = np.zeros((config.NUM_CLIMBING_KEYPOINTS, 2), dtype=np.float32)
    hand_mid = (l_hand_pos + r_hand_pos) / 2
    shoulder_mid = hand_mid - np.array([0, shoulder_offset])

    hsw = bl["half_shoulder_width"]
    hhw = bl["half_hip_width"]

    # Shoulders (climbing idx 0=L, 1=R)
    pose[0] = shoulder_mid + np.array([-hsw, 0])
    pose[1] = shoulder_mid + np.array([hsw, 0])

    # Arms via IK (elbows bend outward: +1 left, -1 right)
    for (root, mid, end), hold, bend in [
        ((0, 2, 4), l_hand_pos, 1.0),
        ((1, 3, 5), r_hand_pos, -1.0),
    ]:
        pose[mid], pose[end] = _solve_two_bone_ik(
            pose[root], hold, bl["upper_arm"], bl["forearm"], bend
        )

    # Torso + legs hang straight down
    pose[6] = shoulder_mid + np.array([-hhw, -bl["torso"]])
    pose[7] = shoulder_mid + np.array([hhw, -bl["torso"]])
    pose[8] = pose[6] - np.array([0, bl["thigh"]])
    pose[9] = pose[7] - np.array([0, bl["thigh"]])
    pose[10] = pose[8] - np.array([0, bl["shin"]])
    pose[11] = pose[9] - np.array([0, bl["shin"]])

    return pose


def hanging_baseline_predictions(
    gt_frames: list[np.ndarray],
    route_holds: list[dict],
    video_stem: str,
    bone_lengths: dict[str, float] | None = None,
    verbose: bool = True,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Naive hanging baseline: body hangs straight down from the hands.

    Interpolates only the moving hand's position between holds, rebuilding
    the full pose at each frame so the anchored hand stays pinned to its
    hold and the body hangs correctly throughout.

    Bone lengths are symmetric (left = right) and computed at the
    RL_BONE_LENGTH_PERCENTILE, matching the RL ragdoll skeleton.

    Args:
        gt_frames: List of T ground truth poses, each (17, 2).
            Used for bone-length estimation when bone_lengths is None.
        route_holds: List of hold dicts with 'x', 'y', 'name', 'role_id'.
        video_stem: Video filename stem, used to load hold order file.
        bone_lengths: Pre-computed bone lengths from compute_rl_bone_lengths.
            When None, computed from gt_frames.
        verbose: Print debug info.

    Returns:
        Tuple of (predictions, target_positions, initial_pose) where
        predictions is a list of T-1 poses (NUM_CLIMBING_KEYPOINTS, 2),
        target_positions is (T-1, 2), and initial_pose is the frame-0
        hanging pose (NUM_CLIMBING_KEYPOINTS, 2).
    """
    T = len(gt_frames)

    # --- Bone geometry ---------------------------------------------------
    if bone_lengths is None:
        bone_lengths = compute_rl_bone_lengths([np.array(gt_frames)])

    arm_reach = bone_lengths["upper_arm"] + bone_lengths["forearm"]
    shoulder_offset = arm_reach * config.HANGING_SHOULDER_DROP

    # --- Hold order ------------------------------------------------------
    override = load_hold_order_edit(video_stem)
    hold_by_name = {
        h["name"]: np.array([h["x"], h["y"]], dtype=np.float32)
        for h in route_holds
    }

    l_pos = hold_by_name[override["start_hands"]["L"]].copy()
    r_pos = hold_by_name[override["start_hands"]["R"]].copy()

    edit_frames = override.get("num_frames") or T
    scale = T / edit_frames if edit_frames else 1.0

    # --- Build transition list -------------------------------------------
    transitions = []
    for seg in override["sequence"]:
        hold_pos = hold_by_name.get(seg["name"])
        if hold_pos is None:
            continue

        b = max(0, min(T - 1, int(round(seg["start_frame"] * scale))))
        if seg["hand"] == "L":
            transitions.append((b, "L", l_pos.copy(), hold_pos))
            l_pos = hold_pos.copy()
        else:
            transitions.append((b, "R", r_pos.copy(), hold_pos))
            r_pos = hold_pos.copy()

    if verbose:
        print(f"  Hanging baseline: {len(transitions)} transitions over {T} frames")

    # --- Rebuild pose at every frame -------------------------------------
    all_poses = np.empty((T, config.NUM_CLIMBING_KEYPOINTS, 2), dtype=np.float32)
    target_positions = np.full((T, 2), np.nan, dtype=np.float32)

    cur_l = hold_by_name[override["start_hands"]["L"]].copy()
    cur_r = hold_by_name[override["start_hands"]["R"]].copy()

    if not transitions:
        for f in range(T):
            all_poses[f] = _build_hanging_pose(
                cur_l, cur_r, bone_lengths, shoulder_offset)
    else:
        # Static initial pose before first transition
        initial = _build_hanging_pose(
            cur_l, cur_r, bone_lengths, shoulder_offset)
        for f in range(transitions[0][0]):
            all_poses[f] = initial

        for i, (b_start, hand, old_pos, new_pos) in enumerate(transitions):
            b_end = transitions[i + 1][0] if i + 1 < len(transitions) else T
            n = b_end - b_start

            if n > 0:
                for j in range(n):
                    t = j / max(1, n - 1) if n > 1 else 1.0
                    moving = old_pos * (1 - t) + new_pos * t

                    if hand == "L":
                        all_poses[b_start + j] = _build_hanging_pose(
                            moving, cur_r, bone_lengths, shoulder_offset)
                    else:
                        all_poses[b_start + j] = _build_hanging_pose(
                            cur_l, moving, bone_lengths, shoulder_offset)

                    target_positions[b_start + j] = new_pos

            if hand == "L":
                cur_l = new_pos.copy()
            else:
                cur_r = new_pos.copy()

    predictions = [all_poses[f] for f in range(1, T)]
    return predictions, target_positions[1:], all_poses[0]