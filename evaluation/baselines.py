import numpy as np

import config

# --- Constants -----------------------------------------------------------

_LIMB_CHAINS = {
    0: (5, 7, 9),    # left arm
    1: (6, 8, 10),   # right arm
    2: (11, 13, 15), # left leg
    3: (12, 14, 16), # right leg
}

_LIMB_NAMES = {0: "L hand", 1: "R hand", 2: "L foot", 3: "R foot"}


# --- Geometry helpers ----------------------------------------------------

def _bone_length(pose: np.ndarray, i: int, j: int) -> float:
    return float(np.linalg.norm(pose[i] - pose[j]))


def _solve_two_bone_ik(
    root: np.ndarray,
    target: np.ndarray,
    len_upper: float,
    len_lower: float,
    bend_sign: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve 2-bone IK in 2D."""
    to_target = target - root
    dist = np.linalg.norm(to_target)

    if dist < 1e-6:
        return root + np.array([0.0, -len_upper]), target.copy()

    max_reach = len_upper + len_lower
    if dist >= max_reach:
        direction = to_target / dist
        return root + direction * len_upper, root + direction * max_reach

    cos_angle = (len_upper**2 + dist**2 - len_lower**2) / (2 * len_upper * dist)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    base_angle = np.arctan2(to_target[1], to_target[0])
    mid_angle = base_angle + bend_sign * angle
    mid = root + len_upper * np.array([np.cos(mid_angle), np.sin(mid_angle)])

    return mid, target.copy()


def _solve_two_bone_ik_closest(
    root: np.ndarray,
    target: np.ndarray,
    len_upper: float,
    len_lower: float,
    current_mid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes both possible bend directions for the 2-bone IK and returns the
    solution that places the mid-joint (elbow/knee) closest to its current position.
    """
    mid_pos, end_pos = _solve_two_bone_ik(root, target, len_upper, len_lower, 1.0)
    mid_neg, end_neg = _solve_two_bone_ik(root, target, len_upper, len_lower, -1.0)
    
    if np.linalg.norm(mid_pos - current_mid) < np.linalg.norm(mid_neg - current_mid):
        return mid_pos, end_pos
    return mid_neg, end_neg


# --- Constraint helpers --------------------------------------------------

def _is_contact_valid(limb_on_hold: dict[int, int | None], ignore_limbs: list[int] = None) -> bool:
    """Strictly enforces at least 1 upper-body limb AND at least 1 lower-body limb."""
    if ignore_limbs is None:
        ignore_limbs = []
        
    has_upper = False
    has_lower = False
    
    for lid, hold_idx in limb_on_hold.items():
        if hold_idx is not None and lid not in ignore_limbs:
            if lid in config.HAND_LIMBS:
                has_upper = True
            elif lid in config.FOOT_LIMBS:
                has_lower = True
                
    return has_upper and has_lower


# --- Baseline -----------------------------------------------------------

def greedy_ik_baseline_predictions(
    gt_frames: list[np.ndarray],
    route_holds: list[dict],
    verbose: bool = True,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Greedy climber + IK baseline (no GT beyond the starting pose).

    Runs a two-pass simulation: first pass determines natural timing,
    second pass rescales speeds so the climb finishes at frame T-1.

    Args:
        gt_frames: List of T ground truth poses, each (17, 2).
            Only gt_frames[0] is used.
        route_holds: List of hold dicts with 'x', 'y', 'role_id'.
        verbose: Print debug info.

    Returns:
        Tuple of (predictions, target_positions).
    """
    # Pass 1: dry run to find how many frames the climb takes
    _, _, finish_frame = _simulate(gt_frames, route_holds, speed_scale=1.0, verbose=False)

    T = len(gt_frames)
    active_frames = max(1, finish_frame)
    # Scale so the climb fills the video (leave ~5% for the final hold pose)
    speed_scale = active_frames / (T * 0.95)

    if verbose:
        print(f"  Pacing: dry run finished at frame {finish_frame}/{T-1}, "
              f"speed_scale={speed_scale:.2f}")

    predictions, target_positions, _ = _simulate(
        gt_frames, route_holds, speed_scale=speed_scale, verbose=verbose
    )
    return predictions, target_positions


def _simulate(
    gt_frames: list[np.ndarray],
    route_holds: list[dict],
    speed_scale: float = 1.0,
    verbose: bool = True,
) -> tuple[list[np.ndarray], np.ndarray, int]:
    T = len(gt_frames)
    start_pose = gt_frames[0]
    hold_positions = np.array([[h["x"], h["y"]] for h in route_holds])

    # Bone lengths per limb
    bone_lengths = {}
    reach = {}
    for limb_id, (root_idx, mid_idx, end_idx) in _LIMB_CHAINS.items():
        upper = _bone_length(start_pose, root_idx, mid_idx)
        lower = _bone_length(start_pose, mid_idx, end_idx)
        bone_lengths[limb_id] = (upper, lower)
        reach[limb_id] = upper + lower

    # Build target queue
    start_indices = [i for i, h in enumerate(route_holds) if h["role_id"] == 12]
    mid_indices = [i for i, h in enumerate(route_holds) if h["role_id"] == 13]
    finish_indices = [i for i, h in enumerate(route_holds) if h["role_id"] == 14]
    foot_indices = [i for i, h in enumerate(route_holds) if h["role_id"] == 15]

    target_queue = (
        sorted(start_indices, key=lambda i: hold_positions[i, 1])
        + sorted(mid_indices, key=lambda i: hold_positions[i, 1])
        + sorted(finish_indices, key=lambda i: hold_positions[i, 1])
    )
    finish_index_set = set(finish_indices)

    def get_valid_holds(lid):
        if lid in config.HAND_LIMBS:
            return start_indices + mid_indices + finish_indices
        return start_indices + mid_indices + finish_indices + foot_indices

    # --- DYNAMIC PACING ---
    frames_per_target = max(5.0, T / max(1, len(target_queue)))
    BODY_SPEED = float(np.clip(30.0 / (frames_per_target * 0.5), 0.2, 3.0)) * speed_scale
    LIMB_SPEED = float(np.clip(60.0 / (frames_per_target * 0.5), 0.5, 10.0)) * speed_scale

    if verbose:
        print(f"  Greedy baseline: {len(target_queue)} targets over {T} frames")

    limb_on_hold: dict[int, int | None] = {lid: None for lid in config.LIMB_KEYPOINTS_COCO}
    current_pose = start_pose.copy()

    # 0. Initialize: Snap limbs logically to starting holds
    for limb_id, kp_idx in config.LIMB_KEYPOINTS_COCO.items():
        candidates = get_valid_holds(limb_id)
        if candidates:
            r, m, e = _LIMB_CHAINS[limb_id]
            reachable = [h for h in candidates if np.linalg.norm(start_pose[r] - hold_positions[h]) <= reach[limb_id] * 0.99]
            if reachable:
                nearest = min(reachable, key=lambda h: np.linalg.norm(start_pose[kp_idx] - hold_positions[h]))
                is_upper = limb_id in config.HAND_LIMBS
                has_type = any(limb_on_hold[l] is not None for l in (config.HAND_LIMBS if is_upper else config.FOOT_LIMBS))
                
                if np.linalg.norm(start_pose[kp_idx] - hold_positions[nearest]) < 80.0 or not has_type:
                    limb_on_hold[limb_id] = nearest
                    current_pose[m], current_pose[e] = _solve_two_bone_ik_closest(
                        current_pose[r], hold_positions[nearest], bone_lengths[limb_id][0], bone_lengths[limb_id][1], current_pose[m]
                    )

    # 0.5 Force Rules: Ensure mathematical constraint is met even if GT frame 0 looks weird
    if not any(limb_on_hold[l] is not None for l in config.HAND_LIMBS):
        best_l, best_h, min_d = None, None, float('inf')
        for l in config.HAND_LIMBS:
            r = _LIMB_CHAINS[l][0]
            for h in start_indices + mid_indices + finish_indices:
                d = np.linalg.norm(start_pose[r] - hold_positions[h])
                if d < min_d: min_d, best_l, best_h = d, l, h
        if best_l is not None:
            limb_on_hold[best_l] = best_h
            r, m, e = _LIMB_CHAINS[best_l]
            current_pose[m], current_pose[e] = _solve_two_bone_ik_closest(current_pose[r], hold_positions[best_h], bone_lengths[best_l][0], bone_lengths[best_l][1], current_pose[m])

    if not any(limb_on_hold[l] is not None for l in config.FOOT_LIMBS):
        best_l, best_h, min_d = None, None, float('inf')
        for l in config.FOOT_LIMBS:
            r = _LIMB_CHAINS[l][0]
            for h in get_valid_holds(l):
                d = np.linalg.norm(start_pose[r] - hold_positions[h])
                if d < min_d: min_d, best_l, best_h = d, l, h
        if best_l is not None:
            limb_on_hold[best_l] = best_h
            r, m, e = _LIMB_CHAINS[best_l]
            current_pose[m], current_pose[e] = _solve_two_bone_ik_closest(current_pose[r], hold_positions[best_h], bone_lengths[best_l][0], bone_lengths[best_l][1], current_pose[m])

    queue_idx = 0
    active_hold_idx = None
    active_limb = None
    finished = False
    finish_frame = T - 1  # default: never finished early

    predictions = []
    target_positions = np.full((T - 1, 2), np.nan, dtype=np.float32)

    for frame in range(T - 1):
        if finished:
            predictions.append(current_pose.copy())
            if frame > 0: target_positions[frame] = target_positions[frame - 1]
            continue

        # 1. Pick next target
        if active_hold_idx is None:
            if queue_idx >= len(target_queue):
                finished = True
                finish_frame = frame
                predictions.append(current_pose.copy())
                continue

            tentative_hold_idx = target_queue[queue_idx]
            hold_role = route_holds[tentative_hold_idx]["role_id"]
            candidate_limbs = config.FOOT_LIMBS if hold_role == 15 else config.HAND_LIMBS
            candidate_limbs = sorted(
                candidate_limbs,
                key=lambda lid: np.linalg.norm(current_pose[config.LIMB_KEYPOINTS_COCO[lid]] - hold_positions[tentative_hold_idx])
            )

            valid_limb = None
            for lid in candidate_limbs:
                if _is_contact_valid(limb_on_hold, ignore_limbs=[lid]):
                    valid_limb = lid
                    break

            # PRE-ANCHORING via Queue Injection
            if valid_limb is None:
                best_free_lid, best_h, best_dist = None, None, float('inf')
                
                for free_lid in candidate_limbs:
                    if limb_on_hold[free_lid] is None:
                        ee_pos = current_pose[config.LIMB_KEYPOINTS_COCO[free_lid]]
                        for h in get_valid_holds(free_lid):
                            if h != tentative_hold_idx:
                                d = np.linalg.norm(ee_pos - hold_positions[h])
                                if d < best_dist:
                                    best_dist, best_free_lid, best_h = d, free_lid, h
                                    
                if best_free_lid is not None:
                    target_queue.insert(queue_idx, best_h)
                    if verbose: print(f"  Frame {frame}: PRE-ANCHOR INJECTED -> {_LIMB_NAMES[best_free_lid]} needs hold {best_h} first.")
                    predictions.append(current_pose.copy())
                    continue  
                else:
                    if verbose: print(f"  Frame {frame}: STUCK! Cannot release any limbs for hold {tentative_hold_idx}.")
                    finished = True
                    finish_frame = frame
                    predictions.append(current_pose.copy())
                    continue
            else:
                active_limb = valid_limb
                active_hold_idx = tentative_hold_idx
                limb_on_hold[active_limb] = None
                queue_idx += 1
                if verbose: print(f"  Frame {frame}: {_LIMB_NAMES[active_limb]} reaching for hold {active_hold_idx}")

        target_positions[frame] = hold_positions[active_hold_idx]
        target_pos = hold_positions[active_hold_idx]
        
        root_idx, mid_idx, end_idx = _LIMB_CHAINS[active_limb]
        root_to_target = np.linalg.norm(current_pose[root_idx] - target_pos)
        limb_reach = reach[active_limb]

        # 2. Shift body
        if root_to_target > limb_reach * 0.85:
            desired_root = target_pos + (current_pose[root_idx] - target_pos) / root_to_target * limb_reach * 0.7
            body_shift_unclamped = desired_root - current_pose[root_idx]
            shift_dist_unclamped = np.linalg.norm(body_shift_unclamped)

            # Ease-out: move faster when far, decelerate near target
            ease_factor = min(1.0, root_to_target / (limb_reach * 0.5))
            eased_speed = BODY_SPEED * (0.3 + 0.7 * ease_factor)
            if shift_dist_unclamped > eased_speed:
                body_shift_unclamped = body_shift_unclamped / shift_dist_unclamped * eased_speed

            # --- Critical Constraint Clamping ---
            best_safe_frac = 1.0
            low, high = 0.0, 1.0
            for _ in range(6):
                mid_val = (low + high) / 2.0
                test_pose = current_pose + body_shift_unclamped * mid_val
                test_release = [l for l, h in limb_on_hold.items() if h is not None 
                                and np.linalg.norm(test_pose[_LIMB_CHAINS[l][0]] - hold_positions[h]) > reach[l] * 0.99]
                if not _is_contact_valid(limb_on_hold, ignore_limbs=test_release):
                    high = mid_val
                else:
                    low = mid_val
                    best_safe_frac = mid_val

            body_shift = body_shift_unclamped * best_safe_frac
            shift_dist_applied = np.linalg.norm(body_shift)

            for kp in range(17):
                current_pose[kp] += body_shift

            # Safely drop over-extended limbs
            limbs_to_drop = []
            for lid, hold_idx in limb_on_hold.items():
                if hold_idx is not None:
                    if np.linalg.norm(current_pose[_LIMB_CHAINS[lid][0]] - hold_positions[hold_idx]) > reach[lid] * 0.99:
                        limbs_to_drop.append(lid)
            
            for lid in limbs_to_drop:
                if _is_contact_valid(limb_on_hold, ignore_limbs=[lid]):
                    limb_on_hold[lid] = None
                    if verbose: print(f"  Frame {frame}: {_LIMB_NAMES[lid]} safely released (over-extended)")

            # Re-pin remaining anchors
            for lid, hold_idx in limb_on_hold.items():
                if hold_idx is not None:
                    r, m, e = _LIMB_CHAINS[lid]
                    current_pose[m], current_pose[e] = _solve_two_bone_ik_closest(
                        current_pose[r], hold_positions[hold_idx], bone_lengths[lid][0], bone_lengths[lid][1], current_pose[m]
                    )

            # Pre-Anchor Rescue via Queue Injection (Stuck Stretching)
            if shift_dist_applied < 0.1 and root_to_target > limb_reach * 0.90:
                critical_types_tearing = set()
                test_pose = current_pose + body_shift_unclamped
                for l, h in limb_on_hold.items():
                    if h is not None and np.linalg.norm(test_pose[_LIMB_CHAINS[l][0]] - hold_positions[h]) > reach[l] * 0.99:
                        if not _is_contact_valid(limb_on_hold, ignore_limbs=[l]):
                            critical_types_tearing.add('hand' if l in config.HAND_LIMBS else 'foot')

                best_free_lid, best_h, best_dist = None, None, float('inf')

                for crit_type in critical_types_tearing:
                    pool = config.HAND_LIMBS if crit_type == 'hand' else config.FOOT_LIMBS
                    for free_lid in pool:
                        if limb_on_hold[free_lid] is None and free_lid != active_limb:
                            ee_pos = current_pose[config.LIMB_KEYPOINTS_COCO[free_lid]]
                            r = _LIMB_CHAINS[free_lid][0]
                            for h in get_valid_holds(free_lid):
                                if h != active_hold_idx and np.linalg.norm(current_pose[r] - hold_positions[h]) <= reach[free_lid] * 0.95:
                                    d = np.linalg.norm(ee_pos - hold_positions[h])
                                    if d < best_dist:
                                        best_dist, best_free_lid, best_h = d, free_lid, h

                if best_free_lid is not None:
                    target_queue.insert(queue_idx, active_hold_idx)
                    target_queue.insert(queue_idx, best_h)
                    active_hold_idx = None
                    active_limb = None
                    if verbose: print(f"  Frame {frame}: STUCK stretching! Aborted and injected PRE-ANCHOR {best_h}")
                    predictions.append(current_pose.copy())
                    continue
                else:
                    if verbose: print(f"  Frame {frame}: STUCK stretching! No free limbs to pre-anchor.")
                    finished = True
                    finish_frame = frame

        # 3. Handle reaching limb (Smooth Interpolation)
        else:
            upper_len, lower_len = bone_lengths[active_limb]
            current_ee = current_pose[end_idx]
            dist_to_hold = np.linalg.norm(target_pos - current_ee)
            
            # Ease-out: decelerate as limb approaches hold
            ease_factor = min(1.0, dist_to_hold / 20.0)
            eased_limb_speed = LIMB_SPEED * (0.3 + 0.7 * ease_factor)
            if dist_to_hold > eased_limb_speed:
                step_target = current_ee + (target_pos - current_ee) / dist_to_hold * eased_limb_speed
            else:
                step_target = target_pos

            current_pose[mid_idx], current_pose[end_idx] = _solve_two_bone_ik_closest(
                current_pose[root_idx], step_target, upper_len, lower_len, current_pose[mid_idx]
            )

            if np.linalg.norm(current_pose[end_idx] - target_pos) < 2.0:
                limb_on_hold[active_limb] = active_hold_idx
                if verbose: print(f"  Frame {frame}: {_LIMB_NAMES[active_limb]} arrived at hold {active_hold_idx}")
                active_hold_idx = None

                if finish_index_set and sum(1 for lid in config.HAND_LIMBS if limb_on_hold[lid] in finish_index_set) >= len(finish_index_set):
                    if verbose: print(f"  Frame {frame}: FINISH")
                    finished = True
                    finish_frame = frame

        # 4. Smooth Opportunistic grab for ALL free limbs (NO TELEPORTING)
        for lid in config.LIMB_KEYPOINTS_COCO:
            if limb_on_hold[lid] is None and lid != active_limb:
                r, m, e = _LIMB_CHAINS[lid]
                possible = [h for h in get_valid_holds(lid) if h != active_hold_idx and np.linalg.norm(current_pose[r] - hold_positions[h]) <= reach[lid] * 0.99]
                if possible:
                    ee_pos = current_pose[config.LIMB_KEYPOINTS_COCO[lid]]
                    best_h = min(possible, key=lambda h: np.linalg.norm(ee_pos - hold_positions[h]))
                    dist_to_h = np.linalg.norm(ee_pos - hold_positions[best_h])
                    
                    # If it drifts within 40 units, it acts like a magnet and smoothly pulls it in
                    if dist_to_h < 40.0:
                        ease_factor = min(1.0, dist_to_h / 20.0)
                        eased_speed = LIMB_SPEED * (0.3 + 0.7 * ease_factor)
                        if dist_to_h > eased_speed:
                            step_target = ee_pos + (hold_positions[best_h] - ee_pos) / dist_to_h * eased_speed
                        else:
                            step_target = hold_positions[best_h]
                            
                        current_pose[m], current_pose[e] = _solve_two_bone_ik_closest(
                            current_pose[r], step_target, bone_lengths[lid][0], bone_lengths[lid][1], current_pose[m]
                        )
                        
                        if np.linalg.norm(current_pose[e] - hold_positions[best_h]) < 2.0:
                            limb_on_hold[lid] = best_h
                            if verbose: print(f"  Frame {frame}: {_LIMB_NAMES[lid]} opportunistically grabbed hold {best_h}")

        predictions.append(current_pose.copy())

    return predictions, target_positions, finish_frame