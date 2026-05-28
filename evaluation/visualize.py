"""
Visualization: render predicted pose sequences as skeleton overlays on the board.

Provides:
    - Route lookup from the Kilter database
    - Board image rendering with holds and route highlights
    - Skeleton drawing in board coordinates
    - Video generation from pose sequences
"""

import math
import re
import sqlite3
from pathlib import Path

import cv2
import numpy as np

import config
from models.world_model import enforce_bone_lengths, check_hand_arrival


# ─── Board rendering constants ───────────────────────────────────────────────

BOARD_EDGES = {"left": 0, "right": 144, "bottom": 0, "top": 156}
KICKBOARD_Y = 12

# Pixels per board unit in the rendered image
RENDER_SCALE = 8

# Padding around the board in board units
RENDER_PAD = 4

# Colors (BGR for OpenCV)
COLOR_HOLD = (160, 160, 160)
COLOR_START = (0, 255, 0)       # green
COLOR_MIDDLE = (255, 255, 0)    # cyan
COLOR_FINISH = (255, 0, 255)    # magenta
COLOR_FOOT = (0, 165, 255)      # orange
COLOR_SKELETON = (0, 230, 255)  # yellow
COLOR_JOINT = (0, 100, 255)     # orange-red
COLOR_BOARD_BG = (40, 40, 40)
COLOR_GRID = (60, 60, 60)
COLOR_KICKBOARD_LINE = (80, 60, 60)

ROLE_COLORS = {
    12: COLOR_START,
    13: COLOR_MIDDLE,
    14: COLOR_FINISH,
    15: COLOR_FOOT,
}


# ─── Route lookup ────────────────────────────────────────────────────────────

import sqlite3
import re
from pathlib import Path

def lookup_route(
    climb_name: str,
    db_path: Path | None = None,
    layout_id: int = 1,
) -> dict:
    """
    Look up a climb by exact name from the Kilter database.

    Args:
        climb_name: Name of the climb (case-sensitive exact match).
        db_path: Path to kilter.db. Defaults to config.DATA_DIR / "kilter.db".
        layout_id: Layout to filter placements by.

    Returns:
        Dict with:
            'name': str - exact climb name from DB
            'frames_str': str - raw frames encoding
            'holds': list of {'x': float, 'y': float, 'name': str, 'role_id': int}
            'grade': str or None - display difficulty grade
    """
    db = db_path or (config.DATA_DIR / "kilter.db")
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row

    normalized = (climb_name.strip()
                  .replace("\u2019", "'").replace("\u2018", "'")
                  .replace("\u201c", '"').replace("\u201d", '"'))

    # Removed COLLATE NOCASE to enforce case-sensitivity
    row = conn.execute("""
        SELECT uuid, name, frames 
        FROM climbs 
        WHERE TRIM(REPLACE(REPLACE(REPLACE(REPLACE(name,
            CHAR(8217), "'"), CHAR(8216), "'"),
            CHAR(8220), '"'), CHAR(8221), '"')) = ?
        LIMIT 1
    """, (normalized,)).fetchone()

    if row is None:
        conn.close()
        raise ValueError(f"No climb found exactly matching '{climb_name}'")

    climb_uuid = row["uuid"]
    climb_name_exact = row["name"]
    frames_str = row["frames"]

    placements = re.findall(r"p(\d+)r(\d+)", frames_str)

    holds = []
    for placement_id, role_id in placements:
        hold = conn.execute("""
            SELECT h.x, h.y, h.name
            FROM placements p
            JOIN holes h ON p.hole_id = h.id
            WHERE p.id = ? AND p.layout_id = ?
        """, (int(placement_id), layout_id)).fetchone()

        if hold:
            holds.append({
                "x": hold["x"],
                "y": hold["y"],
                "name": hold["name"],
                "role_id": int(role_id),
            })

    grade_row = conn.execute("""
        SELECT dg.boulder_name
        FROM climb_stats cs
        JOIN difficulty_grades dg ON dg.difficulty = cs.display_difficulty
        WHERE cs.climb_uuid = ? AND cs.angle = 30
        LIMIT 1
    """, (climb_uuid,)).fetchone()

    grade = grade_row["boulder_name"] if grade_row else None

    conn.close()

    return {
        "name": climb_name_exact,
        "frames_str": frames_str,
        "holds": holds,
        "grade": grade,
    }


def get_all_holds(db_path: Path | None = None) -> list[tuple[float, float]]:
    """
    Get all hold positions on the board (for background rendering).

    Returns:
        List of (x, y) tuples for all 476 holds.
    """
    db = db_path or (config.DATA_DIR / "kilter.db")
    conn = sqlite3.connect(db)
    rows = conn.execute("""
        SELECT h.x, h.y
        FROM holes h
        JOIN leds l ON l.hole_id = h.id
        WHERE l.product_size_id = ?
    """, (config.PRODUCT_SIZE_ID,)).fetchall()
    conn.close()
    return [(r[0], r[1]) for r in rows]


# ─── Board rendering ─────────────────────────────────────────────────────────

def board_to_pixel(
    bx: float, by: float,
    scale: int = RENDER_SCALE,
    pad: int = RENDER_PAD,
) -> tuple[int, int]:
    """Convert board coordinates to render pixel coordinates with bounds checking."""
    # 1. Handle NaNs and Infs caused by exploding model predictions
    if math.isnan(bx) or math.isnan(by) or math.isinf(bx) or math.isinf(by):
        return -9999, -9999  # Render safely off-screen

    px = int((bx - BOARD_EDGES["left"] + pad) * scale)
    # y is flipped: board y increases upward, pixel y increases downward
    py = int((BOARD_EDGES["top"] + pad - by) * scale)

    # 2. Prevent OpenCV integer overflow crashes
    # OpenCV requires coordinates to fit within a 32-bit signed integer.
    # We clip to +/- 30000 so the lines still point in the correct direction off-screen.
    px = max(-30000, min(px, 30000))
    py = max(-30000, min(py, 30000))

    return px, py


def render_board_image(
    route_holds: list[dict] | None = None,
    all_holds: list[tuple[float, float]] | None = None,
    scale: int = RENDER_SCALE,
    pad: int = RENDER_PAD,
) -> np.ndarray:
    """
    Render a static board image with holds.

    Args:
        route_holds: List of hold dicts with 'x', 'y', 'role_id' for the route.
            If None, only background holds are drawn.
        all_holds: List of (x, y) for all holds. Fetched from DB if None.
        scale: Pixels per board unit.
        pad: Padding in board units.

    Returns:
        BGR image of the board.
    """
    if all_holds is None:
        all_holds = get_all_holds()

    w = int((BOARD_EDGES["right"] - BOARD_EDGES["left"] + 2 * pad) * scale)
    h = int((BOARD_EDGES["top"] - BOARD_EDGES["bottom"] + 2 * pad) * scale)
    img = np.full((h, w, 3), COLOR_BOARD_BG, dtype=np.uint8)

    # Grid
    for bx in range(BOARD_EDGES["left"], BOARD_EDGES["right"] + 1, 12):
        px, _ = board_to_pixel(bx, 0, scale, pad)
        cv2.line(img, (px, 0), (px, h), COLOR_GRID, 1)
    for by in range(BOARD_EDGES["bottom"], BOARD_EDGES["top"] + 1, 12):
        _, py = board_to_pixel(0, by, scale, pad)
        cv2.line(img, (0, py), (w, py), COLOR_GRID, 1)

    # Kickboard line
    _, ky = board_to_pixel(0, KICKBOARD_Y, scale, pad)
    cv2.line(img, (0, ky), (w, ky), COLOR_KICKBOARD_LINE, 1)

    # All holds (background)
    for hx, hy in all_holds:
        px, py = board_to_pixel(hx, hy, scale, pad)
        cv2.circle(img, (px, py), max(2, scale // 3), COLOR_HOLD, -1)

    # Route holds (highlighted)
    if route_holds:
        for hold in route_holds:
            px, py = board_to_pixel(hold["x"], hold["y"], scale, pad)
            color = ROLE_COLORS.get(hold["role_id"], COLOR_MIDDLE)
            cv2.circle(img, (px, py), max(4, scale // 2), color, 2)

    return img


def draw_skeleton(
    img: np.ndarray,
    keypoints: np.ndarray,
    scale: int = RENDER_SCALE,
    pad: int = RENDER_PAD,
    skeleton_color: tuple = COLOR_SKELETON,
    joint_color: tuple = COLOR_JOINT,
    thickness: int = 2,
    joint_radius: int = 4,
) -> None:
    """
    Draw a skeleton on a board image (in-place).

    Accepts either full COCO (17, 2) or climbing-only (NUM_CLIMBING_KEYPOINTS, 2) keypoints.

    Args:
        img: Board image (BGR).
        keypoints: (17, 2) or (NUM_CLIMBING_KEYPOINTS, 2) array of board-space coordinates.
        scale: Pixels per board unit (must match render_board_image).
        pad: Padding in board units (must match render_board_image).
        skeleton_color: BGR color for skeleton lines.
        joint_color: BGR color for joint circles.
        thickness: Line thickness in pixels.
        joint_radius: Joint circle radius in pixels.
    """
    n_kp = keypoints.shape[0]
    skeleton = config.COCO_SKELETON if n_kp == 17 else config.CLIMBING_SKELETON

    for i, j in skeleton:
        pt1 = board_to_pixel(keypoints[i, 0], keypoints[i, 1], scale, pad)
        pt2 = board_to_pixel(keypoints[j, 0], keypoints[j, 1], scale, pad)
        cv2.line(img, pt1, pt2, skeleton_color, thickness)

    for kp in keypoints:
        pt = board_to_pixel(kp[0], kp[1], scale, pad)
        cv2.circle(img, pt, joint_radius, joint_color, -1)
        

def render_pose_video(
    poses: list[np.ndarray],
    output_path: Path,
    route_holds: list[dict] | None = None,
    fps: float = 30.0,
    title: str | None = None,
    scale: int = RENDER_SCALE,
    pad: int = RENDER_PAD,
    gt_poses: list[np.ndarray] | None = None,
) -> None:
    """
    Render a sequence of poses as a video with skeleton overlay on the board.

    Args:
        poses: List of (17, 2) arrays in board-space coordinates.
        output_path: Path for the output .mp4 file.
        route_holds: Route holds to highlight (from lookup_route).
        fps: Output video frame rate.
        title: Optional title text shown on the video.
        scale: Pixels per board unit.
        pad: Padding in board units.
        gt_poses: Optional list of (17, 2) GT arrays. If provided, drawn
            as a dimmer skeleton underneath the predicted skeleton.
    """
    all_holds = get_all_holds()
    board_img = render_board_image(route_holds, all_holds, scale, pad)
    h, w = board_img.shape[:2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    for frame_idx, pose in enumerate(poses):
        frame = board_img.copy()
        if gt_poses is not None and frame_idx < len(gt_poses):
            draw_skeleton(
                frame, gt_poses[frame_idx], scale, pad,
                skeleton_color=(80, 80, 80),
                joint_color=(60, 60, 60),
                thickness=1,
                joint_radius=3,
            )
        draw_skeleton(frame, pose, scale, pad)

        # Frame counter
        cv2.putText(
            frame, f"Frame {frame_idx}",
            (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (200, 200, 200), 1,
        )

        if title:
            cv2.putText(
                frame, title,
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1,
            )

        writer.write(frame)

    writer.release()
    print(f"Saved {len(poses)}-frame video to {output_path}")
    
    
def draw_target_hold(
    img: np.ndarray,
    target_board_xy: tuple[float, float],
    scale: int = RENDER_SCALE,
    pad: int = RENDER_PAD,
) -> None:
    """
    Draw target hold marker on a board image (in-place).

    Args:
        img: Board image (BGR).
        target_board_xy: (x, y) target hold in board coordinates.
        scale: Pixels per board unit.
        pad: Padding in board units.
    """
    tgt_px = board_to_pixel(target_board_xy[0], target_board_xy[1], scale, pad)
    # Bright ring
    cv2.circle(img, tgt_px, max(6, scale), (0, 200, 255), 3)
    # Inner dot
    cv2.circle(img, tgt_px, max(2, scale // 2), (0, 200, 255), -1)


def render_pose_video_with_targets(
    poses: list[np.ndarray],
    target_positions: np.ndarray,
    output_path: Path,
    route_holds: list[dict] | None = None,
    fps: float = 30.0,
    title: str | None = None,
    scale: int = RENDER_SCALE,
    pad: int = RENDER_PAD,
    gt_poses: list[np.ndarray] | None = None,
) -> None:
    """
    Render poses with target hold overlay.

    Args:
        poses: List of (17, 2) arrays in board-space coordinates.
        target_positions: (T, 2) board-space target hold positions per frame.
        output_path: Path for the output .mp4 file.
        route_holds: Route holds to highlight.
        fps: Output video frame rate.
        title: Optional title text.
        scale: Pixels per board unit.
        pad: Padding in board units.
        gt_poses: Optional list of (17, 2) GT arrays. If provided, drawn
            as a dimmer skeleton underneath the predicted skeleton.
    """
    all_holds = get_all_holds()
    board_img = render_board_image(route_holds, all_holds, scale, pad)
    h, w = board_img.shape[:2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    for frame_idx, pose in enumerate(poses):
        frame = board_img.copy()
        if gt_poses is not None and frame_idx < len(gt_poses):
            draw_skeleton(
                frame, gt_poses[frame_idx], scale, pad,
                skeleton_color=(80, 80, 80),
                joint_color=(60, 60, 60),
                thickness=1,
                joint_radius=3,
            )
        draw_skeleton(frame, pose, scale, pad)
        draw_target_hold(frame, target_positions[frame_idx], scale, pad)

        cv2.putText(
            frame, f"Frame {frame_idx}",
            (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (200, 200, 200), 1,
        )
        if title:
            cv2.putText(
                frame, title,
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1,
            )

        writer.write(frame)

    writer.release()
    print(f"Saved {len(poses)}-frame target video to {output_path}")


# ─── Inference helpers ────────────────────────────────────────────────────────

def autoregressive_rollout(
    model,
    seed_poses: np.ndarray,
    n_frames: int,
    hold_positions: np.ndarray,
    hold_roles: np.ndarray,
    device=None,
    max_bone_lengths: np.ndarray | None = None,
) -> list[np.ndarray]:
    """
    Run autoregressive rollout from a seed sequence.

    Args:
        model: PoseTransformer (or any model with predict_absolute).
        seed_poses: (context_len, 17, 2) initial ground truth poses.
        n_frames: Total number of frames to generate (including seed).
        hold_positions: (N, 2) normalized hold positions for the route.
        hold_roles: (N,) role IDs for each hold.
        device: Torch device. Inferred from model if None.
        max_bone_lengths: (n_bones,) max valid bone lengths. If provided,
            each prediction is clamped before being fed back as input,
            preventing skeleton explosion from error accumulation.

    Returns:
        List of n_frames (NUM_CLIMBING_KEYPOINTS, 2) arrays: seed frames followed by predictions.
    """
    import torch

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    context_len = model.context_len

    # Prepare hold tensors (same for every frame)
    n_holds = len(hold_positions)
    padded_pos = np.zeros((config.MAX_ROUTE_HOLDS, 2), dtype="float32")
    padded_roles = np.zeros(config.MAX_ROUTE_HOLDS, dtype="int64")
    mask = np.ones(config.MAX_ROUTE_HOLDS, dtype=bool)
    padded_pos[:n_holds] = hold_positions
    padded_roles[:n_holds] = hold_roles
    mask[:n_holds] = False

    h_pos_t = torch.from_numpy(padded_pos).unsqueeze(0).to(device)
    h_roles_t = torch.from_numpy(padded_roles).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)

    stride = config.ROLLOUT_STRIDE

    n_seed = min(context_len * stride, len(seed_poses))
    history = [seed_poses[t][config.CLIMBING_KEYPOINT_INDICES].reshape(-1).astype("float32") for t in range(n_seed)]
    all_poses = [seed_poses[t][config.CLIMBING_KEYPOINT_INDICES] for t in range(n_seed)]

    with torch.no_grad():
        while len(all_poses) < n_frames:
            indices = list(range(len(history) - context_len * stride, len(history), stride))
            indices = [max(0, i) for i in indices]
            context = torch.from_numpy(
                np.array([history[i] for i in indices])
            ).unsqueeze(0).to(device)

            pred_flat = model.predict_absolute(
                context, h_pos_t, h_roles_t, mask_t
            ).squeeze(0).cpu().numpy()
            pred_pose = pred_flat.reshape(config.NUM_CLIMBING_KEYPOINTS, 2)

            if max_bone_lengths is not None:
                pred_pose = enforce_bone_lengths(pred_pose, max_bone_lengths)
                pred_flat = pred_pose.reshape(-1)

            pred_flat = pred_pose.reshape(-1)
            history.append(pred_flat)

            prev_pose = all_poses[-1]
            for s in range(1, stride + 1):
                all_poses.append(prev_pose * (1 - s / stride) + pred_pose * (s / stride))

    return all_poses[:n_frames]

def autoregressive_rollout_structured(
    model,
    seed_poses: np.ndarray,
    n_frames: int,
    hold_positions: np.ndarray,
    hold_roles: np.ndarray,
    hold_sequence: list[dict],
    device=None,
    max_bone_lengths: np.ndarray | None = None,
    gt_poses: np.ndarray | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Run autoregressive rollout for the structured model variant.

    Advances the target hold based on the model's own predicted poses:
    when any limb endpoint stays within the arrival threshold for enough
    consecutive frames, the target advances to the next hold in sequence.

    Args:
        model: StructuredPoseTransformer.
        seed_poses: (context_len, 17, 2) initial ground truth poses.
        n_frames: Minimum number of frames to generate. Rollout continues
            beyond this until the hold sequence is exhausted, up to
            ROLLOUT_MAX_FRAMES.
        hold_positions: (N, 2) normalized hold positions for the route.
        hold_roles: (N,) role IDs for each hold.
        hold_sequence: Ordered list of hold dicts in board coordinates.
        device: Torch device.
        max_bone_lengths: (n_bones,) max valid bone lengths for clamping.
        gt_poses: Unused. Kept for call-site compatibility.

    Returns:
        Tuple of (poses, targets) where poses is a list of n_frames
        (NUM_CLIMBING_KEYPOINTS, 2) arrays and targets is (n_frames, 2)
        board-space target hold positions per frame.
    """
    import torch
    from pipeline.routes import normalize_board_coords

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    context_len = model.context_len

    n_holds = len(hold_positions)
    padded_pos = np.zeros((config.MAX_ROUTE_HOLDS, 2), dtype="float32")
    padded_roles = np.zeros(config.MAX_ROUTE_HOLDS, dtype="int64")
    mask = np.ones(config.MAX_ROUTE_HOLDS, dtype=bool)
    padded_pos[:n_holds] = hold_positions
    padded_roles[:n_holds] = hold_roles
    mask[:n_holds] = False

    h_pos_t = torch.from_numpy(padded_pos).unsqueeze(0).to(device)
    h_roles_t = torch.from_numpy(padded_roles).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)

    stride = config.ROLLOUT_STRIDE

    n_seed = min(context_len * stride, len(seed_poses))
    history = [seed_poses[t][config.CLIMBING_KEYPOINT_INDICES].reshape(-1).astype("float32") for t in range(n_seed)]
    all_poses = [seed_poses[t][config.CLIMBING_KEYPOINT_INDICES] for t in range(n_seed)]

    # Target hold tracking (prediction-based arrival detection)
    seq_idx = 0
    consecutive_near = 0
    arrival_needed = max(1, config.HOLD_ARRIVAL_FRAMES // stride)
    frames_on_current = 0

    initial_target = np.array([hold_sequence[0]["x"], hold_sequence[0]["y"]], dtype=np.float32)
    all_targets = [initial_target for _ in range(n_seed)]

    with torch.no_grad():
        while len(all_poses) < n_frames or seq_idx < len(hold_sequence):
            if len(all_poses) >= n_frames * 3:
                break  # defensive cap
            indices = list(range(len(history) - context_len * stride, len(history), stride))
            indices = [max(0, i) for i in indices]
            context = torch.from_numpy(
                np.array([history[i] for i in indices])
            ).unsqueeze(0).to(device)

            # Current target from hold sequence
            target_hold = hold_sequence[min(seq_idx, len(hold_sequence) - 1)]
            tgt_board = np.array([target_hold["x"], target_hold["y"]], dtype=np.float32)
            tgt_norm = normalize_board_coords(tgt_board.reshape(1, 2)).reshape(2)
            tgt_t = torch.from_numpy(tgt_norm).unsqueeze(0).to(device)

            pred_flat = model.predict_absolute(
                context, h_pos_t, h_roles_t, tgt_t, mask_t
            ).squeeze(0).cpu().numpy()
            pred_pose = pred_flat.reshape(config.NUM_CLIMBING_KEYPOINTS, 2)

            if max_bone_lengths is not None:
                pred_pose = enforce_bone_lengths(pred_pose, max_bone_lengths)

            pred_flat = pred_pose.reshape(-1)
            history.append(pred_flat)

            prev_pose = all_poses[-1]
            for s in range(1, stride + 1):
                all_poses.append(prev_pose * (1 - s / stride) + pred_pose * (s / stride))
                all_targets.append(tgt_board.copy())

            # Advance hold sequence based on predicted pose
            if seq_idx < len(hold_sequence):
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

    return all_poses, np.array(all_targets)