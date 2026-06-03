"""
Visualization: render predicted pose sequences as skeleton overlays on the board.

Provides:
    - Route lookup from the Kilter database
    - Board image rendering with holds and route highlights
    - Skeleton drawing in board coordinates
    - Video generation from pose sequences
"""

import math
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

import config
from models.world_model import enforce_bone_lengths, check_hand_arrival
from evaluation.metrics import _batch_procrustes_distances, _HIP_L, _HIP_R
from pipeline.routes import prepare_holds_for_model


# ─── Board rendering constants ───────────────────────────────────────────────

BOARD_EDGES = {"left": 0, "right": 144, "bottom": 0, "top": 156}
KICKBOARD_Y = 12

# Pixels per board unit in the rendered image
RENDER_SCALE = 8

# Padding around the board in board units
RENDER_PAD = 4

_HEAD_RADIUS_BU = config.RL_HEAD_RADIUS / config.RL_BOARD_UNIT_TO_METERS

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

from pipeline.routes import lookup_route_by_name


def lookup_route(climb_name: str, db_path: Path | None = None) -> dict:
    """Look up a climb by exact name. Raises ValueError if not found."""
    result = lookup_route_by_name(climb_name, db_path)
    if result is None:
        raise ValueError(f"No climb found exactly matching '{climb_name}'")
    return result


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
    head_pos: np.ndarray | None = None,
    draw_head=True,
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
        head_pos: When None and draw_head is True, auto-computed from
            shoulder midpoint for both 17-kp COCO and 12-kp climbing formats.
    """
    n_kp = keypoints.shape[0]
    skeleton = config.COCO_SKELETON if n_kp == 17 else config.CLIMBING_SKELETON

    # Head keypoint indices to skip for 17-kp COCO (nose, eyes, ears)
    head_kp = {0, 1, 2, 3, 4} if n_kp == 17 else set()

    for i, j in skeleton:
        if i in head_kp or j in head_kp:
            continue
        pt1 = board_to_pixel(keypoints[i, 0], keypoints[i, 1], scale, pad)
        pt2 = board_to_pixel(keypoints[j, 0], keypoints[j, 1], scale, pad)
        cv2.line(img, pt1, pt2, skeleton_color, thickness)

    for idx, kp in enumerate(keypoints):
        if idx in head_kp:
            continue
        pt = board_to_pixel(kp[0], kp[1], scale, pad)
        cv2.circle(img, pt, joint_radius, joint_color, -1)

    # Auto-compute head position when not explicitly provided
    if head_pos is None and draw_head:
        if n_kp == config.NUM_CLIMBING_KEYPOINTS:
            shoulder_mid = (keypoints[0] + keypoints[1]) / 2
        elif n_kp == 17:
            shoulder_mid = (keypoints[5] + keypoints[6]) / 2
        else:
            shoulder_mid = None
        if shoulder_mid is not None:
            head_pos = shoulder_mid + np.array([0, _HEAD_RADIUS_BU])

    if head_pos is not None and draw_head:
        pt = board_to_pixel(head_pos[0], head_pos[1], scale, pad)
        r_px = max(1, int(_HEAD_RADIUS_BU * scale))
        cv2.circle(img, pt, r_px, joint_color, 2)
        

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
    hand: str | None = None,
) -> None:
    """Draw target hold marker on a board image (in-place).

    Args:
        img: Board image (BGR).
        target_board_xy: (x, y) target hold in board coordinates.
        scale: Pixels per board unit.
        pad: Padding in board units.
        hand: ``"L"`` or ``"R"`` to color by hand. When None,
            uses the default orange-yellow.
    """
    COLOR_LEFT_HAND = (255, 150, 0)    # blue-ish (BGR)
    COLOR_RIGHT_HAND = (0, 100, 255)   # red-ish (BGR)
    COLOR_DEFAULT = (0, 200, 255)      # orange-yellow

    if hand == "L":
        color = COLOR_LEFT_HAND
    elif hand == "R":
        color = COLOR_RIGHT_HAND
    else:
        color = COLOR_DEFAULT

    tgt_px = board_to_pixel(target_board_xy[0], target_board_xy[1], scale, pad)
    cv2.circle(img, tgt_px, max(6, scale), color, 3)
    cv2.circle(img, tgt_px, max(2, scale // 2), color, -1)


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
    target_hands: list[str | None] | None = None,
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
        target_hands: Per-frame hand assignment ('L' or 'R') for coloring
            the target hold marker. When None, uses default color.
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
        hand = target_hands[frame_idx] if target_hands is not None else None
        draw_target_hold(frame, target_positions[frame_idx], scale, pad, hand=hand)

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
    
    
def draw_rl_frame(
    img: np.ndarray,
    keypoints: np.ndarray,
    scale: int = RENDER_SCALE,
    pad: int = RENDER_PAD,
    head_pos: np.ndarray | None = None,
    head_radius_bu: float | None = None,
    target_pos: tuple[float, float] | None = None,
    target_hand: str | None = None,
    cog_bu: np.ndarray | None = None,
    support_bu: np.ndarray | None = None,
    segment_cogs_bu: np.ndarray | None = None,
    show_segment_cogs: bool = False,
) -> None:
    """Draw skeleton with RL overlays on a board image (in-place).

    Draws stability overlay (behind), skeleton, head circle, and
    target hold marker.  Overlays are skipped when their data is None.

    Args:
        img: Board image (BGR).
        keypoints: (12, 2) climbing keypoints in board units.
        scale: Pixels per board unit.
        pad: Padding in board units.
        head_pos: (2,) head centre in board units.
        head_radius_bu: Head circle radius in board units.  When None
            a cosmetic default is used.  To get the physical radius,
            pass ``RL_HEAD_RADIUS / RL_BOARD_UNIT_TO_METERS``.
        target_pos: (x, y) target hold in board units.
        target_hand: ``"L"`` or ``"R"`` indicating which hand should
            grab the target.  Controls the marker color.
        cog_bu: (2,) overall centre of gravity in board units.
        support_bu: (N, 2) support polygon vertices in board units.
            Drawn as a line for 2 contacts, shaded polygon for 3+.
        segment_cogs_bu: (M, 2) per-segment CoGs in board units.
        show_segment_cogs: Draw per-segment CoG dots when True and
            *segment_cogs_bu* is provided.
    """
    # 1. Support polygon (behind everything)
    if support_bu is not None and len(support_bu) >= 2:
        pts_px = np.array(
            [board_to_pixel(p[0], p[1], scale, pad) for p in support_bu],
            dtype=np.int32,
        )
        if len(support_bu) == 2:
            cv2.line(img, tuple(pts_px[0]), tuple(pts_px[1]),
                     (0, 0, 200), 2)
        else:
            ordered_px = pts_px.reshape(-1, 1, 2)
            overlay = img.copy()
            cv2.fillPoly(overlay, [ordered_px], (0, 0, 180))
            cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
            cv2.polylines(img, [ordered_px], isClosed=True,
                          color=(0, 0, 200), thickness=2)
        for pt in pts_px:
            cv2.circle(img, tuple(pt), 5, (0, 0, 220), -1)

    # 2. Per-segment CoGs
    if show_segment_cogs and segment_cogs_bu is not None:
        for seg_cog in segment_cogs_bu:
            px = board_to_pixel(seg_cog[0], seg_cog[1], scale, pad)
            cv2.circle(img, px, 3, (255, 255, 255), -1)

    # 3. Overall CoG
    if cog_bu is not None:
        cog_px = board_to_pixel(cog_bu[0], cog_bu[1], scale, pad)
        cv2.circle(img, cog_px, 6, (0, 0, 255), -1)
        cv2.circle(img, cog_px, 6, (255, 255, 255), 1)

    # 4. Skeleton (without head — drawn separately with physical radius)
    draw_skeleton(img, keypoints, scale, pad, draw_head=False)

    # 5. Head circle
    if head_pos is not None:
        pt = board_to_pixel(head_pos[0], head_pos[1], scale, pad)
        if head_radius_bu is not None:
            r_px = max(1, int(head_radius_bu * scale))
        else:
            r_px = max(6, int(scale * 0.4))
        cv2.circle(img, pt, r_px, COLOR_JOINT, 2)

    # 6. Target hold marker
    if target_pos is not None:
        draw_target_hold(img, target_pos, scale, pad, hand=target_hand)


def _interpolate_rl_data(poses, heads, targets, cogs, supports,
                         segment_cogs, substeps):
    """Linearly interpolate per-step RL data for smooth video playback."""
    def _lerp_lists(a_list, b_list, substeps):
        out = []
        for i in range(len(a_list) - 1):
            for s in range(substeps):
                alpha = s / substeps
                out.append(a_list[i] * (1 - alpha) + b_list[i + 1] * alpha)
        out.append(a_list[-1])
        return out

    def _snap_list(lst, substeps):
        out = []
        for i in range(len(lst) - 1):
            out.extend([lst[i]] * substeps)
        out.append(lst[-1])
        return out

    sm_poses = _lerp_lists(poses, poses, substeps)
    sm_heads = _lerp_lists(heads, heads, substeps) if heads else None
    sm_targets = _snap_list(targets, substeps) if targets else None
    sm_cogs = _lerp_lists(cogs, cogs, substeps) if cogs else None

    sm_supports = None
    if supports:
        sm_supports = []
        for i in range(len(supports) - 1):
            a, b = supports[i], supports[i + 1]
            can_lerp = len(a) == len(b) and len(a) > 0
            for s in range(substeps):
                alpha = s / substeps
                sm_supports.append(
                    a * (1 - alpha) + b * alpha if can_lerp else a,
                )
        sm_supports.append(supports[-1])

    sm_seg = _lerp_lists(segment_cogs, segment_cogs, substeps) \
        if segment_cogs else None

    return sm_poses, sm_heads, sm_targets, sm_cogs, sm_supports, sm_seg


def render_rl_video(
    poses: list[np.ndarray],
    output_path: Path,
    route_holds: list[dict] | None = None,
    fps: float = 30.0,
    title: str | None = None,
    head_positions: list[np.ndarray] | None = None,
    head_radius_bu: float | None = None,
    target_positions: list[tuple[float, float]] | None = None,
    target_hands: list[str] | None = None,
    cog_positions: list[np.ndarray] | None = None,
    support_polygons: list[np.ndarray] | None = None,
    segment_cog_positions: list[np.ndarray] | None = None,
    show_segment_cogs: bool = False,
    board_y_min: int | None = None,
    scale: int = RENDER_SCALE,
    pad: int = RENDER_PAD,
    interpolate: tuple[int, int] | None = None,
    reward_breakdowns: list[dict[str, float]] | None = None,
) -> None:
    """Render an RL episode as a skeleton overlay video.

    Combines the skeleton, head circle, target hold marker, CoG dot,
    and support polygon into a single video.  All overlay data is
    optional — omit any list to skip that overlay.

    Args:
        poses: List of (12, 2) keypoint arrays in board units.
        output_path: Path for the output .mp4 file.
        route_holds: Route holds to highlight on the board.
        fps: Output video frame rate.
        title: Optional title text shown on each frame.
        head_positions: Per-frame (2,) head centre in board units.
        head_radius_bu: Head circle radius in board units.
        target_positions: Per-frame (x, y) target hold position.
        cog_positions: Per-frame (2,) overall CoG in board units.
        support_polygons: Per-frame (N, 2) support polygon vertices.
        segment_cog_positions: Per-frame (M, 2) segment CoGs.
        show_segment_cogs: Draw per-segment CoG dots.
        board_y_min: Extend the viewport to this board-unit y value
            (e.g. ``RL_GROUND_Y - 5`` to show ground-level feet).
        scale: Pixels per board unit.
        pad: Padding in board units.
        interpolate: ``(physics_hz, control_hz)`` tuple.  When provided,
            linearly interpolates all per-frame data to produce
            ``physics_hz / control_hz`` smooth frames per input frame.
            The *fps* should then be set to *physics_hz* for real-time
            playback.
        reward_breakdowns: Per-step reward component dicts with keys
            ``step``, ``hand_prox``, ``foot_prox``, ``contact``,
            ``grab_bonus``.  One fewer entry than *poses* (no
            breakdown for the initial reset frame).  Overlaid as
            text on each video frame when provided.
    """
    original_bottom = BOARD_EDGES["bottom"]
    if board_y_min is not None:
        BOARD_EDGES["bottom"] = int(min(original_bottom, board_y_min))

    try:
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
        
        if interpolate is not None:
            phys_hz, ctrl_hz = interpolate
            substeps = phys_hz // ctrl_hz
            poses, head_positions, target_positions, cog_positions, \
                support_polygons, segment_cog_positions = _interpolate_rl_data(
                    poses, head_positions, target_positions, cog_positions,
                    support_polygons, segment_cog_positions, substeps,
                )
            if reward_breakdowns:
                interp_bd = []
                for bd in reward_breakdowns:
                    interp_bd.extend([bd] * substeps)
                reward_breakdowns = interp_bd
            if target_hands:
                snapped = []
                for th in target_hands[:-1]:
                    snapped.extend([th] * substeps)
                snapped.append(target_hands[-1])
                target_hands = snapped

        for i, pose in enumerate(poses):
            frame = board_img.copy()

            draw_rl_frame(
                frame, pose, scale, pad,
                head_pos=(head_positions[i]
                          if head_positions and i < len(head_positions)
                          else None),
                head_radius_bu=head_radius_bu,
                target_pos=(target_positions[i]
                            if target_positions and i < len(target_positions)
                            else None),
                target_hand=(target_hands[i]
                             if target_hands and i < len(target_hands)
                             else None),
                cog_bu=(cog_positions[i]
                        if cog_positions and i < len(cog_positions)
                        else None),
                support_bu=(support_polygons[i]
                            if support_polygons and i < len(support_polygons)
                            else None),
                segment_cogs_bu=(segment_cog_positions[i]
                                 if segment_cog_positions
                                 and i < len(segment_cog_positions)
                                 else None),
                show_segment_cogs=show_segment_cogs,
            )

            if title:
                cv2.putText(frame, title, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 1)
            cv2.putText(frame, f"Frame {i}", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1)
            
            if reward_breakdowns is not None:
                # First pose is the reset frame (no breakdown yet)
                bd_idx = i - 1 if not interpolate else i - (interpolate[0] // interpolate[1])
                if 0 <= bd_idx < len(reward_breakdowns):
                    bd = reward_breakdowns[bd_idx]
                    cumulative_reward = bd.get("cumulative", 0.0)
                    step_r = (bd.get("step", 0) + bd.get("hand_prox", 0)
                              + bd.get("foot_prox", 0) + bd.get("contact", 0)
                              + bd.get("grab_bonus", 0) + bd.get("fall_penalty", 0)
                              + bd.get("timeout_penalty", 0))
                    remaining = int(bd.get("steps_remaining", 0))
                    lines = [
                        f"Total: {step_r:+.3f}  Cumulative: {cumulative_reward:+.1f}",
                        f"hand:{bd.get('hand_prox',0):+.3f}  foot:{bd.get('foot_prox',0):+.3f}",
                        f"stability:{bd.get('contact',0):+.3f}  step:{bd.get('step',0):.3f}",
                        f"Target budget: {remaining}/{config.RL_STEPS_PER_TARGET}",
                    ]
                    grab = bd.get("grab_bonus", 0)
                    fall = bd.get("fall_penalty", 0)
                    if grab > 0:
                        lines.append(f"GRAB BONUS: +{grab:.1f}")
                    if fall < 0:
                        lines.append(f"FALL PENALTY: {fall:.1f}")

                    for j, line in enumerate(lines):
                        cv2.putText(frame, line, (10, h - 30 - (len(lines) - 1 - j) * 18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    (180, 255, 180), 1)

            writer.write(frame)

        writer.release()
        print(f"Saved {len(poses)}-frame RL video to {output_path}")

    finally:
        BOARD_EDGES["bottom"] = original_bottom


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
    h_pos_t, h_roles_t, mask_t = prepare_holds_for_model(hold_positions, hold_roles, device)

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

    h_pos_t, h_roles_t, mask_t = prepare_holds_for_model(hold_positions, hold_roles, device)

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


def plot_nn_pose_comparison(
    method_poses: dict[str, np.ndarray],
    bank: np.ndarray,
    bank_norm: np.ndarray,
    output_path: Path | None = None,
    alignment: str = "procrustes",
) -> "plt.Figure":
    """
    Presentation figure: for each method, show a sample pose, its nearest
    GT neighbor from the bank, and both overlaid after Procrustes alignment.

    Produces a (n_methods × 3) grid:
        Col 1: predicted pose (hip-centered)
        Col 2: nearest GT neighbor (hip-centered)
        Col 3: overlay after Procrustes alignment, with distance annotation

    All poses are drawn in climbing-keypoint space with no board context,
    centered at the origin.

    Args:
        method_poses: Maps method name → (12, 2) single sample pose in
            climbing-keypoint space (NOT hip-centered — this function centers).
        bank: (N, 12, 2) hip-centered reference bank from build_pose_bank.
        bank_norm: (N, 12, 2) unit-normalized bank from build_pose_bank.
        output_path: Save figure here. If None, calls plt.show().
        alignment: 'procrustes' for rotation+scale-aligned overlay, or
    """
    methods = list(method_poses.keys())
    n = len(methods)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = [
        "Predicted (hip-centered)",
        "Nearest GT neighbor",
        "Overlay (Procrustes)" if alignment == "procrustes" else "Overlay (raw)",
    ]

    for row, name in enumerate(methods):
        pose = method_poses[name]
        hip = pose[[_HIP_L, _HIP_R]].mean(axis=0)
        centered = pose - hip

        if alignment == "procrustes":
            dists = _batch_procrustes_distances(centered, bank_norm)
            best_idx = dists.argmin()
            best_dist = dists[best_idx]
            best_ref = bank[best_idx]
            # Compute aligned query for overlay
            q_norm = np.linalg.norm(centered)
            r_norm = np.linalg.norm(best_ref)
            if q_norm > 1e-8 and r_norm > 1e-8:
                q = centered / q_norm
                r = best_ref / r_norm
                U, _, Vt = np.linalg.svd(r.T @ q)
                d_sign = np.linalg.det(U @ Vt)
                D = np.diag([1.0, np.sign(d_sign)])
                R = U @ D @ Vt
                best_overlay = (q @ R.T) * r_norm
            else:
                best_overlay = centered
        else:
            dists = np.linalg.norm(
                bank.reshape(len(bank), -1) - centered.ravel(), axis=1
            )
            best_idx = dists.argmin()
            best_dist = dists[best_idx]
            best_ref = bank[best_idx]
            best_overlay = centered

        # --- Draw ---
        panels = [
            (centered, None, None),
            (best_ref, None, None),
            (best_ref, best_overlay, best_dist),
        ]

        for col, (primary, overlay, dist) in enumerate(panels):
            ax = axes[row, col]
            _draw_skeleton_mpl(ax, primary, color="steelblue",
                               label="GT neighbor" if col == 2 else None)
            if overlay is not None:
                _draw_skeleton_mpl(ax, overlay, color="tomato", alpha=0.8,
                                   label="Predicted (aligned)" if alignment == "procrustes"
                                   else "Predicted (hip-centered)")
                ax.legend(fontsize=7, loc="upper right")
            if col == 0:
                _draw_skeleton_mpl(ax, centered, color="tomato")

            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                ax.set_title(col_titles[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(name, fontsize=11, fontweight="bold")
            if dist is not None:
                ax.annotate(
                    f"d = {dist:.3f}",
                    xy=(0.5, 0.02), xycoords="axes fraction",
                    ha="center", fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                )

    fig.suptitle(
        f"NN Pose Distance — {'Procrustes' if alignment == 'procrustes' else 'Raw (hip-centered)'}",
        fontsize=13, fontweight="bold",
    )

    fig.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved NN pose comparison to {output_path}")

    return fig


def _draw_skeleton_mpl(
    ax: plt.Axes,
    pose: np.ndarray,
    color: str = "steelblue",
    alpha: float = 1.0,
    label: str | None = None,
) -> None:
    """Draw a climbing skeleton on a matplotlib axes.

    Args:
        ax: Matplotlib axes.
        pose: (12, 2) climbing-keypoint array.
        color: Line and joint color.
        alpha: Opacity.
        label: Legend label (applied to first bone only).
    """
    first = True
    for i, j in config.CLIMBING_SKELETON:
        ax.plot(
            [pose[i, 0], pose[j, 0]],
            [pose[i, 1], pose[j, 1]],
            color=color, alpha=alpha, linewidth=2,
            label=label if first else None,
        )
        first = False
    ax.scatter(pose[:, 0], pose[:, 1], color=color, alpha=alpha, s=20, zorder=3)
    

# ─── Comparison visualization ────────────────────────────────────────────────

# Colors for method trajectories (BGR for OpenCV)
TRAJECTORY_COLORS = {
    "Ground Truth": (255, 255, 255),   # white
    "Hands-Only":   (0, 165, 255),     # orange
    "World Model":  (255, 100, 0),     # blue
    "RL":           (0, 255, 100),     # green
}


def render_trajectory_comparison_image(
    method_sequences: dict[str, list[np.ndarray]],
    route_holds: list[dict],
    output_path: Path,
    centroid_indices: list[int] | None = None,
    scale: int = RENDER_SCALE,
    pad: int = RENDER_PAD,
) -> None:
    """
    Render the board with route holds and overlaid torso-centroid trajectories.

    Draws each method's trajectory as a colored polyline on top of the
    standard board rendering (grid + background holds + highlighted route holds).

    Args:
        method_sequences: Maps method label → list of (12, 2) climbing-keypoint
            poses. Keys should match TRAJECTORY_COLORS.
        route_holds: Route hold dicts with 'x', 'y', 'role_id'.
        output_path: Save the image here (PNG).
        centroid_indices: Keypoint indices to average for centroid path.
            Defaults to config.TORSO_CENTROID_INDICES.
        scale: Pixels per board unit.
        pad: Padding in board units.
    """
    if centroid_indices is None:
        centroid_indices = config.TORSO_CENTROID_INDICES

    all_holds = get_all_holds()
    img = render_board_image(route_holds, all_holds, scale, pad)

    for label, poses in method_sequences.items():
        color = TRAJECTORY_COLORS.get(label, (200, 200, 200))
        centroids = [pose[centroid_indices].mean(axis=0) for pose in poses]
        for k in range(len(centroids) - 1):
            pt1 = board_to_pixel(centroids[k][0], centroids[k][1], scale, pad)
            pt2 = board_to_pixel(centroids[k + 1][0], centroids[k + 1][1], scale, pad)
            cv2.line(img, pt1, pt2, color, 2)

    # Legend
    h = img.shape[0]
    y = 30
    for label, color in TRAJECTORY_COLORS.items():
        if label in method_sequences:
            cv2.line(img, (10, y), (40, y), color, 3)
            cv2.putText(img, label, (50, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += 25

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    print(f"Saved trajectory comparison to {output_path}")
    
    
def render_path_aligned_comparison_video(
    method_sequences: dict[str, list[np.ndarray]],
    output_path: Path,
    fps: float = 10.0,
    n_frames: int = 200,
    align: str = "procrustes",
) -> None:
    """
    Render side-by-side Procrustes-aligned poses synchronized by path progress.

    All sequences are normalized by their centroid arc length so that
    at each video frame, every panel shows the pose at the same fraction
    of climb completion. Each panel shows the Procrustes-aligned pose
    on a white background.

    Args:
        method_sequences: Maps label → list of (12, 2) poses. Should
            include "Ground Truth" plus prediction methods.
        output_path: Save the MP4 here.
        fps: Output frame rate.
        n_frames: Number of video frames (progress steps from 0 to 1).
        align: Per-frame spatial alignment mode for non-GT panels.
    """
    from evaluation.metrics import _compute_path_progress, _align_pose_pair

    labels = list(method_sequences.keys())
    n_panels = len(labels)

    # Pre-compute progress arrays and find GT
    progress = {
        label: _compute_path_progress(seq)
        for label, seq in method_sequences.items()
    }

    gt_label = "Ground Truth"
    gt_seq = method_sequences[gt_label]
    gt_progress = progress[gt_label]

    panel_w, panel_h = 250, 400
    total_w = panel_w * n_panels
    header = 30

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (total_w, panel_h + header),
    )

    sample_points = np.linspace(0.0, 1.0, n_frames)

    for step, p in enumerate(sample_points):
        frame = np.full((panel_h + header, total_w, 3), 255, dtype=np.uint8)

        # Find GT frame at this progress
        gi = int(np.argmin(np.abs(gt_progress - p)))

        for col, label in enumerate(labels):
            seq = method_sequences[label]
            pi = int(np.argmin(np.abs(progress[label] - p)))

            if label == gt_label:
                pose = _center_pose_for_panel(seq[pi])
                color = (100, 100, 100)
            else:
                ap, ag = _align_pose_pair(seq[pi], gt_seq[gi], align)
                # Draw GT match (gray) then prediction (colored)
                _draw_panel_skeleton(
                    frame, ag, col=col, panel_w=panel_w,
                    panel_h=panel_h, margin=header, color=(200, 200, 200),
                )
                pose = ap
                color = TRAJECTORY_COLORS.get(label, (200, 100, 0))

            _draw_panel_skeleton(
                frame, pose, col=col, panel_w=panel_w,
                panel_h=panel_h, margin=header, color=color, label=label,
            )

        pct = int(p * 100)
        cv2.putText(
            frame, f"{pct}%",
            (total_w - 60, panel_h + header - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1,
        )
        writer.write(frame)

    writer.release()
    print(f"Saved path-aligned comparison video ({n_frames} frames) to {output_path}")
    
    
def _center_pose_for_panel(pose: np.ndarray) -> np.ndarray:
    """Hip-center a (12,2) pose for panel rendering."""
    hip = (pose[_HIP_L] + pose[_HIP_R]) / 2
    return pose - hip


def _draw_panel_skeleton(
    frame: np.ndarray,
    pose: np.ndarray,
    col: int,
    panel_w: int,
    panel_h: int,
    margin: int,
    color: tuple,
    label: str | None = None,
) -> None:
    """Draw a climbing skeleton into a specific panel of the comparison frame.

    Poses are in centered board units; mapped to panel pixel coords with
    a fixed scale and centered in the panel.

    Args:
        frame: Full video frame (modified in-place).
        pose: (12, 2) centered climbing-keypoint pose.
        col: Panel column index (0-based).
        panel_w: Panel width in pixels.
        panel_h: Panel height in pixels.
        margin: Top margin in pixels (for labels).
        color: BGR color.
        label: If provided, drawn as the panel header.
    """
    # Map board units → panel pixels (20 px per board unit, centered)
    px_scale = 6.0
    cx = col * panel_w + panel_w // 2
    cy = margin + panel_h // 2

    def to_px(pt):
        return (int(cx + pt[0] * px_scale), int(cy - pt[1] * px_scale))

    for i, j in config.CLIMBING_SKELETON:
        cv2.line(frame, to_px(pose[i]), to_px(pose[j]), color, 2)
    for kp in pose:
        cv2.circle(frame, to_px(kp), 3, color, -1)

    if label:
        cv2.putText(
            frame, label,
            (col * panel_w + 10, margin - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1,
        )
        
        
def render_2x2_comparison_video(
    panels: list[dict],
    route_holds: list[dict],
    output_path: Path,
    fps: float = 30.0,
    n_frames: int | None = None,
    scale: int = 4,
    pad: int = RENDER_PAD,
) -> None:
    """
    Render a 2x2 grid video comparing 4 methods on the same climb.

    Each panel shows the board with route holds, a skeleton overlay,
    and an optional target hold marker. All sequences are linearly
    time-normalized to the same video length.

    Args:
        panels: List of exactly 4 dicts in order
            [top-left, top-right, bottom-left, bottom-right], each with:
            'label': str — method name drawn on the panel.
            'poses': list of (K, 2) arrays in board-space coordinates.
            'targets': optional (T, 2) array of per-frame target positions.
            'target_hands': optional list of 'L'/'R'/None per frame.
        route_holds: Route hold dicts for board rendering.
        output_path: Save the MP4 here.
        fps: Output frame rate.
        n_frames: Video length. Defaults to the first panel's sequence length.
        scale: Pixels per board unit for each panel.
        pad: Padding in board units.
    """
    assert len(panels) == 4, f"Expected 4 panels, got {len(panels)}"

    if n_frames is None:
        n_frames = len(panels[0]["poses"])

    all_holds = get_all_holds()
    board_img = render_board_image(route_holds, all_holds, scale, pad)
    panel_h, panel_w = board_img.shape[:2]

    label_h = 30
    cell_h = panel_h + label_h
    total_h = cell_h * 2
    total_w = panel_w * 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (total_w, total_h),
    )

    for frame_idx in range(n_frames):
        grid = np.full((total_h, total_w, 3), 40, dtype=np.uint8)

        for p_idx, panel in enumerate(panels):
            row, col = divmod(p_idx, 2)
            poses = panel["poses"]
            src_idx = min(int(frame_idx * len(poses) / n_frames), len(poses) - 1)

            cell = board_img.copy()
            draw_skeleton(cell, poses[src_idx], scale, pad)

            targets = panel.get("targets")
            target_hands = panel.get("target_hands")
            if targets is not None and src_idx < len(targets):
                tgt = targets[src_idx]
                if not (np.isnan(tgt[0]) or np.isnan(tgt[1])):
                    hand = (target_hands[src_idx]
                            if target_hands and src_idx < len(target_hands)
                            else None)
                    draw_target_hold(cell, tgt, scale, pad, hand=hand)

            x0 = col * panel_w
            y0 = row * cell_h + label_h
            grid[y0:y0 + panel_h, x0:x0 + panel_w] = cell

            cv2.putText(
                grid, panel["label"],
                (x0 + 10, row * cell_h + label_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
            )

        writer.write(grid)

    writer.release()
    print(f"Saved 2x2 comparison video ({n_frames} frames) to {output_path}")
    
    
def interpolate_pose_sequence(
    poses: list[np.ndarray],
    factor: int,
    targets: np.ndarray | None = None,
    target_hands: list[str | None] | None = None,
) -> tuple[list[np.ndarray], np.ndarray | None, list[str | None] | None]:
    """
    Linearly interpolate a pose sequence for smoother playback.

    Inserts factor-1 intermediate frames between each consecutive pair
    by linear interpolation. Targets and target hands are held constant
    (snapped, not interpolated) across sub-steps.

    Args:
        poses: List of (K, 2) pose arrays.
        factor: Number of output frames per input frame.
        targets: Optional (T, 2) per-frame target positions.
        target_hands: Optional per-frame hand labels.

    Returns:
        (interpolated_poses, interpolated_targets, interpolated_hands).
    """
    if factor <= 1 or len(poses) < 2:
        return poses, targets, target_hands

    out_poses = []
    for i in range(len(poses) - 1):
        for s in range(factor):
            alpha = s / factor
            out_poses.append(poses[i] * (1 - alpha) + poses[i + 1] * alpha)
    out_poses.append(poses[-1])

    out_targets = None
    if targets is not None:
        out_targets = np.repeat(targets, factor, axis=0)[:len(out_poses)]

    out_hands = None
    if target_hands is not None:
        out_hands = [h for h in target_hands for _ in range(factor)][:len(out_poses)]

    return out_poses, out_targets, out_hands