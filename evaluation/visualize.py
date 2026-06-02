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
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

import config
from models.world_model import enforce_bone_lengths, check_hand_arrival
from evaluation.metrics import _batch_procrustes_distances, _HIP_L, _HIP_R



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
        head_pos: Optional (2,) board-space position to draw a head circle.
            Used by the RL baseline; ignored when None.
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
        
    # Auto-compute head for climbing-keypoint poses
    if head_pos is None and draw_head and n_kp == config.NUM_CLIMBING_KEYPOINTS:
        shoulder_mid = (keypoints[0] + keypoints[1]) / 2
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