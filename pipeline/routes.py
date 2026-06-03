"""
Route utilities: decode Kilter DB climbs into hold positions and roles.

Used by the dataset pipeline to attach route context to each training sequence,
and by the visualization tools to look up climbs by name or UUID.
"""

import json
import re
import sqlite3
from pathlib import Path
import torch

import numpy as np

import config

# Board dimensions for normalizing hold positions to [0, 1]
BOARD_X_MIN = 0
BOARD_X_MAX = 144
BOARD_Y_MIN = 0
BOARD_Y_MAX = 156


def decode_frames_string(
    frames_str: str,
    conn: sqlite3.Connection,
    layout_id: int = 1,
) -> list[dict]:
    """
    Decode a climbs.frames string into a list of hold dicts.

    Args:
        frames_str: Raw frames encoding like 'p1145r12p1216r13...'.
        conn: Open sqlite3 connection to kilter.db.
        layout_id: Layout to filter placements by.

    Returns:
        List of {'x': float, 'y': float, 'name': str, 'role_id': int} dicts.
    """
    pairs = re.findall(r"p(\d+)r(\d+)", frames_str)
    holds = []

    for placement_id, role_id in pairs:
        row = conn.execute("""
            SELECT h.x, h.y, h.name
            FROM placements p
            JOIN holes h ON p.hole_id = h.id
            WHERE p.id = ? AND p.layout_id = ?
        """, (int(placement_id), layout_id)).fetchone()

        if row:
            holds.append({
                "x": row[0],
                "y": row[1],
                "name": row[2],
                "role_id": int(role_id),
            })

    return holds


def lookup_route_by_uuid(
    climb_uuid: str,
    db_path: Path | None = None,
) -> dict | None:
    """
    Look up a route by its climb UUID.

    Args:
        climb_uuid: The climb's UUID from the Kilter database.
        db_path: Path to kilter.db.

    Returns:
        Dict with 'name', 'holds', 'grade', or None if not found.
    """
    db = db_path or (config.DATA_DIR / "kilter.db")
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row

    row = conn.execute(
        "SELECT name, frames FROM climbs WHERE uuid = ?",
        (climb_uuid,),
    ).fetchone()

    if row is None:
        conn.close()
        return None

    holds = decode_frames_string(row["frames"], conn)

    grade_row = conn.execute("""
        SELECT dg.boulder_name
        FROM climb_stats cs
        JOIN difficulty_grades dg ON dg.difficulty = CAST(cs.display_difficulty AS INT)
        WHERE cs.climb_uuid = ? AND cs.angle = 30
        LIMIT 1
    """, (climb_uuid,)).fetchone()

    conn.close()

    return {
        "name": row["name"],
        "holds": holds,
        "grade": grade_row["boulder_name"] if grade_row else None,
    }


def lookup_route_by_name(
    climb_name: str,
    db_path: Path | None = None,
) -> dict | None:
    """
    Look up a route by exact name (case-sensitive, matches check_problem_names.py logic).
    Normalizes quotes, strips whitespace, and requires an exact match.
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
        return None

    holds = decode_frames_string(row["frames"], conn)

    grade_row = conn.execute("""
        SELECT dg.boulder_name
        FROM climb_stats cs
        JOIN difficulty_grades dg ON dg.difficulty = CAST(cs.display_difficulty AS INT)
        WHERE cs.climb_uuid = ? AND cs.angle = 30
        LIMIT 1
    """, (row["uuid"],)).fetchone()

    conn.close()

    return {
        "name": row["name"],
        "holds": holds,
        "grade": grade_row["boulder_name"] if grade_row else None,
    }


def holds_to_array(
    holds: list[dict],
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert hold dicts to arrays for model input.

    Args:
        holds: List of hold dicts from decode_frames_string.
        normalize: If True, normalize x/y to [0, 1] range.

    Returns:
        (positions, roles) where:
            positions: (N, 2) float32 array of (x, y)
            roles: (N,) int array of role IDs
    """
    positions = np.array([[h["x"], h["y"]] for h in holds], dtype=np.float32)
    roles = np.array([config.ROLE_ID_MAP.get(h["role_id"], 13) for h in holds], dtype=np.int64)

    if normalize and len(positions) > 0:
        positions[:, 0] = (positions[:, 0] - BOARD_X_MIN) / (BOARD_X_MAX - BOARD_X_MIN)
        positions[:, 1] = (positions[:, 1] - BOARD_Y_MIN) / (BOARD_Y_MAX - BOARD_Y_MIN)

    return positions, roles


def pad_holds(
    positions: np.ndarray,
    roles: np.ndarray,
    max_holds: int = config.MAX_ROUTE_HOLDS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pad hold arrays to a fixed length with a boolean mask.

    Args:
        positions: (N, 2) hold positions.
        roles: (N,) role IDs.
        max_holds: Pad to this length.

    Returns:
        (padded_positions, padded_roles, mask) where mask is True for padding.
    """
    n = len(positions)
    padded_pos = np.zeros((max_holds, 2), dtype=np.float32)
    padded_roles = np.zeros(max_holds, dtype=np.int64)
    mask = np.ones(max_holds, dtype=bool)
    padded_pos[:n] = positions
    padded_roles[:n] = roles
    mask[:n] = False
    return padded_pos, padded_roles, mask


def prepare_holds_for_model(
    positions: np.ndarray,
    roles: np.ndarray,
    device: torch.device,
    max_holds: int = config.MAX_ROUTE_HOLDS,
) -> tuple:
    """Pad, batch, and move hold arrays to device for single-sample inference.

    Returns:
        (h_pos, h_roles, mask) each as (1, max_holds, ...) tensors on device.
    """
    padded_pos, padded_roles, mask = pad_holds(positions, roles, max_holds)
    return (
        torch.from_numpy(padded_pos).unsqueeze(0).to(device),
        torch.from_numpy(padded_roles).unsqueeze(0).to(device),
        torch.from_numpy(mask).unsqueeze(0).to(device),
    )


def normalize_board_coords(coords: np.ndarray) -> np.ndarray:
    """
    Normalize board-space coordinates to [0, 1].

    Args:
        coords: (..., 2) array with board-space (x, y).

    Returns:
        Copy with x and y normalized to [0, 1].
    """
    out = coords.copy()
    out[..., 0] = (out[..., 0] - BOARD_X_MIN) / (BOARD_X_MAX - BOARD_X_MIN)
    out[..., 1] = (out[..., 1] - BOARD_Y_MIN) / (BOARD_Y_MAX - BOARD_Y_MIN)
    return out


def apply_route_edits(
    holds: list[dict],
    video_stem: str,
    edits_dir: Path | None = None,
) -> list[dict]:
    """
    Filter out manually excluded holds for a specific video.

    Loads data/route_edits/{video_stem}_route_edit.json if it exists
    and removes holds whose 'name' appears in 'excluded_holds'.

    Args:
        holds: List of hold dicts from decode_frames_string.
        video_stem: Video filename without extension.
        edits_dir: Directory containing edit JSONs.
            Defaults to config.ROUTE_EDITS_DIR.

    Returns:
        Filtered list of hold dicts (unmodified if no edit file exists).
    """
    edits_dir = edits_dir or config.ROUTE_EDITS_DIR
    edit_path = edits_dir / f"{video_stem}_route_edit.json"

    if not edit_path.exists():
        # Search subdirectories
        matches = list(edits_dir.rglob(f"{video_stem}_route_edit.json"))
        if not matches:
            return holds
        edit_path = matches[0]

    with open(edit_path) as f:
        edits = json.load(f)

    excluded = set(edits.get("excluded_holds", []))
    if not excluded:
        return holds

    return [h for h in holds if h["name"] not in excluded]

def load_hold_order_edit(
    video_stem: str,
    orders_dir: Path | None = None,
) -> dict | None:
    """
    Load a manual hold order/timing override for a video, if one exists.

    Looks for data/hold_orders/{video_stem}_hold_order.json, searching
    subdirectories as a fallback (matching apply_route_edits).

    Args:
        video_stem: Video filename without extension.
        orders_dir: Directory containing override JSONs.
            Defaults to config.HOLD_ORDERS_DIR.

    Returns:
        The parsed override dict, or None if no override file exists.
    """
    orders_dir = orders_dir or config.HOLD_ORDERS_DIR
    edit_path = orders_dir / f"{video_stem}_hold_order.json"

    if not edit_path.exists():
        matches = list(orders_dir.rglob(f"{video_stem}_hold_order.json"))
        if not matches:
            return None
        edit_path = matches[0]

    with open(edit_path) as f:
        return json.load(f)


def apply_hold_order_edit(
    override: dict,
    route_holds: list[dict],
    num_frames: int,
) -> tuple[list[dict], np.ndarray]:
    """
    Expand a hold order/timing override into an ordered hold sequence and
    per-frame target positions.

    The override stores an ordered list of segments, each naming a route
    hold, the pose-frame index at which it becomes the active target, and
    which hand (L/R) reaches it. Segments are contiguous: segment i is
    the target from its start_frame until the next segment's start_frame;
    the last runs to the end. The earliest segment is forced to cover
    from frame 0.

    Hold names that no longer match any route hold (e.g. excluded by a
    later route edit) are skipped. Start frames are rescaled proportionally
    when the override's stored frame count differs from num_frames, so an
    override survives a dataset rebuild that changes the cleaned frame count.

    Args:
        override: Parsed override from load_hold_order_edit, with keys
            'sequence' (list of {'name': str, 'start_frame': int,
            'hand': 'L'|'R'}) and 'num_frames' (int, pose-frame count
            at edit time).
        route_holds: Unordered list of hold dicts with 'x', 'y', 'name'.
        num_frames: Current pose-frame count T for the target array.

    Returns:
        (hold_seq, targets) where hold_seq is the ordered list of hold
        dicts (each augmented with a 'hand' key: 'L' or 'R') and targets
        is a (num_frames, 2) board-space array of the active target per
        frame. hold_seq is empty if no override segment resolves to a
        current route hold.
    """
    by_name = {h["name"]: h for h in route_holds}
    edit_frames = override.get("num_frames") or num_frames
    scale = num_frames / edit_frames if edit_frames else 1.0

    segments = []
    for seg in override.get("sequence", []):
        hold = by_name.get(seg.get("name"))
        if hold is None:
            continue
        start = int(round(seg.get("start_frame", 0) * scale))
        start = max(0, min(num_frames - 1, start))
        hand = seg.get("hand", "L")
        segments.append((start, {**hold, "hand": hand}))

    segments.sort(key=lambda s: s[0])
    hold_seq = [hold for _, hold in segments]

    targets = np.zeros((num_frames, 2), dtype=np.float32)
    if not segments:
        return hold_seq, targets

    # Earliest segment always covers from frame 0.
    segments[0] = (0, segments[0][1])
    for i, (start, hold) in enumerate(segments):
        end = segments[i + 1][0] if i + 1 < len(segments) else num_frames
        targets[start:end] = [hold["x"], hold["y"]]

    return hold_seq, targets