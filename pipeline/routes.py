"""
Route utilities: decode Kilter DB climbs into hold positions and roles.

Used by the dataset pipeline to attach route context to each training sequence,
and by the visualization tools to look up climbs by name or UUID.
"""

import re
import sqlite3
from pathlib import Path

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
    role_map = {
        12: 12, 13: 13, 14: 14, 15: 15,
        20: 12, 21: 13, 22: 14, 23: 15,
        24: 12, 25: 13, 26: 14, 27: 15,
        28: 12, 29: 13, 30: 14, 31: 15,
        32: 12, 33: 13, 34: 14, 35: 15,
        42: 12, 43: 13, 44: 14, 45: 15,
    }
    roles = np.array([role_map.get(h["role_id"], 13) for h in holds], dtype=np.int64)

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