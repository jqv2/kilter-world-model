"""
Export hold database to JSON for the calibration UI.

Usage:
    python scripts/export_holds_json.py
"""

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

DB_PATH = config.DATA_DIR / "kilter.db"
PRODUCT_SIZE_ID = 10
OUTPUT_PATH = config.DATA_DIR / "holds.json"

BOARD_CORNERS = {
    "MTL": [0, 156], "MTR": [144, 156],
    "MBL": [0, 12], "MBR": [144, 12],
    "KTL": [0, 12], "KTR": [144, 12],
    "KBL": [0, 0], "KBR": [144, 0],
}


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT h.name, h.x, h.y
        FROM holes h
        JOIN leds l ON l.hole_id = h.id
        WHERE l.product_size_id = ?
    """, (PRODUCT_SIZE_ID,)).fetchall()
    conn.close()

    holds = {row["name"]: [row["x"], row["y"]] for row in rows}

    data = {
        "holds": holds,
        "corners": BOARD_CORNERS,
        "board_edges": {"left": 0, "right": 144, "bottom": 0, "top": 156},
        "kickboard_boundary_y": 12,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Exported {len(holds)} holds + {len(BOARD_CORNERS)} corners to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()