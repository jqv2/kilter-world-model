"""
Quick check to see climbs in climb_log.csv are found in the db

Usage:
    python scripts/check_problem_names.py
"""

import csv
import sqlite3
from pathlib import Path
from collections import defaultdict

import config

DB_PATH = config.DATA_DIR / "kilter.db"
CSV_PATH = config.RAW_VIDEO_DIR / "climb_log.csv"

def normalize_quotes(s: str) -> str:
    return (s.strip()
            .replace("\u2019", "'").replace("\u2018", "'")
            .replace("\u201c", '"').replace("\u201d", '"'))

def main():
    conn = sqlite3.connect(DB_PATH)
    
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Found {len(rows)} entries in climb_log.csv\n")

    # Load all 30° climbs and build normalized lookup
    all_climbs = conn.execute("""
        SELECT c.uuid, c.name, cs.display_difficulty
        FROM climbs c
        JOIN climb_stats cs ON cs.climb_uuid = c.uuid
        WHERE cs.angle = 30
    """).fetchall()
    conn.close()

    name_lookup = defaultdict(list)
    for uuid, db_name, diff in all_climbs:
        name_lookup[normalize_quotes(db_name)].append((uuid, db_name, diff))

    for row in rows:
        name = normalize_quotes(row["route_name"])
        matches = name_lookup.get(name, [])

        if len(matches) == 0:
            print(f"  MISS  | {name}")
        elif len(matches) == 1:
            uuid, db_name, diff = matches[0]
            print(f"  OK    | {name} -> {uuid} (difficulty {diff})")
        else:
            print(f"  DUPE  | {name} ({len(matches)} matches)")
            for uuid, db_name, diff in matches[:5]:
                print(f"          {uuid} (difficulty {diff})")
            if len(matches) > 5:
                print(f"          ... and {len(matches) - 5} more")

if __name__ == "__main__":
    main()