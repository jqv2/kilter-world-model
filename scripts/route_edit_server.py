"""
Local server for the route hold editing UI.

Serves pre-warped board-space videos, route hold data, and accepts
hold exclusion edits.

Prerequisites:
    - Run warp_videos.py to generate board-space videos
    - Run export_holds_json.py to generate holds.json
    - climb_log.csv must exist with route info

Usage:
    python scripts/route_edit_server.py
    # Then open http://localhost:5002
"""

import csv
import json
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from pipeline.routes import lookup_route_by_uuid, lookup_route_by_name

app = Flask(__name__)

HOLDS_PATH = config.DATA_DIR / "holds.json"


def load_climb_log() -> dict[str, dict]:
    """Load climb_log.csv into a dict keyed by filename stem."""
    log_path = config.RAW_VIDEO_DIR / "climb_log.csv"
    if not log_path.exists():
        return {}
    log = {}
    with open(log_path, newline="") as f:
        for row in csv.DictReader(f):
            stem = Path(row.get("filename", "")).stem
            if stem:
                log[stem] = dict(row)
    return log


def get_video_list():
    """
    Build list of videos with warped-video availability and edit status.

    Scans data/warped/ for pre-warped videos, cross-references climb_log.csv
    for route info, and checks data/route_edits/ for existing edits.
    """
    warped_videos = sorted(config.WARPED_DIR.rglob("*.mp4"))
    climb_log = load_climb_log()
    videos = []

    for wp in warped_videos:
        rel = wp.relative_to(config.WARPED_DIR)
        stem = wp.stem
        log_entry = climb_log.get(stem, {})

        edit_path = config.ROUTE_EDITS_DIR / rel.parent / f"{stem}_route_edit.json"

        videos.append({
            "id": str(rel.with_suffix("")),
            "stem": stem,
            "subfolder": str(rel.parent) if str(rel.parent) != "." else "",
            "route_name": log_entry.get("route_name", ""),
            "grade": log_entry.get("grade", ""),
            "climb_uuid": log_entry.get("climb_uuid", ""),
            "has_edit": edit_path.exists(),
        })

    return videos


def resolve_holds(stem: str) -> list[dict] | None:
    """Look up route holds for a video stem via climb_log."""
    climb_log = load_climb_log()
    entry = climb_log.get(stem)
    if not entry:
        return None

    uuid = entry.get("climb_uuid", "").strip()
    if uuid:
        result = lookup_route_by_uuid(uuid)
        if result:
            return result["holds"]

    name = entry.get("route_name", "").strip()
    if name:
        result = lookup_route_by_name(name)
        if result:
            return result["holds"]

    return None


@app.route("/")
def index():
    return send_file(Path(__file__).parent.parent / "route_edit_ui.html")


@app.route("/api/videos")
def list_videos():
    return jsonify(get_video_list())


@app.route("/api/warped/<path:video_id>.mp4")
def get_warped_video(video_id):
    video_path = config.WARPED_DIR / f"{video_id}.mp4"
    if not video_path.exists():
        return "Not found", 404
    return send_file(video_path, mimetype="video/mp4")


@app.route("/api/holds/<path:video_id>")
def get_route_holds(video_id):
    """Return route holds with current exclusion state."""
    stem = Path(video_id).name
    holds = resolve_holds(stem)
    if holds is None:
        return jsonify({"error": "No route found", "holds": []})

    # Load existing edits if any
    edit_path = config.ROUTE_EDITS_DIR / f"{video_id}_route_edit.json"
    excluded = set()
    if edit_path.exists():
        with open(edit_path) as f:
            edits = json.load(f)
        excluded = set(edits.get("excluded_holds", []))

    # Annotate each hold with its exclusion state
    for h in holds:
        h["excluded"] = h["name"] in excluded

    return jsonify({"holds": holds})


@app.route("/api/all_holds")
def get_all_holds():
    """Return all board holds for the schematic overlay."""
    if not HOLDS_PATH.exists():
        return "Run export_holds_json.py first", 404
    return send_file(HOLDS_PATH, mimetype="application/json")


@app.route("/api/edit/<path:video_id>", methods=["POST"])
def save_edit(video_id):
    """Save hold exclusion list for a video."""
    edit_path = config.ROUTE_EDITS_DIR / f"{video_id}_route_edit.json"
    edit_path.parent.mkdir(parents=True, exist_ok=True)

    data = request.get_json()
    with open(edit_path, "w") as f:
        json.dump(data, f, indent=2)

    return jsonify({"status": "saved", "path": str(edit_path)})


@app.route("/api/edit/<path:video_id>", methods=["GET"])
def get_edit(video_id):
    """Load existing edits for a video."""
    edit_path = config.ROUTE_EDITS_DIR / f"{video_id}_route_edit.json"
    if not edit_path.exists():
        return jsonify(None)
    return send_file(edit_path, mimetype="application/json")


if __name__ == "__main__":
    print(f"Warped dir: {config.WARPED_DIR}")
    print(f"Edits dir:  {config.ROUTE_EDITS_DIR}")
    print(f"Holds:      {HOLDS_PATH}")
    print()
    app.run(port=5002, debug=True)