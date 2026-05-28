"""
Local server for the hold order + timing editing UI.

Serves pre-warped board-space videos, the automatically detected hold
visit sequence and per-frame target timing (as draggable segments), and
accepts manual ordering/timing overrides written to data/hold_orders/.
Overrides are consumed by the structured world model and RL baseline via
resolve_hold_sequence_and_targets().

Prerequisites:
    - Run warp_videos.py to generate board-space videos
    - Run export_holds_json.py to generate holds.json
    - Pose JSONs and calibrations must exist for each video
    - climb_log.csv must exist with route info

Usage:
    python scripts/hold_order_server.py
    # Then open http://localhost:5003
"""

import json
import sys
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from pipeline.dataset import load_climb_log, process_video
from models.world_model import derive_hold_sequence, extract_move_targets

app = Flask(__name__)

HOLDS_PATH = config.DATA_DIR / "holds.json"


def get_video_list():
    """
    Build list of videos with warped-video availability and edit status.

    Scans data/warped/ for pre-warped videos, cross-references climb_log.csv
    for route info, and checks data/hold_orders/ for existing overrides.
    """
    warped_videos = sorted(config.WARPED_DIR.rglob("*.mp4"))
    climb_log = load_climb_log()
    videos = []

    for wp in warped_videos:
        rel = wp.relative_to(config.WARPED_DIR)
        stem = wp.stem
        log_entry = climb_log.get(stem, {})

        edit_path = config.HOLD_ORDERS_DIR / rel.parent / f"{stem}_hold_order.json"

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


def find_pose_json(stem: str) -> Path | None:
    """Locate the (non-overlay) pose JSON for a video stem."""
    matches = [
        p for p in config.POSES_DIR.rglob(f"{stem}.json")
        if not p.stem.endswith("_overlay")
    ]
    return matches[0] if matches else None


def auto_sequence(stem: str):
    """
    Resolve route holds and the automatically detected sequence/timing.

    Runs the same cleaning + calibration + route-edit pipeline the dataset
    uses (process_video), so the seed shown in the UI matches what training
    consumes. Returns (route_holds, segments, num_frames) where route_holds
    is the post-route-edit hold list (board coords) and segments is an
    ordered list of {'name', 'start_frame'} from automatic detection.
    Returns None on failure (missing pose/calibration/route, too few frames).
    """
    pose_json = find_pose_json(stem)
    if pose_json is None:
        return None

    climb_log = load_climb_log()
    try:
        result = process_video(pose_json, stem, climb_log, normalize_holds=False)
    except Exception:
        return None
    if result is None:
        return None

    route_holds = result["route_holds_raw"]
    seq_filtered = result["keypoints"][:, config.CLIMBING_KEYPOINT_INDICES, :]
    T = seq_filtered.shape[0]

    hold_seq = derive_hold_sequence(seq_filtered, route_holds)
    targets = (
        extract_move_targets(seq_filtered, hold_seq)
        if hold_seq else np.zeros((T, 2), dtype=np.float32)
    )

    # Build a position -> (hold, index) lookup from hold_seq
    pos_to_hold = {}
    for i, h in enumerate(hold_seq):
        key = (round(h["x"], 4), round(h["y"], 4))
        if key not in pos_to_hold:
            pos_to_hold[key] = h

    segments = []
    prev_key = None
    for t in range(T):
        key = (round(float(targets[t, 0]), 4), round(float(targets[t, 1]), 4))
        if key != prev_key:
            h = pos_to_hold.get(key)
            if h is not None:
                segments.append({
                    "name": h["name"],
                    "start_frame": t,
                    "hand": h.get("hand", "L"),
                })
            prev_key = key

    return route_holds, segments, T


def edit_path_for(video_id: str) -> Path:
    """Resolve the override path for a video id, searching subdirs."""
    direct = config.HOLD_ORDERS_DIR / f"{video_id}_hold_order.json"
    if direct.exists():
        return direct
    stem = Path(video_id).name
    matches = list(config.HOLD_ORDERS_DIR.rglob(f"{stem}_hold_order.json"))
    return matches[0] if matches else direct


@app.route("/")
def index():
    return send_file(Path(__file__).parent.parent / "hold_order_ui.html")


@app.route("/api/videos")
def list_videos():
    return jsonify(get_video_list())


@app.route("/api/warped/<path:video_id>.mp4")
def get_warped_video(video_id):
    video_path = config.WARPED_DIR / f"{video_id}.mp4"
    if not video_path.exists():
        return "Not found", 404
    return send_file(video_path, mimetype="video/mp4")


@app.route("/api/all_holds")
def get_all_holds():
    """Return all board holds for the schematic overlay."""
    if not HOLDS_PATH.exists():
        return "Run export_holds_json.py first", 404
    return send_file(HOLDS_PATH, mimetype="application/json")


@app.route("/api/holds/<path:video_id>")
def get_route_holds(video_id):
    """
    Return route holds, the automatically detected sequence/timing seed,
    the pose-frame count, and any existing override for the video.
    """
    stem = Path(video_id).name
    auto = auto_sequence(stem)
    if auto is None:
        return jsonify({
            "error": "Could not resolve route/poses for this video",
            "holds": [], "auto": [], "num_frames": 0, "override": None,
        })

    route_holds, segments, T = auto
    holds_out = [
        {"name": h["name"], "x": h["x"], "y": h["y"], "role_id": h["role_id"]}
        for h in route_holds
    ]

    override = None
    ep = edit_path_for(video_id)
    if ep.exists():
        with open(ep) as f:
            override = json.load(f)

    return jsonify({
        "holds": holds_out,
        "auto": segments,
        "num_frames": T,
        "override": override,
    })


@app.route("/api/edit/<path:video_id>", methods=["POST"])
def save_edit(video_id):
    """Save the hold order/timing override for a video."""
    edit_path = config.HOLD_ORDERS_DIR / f"{video_id}_hold_order.json"
    edit_path.parent.mkdir(parents=True, exist_ok=True)

    data = request.get_json()
    with open(edit_path, "w") as f:
        json.dump(data, f, indent=2)

    return jsonify({"status": "saved", "path": str(edit_path)})


@app.route("/api/edit/<path:video_id>", methods=["GET"])
def get_edit(video_id):
    """Load an existing override for a video."""
    ep = edit_path_for(video_id)
    if not ep.exists():
        return jsonify(None)
    return send_file(ep, mimetype="application/json")


@app.route("/api/edit/<path:video_id>", methods=["DELETE"])
def delete_edit(video_id):
    """Delete the hold order override for a video, reverting to auto."""
    ep = edit_path_for(video_id)
    if ep.exists():
        ep.unlink()
    return jsonify({"status": "deleted"})


if __name__ == "__main__":
    print(f"Warped dir:  {config.WARPED_DIR}")
    print(f"Orders dir:  {config.HOLD_ORDERS_DIR}")
    print(f"Holds:       {HOLDS_PATH}")
    print()
    app.run(port=5003, debug=True)