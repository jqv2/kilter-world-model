"""
Local server for the calibration UI.

Serves calibration frames, hold data, detected features,
and accepts calibration saves.

Usage:
    python scripts/calibration_server.py
    # Then open http://localhost:5001
"""

import json
import sys
from pathlib import Path
import cv2
import numpy as np

from flask import Flask, jsonify, request, send_file, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

app = Flask(__name__)

HOLDS_PATH = config.DATA_DIR / "holds.json"


def get_video_list():
    """
    Build list of videos with their calibration status.
    
    Scans data/calibration_frames/ for extracted frames,
    checks data/calibrations/ for existing calibrations.
    """
    frames = sorted(config.CALIBRATION_FRAMES_DIR.rglob("*.jpg"))
    videos = []
    for frame_path in frames:
        rel = frame_path.relative_to(config.CALIBRATION_FRAMES_DIR)
        stem = frame_path.stem
        # Use / as separator for subfolder paths in the API
        video_id = str(rel.with_suffix(""))

        cal_path = config.CALIBRATIONS_DIR / rel.parent / f"{stem}_calibration.json"
        features_path = config.DETECTED_FEATURES_DIR / rel.parent / f"{stem}_features.json"

        entry = {
            "id": video_id,
            "name": stem,
            "subfolder": str(rel.parent) if str(rel.parent) != "." else "",
            "has_calibration": cal_path.exists(),
            "has_features": features_path.exists(),
        }
        videos.append(entry)
    return videos


@app.route("/")
def index():
    return send_file(Path(__file__).resolve().parent.parent / "calibration_ui.html")


@app.route("/api/videos")
def list_videos():
    return jsonify(get_video_list())


@app.route("/api/frame/<path:video_id>")
def get_frame(video_id):
    frame_path = config.CALIBRATION_FRAMES_DIR / f"{video_id}.jpg"
    if not frame_path.exists():
        return "Not found", 404
    return send_file(frame_path, mimetype="image/jpeg")


@app.route("/api/holds")
def get_holds():
    if not HOLDS_PATH.exists():
        return "Run export_holds_json.py first", 404
    return send_file(HOLDS_PATH, mimetype="application/json")


@app.route("/api/features/<path:video_id>")
def get_features(video_id):
    features_path = config.DETECTED_FEATURES_DIR / f"{video_id}_features.json"
    if not features_path.exists():
        return jsonify({"hold_centers": [], "board_corners": None})
    return send_file(features_path, mimetype="application/json")


@app.route("/api/calibration/<path:video_id>", methods=["GET"])
def get_calibration(video_id):
    cal_path = config.CALIBRATIONS_DIR / f"{video_id}_calibration.json"
    if not cal_path.exists():
        return jsonify(None)
    return send_file(cal_path, mimetype="application/json")


@app.route("/api/calibration/<path:video_id>", methods=["POST"])
def save_calibration(video_id):
    cal_path = config.CALIBRATIONS_DIR / f"{video_id}_calibration.json"
    cal_path.parent.mkdir(parents=True, exist_ok=True)

    data = request.get_json()
    with open(cal_path, "w") as f:
        json.dump(data, f, indent=2)

    return jsonify({"status": "saved", "path": str(cal_path)})


@app.route("/api/preview_warp/<path:video_id>/<plane>")
def preview_warp(video_id, plane):
    """
    Compute homography from saved calibration points and return
    the frame warped into board space, with grid overlay.
    """
    cal_path = config.CALIBRATIONS_DIR / f"{video_id}_calibration.json"
    frame_path = config.CALIBRATION_FRAMES_DIR / f"{video_id}.jpg"

    if not cal_path.exists() or not frame_path.exists():
        return "Not found", 404

    with open(cal_path) as f:
        cal = json.load(f)

    key = "main_points" if plane == "main" else "kick_points"
    pts = cal.get(key, [])
    if len(pts) < 4:
        return "Need at least 4 points", 400

    pixel_pts = np.array([p["pixel"] for p in pts], dtype=np.float64)
    board_pts = np.array([p["board"] for p in pts], dtype=np.float64)

    H, _ = cv2.findHomography(pixel_pts, board_pts, cv2.RANSAC, 3.0)
    if H is None:
        return "Homography computation failed", 500

    img = cv2.imread(str(frame_path))

    # Board dimensions in board units
    if plane == "main":
        bx_min, bx_max = 0, 144
        by_min, by_max = 12, 156
    else:
        bx_min, bx_max = 0, 144
        by_min, by_max = 0, 12

    # Padding in board units. Shows area beyond the board edges
    pad = 20

    # Scale: pixels per board unit in the output image
    scale = 6
    out_w = int((bx_max - bx_min + 2 * pad) * scale)
    out_h = int((by_max - by_min + 2 * pad) * scale)

    # Adjust homography to account for offset and scale:
    # We want to map board coords to output pixel coords.
    # output_px = (board_coord - min) * scale
    # So we need to warp with H_adjusted = S @ H
    # where S maps board space to output pixels.
    S = np.array([
        [scale, 0, -(bx_min - pad) * scale],
        [0, -scale, (by_max + pad) * scale],
        [0, 0, 1]
    ], dtype=np.float64)

    H_adjusted = S @ H
    warped = cv2.warpPerspective(img, H_adjusted, (out_w, out_h))

    # Draw grid (only within actual board bounds)
    for bx in range(bx_min, bx_max + 1, 12):
        px = int((bx - bx_min + pad) * scale)
        cv2.line(warped, (px, 0), (px, out_h), (60, 60, 60), 1)
    for by in range(by_min, by_max + 1, 12):
        py = int((by_max + pad - by) * scale)
        cv2.line(warped, (0, py), (out_w, py), (60, 60, 60), 1)

    # Draw board boundary as a bright rectangle
    brd_left = int(pad * scale)
    brd_right = int((bx_max - bx_min + pad) * scale)
    brd_top = int(pad * scale)
    brd_bottom = int((by_max - by_min + pad) * scale)
    cv2.rectangle(warped, (brd_left, brd_top), (brd_right, brd_bottom), (0, 255, 255), 2)

    # X axis labels
    for bx in range(bx_min, bx_max + 1, 24):
        px = int((bx - bx_min + pad) * scale)
        py = out_h - 5
        cv2.putText(warped, str(bx), (px + 2, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1)
    # Y axis labels
    for by in range(by_min, by_max + 1, 24):
        py = int((by_max + pad - by) * scale)
        cv2.putText(warped, str(by), (5, py - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1)

    # Origin marker
    ox = int(pad * scale)
    oy = int((by_max - by_min + pad) * scale)
    cv2.drawMarker(warped, (ox, oy), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
    cv2.putText(warped, "origin", (ox + 5, oy - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Calibration points
    for p in pts:
        bx, by = p["board"]
        px = int((bx - bx_min + pad) * scale)
        py = int((by_max + pad - by) * scale)
        cv2.circle(warped, (px, py), 4, (0, 255, 0), -1)
        cv2.putText(warped, p["label"], (px + 6, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
    # Draw expected hold positions from database
    with open(HOLDS_PATH) as f:
        holds_data = json.load(f)

    for name, (hx, hy) in holds_data["holds"].items():
        if bx_min - pad <= hx <= bx_max + pad and by_min - pad <= hy <= by_max + pad:
            px = int((hx - bx_min + pad) * scale)
            py = int((by_max + pad - hy) * scale)
            cv2.circle(warped, (px, py), 3, (100, 100, 255), 1)

    # Encode as JPEG and return
    _, buf = cv2.imencode('.jpg', warped, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes(), 200, {'Content-Type': 'image/jpeg'}


@app.route("/api/board_map")
def board_map():
    """Render a reference image of all holds with names."""
    if not HOLDS_PATH.exists():
        return "Run export_holds_json.py first", 404

    with open(HOLDS_PATH) as f:
        data = json.load(f)

    holds = data["holds"]
    edges = data["board_edges"]
    kick_y = data["kickboard_boundary_y"]

    pad = 10
    scale = 7
    bx_min, bx_max = edges["left"], edges["right"]
    by_min, by_max = edges["bottom"], edges["top"]
    out_w = int((bx_max - bx_min + 2 * pad) * scale)
    out_h = int((by_max - by_min + 2 * pad) * scale)

    img = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)

    def board_to_px(bx, by):
        px = int((bx - bx_min + pad) * scale)
        py = int((by_max + pad - by) * scale)
        return px, py

    # Board outline
    tl = board_to_px(bx_min, by_max)
    br = board_to_px(bx_max, by_min)
    cv2.rectangle(img, tl, br, (80, 80, 80), 1)

    # Kickboard boundary
    kl = board_to_px(bx_min, kick_y)
    kr = board_to_px(bx_max, kick_y)
    cv2.line(img, kl, kr, (60, 60, 100), 1)

    # Grid
    for bx in range(bx_min, bx_max + 1, 12):
        top = board_to_px(bx, by_max)
        bot = board_to_px(bx, by_min)
        cv2.line(img, top, bot, (40, 40, 40), 1)
    for by in range(by_min, by_max + 1, 12):
        left = board_to_px(bx_min, by)
        right = board_to_px(bx_max, by)
        cv2.line(img, left, right, (40, 40, 40), 1)
    # Coordinate axes with tick marks
    tick_len = 5  # pixels
    axis_color = (0, 200, 200)
    label_color = (0, 200, 200)

    # X-axis ticks and labels along bottom edge
    for bx in range(bx_min, bx_max + 1, 4):
        px, py = board_to_px(bx, by_min)
        cv2.line(img, (px, py), (px, py + tick_len), axis_color, 1)
        cv2.putText(img, str(bx), (px - 8, py + tick_len + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, label_color, 1)

    # Y-axis ticks and labels along left edge
    for by in range(by_min, by_max + 1, 4):
        px, py = board_to_px(bx_min, by)
        cv2.line(img, (px - tick_len, py), (px, py), axis_color, 1)
        cv2.putText(img, str(by), (px - tick_len - 28, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, label_color, 1)

    # Ensure exact edge values are labelled even if not on a 12-unit boundary
    for bx in [bx_min, bx_max]:
        if bx % 12 != 0:
            px, py = board_to_px(bx, by_min)
            cv2.line(img, (px, py), (px, py + tick_len), axis_color, 1)
            cv2.putText(img, str(bx), (px - 8, py + tick_len + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, label_color, 1)
    for by in [by_min, by_max]:
        if by % 12 != 0:
            px, py = board_to_px(bx_min, by)
            cv2.line(img, (px - tick_len, py), (px, py), axis_color, 1)
            cv2.putText(img, str(by), (px - tick_len - 28, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, label_color, 1)

    # Corner labels for clarity
    ox, oy = board_to_px(bx_min, by_min)
    cv2.putText(img, f"({bx_min},{by_min})", (ox + 4, oy + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    ox, oy = board_to_px(bx_max, by_max)
    cv2.putText(img, f"({bx_max},{by_max})", (ox - 50, oy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

    # Holds
    for name, (hx, hy) in holds.items():
        px, py = board_to_px(hx, hy)
        cv2.circle(img, (px, py), 3, (100, 180, 220), -1)
        cv2.putText(img, name, (px + 5, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 200, 240), 1)

    _, buf = cv2.imencode('.png', img)
    return buf.tobytes(), 200, {'Content-Type': 'image/png'}


if __name__ == "__main__":
    print(f"Frames dir: {config.CALIBRATION_FRAMES_DIR}")
    print(f"Calibrations dir: {config.CALIBRATIONS_DIR}")
    print(f"Holds: {HOLDS_PATH}")
    print()
    app.run(port=5001, debug=True)