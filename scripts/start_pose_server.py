"""
Local server for the RL start pose editing UI.

Serves a list of routes that have hold order edits (start_hands +
start_feet defined), computes the auto-generated starting pose via
the same IK heuristics reset_pose uses, and accepts manual pose
overrides (torso position + mid-joint preferences) written back to
the existing hold order JSON.

Prerequisites:
    - Run build_dataset.py to generate dataset.npz
    - Hold order edits must exist for routes you want to edit

Usage:
    python scripts/start_pose_server.py
    python scripts/start_pose_server.py --dataset data/dataset.npz
    # Then open http://localhost:5004
"""

import json
import math
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from pipeline.dataset import load_dataset, load_climb_log
from pipeline.routes import load_hold_order_edit
from models.rl_baseline import (
    compute_rl_bone_lengths,
    _COM_PROXIMAL,
    _TRUNK_MASS_FRAC,
    _HEAD_MASS_FRAC,
    _MASS_FRACTIONS,
    _ROLE_MAP,
    _ROLE_START,
)
from models.world_model import resolve_hold_sequence_and_targets

app = Flask(__name__)

# ── Module-level state (populated in main) ──

_bone_lengths: dict[str, float] = {}
_geometry: dict[str, float] = {}
_routes: list[dict] = []


def _compute_geometry(bone_lengths: dict[str, float]) -> dict[str, float]:
    """Derive skeleton geometry constants from bone lengths (board units).

    Returns a dict with shoulder_offset, hip_offset, and the raw bone
    lengths — everything the JS IK solver needs.
    """
    bu2m = config.RL_BOARD_UNIT_TO_METERS
    L_torso_m = bone_lengths["torso"] * bu2m
    R_head = config.RL_HEAD_RADIUS
    M = config.RL_BODY_MASS_KG

    m_trunk = _TRUNK_MASS_FRAC * M
    m_head = _HEAD_MASS_FRAC * M
    m_torso = _MASS_FRACTIONS["torso"] * M

    trunk_com = _COM_PROXIMAL["trunk"] * L_torso_m
    head_com = L_torso_m + _COM_PROXIMAL["head"] * (2 * R_head)
    torso_com_m = (m_trunk * trunk_com + m_head * head_com) / m_torso
    torso_com_bu = torso_com_m / bu2m

    return {
        "upper_arm": bone_lengths["upper_arm"],
        "forearm": bone_lengths["forearm"],
        "thigh": bone_lengths["thigh"],
        "shin": bone_lengths["shin"],
        "torso": bone_lengths["torso"],
        "half_shoulder_width": bone_lengths["half_shoulder_width"],
        "half_hip_width": bone_lengths["half_hip_width"],
        "shoulder_offset": bone_lengths["torso"] - torso_com_bu,
        "hip_offset": torso_com_bu,
    }


def _solve_ik_2bone(rx, ry, tx, ty, len_a, len_b, bend_sign):
    """Pure-Python 2-bone IK (mirrors rl_baseline._solve_ik_2bone)."""
    dx, dy = tx - rx, ty - ry
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return (rx, ry - len_a)
    max_reach = len_a + len_b
    if dist >= max_reach:
        s = len_a / max_reach
        return (rx + dx * s, ry + dy * s)
    cos_a = (len_a**2 + dist**2 - len_b**2) / (2 * len_a * dist)
    cos_a = max(-1.0, min(1.0, cos_a))
    angle = math.acos(cos_a)
    base = math.atan2(dy, dx)
    mid = base + bend_sign * angle
    return (rx + len_a * math.cos(mid), ry + len_a * math.sin(mid))


def _auto_pose(hand_holds, foot_holds, geom):
    """Compute the auto-generated torso position and mid-joint positions.

    Mirrors the heuristics in reset_pose but works in board units
    without Pymunk, producing the data the UI needs as its initial state.

    Returns dict with torso_board and mid_joints.
    """
    so = geom["shoulder_offset"]
    ho = geom["hip_offset"]
    hsw = geom["half_shoulder_width"]
    hhw = geom["half_hip_width"]
    arm_reach = geom["upper_arm"] + geom["forearm"]

    contacts = list(hand_holds.values()) + list(foot_holds.values())
    tx = sum(p[0] for p in contacts) / len(contacts)

    if foot_holds:
        ty = sum(p[1] for p in contacts) / len(contacts)
        max_hy = max(p[1] for p in hand_holds.values())
        ty = min(ty, max_hy - so)
    else:
        mid_y = sum(p[1] for p in hand_holds.values()) / len(hand_holds)
        ty = mid_y - so - 0.7 * arm_reach

    # Lower if arms can't reach
    for _ in range(10):
        all_ok = True
        shoulder_y = ty + so
        for side, pos in hand_holds.items():
            sx = tx - hsw if "left" in side else tx + hsw
            d = math.hypot(pos[0] - sx, pos[1] - shoulder_y)
            if d >= arm_reach * 0.98:
                all_ok = False
                break
        if all_ok:
            break
        ty -= 0.05 * arm_reach

    shoulder_y = ty + so
    hip_y = ty - ho
    shoulders = {
        "left": (tx - hsw, shoulder_y),
        "right": (tx + hsw, shoulder_y),
    }
    hips = {
        "left": (tx - hhw, hip_y),
        "right": (tx + hhw, hip_y),
    }

    mid_joints = {}

    # Arms
    for limb, pos in hand_holds.items():
        side = "left" if "left" in limb else "right"
        s = shoulders[side]
        ea = _solve_ik_2bone(s[0], s[1], pos[0], pos[1],
                             geom["upper_arm"], geom["forearm"], -1.0)
        eb = _solve_ik_2bone(s[0], s[1], pos[0], pos[1],
                             geom["upper_arm"], geom["forearm"], 1.0)
        if side == "left":
            elbow = ea if ea[0] <= eb[0] else eb
        else:
            elbow = ea if ea[0] >= eb[0] else eb
        mid_joints[f"{side}_elbow"] = list(elbow)

    # Legs
    for side in ("left", "right"):
        limb = f"{side}_foot"
        hip = hips[side]
        if limb in foot_holds:
            fp = foot_holds[limb]
            ka = _solve_ik_2bone(hip[0], hip[1], fp[0], fp[1],
                                 geom["thigh"], geom["shin"], -1.0)
            kb = _solve_ik_2bone(hip[0], hip[1], fp[0], fp[1],
                                 geom["thigh"], geom["shin"], 1.0)
            hip_below = hip[1] < fp[1]
            if side == "right":
                prefer_smaller = hip_below
            else:
                prefer_smaller = not hip_below
            knee = (ka if ka[0] <= kb[0] else kb) if prefer_smaller else (
                ka if ka[0] >= kb[0] else kb)
            mid_joints[f"{side}_knee"] = list(knee)
        else:
            # Free leg: default hang-down positions
            mid_joints[f"{side}_knee"] = [hip[0], hip[1] - geom["thigh"]]
            mid_joints[f"{side}_ankle"] = [hip[0], hip[1] - geom["thigh"] - geom["shin"]]

    return {"torso_board": [tx, ty], "torso_angle": 0, "mid_joints": mid_joints}


def _build_routes(dataset):
    """Build the route list for the UI from the dataset."""
    climb_log = load_climb_log()
    routes = []
    all_seqs = list(dataset["train_sequences"]) + list(dataset.get("test_sequences", []))
    all_rh = list(dataset["train_route_holds"]) + list(dataset.get("test_route_holds", []))
    all_stems = list(dataset["train_stems"]) + list(dataset.get("test_stems", []))

    for seq, route_holds, stem in zip(all_seqs, all_rh, all_stems):
        override = load_hold_order_edit(stem)
        if override is None:
            continue  # need start_hands at minimum

        start_hands = override.get("start_hands")
        start_feet = override.get("start_feet")
        if not start_hands:
            continue

        # Need hold_sequence to be valid
        climbing_seq = seq[:, config.CLIMBING_KEYPOINT_INDICES, :]
        hold_seq, _ = resolve_hold_sequence_and_targets(
            climbing_seq, route_holds, video_stem=stem,
        )
        if not hold_seq:
            continue

        name_to_pos = {h["name"]: (h["x"], h["y"]) for h in route_holds}

        hand_holds = {}
        for side_key, hname in start_hands.items():
            limb = "left_hand" if side_key == "L" else "right_hand"
            if hname in name_to_pos:
                hand_holds[limb] = name_to_pos[hname]
        if len(hand_holds) < 2:
            continue

        foot_holds = {}
        if start_feet:
            for side_key, hname in start_feet.items():
                limb = "left_foot" if side_key == "L" else "right_foot"
                if hname in name_to_pos:
                    foot_holds[limb] = name_to_pos[hname]

        log_entry = climb_log.get(stem, {})
        auto = _auto_pose(hand_holds, foot_holds, _geometry)

        routes.append({
            "stem": stem,
            "route_name": log_entry.get("route_name", stem),
            "grade": log_entry.get("grade", ""),
            "holds": [
                {"name": h["name"], "x": h["x"], "y": h["y"],
                 "role_id": _ROLE_MAP.get(h["role_id"], 13)}
                for h in route_holds
            ],
            "start_hands": start_hands,
            "start_feet": start_feet or {},
            "auto_pose": auto,
            "existing_override": override.get("start_pose_override"),
        })

    return routes


# ── Flask routes ──

@app.route("/")
def index():
    return send_file(Path(__file__).parent.parent / "start_pose_ui.html")


@app.route("/api/geometry")
def get_geometry():
    return jsonify(_geometry)


@app.route("/api/routes")
def list_routes():
    return jsonify([
        {
            "stem": r["stem"],
            "route_name": r["route_name"],
            "grade": r["grade"],
            "has_override": r["existing_override"] is not None,
        }
        for r in _routes
    ])


@app.route("/api/route/<stem>")
def get_route(stem):
    route = next((r for r in _routes if r["stem"] == stem), None)
    if route is None:
        return jsonify({"error": "Route not found"}), 404
    return jsonify(route)


@app.route("/api/all_holds")
def get_all_holds():
    from evaluation.visualize import get_all_holds as _get
    return jsonify(_get())


@app.route("/api/save/<stem>", methods=["POST"])
def save_override(stem):
    """Merge start_pose_override into the existing hold order JSON."""
    route = next((r for r in _routes if r["stem"] == stem), None)
    if route is None:
        return jsonify({"error": "Route not found"}), 404

    pose_data = request.get_json()

    # Load existing hold order
    override = load_hold_order_edit(stem) or {}
    override["start_pose_override"] = pose_data

    # Write back
    edit_path = config.HOLD_ORDERS_DIR / f"{stem}_hold_order.json"
    if not edit_path.exists():
        matches = list(config.HOLD_ORDERS_DIR.rglob(f"{stem}_hold_order.json"))
        if matches:
            edit_path = matches[0]
    edit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(edit_path, "w") as f:
        json.dump(override, f, indent=2)

    # Update in-memory state
    route["existing_override"] = pose_data

    return jsonify({"status": "saved", "path": str(edit_path)})


@app.route("/api/delete/<stem>", methods=["DELETE"])
def delete_override(stem):
    """Remove start_pose_override from the hold order JSON."""
    route = next((r for r in _routes if r["stem"] == stem), None)
    if route is None:
        return jsonify({"error": "Route not found"}), 404

    override = load_hold_order_edit(stem)
    if override and "start_pose_override" in override:
        del override["start_pose_override"]
        edit_path = config.HOLD_ORDERS_DIR / f"{stem}_hold_order.json"
        if not edit_path.exists():
            matches = list(config.HOLD_ORDERS_DIR.rglob(f"{stem}_hold_order.json"))
            if matches:
                edit_path = matches[0]
        with open(edit_path, "w") as f:
            json.dump(override, f, indent=2)

    route["existing_override"] = None
    return jsonify({"status": "deleted"})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start pose editor server")
    parser.add_argument("--dataset", default=str(config.DATA_DIR / "dataset.npz"))
    parser.add_argument("--port", type=int, default=5004)
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset}...")
    dataset = load_dataset(Path(args.dataset))

    _bone_lengths = compute_rl_bone_lengths(dataset["train_sequences"])
    _geometry = _compute_geometry(_bone_lengths)
    print(f"Skeleton geometry: {_geometry}")

    _routes = _build_routes(dataset)
    print(f"Found {len(_routes)} routes with hold order edits")
    for r in _routes:
        tag = " [override]" if r["existing_override"] else ""
        print(f"  {r['stem']}: {r['route_name']} {r['grade']}{tag}")

    print(f"\nStarting server on http://localhost:{args.port}")
    app.run(port=args.port, debug=True)