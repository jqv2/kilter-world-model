"""
Quick visual check of RL starting poses for every training route.

Renders a single PNG per route showing the ragdoll's initial pose
(after physics settling) on the board with route holds highlighted.

Usage:
    python scripts/verify_start_poses.py
    python scripts/verify_start_poses.py --no-settle   # skip settling

Output:
    data/rl_viz/start_poses/<stem>.png
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from pipeline.dataset import load_dataset
from models.rl_baseline import (
    prepare_routes_for_rl,
    RouteConfig,
    create_space,
    create_ragdoll,
    reset_pose,
    extract_keypoints,
    extract_head_position,
    set_motor_rate,
    compute_rl_bone_lengths,
    _JOINT_NAMES,
    _ROLE_MAP,
    _ROLE_START,
)
from pipeline.routes import load_hold_order_edit
from evaluation.visualize import (
    render_board_image,
    draw_skeleton,
    get_all_holds,
    board_to_pixel,
    RENDER_SCALE,
    RENDER_PAD,
    BOARD_EDGES,
)


def render_start_pose(route, bone_lengths, all_holds, settle=True):
    """Reset a route and return the starting pose image."""
    name_to_pos = {h["name"]: (h["x"], h["y"]) for h in route.holds}

    if route.start_hands is not None:
        hand_holds = {
            "left_hand": name_to_pos[route.start_hands["L"]],
            "right_hand": name_to_pos[route.start_hands["R"]],
        }
    else:
        roles = [_ROLE_MAP.get(h["role_id"], 13) for h in route.holds]
        start_indices = [i for i, r in enumerate(roles) if r == _ROLE_START]
        if len(start_indices) < 2:
            start_indices = list(range(min(2, len(route.holds))))
        positions = np.array(
            [(route.holds[i]["x"], route.holds[i]["y"]) for i in start_indices[:2]]
        )
        order = np.argsort(positions[:, 0])
        hand_holds = {
            "left_hand": tuple(positions[order[0]]),
            "right_hand": tuple(positions[order[-1]]),
        }

    foot_holds = {}
    if route.start_feet is not None:
        for side_key, hold_name in route.start_feet.items():
            limb = "left_foot" if side_key == "L" else "right_foot"
            if hold_name in name_to_pos:
                foot_holds[limb] = name_to_pos[hold_name]

    space = create_space()
    ragdoll = create_ragdoll(space, bone_lengths)
    reset_pose(ragdoll, space, hand_holds, foot_holds if foot_holds else None)

    if settle:
        for jname in _JOINT_NAMES:
            set_motor_rate(ragdoll, jname, 0.0)
        dt = 1.0 / config.RL_PHYSICS_HZ
        for _ in range(config.RL_SETTLE_STEPS):
            space.step(dt)

    kp = extract_keypoints(ragdoll)
    head = extract_head_position(ragdoll)

    # Expand padding so all keypoints are visible
    all_pts = np.vstack([kp, head.reshape(1, 2)])
    margin = 8  # board units of breathing room
    pad = max(
        RENDER_PAD,
        int(np.ceil(BOARD_EDGES["left"] - all_pts[:, 0].min() + margin)),
        int(np.ceil(all_pts[:, 0].max() - BOARD_EDGES["right"] + margin)),
        int(np.ceil(BOARD_EDGES["bottom"] - all_pts[:, 1].min() + margin)),
        int(np.ceil(all_pts[:, 1].max() - BOARD_EDGES["top"] + margin)),
    )

    board_img = render_board_image(route.holds, all_holds, pad=pad)
    draw_skeleton(board_img, kp, pad=pad, head_pos=head)

    for limb, pos in foot_holds.items():
        px, py = board_to_pixel(pos[0], pos[1], RENDER_SCALE, pad)
        cv2.circle(board_img, (px, py), max(3, RENDER_SCALE // 2), (0, 165, 255), 2)

    return board_img


def _build_route_configs(sequences, route_holds_list, stems):
    """Build RouteConfigs from a split (same logic as prepare_routes_for_rl)."""
    routes = []
    for seq, route_holds, stem in zip(sequences, route_holds_list, stems):
        climbing_seq = seq[:, config.CLIMBING_KEYPOINT_INDICES, :]
        from models.world_model import resolve_hold_sequence_and_targets
        hold_seq, _ = resolve_hold_sequence_and_targets(
            climbing_seq, route_holds, video_stem=stem,
        )
        if not hold_seq:
            continue

        start_hands = None
        start_feet = None
        override = load_hold_order_edit(stem)
        if override is not None:
            if "start_hands" in override:
                start_hands = override["start_hands"]
            if "start_feet" in override:
                start_feet = override["start_feet"]

        routes.append(RouteConfig(
            holds=route_holds,
            hold_sequence=hold_seq,
            start_hands=start_hands,
            start_feet=start_feet,
            stem=stem,
        ))
    return routes


def main():
    parser = argparse.ArgumentParser(description="Verify RL starting poses")
    parser.add_argument("--no-settle", action="store_true",
                        help="Skip physics settling (show pure IK pose)")
    parser.add_argument("--dataset", default=str(config.DATA_DIR / "dataset.npz"))
    args = parser.parse_args()

    out_dir = config.DATA_DIR / "rl_viz" / "start_poses"
    out_dir.mkdir(parents=True, exist_ok=True)

    stem_to_name = {}
    climb_log_path = config.DATA_DIR / "raw" / "climb_log.csv"
    if climb_log_path.is_file():
        with open(climb_log_path) as f:
            for row in csv.DictReader(f):
                stem_to_name[Path(row["filename"]).stem] = row["route_name"]

    dataset = load_dataset(Path(args.dataset))

    # Bone lengths from training data (the reference)
    bone_lengths = compute_rl_bone_lengths(dataset["train_sequences"])

    # Build routes from both splits
    routes = _build_route_configs(
        dataset["train_sequences"], dataset["train_route_holds"], dataset["train_stems"],
    ) + _build_route_configs(
        dataset["test_sequences"], dataset["test_route_holds"], dataset["test_stems"],
    )

    all_holds = get_all_holds()
    print(f"Rendering {len(routes)} starting poses → {out_dir}/")

    for i, route in enumerate(routes):
        img = render_start_pose(route, bone_lengths, all_holds, settle=not args.no_settle)

        name = stem_to_name.get(route.stem, route.stem)
        feet_info = f"feet: {route.start_feet}" if route.start_feet else "no feet"
        label = f"{name}  ({feet_info})"
        cv2.putText(img, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        path = out_dir / f"{route.stem}.png"
        cv2.imwrite(str(path), img)
        print(f"  [{i+1}/{len(routes)}] {route.stem}")

    print(f"Done. Check {out_dir}/")


if __name__ == "__main__":
    main()