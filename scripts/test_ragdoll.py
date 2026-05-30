"""
Ragdoll smoke test — produces videos for visual inspection.

Run from project root:
    python test_ragdoll.py

Outputs:
    data/rl_viz/ragdoll_freefall.mp4   — ragdoll falling under gravity
    data/rl_viz/ragdoll_hanging.mp4    — hands attached, body settles
    data/rl_viz/ragdoll_motor.mp4      — left elbow bending while hanging
"""

from pathlib import Path

import cv2
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from pipeline.dataset import load_dataset
from models.rl_baseline import (
    compute_rl_bone_lengths, create_space, create_ragdoll,
    extract_keypoints, extract_head_position,
    create_hold_joint, set_motor_rate, reset_pose,
)
from evaluation.visualize import (
    render_board_image, draw_skeleton, get_all_holds, board_to_pixel,
    RENDER_SCALE, RENDER_PAD, ROLE_COLORS, COLOR_MIDDLE,
)


DT = 1 / config.RL_PHYSICS_HZ
OUT_DIR = config.DATA_DIR / "rl_viz"


def collect_frames(space, ragdoll, n_steps, substeps=1):
    """Step physics and collect keypoints + head position per step."""
    frames, heads = [], []
    for _ in range(n_steps):
        for _ in range(substeps):
            space.step(DT)
        frames.append(extract_keypoints(ragdoll))
        heads.append(extract_head_position(ragdoll))
    return frames, heads


def save_video(frames, heads, path, route_holds=None, title=None):
    """Render a skeleton video with head circle and optional route holds."""
    all_holds = get_all_holds()
    board_img = render_board_image(route_holds, all_holds)
    h, w = board_img.shape[:2]

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"),
        config.RL_PHYSICS_HZ, (w, h),
    )

    for i, (kp, head) in enumerate(zip(frames, heads)):
        img = board_img.copy()
        draw_skeleton(img, kp, head_pos=head)
        if title:
            cv2.putText(img, title, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, f"Frame {i}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        writer.write(img)

    writer.release()
    print(f"Saved {len(frames)}-frame video to {path}")


def main():
    ds = load_dataset(config.DATA_DIR / "dataset.npz")
    bl = compute_rl_bone_lengths(ds["train_sequences"])
    print("Bone lengths (board units):")
    for k, v in bl.items():
        print(f"  {k}: {v:.2f}")

    lh_hold = (60.0, 120.0)
    rh_hold = (84.0, 120.0)
    lf_hold = (60.0, 52.0)
    rf_hold = (104.0, 72.0)
    route_holds = [
        {"x": lh_hold[0], "y": lh_hold[1], "role_id": 12},
        {"x": rh_hold[0], "y": rh_hold[1], "role_id": 12},
        {"x": lf_hold[0], "y": lf_hold[1], "role_id": 15},
        {"x": rf_hold[0], "y": rf_hold[1], "role_id": 15},
    ]

    # ── 1. Free-fall ───────────────────────────────────────────────
    space = create_space()
    start_m = (72 * config.RL_BOARD_UNIT_TO_METERS,
               100 * config.RL_BOARD_UNIT_TO_METERS)
    ragdoll = create_ragdoll(space, bl, position=start_m)

    init_kp = extract_keypoints(ragdoll)
    init_head = extract_head_position(ragdoll)
    frames, heads = collect_frames(space, ragdoll, n_steps=120)
    frames.insert(0, init_kp)
    heads.insert(0, init_head)
    save_video(frames, heads, OUT_DIR / "ragdoll_freefall.mp4",
               title="Free-fall (no holds)")
    print(f"Hip dropped from y={frames[0][6,1]:.1f}"
          f" to y={frames[-1][6,1]:.1f}")

    # ── 2. Hanging from two hand holds ─────────────────────────────
    space = create_space()
    ragdoll = create_ragdoll(space, bl, position=(0.0, 0.0))
    reset_pose(ragdoll, space, {"left_hand": lh_hold, "right_hand": rh_hold})

    init_kp = extract_keypoints(ragdoll)
    init_head = extract_head_position(ragdoll)
    frames, heads = collect_frames(space, ragdoll, n_steps=120)
    frames.insert(0, init_kp)
    heads.insert(0, init_head)
    save_video(frames, heads, OUT_DIR / "ragdoll_hanging.mp4",
               route_holds=route_holds,
               title="Hanging from start holds")
    kp = frames[-1]
    print(f"Final — shoulders y={kp[0,1]:.1f},{kp[1,1]:.1f}  "
          f"hips y={kp[6,1]:.1f},{kp[7,1]:.1f}  "
          f"ankles y={kp[10,1]:.1f},{kp[11,1]:.1f}")

    # ── 3. Bend left elbow while hanging ───────────────────────────
    space = create_space()
    ragdoll = create_ragdoll(space, bl, position=(0.0, 0.0))
    reset_pose(ragdoll, space, {"left_hand": lh_hold, "right_hand": rh_hold})

    # Let it settle before applying motor
    for _ in range(120):
        space.step(DT)

    set_motor_rate(ragdoll, "left_elbow", 5.0)
    frames_a, heads_a = collect_frames(space, ragdoll, n_steps=120)

    set_motor_rate(ragdoll, "left_elbow", -5.0)
    frames_b, heads_b = collect_frames(space, ragdoll, n_steps=120)

    frames_motor = frames_a + frames_b
    heads_motor = heads_a + heads_b
    save_video(frames_motor, heads_motor, OUT_DIR / "ragdoll_left_elbow.mp4",
               route_holds=route_holds, title="Left elbow bend/extend")
    print(f"\nMotor test — left elbow moved "
          f"{np.linalg.norm(frames_motor[0][2] - frames_motor[60][2]):.1f} bu "
          f"over 1 second")
    
    # ── 4. Bend right elbow while hanging ───────────────────────────
    space = create_space()
    ragdoll = create_ragdoll(space, bl, position=(0.0, 0.0))
    reset_pose(ragdoll, space, {"left_hand": lh_hold, "right_hand": rh_hold})

    # Let it settle before applying motor
    for _ in range(120):
        space.step(DT)

    set_motor_rate(ragdoll, "right_elbow", 5.0)
    frames_a, heads_a = collect_frames(space, ragdoll, n_steps=120)

    set_motor_rate(ragdoll, "right_elbow", -5.0)
    frames_b, heads_b = collect_frames(space, ragdoll, n_steps=120)

    frames_motor = frames_a + frames_b
    heads_motor = heads_a + heads_b
    save_video(frames_motor, heads_motor, OUT_DIR / "ragdoll_right_elbow.mp4",
               route_holds=route_holds, title="right elbow bend/extend")
    print(f"\nMotor test — right elbow moved "
          f"{np.linalg.norm(frames_motor[0][2] - frames_motor[60][2]):.1f} bu "
          f"over 1 second")
    
    # ── 5. Bend left shoulder while hanging ───────────────────────────
    space = create_space()
    ragdoll = create_ragdoll(space, bl, position=(0.0, 0.0))
    reset_pose(ragdoll, space, {"left_hand": lh_hold, "right_hand": rh_hold})

    # Let it settle before applying motor
    for _ in range(120):
        space.step(DT)

    set_motor_rate(ragdoll, "left_shoulder", 5.0)
    frames_a, heads_a = collect_frames(space, ragdoll, n_steps=120)

    set_motor_rate(ragdoll, "left_shoulder", -5.0)
    frames_b, heads_b = collect_frames(space, ragdoll, n_steps=120)

    frames_motor = frames_a + frames_b
    heads_motor = heads_a + heads_b
    save_video(frames_motor, heads_motor, OUT_DIR / "ragdoll_left_shoulder.mp4",
               route_holds=route_holds, title="Left elbow bend/extend")
    print(f"\nMotor test — left elbow moved "
          f"{np.linalg.norm(frames_motor[0][2] - frames_motor[60][2]):.1f} bu "
          f"over 1 second")
    
    # ── 6. Bend right shoulder while hanging ───────────────────────────
    space = create_space()
    ragdoll = create_ragdoll(space, bl, position=(0.0, 0.0))
    reset_pose(ragdoll, space, {"left_hand": lh_hold, "right_hand": rh_hold})

    # Let it settle before applying motor
    for _ in range(120):
        space.step(DT)

    set_motor_rate(ragdoll, "right_shoulder", 5.0)
    frames_a, heads_a = collect_frames(space, ragdoll, n_steps=120)

    set_motor_rate(ragdoll, "right_shoulder", -5.0)
    frames_b, heads_b = collect_frames(space, ragdoll, n_steps=120)

    frames_motor = frames_a + frames_b
    heads_motor = heads_a + heads_b
    save_video(frames_motor, heads_motor, OUT_DIR / "ragdoll_right_shoulder.mp4",
               route_holds=route_holds, title="right elbow bend/extend")
    print(f"\nMotor test — right elbow moved "
          f"{np.linalg.norm(frames_motor[0][2] - frames_motor[60][2]):.1f} bu "
          f"over 1 second")
    
    # ── 7. Bend left knee while hanging ───────────────────────────
    space = create_space()
    ragdoll = create_ragdoll(space, bl, position=(0.0, 0.0))
    reset_pose(ragdoll, space, {"left_hand": lh_hold, "right_hand": rh_hold})

    # Let it settle before applying motor
    for _ in range(120):
        space.step(DT)

    set_motor_rate(ragdoll, "left_knee", 5.0)
    frames_a, heads_a = collect_frames(space, ragdoll, n_steps=120)

    set_motor_rate(ragdoll, "left_knee", -5.0)
    frames_b, heads_b = collect_frames(space, ragdoll, n_steps=120)

    frames_motor = frames_a + frames_b
    heads_motor = heads_a + heads_b
    save_video(frames_motor, heads_motor, OUT_DIR / "ragdoll_left_knee.mp4",
               route_holds=route_holds, title="Left elbow bend/extend")
    print(f"\nMotor test — left elbow moved "
          f"{np.linalg.norm(frames_motor[0][2] - frames_motor[60][2]):.1f} bu "
          f"over 1 second")
    
    # ── 8. Bend right knee while hanging ───────────────────────────
    space = create_space()
    ragdoll = create_ragdoll(space, bl, position=(0.0, 0.0))
    reset_pose(ragdoll, space, {"left_hand": lh_hold, "right_hand": rh_hold})

    # Let it settle before applying motor
    for _ in range(120):
        space.step(DT)

    set_motor_rate(ragdoll, "right_knee", 5.0)
    frames_a, heads_a = collect_frames(space, ragdoll, n_steps=120)

    set_motor_rate(ragdoll, "right_knee", -5.0)
    frames_b, heads_b = collect_frames(space, ragdoll, n_steps=120)

    frames_motor = frames_a + frames_b
    heads_motor = heads_a + heads_b
    save_video(frames_motor, heads_motor, OUT_DIR / "ragdoll_right_knee.mp4",
               route_holds=route_holds, title="right elbow bend/extend")
    print(f"\nMotor test — right elbow moved "
          f"{np.linalg.norm(frames_motor[0][2] - frames_motor[60][2]):.1f} bu "
          f"over 1 second")
    
    # ── 9. Bend left hip while hanging ───────────────────────────
    space = create_space()
    ragdoll = create_ragdoll(space, bl, position=(0.0, 0.0))
    reset_pose(ragdoll, space, {"left_hand": lh_hold, "right_hand": rh_hold})

    # Let it settle before applying motor
    for _ in range(120):
        space.step(DT)

    set_motor_rate(ragdoll, "left_hip", 5.0)
    frames_a, heads_a = collect_frames(space, ragdoll, n_steps=120)

    set_motor_rate(ragdoll, "left_hip", -5.0)
    frames_b, heads_b = collect_frames(space, ragdoll, n_steps=120)

    frames_motor = frames_a + frames_b
    heads_motor = heads_a + heads_b
    save_video(frames_motor, heads_motor, OUT_DIR / "ragdoll_left_hip.mp4",
               route_holds=route_holds, title="Left elbow bend/extend")
    print(f"\nMotor test — left elbow moved "
          f"{np.linalg.norm(frames_motor[0][2] - frames_motor[60][2]):.1f} bu "
          f"over 1 second")
    
    # ── 10. Bend right hip while hanging ───────────────────────────
    space = create_space()
    ragdoll = create_ragdoll(space, bl, position=(0.0, 0.0))
    reset_pose(ragdoll, space, {"left_hand": lh_hold, "right_hand": rh_hold})

    # Let it settle before applying motor
    for _ in range(120):
        space.step(DT)

    set_motor_rate(ragdoll, "right_hip", 5.0)
    frames_a, heads_a = collect_frames(space, ragdoll, n_steps=120)

    set_motor_rate(ragdoll, "right_hip", -5.0)
    frames_b, heads_b = collect_frames(space, ragdoll, n_steps=120)

    frames_motor = frames_a + frames_b
    heads_motor = heads_a + heads_b
    save_video(frames_motor, heads_motor, OUT_DIR / "ragdoll_right_hip.mp4",
               route_holds=route_holds, title="right elbow bend/extend")
    print(f"\nMotor test — right elbow moved "
          f"{np.linalg.norm(frames_motor[0][2] - frames_motor[60][2]):.1f} bu "
          f"over 1 second")
    
    # ── 11. Established on three holds ─────────────────────────────
    space = create_space()
    ragdoll = create_ragdoll(space, bl, position=(0.0, 0.0))
    reset_pose(ragdoll, space, {"left_hand": lh_hold, "right_hand": rh_hold, "right_foot": rf_hold})

    init_kp = extract_keypoints(ragdoll)
    init_head = extract_head_position(ragdoll)
    frames, heads = collect_frames(space, ragdoll, n_steps=300)
    frames.insert(0, init_kp)
    heads.insert(0, init_head)
    save_video(frames, heads, OUT_DIR / "ragdoll_3_points.mp4",
               route_holds=route_holds,
               title="Established on 3 holds.")
    kp = frames[-1]
    print(f"Final — shoulders y={kp[0,1]:.1f},{kp[1,1]:.1f}  "
          f"hips y={kp[6,1]:.1f},{kp[7,1]:.1f}  "
          f"ankles y={kp[10,1]:.1f},{kp[11,1]:.1f}")
    
    # ── 12. Established on four holds ─────────────────────────────
    space = create_space()
    ragdoll = create_ragdoll(space, bl, position=(0.0, 0.0))
    reset_pose(ragdoll, space, {"left_hand": lh_hold, "right_hand": rh_hold, "left_foot": lf_hold, "right_foot": rf_hold})

    init_kp = extract_keypoints(ragdoll)
    init_head = extract_head_position(ragdoll)
    frames, heads = collect_frames(space, ragdoll, n_steps=300)
    frames.insert(0, init_kp)
    heads.insert(0, init_head)
    save_video(frames, heads, OUT_DIR / "ragdoll_4_points.mp4",
               route_holds=route_holds,
               title="Established on 4 holds.")
    kp = frames[-1]
    print(f"Final — shoulders y={kp[0,1]:.1f},{kp[1,1]:.1f}  "
          f"hips y={kp[6,1]:.1f},{kp[7,1]:.1f}  "
          f"ankles y={kp[10,1]:.1f},{kp[11,1]:.1f}")
    
    # ── 13. Use motors while established on three holds ─────────────────────────────
    space = create_space()
    ragdoll = create_ragdoll(space, bl, position=(0.0, 0.0))
    reset_pose(ragdoll, space, {"left_hand": lh_hold, "right_hand": rh_hold, "right_foot": rf_hold})

    # Let it settle before applying motor
    for _ in range(200):
        space.step(DT)

    set_motor_rate(ragdoll, "right_knee", 5.0)
    frames_a, heads_a = collect_frames(space, ragdoll, n_steps=120)

    set_motor_rate(ragdoll, "right_knee", -2.0)
    frames_b, heads_b = collect_frames(space, ragdoll, n_steps=200)

    frames_motor = frames_a + frames_b
    heads_motor = heads_a + heads_b
    save_video(frames_motor, heads_motor, OUT_DIR / "ragdoll_3_point_motor.mp4",
               route_holds=route_holds, title="Left elbow bend/extend")
    print(f"\nMotor test — left elbow moved "
          f"{np.linalg.norm(frames_motor[0][2] - frames_motor[60][2]):.1f} bu "
          f"over 1 second")


if __name__ == "__main__":
    main()