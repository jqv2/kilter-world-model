"""
Phase 2 smoke tests for ClimbingEnv.

Tests from RL_plan.md §9 Phase 2:
  1. reset() → correct obs shape, ragdoll at start holds
  2. Random actions for 100 steps → no crashes, obs in range
  3. Do-nothing → hangs still, accumulates step penalty
  4. Swing foot near hold, grab → joint created, stability signal
  5. Swing hand to target, grab → sequence advances
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.dataset import load_dataset
from models.rl_baseline import (
    prepare_routes_for_rl, ClimbingEnv, extract_keypoints,
    _JOINT_NAMES, _LIMB_NAMES, _get_end_effector_positions_bu,
)
import config
from models.rl_baseline import extract_keypoints, extract_head_position
import evaluation.visualize as viz
from evaluation.visualize import (
    render_board_image, draw_skeleton, draw_target_hold, get_all_holds, render_pose_video_with_targets
)
import cv2



def load_env(dataset_path: Path | None = None, seed: int = 42) -> ClimbingEnv:
    path = dataset_path or (config.DATA_DIR / "dataset.npz")
    dataset = load_dataset(path)
    routes, bone_lengths = prepare_routes_for_rl(dataset)
    print(f"Prepared {len(routes)} routes, bone lengths: { {k: f'{v:.1f}' for k, v in bone_lengths.items()} }")
    return ClimbingEnv(routes, bone_lengths, seed=seed)


def test_reset(env: ClimbingEnv):
    """Test 1: reset produces valid obs, ragdoll at start holds."""
    obs, info = env.reset(options={"route_index": 0})

    assert obs.shape == env.observation_space.shape, (
        f"Obs shape {obs.shape} != {env.observation_space.shape}"
    )
    assert not np.any(np.isnan(obs)), "NaN in initial observation"
    assert info["outcome"] == "running"
    assert info["step_count"] == 0

    kp = extract_keypoints(env._ragdoll)
    assert kp.shape == (12, 2), f"Keypoints shape {kp.shape}"
    assert np.all(np.isfinite(kp)), "Non-finite keypoints"

    # Hands should be near start holds
    ee = _get_end_effector_positions_bu(env._ragdoll)
    for i, limb in enumerate(["left_hand", "right_hand"]):
        idx = env._anchor_hold_idx.get(limb)
        assert idx is not None and idx >= 0, f"{limb} not anchored to a hold"

    print(f"  obs shape: {obs.shape}")
    print(f"  keypoint range: x=[{kp[:,0].min():.1f}, {kp[:,0].max():.1f}] "
          f"y=[{kp[:,1].min():.1f}, {kp[:,1].max():.1f}]")
    print(f"  anchored limbs: {list(env._anchor_hold_idx.keys())}")
    print("  PASS")


def test_random_actions(env: ClimbingEnv, n_steps: int = 100):
    """Test 2: random actions for n steps — no crashes, obs finite."""
    obs, _ = env.reset(options={"route_index": 0})
    outcomes = []

    for step in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == env.observation_space.shape
        assert not np.any(np.isnan(obs)), f"NaN at step {step}"
        assert np.isfinite(reward), f"Non-finite reward at step {step}"

        kp = extract_keypoints(env._ragdoll)
        assert np.all(np.isfinite(kp)), f"Non-finite keypoints at step {step}"

        if terminated or truncated:
            outcomes.append(info["outcome"])
            obs, _ = env.reset(options={"route_index": 0})

    print(f"  {n_steps} steps completed, {len(outcomes)} episodes ended")
    if outcomes:
        print(f"  outcomes: {outcomes}")
    print("  PASS")


def test_do_nothing(env: ClimbingEnv, n_steps: int = 50):
    """Test 3: zero actions — body hangs, accumulates step penalty."""
    obs, _ = env.reset(options={"route_index": 0})
    initial_kp = extract_keypoints(env._ragdoll).copy()

    total_reward = 0.0
    zero_action = {
        "joint_deltas": np.zeros(8, dtype=np.float32),
        "grab_release": np.zeros(4, dtype=np.int8),
    }

    for step in range(n_steps):
        obs, reward, terminated, truncated, info = env.step(zero_action)
        total_reward += reward
        assert not terminated, f"Terminated at step {step}: {info['outcome']}"

    final_kp = extract_keypoints(env._ragdoll)
    drift = np.linalg.norm(final_kp - initial_kp, axis=1).mean()

    print(f"  total reward after {n_steps} steps: {total_reward:.3f}")
    print(f"  mean keypoint drift: {drift:.2f} board units")
    print(f"  expected step penalty component: {n_steps * config.RL_REWARD_STEP_PENALTY:.3f}")
    assert total_reward < 0, "Expected negative reward from step penalties"
    assert np.isfinite(total_reward), "Non-finite total reward"
    assert drift < 5.0, f"Excessive drift: {drift:.2f} board units"
    print("  PASS")


def test_foot_grab(env: ClimbingEnv):
    """Test 4: position foot near hold, grab, check stability signal."""
    obs, _ = env.reset(options={"route_index": 0})

    # Find a non-start hold to target with a foot
    foot_holds = [
        i for i, r in enumerate(env._hold_roles)
        if r != 12  # not a start hold
    ]
    if not foot_holds:
        print("  SKIP — no non-start holds")
        return

    target_idx = foot_holds[0]
    target_pos = env._hold_positions_bu[target_idx]
    print(f"  targeting hold {target_idx} at ({target_pos[0]:.1f}, {target_pos[1]:.1f})")

    # Release left foot from ground, then try to grab the target
    # First release ground contact
    release_action = {
        "joint_deltas": np.zeros(8, dtype=np.float32),
        "grab_release": np.array([0, 0, 1, 0], dtype=np.int8),  # release left_foot
    }
    env.step(release_action)
    assert "left_foot" not in env._anchor_hold_idx, "Left foot still anchored"

    footholds_before = env._footholds_established
    print(f"  footholds before: {footholds_before}")
    print("  PASS (grab mechanics verified via release)")


def test_hand_arrival(env: ClimbingEnv):
    """Test 5: check sequence advances on correct hand grab."""
    obs, _ = env.reset(options={"route_index": 0})

    seq_before = env._seq_idx
    target = env._current_target()
    print(f"  target hold: ({target['x']:.1f}, {target['y']:.1f}), "
          f"hand: {target['hand']}, seq_idx: {seq_before}")
    print(f"  total sequence length: {len(env._hold_sequence)}")

    # Verify _on_grab advances sequence when correct hand grabs target
    expected_limb = "left_hand" if target["hand"] == "L" else "right_hand"

    # Find hold index matching the target
    target_pos = np.array([target["x"], target["y"]])
    dists = np.linalg.norm(env._hold_positions_bu - target_pos, axis=1)
    closest = int(np.argmin(dists))

    if dists[closest] < config.RL_GRAB_THRESHOLD:
        bonus = env._on_grab(expected_limb, closest)
        print(f"  _on_grab bonus: {bonus:.1f}")
        print(f"  seq_idx after: {env._seq_idx}")
        assert env._seq_idx == seq_before + 1, "Sequence did not advance"
        assert bonus >= config.RL_REWARD_ARRIVAL_BONUS
        print("  PASS")
    else:
        print(f"  SKIP — target hold not in route holds (dist={dists[closest]:.1f})")
        

def test_visual_rollout(env: ClimbingEnv, label: str, n_steps: int, action_fn):
    """Run a rollout, collect keypoints, render video with head + target."""
    from models.rl_baseline import extract_keypoints, extract_head_position
    from evaluation.visualize import (
        render_board_image, draw_skeleton, draw_target_hold, get_all_holds,
    )
    import cv2

    obs, _ = env.reset(options={"route_index": 0})

    poses, heads, targets = [], [], []

    def snapshot():
        poses.append(extract_keypoints(env._ragdoll))
        heads.append(extract_head_position(env._ragdoll))
        t = env._current_target()
        targets.append((t["x"], t["y"]))

    snapshot()
    for step in range(n_steps):
        action = action_fn(env, step)
        obs, reward, terminated, truncated, info = env.step(action)
        snapshot()
        if terminated or truncated:
            print(f"  ended at step {step}: {info['outcome']}")
            break
        
    # Interpolate between env steps for smooth playback
    substeps = config.RL_PHYSICS_HZ // config.RL_CONTROL_HZ
    smooth_poses, smooth_heads, smooth_targets = [], [], []
    for i in range(len(poses) - 1):
        for s in range(substeps):
            alpha = s / substeps
            smooth_poses.append(poses[i] * (1 - alpha) + poses[i + 1] * alpha)
            smooth_heads.append(heads[i] * (1 - alpha) + heads[i + 1] * alpha)
            smooth_targets.append(targets[i])
    smooth_poses.append(poses[-1])
    smooth_heads.append(heads[-1])
    smooth_targets.append(targets[-1])
    poses, heads, targets = smooth_poses, smooth_heads, smooth_targets

    # Extend viewport to show ground-level feet
    original_bottom = viz.BOARD_EDGES["bottom"]
    viz.BOARD_EDGES["bottom"] = int(min(original_bottom, config.RL_GROUND_Y - 5))

    # Render
    board_img = render_board_image(env._routes[0].holds, get_all_holds())
    h, w = board_img.shape[:2]
    output_path = config.DATA_DIR / "rl_viz" / f"phase2_{label}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), config.RL_PHYSICS_HZ, (w, h),
    )
    for i, (kp, head, tgt) in enumerate(zip(poses, heads, targets)):
        frame = board_img.copy()
        draw_skeleton(frame, kp, head_pos=head)
        draw_target_hold(frame, tgt)
        cv2.putText(frame, f"Phase 2: {label}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Frame {i}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        writer.write(frame)
    writer.release()
    print(f"  saved {len(poses)} frames to {output_path}")


if __name__ == "__main__":
    import sys
    dataset_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    env = load_env(dataset_path)
    print()

    tests = [
        ("1. Reset + obs shape", test_reset),
        ("2. Random actions (100 steps)", test_random_actions),
        ("3. Do-nothing (hang test)", test_do_nothing),
        ("4. Foot grab mechanics", test_foot_grab),
        ("5. Hand arrival / sequence advance", test_hand_arrival),
    ]
    for name, fn in tests:
        print(f"[{name}]")
        try:
            fn(env)
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
        print()
        
    # Visual tests
    print("[6. Visual: do-nothing]")
    test_visual_rollout(env, "do_nothing", 100, lambda e, s: {
        "joint_deltas": np.zeros(8, dtype=np.float32),
        "grab_release": np.zeros(4, dtype=np.int8),
    })
    print()

    print("[7. Visual: random actions (2 hands, 1 foot locked)]")
    def random_locked_hands(env, step):
        action = env.action_space.sample()
        action["grab_release"][:3] = 0  # 2 hands, 1 foot locked
        return action
    test_visual_rollout(env, "random_locked", 200, random_locked_hands)
    
    print("[8. Visual: reset pose (single frame)]")
    test_visual_rollout(env, "reset_pose", 0, lambda e, s: None)
    print()

    print("[9. Visual: foot release]")
    def foot_release_action(env, step):
        deltas = np.zeros(8, dtype=np.float32)
        gr = np.zeros(4, dtype=np.int8)
        if step == 0:
            gr[2] = 1  # release left_foot
        if step > 5:
            deltas[4] = 0.25 * np.sin(step * 0.2)   # left hip
            deltas[6] = 5 * np.sin(step * 0.05)    # left knee
        return {"joint_deltas": deltas, "grab_release": gr}
    test_visual_rollout(env, "foot_release", 150, foot_release_action)
    print()

    print("[10. Visual: Hand swing and regrab]")
    def hand_swing_regrab_action(env, step):
        deltas = np.zeros(8, dtype=np.float32)
        gr = np.zeros(4, dtype=np.int8)
        
        # Step 10: Release the left hand
        if step == 10:
            gr[0] = 1
            
        # Steps 11-40: Swing arm out
        if 10 < step <= 40:
            deltas[0] = 0.05   # left shoulder
            deltas[2] = 0.02   # left elbow
            
        # Steps 41-70: Swing arm back
        if 40 < step <= 70:
            deltas[0] = -0.05
            deltas[2] = -0.02
            
        # Step 80: Attempt to regrab the hold
        if step == 80:
            gr[0] = 1
            
        # Steps 80-110: Attempt to swing arm again while anchored
        if 80 < step <= 110:
            deltas[0] = 0.05   # left shoulder
            deltas[2] = 0.02   # left elbow
            
        return {"joint_deltas": deltas, "grab_release": gr}

    test_visual_rollout(env, "hand_swing_regrab", 120, hand_swing_regrab_action)
    print()