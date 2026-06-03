"""Render stochastic rollouts from a saved checkpoint.

Usage:
    python -m scripts.render_stochastic \
        --checkpoint data/rl_checkpoints/milestone_00500000_stem_4_of_8.pt \
        --dataset data/dataset.npz \
        --route-stem stem_name \
        --attempts 1 \
        --no-start-pose-edits
        --save-keypoints
        -edeterministic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import config
from pipeline.dataset import load_dataset
from models.rl_baseline import (
    prepare_routes_for_rl, ClimbingEnv, rollout_episode,
)
from evaluation.visualize import render_rl_video
from scripts.train_rl import PolicyNetwork, ValueNetwork


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default=str(config.DATA_DIR / "dataset.npz"))
    parser.add_argument("--route-stem", default=None,
                        help="Specific route stem. If omitted, runs all routes.")
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-start-pose-edits", action="store_true",
                        help="Ignore manual start pose overrides; use auto IK poses.")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use mean actions (no sampling), matching training eval behavior.")
    parser.add_argument("--save-keypoints", action="store_true",
                        help="Save best-attempt keypoints as .npz for metric evaluation.")
    args = parser.parse_args()

    device = config.get_device(args.device)
    dataset = load_dataset(Path(args.dataset))
    routes, bone_lengths = prepare_routes_for_rl(dataset)
    if args.no_start_pose_edits:
        for r in routes:
            r.start_pose_override = None
    env = ClimbingEnv(routes, bone_lengths)

    obs_dim = env.observation_space.shape[0]
    policy = PolicyNetwork(obs_dim).to(device)
    value_net = ValueNetwork(obs_dim).to(device)
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value_net.parameters()),
        lr=config.RL_PPO_LR,
    )
    ckpt = torch.load(args.checkpoint, weights_only=False)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    def stochastic_action_fn(env, step):
        obs = env._build_obs()
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            if args.deterministic:
                mean, _, logits = policy._heads(obs_t)
                return {
                    "joint_deltas": mean.cpu().numpy(),
                    "grab_release": (logits > 0).int().cpu().numpy(),
                }
            else:
                cont_a, disc_a, _ = policy.sample(obs_t)
                return {
                    "joint_deltas": cont_a.cpu().numpy(),
                    "grab_release": disc_a.cpu().numpy().astype(np.int32),
                }

    stem_to_idx = {r.stem: i for i, r in enumerate(routes)}
    if args.route_stem:
        targets = [(args.route_stem, stem_to_idx[args.route_stem])]
    else:
        targets = [(r.stem, i) for i, r in enumerate(routes)]

    ckpt_name = Path(args.checkpoint).stem
    mode = "deterministic" if args.deterministic else "stochastic"
    out_dir = config.RL_VIZ_DIR / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    for stem, route_idx in targets:
        best_data, best_hv = None, -1
        for a in range(args.attempts):
            data = rollout_episode(
                env, stochastic_action_fn, route_index=route_idx,
                max_steps=config.RL_STEP_LIMIT,
            )
            hv = data["info"]["holds_visited"]
            th = data["info"]["total_holds"]
            print(f"  {stem} attempt {a+1}/{args.attempts}: "
                  f"{hv}/{th} holds, {data['outcome']}")
            if hv > best_hv:
                best_hv = hv
                best_data = data

        th = best_data["info"]["total_holds"]

        if args.save_keypoints:
            kp_array = np.stack(best_data["poses"])  # (T, 12, 2)
            kp_path = out_dir / f"{ckpt_name}_{stem}_keypoints.npz"
            np.savez_compressed(
                kp_path,
                poses=kp_array,
                stem=stem,
                holds_visited=best_hv,
                total_holds=th,
                outcome=best_data["outcome"],
            )
            print(f"  Saved keypoints {kp_array.shape} → {kp_path}")

        out_path = out_dir / f"{ckpt_name}_{stem}_{best_hv}_of_{th}.mp4"
        render_rl_video(
            best_data["poses"],
            out_path,
            route_holds=best_data["route_holds"],
            head_positions=best_data["head_positions"],
            head_radius_bu=config.RL_HEAD_RADIUS / config.RL_BOARD_UNIT_TO_METERS,
            target_positions=best_data["targets"],
            cog_positions=best_data["cog_positions"],
            support_polygons=best_data["support_polygons"],
            interpolate=(config.RL_PHYSICS_HZ, config.RL_CONTROL_HZ),
            fps=float(config.RL_PHYSICS_HZ),
            title=f"{ckpt_name} | best of {args.attempts} | {best_data['outcome']}",
            reward_breakdowns=best_data.get("reward_breakdowns"),
        )
        print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()