"""PPO training for the RL climbing baseline.

Implements PPO with GAE directly in PyTorch.  The policy has two
distribution heads (Normal for 8 joint deltas, Bernoulli for 4
grab/release) with proper per-head gradients and entropy bonuses.

Usage::

    python -m scripts.train_rl --dataset data/dataset.npz
    python -m scripts.train_rl --total-frames 500000 --device cuda
"""

from __future__ import annotations
from torch.utils.tensorboard import SummaryWriter

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli

import config
from pipeline.dataset import load_dataset
from models.rl_baseline import prepare_routes_for_rl, ClimbingEnv, rollout_episode, extract_head_position
from evaluation.visualize import render_rl_video


################################################
# Networks
################################################

def _build_mlp(in_dim: int, hidden_dim: int, hidden_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(hidden_layers):
        layers += [nn.Linear(d, hidden_dim), nn.Tanh()]
        d = hidden_dim
    return nn.Sequential(*layers)


def _init_weights(module: nn.Module, gain: float = np.sqrt(2), final_gain: float = 0.01):
    """Orthogonal initialization for PPO networks.

    Hidden layers get *gain* (sqrt(2) for Tanh activations).  The last
    Linear layer gets *final_gain* (small for policy heads so the agent
    starts with near-zero actions, 1.0 for the value head).
    """
    linears = [m for m in module.modules() if isinstance(m, nn.Linear)]
    for layer in linears[:-1]:
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)
    if linears:
        nn.init.orthogonal_(linears[-1].weight, gain=final_gain)
        nn.init.zeros_(linears[-1].bias)
        
        
def _make_eval_action_fn(policy: nn.Module, device: torch.device):
    """Build a deterministic action function for evaluation rollouts.

    Uses the Normal mean (no sampling) and thresholds Bernoulli logits
    at 0 (no stochasticity), so the rollout is fully repeatable.
    """
    def action_fn(env, step):
        obs = env._build_obs()
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            mean, _, logits = policy._heads(obs_t)
        return {
            "joint_deltas": mean.cpu().numpy(),
            "grab_release": (logits > 0).int().cpu().numpy(),
        }
    return action_fn


def _render_eval_videos(
    policy: PolicyNetwork,
    env: ClimbingEnv,
    routes: list,
    reference_routes: list[int],
    viz_dir: Path,
    frame_label: str,
    device: torch.device,
):
    """Render deterministic eval rollouts for each reference route.

    Args:
        policy: Policy network (switched to eval mode internally).
        env: ClimbingEnv instance.
        routes: Full list of RouteData objects.
        reference_routes: Indices into *routes* to render.
        viz_dir: Root visualization directory (e.g. data/rl_viz).
        frame_label: Frame/episode prefix (e.g. "ep00100").  Hold
            counts are appended automatically from rollout info.
        device: Torch device.
    """
    policy.eval()
    action_fn = _make_eval_action_fn(policy, device)
    for ref_idx in reference_routes:
        out_dir = viz_dir / routes[ref_idx].stem
        out_dir.mkdir(parents=True, exist_ok=True)
        data = rollout_episode(
            env, action_fn, route_index=ref_idx,
            max_steps=config.RL_STEP_LIMIT,
        )
        hv = data["info"]["holds_visited"]
        th = data["info"]["total_holds"]
        filename = f"{frame_label}_{hv}_of_{th}.mp4"
        render_rl_video(
            data["poses"],
            out_dir / filename,
            route_holds=data["route_holds"],
            head_positions=data["head_positions"],
            head_radius_bu=config.RL_HEAD_RADIUS / config.RL_BOARD_UNIT_TO_METERS,
            target_positions=data["targets"],
            cog_positions=data["cog_positions"],
            support_polygons=data["support_polygons"],
            board_y_min=int(config.RL_GROUND_Y - 5),
            interpolate=(config.RL_PHYSICS_HZ, config.RL_CONTROL_HZ),
            fps=float(config.RL_PHYSICS_HZ),
            title=f"{frame_label} | {data['outcome']}",
            reward_breakdowns=data.get("reward_breakdowns"),
        )


class PolicyNetwork(nn.Module):
    """Shared encoder with Normal (joint deltas) and Bernoulli (grab) heads."""

    def __init__(
        self,
        obs_dim: int,
        n_continuous: int = 8,
        n_discrete: int = 4,
        hidden_dim: int = config.RL_HIDDEN_DIM,
        hidden_layers: int = config.RL_HIDDEN_LAYERS,
    ):
        super().__init__()
        self.encoder = _build_mlp(obs_dim, hidden_dim, hidden_layers)
        self.cont_mean = nn.Linear(hidden_dim, n_continuous)
        self.cont_log_std = nn.Parameter(torch.zeros(n_continuous))
        self.disc_logits = nn.Linear(hidden_dim, n_discrete)
        _init_weights(self.encoder, final_gain=np.sqrt(2))
        _init_weights(self.cont_mean, final_gain=0.01)
        _init_weights(self.disc_logits, final_gain=0.01)
        
        # Bias logits to 2.2. 10% chance to release.
        # This makes the agent default to "Hold On" instead of randomly dropping.
        nn.init.constant_(self.disc_logits.bias, 2.2)

    def _heads(self, obs: torch.Tensor):
        h = self.encoder(obs)
        mean = self.cont_mean(h)
        std = self.cont_log_std.exp().expand_as(mean)
        logits = self.disc_logits(h)
        return mean, std, logits

    def sample(self, obs: torch.Tensor):
        """Sample actions and return (cont, disc, log_prob)."""
        mean, std, logits = self._heads(obs)
        cont_dist = Normal(mean, std)
        disc_dist = Bernoulli(logits=logits)
        cont_a = cont_dist.sample()
        disc_a = disc_dist.sample()
        log_prob = cont_dist.log_prob(cont_a).sum(-1) + disc_dist.log_prob(disc_a).sum(-1)
        return cont_a, disc_a, log_prob

    def evaluate(self, obs: torch.Tensor, cont_a: torch.Tensor, disc_a: torch.Tensor):
        """Compute log_prob and entropy for stored actions."""
        mean, std, logits = self._heads(obs)
        cont_dist = Normal(mean, std)
        disc_dist = Bernoulli(logits=logits)
        log_prob = cont_dist.log_prob(cont_a).sum(-1) + disc_dist.log_prob(disc_a).sum(-1)
        entropy = cont_dist.entropy().sum(-1) + disc_dist.entropy().sum(-1)
        return log_prob, entropy


class ValueNetwork(nn.Module):
    """Separate value function (no shared backbone with policy)."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = config.RL_HIDDEN_DIM,
        hidden_layers: int = config.RL_HIDDEN_LAYERS,
    ):
        super().__init__()
        self.net = nn.Sequential(
            *_build_mlp(obs_dim, hidden_dim, hidden_layers),
            nn.Linear(hidden_dim, 1),
        )
        _init_weights(self.net, final_gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


################################################
# Rollout buffer
################################################

class RolloutBuffer:
    """Collects transitions and computes GAE advantages."""

    def __init__(self):
        self._obs, self._cont_a, self._disc_a = [], [], []
        self._log_probs, self._rewards, self._dones, self._values = [], [], [], []

    def add(self, obs, cont_a, disc_a, log_prob, reward, done, value):
        self._obs.append(obs)
        self._cont_a.append(cont_a)
        self._disc_a.append(disc_a)
        self._log_probs.append(log_prob)
        self._rewards.append(reward)
        self._dones.append(done)
        self._values.append(value)

    def finalize(self, last_value: float, gamma: float, lam: float, device: torch.device):
        """Convert to tensors and compute GAE returns/advantages."""
        self.obs = torch.as_tensor(np.array(self._obs), dtype=torch.float32, device=device)
        self.cont_a = torch.stack(self._cont_a).to(device)
        self.disc_a = torch.stack(self._disc_a).to(device)
        self.log_probs = torch.stack(self._log_probs).to(device)

        rewards = np.array(self._rewards, dtype=np.float64)
        dones = np.array(self._dones, dtype=np.float64)
        values = np.array([v for v in self._values] + [last_value], dtype=np.float64)

        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float64)
        gae = 0.0
        for t in reversed(range(n)):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:n]
        self.advantages = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        self.returns = torch.as_tensor(returns, dtype=torch.float32, device=device)

    def batches(self, batch_size: int):
        n = len(self.obs)
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            yield (
                self.obs[idx],
                self.cont_a[idx],
                self.disc_a[idx],
                self.log_probs[idx],
                self.advantages[idx],
                self.returns[idx],
            )


################################################
# Milestone tracking
################################################

class MilestoneTracker:
    """Saves tagged checkpoints on first occurrence of training milestones.

    Tracks both global milestones (first foothold, first completion,
    best return, best HVR) and per-route milestones (new max holds
    visited on a specific route).
    """

    def __init__(self, ckpt_dir: Path):
        self._dir = ckpt_dir
        self._first_foothold = False
        self._holds_reached: set[int] = set()
        self._completed = False
        self._best_return = -float("inf")
        self._best_hvr = 0.0
        self._best_holds_per_route: dict[str, int] = {}

    def check(self, info: dict, total_frames: int, save_fn):
        """Check episode info for milestones; call *save_fn(tag)* on hit."""
        fh = info.get("footholds_established", 0)
        hv = info.get("holds_visited", 0)
        total_holds = info.get("total_holds", 0)
        hvr = info.get("hold_visit_rate", 0.0)
        ret = info.get("cumulative_reward", -float("inf"))
        outcome = info.get("outcome", "")
        stem = info.get("route_stem", "unknown")

        # Global milestones across all climbs
        if fh > 0 and not self._first_foothold:
            self._first_foothold = True
            save_fn("first_foothold")

        for h in range(1, hv + 1):
            if h not in self._holds_reached:
                self._holds_reached.add(h)
                save_fn(f"hold_{h}")

        if outcome == "success" and not self._completed:
            self._completed = True
            save_fn("first_completion")

        if ret > self._best_return + config.RL_BEST_RETURN_THRESHOLD:
            self._best_return = ret
            save_fn(f"best_return_r{ret:+.0f}")

        if hvr > self._best_hvr:
            self._best_hvr = hvr
            save_fn(f"best_hvr_{hvr:.2f}")

        # Per-route: new max holds visited
        prev_best = self._best_holds_per_route.get(stem, 0)
        if hv > prev_best and hv > 0:
            self._best_holds_per_route[stem] = hv
            save_fn(f"{hv}_of_{total_holds}")
            
    def state_dict(self) -> dict:
        """Serializable milestone state for checkpointing."""
        return {
            "first_foothold": self._first_foothold,
            "holds_reached": sorted(self._holds_reached),
            "completed": self._completed,
            "best_return": self._best_return,
            "best_hvr": self._best_hvr,
            "best_holds_per_route": self._best_holds_per_route,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore milestone state from checkpoint."""
        self._first_foothold = state["first_foothold"]
        self._holds_reached = set(state["holds_reached"])
        self._completed = state["completed"]
        self._best_return = state["best_return"]
        self._best_hvr = state["best_hvr"]
        self._best_holds_per_route = state["best_holds_per_route"]


################################################
# Checkpointing
################################################

def save_checkpoint(
    path: Path,
    policy: PolicyNetwork,
    value_net: ValueNetwork,
    optimizer: torch.optim.Optimizer,
    total_frames: int,
    episode_count: int,
    milestones: MilestoneTracker | None = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "policy": policy.state_dict(),
        "value": value_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "total_frames": total_frames,
        "episode_count": episode_count,
    }
    if milestones is not None:
        data["milestones"] = milestones.state_dict()
    torch.save(data, path)


def load_checkpoint(
    path: Path,
    policy: PolicyNetwork,
    value_net: ValueNetwork,
    optimizer: torch.optim.Optimizer,
    milestones: MilestoneTracker | None = None,
) -> tuple[int, int]:
    """Load checkpoint, return (total_frames, episode_count).

    Restores milestone state if present in the checkpoint and a
    tracker is provided.  Older checkpoints without milestone data
    are loaded normally (tracker starts fresh).
    """
    ckpt = torch.load(path, weights_only=False)
    policy.load_state_dict(ckpt["policy"])
    value_net.load_state_dict(ckpt["value"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if milestones is not None and "milestones" in ckpt:
        milestones.load_state_dict(ckpt["milestones"])
    return ckpt["total_frames"], ckpt["episode_count"]


def _resolve_reference_routes(ref_path: str, routes: list) -> list[int]:
    """Resolve reference routes from a file of climb names.

    Each line in the file is matched against ``climb_log.csv`` to find
    the video stem, then matched against loaded routes.  Matching is
    case-insensitive.  Lines starting with ``#`` and blank lines are
    ignored.  Falls back to the first 2 routes if the file is missing.
    """
    path = Path(ref_path)
    if not path.is_file():
        print(f"  No reference routes file at {path}, using first 2 routes")
        return list(range(min(2, len(routes))))

    # climb_log: climb name → video stem
    climb_log_path = config.DATA_DIR / "raw" / "climb_log.csv"
    name_to_stem: dict[str, str] = {}
    if climb_log_path.is_file():
        with open(climb_log_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                stem = Path(row["filename"]).stem
                name_to_stem[row["route_name"].strip().lower()] = stem

    # loaded routes: stem → index
    stem_to_idx = {r.stem: i for i, r in enumerate(routes)}

    queries = [
        line.strip() for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    indices = []
    for query in queries:
        stem = name_to_stem.get(query.strip().lower())
        if stem is None:
            print(f"  Warning: '{query}' not found in climb_log.csv, skipping")
        elif stem not in stem_to_idx:
            print(f"  Warning: '{query}' ({stem}) not in training routes, skipping")
        else:
            indices.append(stem_to_idx[stem])

    return indices


################################################
# Training loop
################################################

def train(args):
    device = config.get_device(args.device)
    print(f"Device: {device}")

    # Load data + env
    dataset = load_dataset(Path(args.dataset))
    routes, bone_lengths = prepare_routes_for_rl(dataset)
    
    # Build stem → climb name mapping for display
    climb_log_path = config.DATA_DIR / "raw" / "climb_log.csv"
    stem_to_name: dict[str, str] = {}
    if climb_log_path.is_file():
        with open(climb_log_path) as f:
            for row in csv.DictReader(f):
                stem_to_name[Path(row["filename"]).stem] = row["route_name"]

    print(f"Prepared {len(routes)} training routes:")
    for i, r in enumerate(routes):
        name = stem_to_name.get(r.stem, "?")
        print(f"  [{i:>2d}] {name}  ({len(r.hold_sequence)} holds)")
    
    env = ClimbingEnv(routes, bone_lengths, seed=args.seed)
    
    # Reference routes for periodic eval videos
    reference_routes = _resolve_reference_routes(args.ref_routes, routes)
    print(f"Eval video routes: {[routes[i].stem for i in reference_routes]}")
    viz_dir = config.RL_VIZ_DIR
    viz_dir.mkdir(parents=True, exist_ok=True)

    obs_dim = env.observation_space.shape[0]
    policy = PolicyNetwork(obs_dim).to(device)
    value_net = ValueNetwork(obs_dim).to(device)
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value_net.parameters()),
        lr=config.RL_PPO_LR,
    )

    # Resume from checkpoint if provided
    total_frames = 0
    episode_count = 0
    ckpt_dir = config.RL_CHECKPOINT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    milestones = MilestoneTracker(ckpt_dir)
    if args.resume:
        total_frames, episode_count = load_checkpoint(
            Path(args.resume), policy, value_net, optimizer, milestones,
        )
        print(f"Resumed from {args.resume} at frame {total_frames}")

    # Logging
    log_dir = config.RL_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train_log.csv"
    log_fields = [
        "total_frames", "episodes", "episode_return", "episode_length",
        "holds_visited", "total_holds", "hold_visit_rate",
        "footholds_established", "outcome", "wall_time_s",
    ]
    log_file = open(log_path, "a", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    if log_path.stat().st_size == 0:
        log_writer.writeheader()
    
    # Tensorboard
    tb_writer = SummaryWriter(log_dir=config.DATA_DIR / "tb_logs")

    t_start = time.time()
    next_ckpt = total_frames + config.RL_CHECKPOINT_INTERVAL

    # Collect + train
    obs, info = env.reset()
    ep_return, ep_length = 0.0, 0

    while total_frames < args.total_frames:
        # Collect rollout
        buf = RolloutBuffer()
        policy.eval()

        for _ in range(config.RL_PPO_BATCH_SIZE):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                cont_a, disc_a, log_prob = policy.sample(obs_t)
                value = value_net(obs_t)

            action = {
                "joint_deltas": cont_a.cpu().numpy(),
                "grab_release": disc_a.cpu().numpy().astype(np.int32),
            }
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buf.add(obs, cont_a.cpu(), disc_a.cpu(), log_prob.cpu(), reward, done, value.item())
            ep_return += reward
            ep_length += 1
            total_frames += 1

            if done:
                # Tensorboard logs
                tb_writer.add_scalar("Rollout/Episode_Return", ep_return, total_frames)
                tb_writer.add_scalar("Rollout/Episode_Length", ep_length, total_frames)
                tb_writer.add_scalar("Rollout/Hold_Visit_Rate", info['hold_visit_rate'], total_frames)
                
                episode_count += 1
                log_writer.writerow({
                    "total_frames": total_frames,
                    "episodes": episode_count,
                    "episode_return": f"{ep_return:.2f}",
                    "episode_length": ep_length,
                    "holds_visited": info["holds_visited"],
                    "total_holds": info["total_holds"],
                    "hold_visit_rate": f"{info['hold_visit_rate']:.3f}",
                    "footholds_established": info["footholds_established"],
                    "outcome": info["outcome"],
                    "wall_time_s": f"{time.time() - t_start:.1f}",
                })
                log_file.flush()

                def _save_milestone(tag: str):
                    stem = info.get("route_stem", "unknown")
                    save_checkpoint(
                        ckpt_dir / f"milestone_{total_frames:08d}_{stem}_{tag}.pt",
                        policy, value_net, optimizer, total_frames, episode_count,
                        milestones,
                    )
                    print(f"  Milestone: {tag} (route: {stem})")

                milestones.check(info, total_frames, _save_milestone)

                if episode_count % 10 == 0:
                    print(
                        f"[{total_frames:>8d}] ep {episode_count:>5d}  "
                        f"ret={ep_return:>7.1f}  len={ep_length:>4d}  "
                        f"hvr={info['hold_visit_rate']:.2f}  "
                        f"out={info['outcome']}"
                    )

                # Periodic eval video
                if episode_count % config.RL_EVAL_VIDEO_INTERVAL == 0:
                    _render_eval_videos(
                        policy, env, routes, reference_routes, viz_dir,
                        frame_label=f"ep{episode_count:05d}",
                        device=device,
                    )
                    print(f"  Eval videos saved for episode {episode_count}")

                obs, info = env.reset()
                ep_return, ep_length = 0.0, 0
            else:
                obs = next_obs

        # GAE + PPO update
        with torch.no_grad():
            last_val = value_net(
                torch.as_tensor(obs, dtype=torch.float32, device=device),
            ).item()
        buf.finalize(last_val, config.RL_PPO_GAMMA, config.RL_PPO_GAE_LAMBDA, device)

        policy.train()
        value_net.train()
        adv_mean = buf.advantages.mean()
        adv_std = buf.advantages.std() + 1e-8

        n_updates = 0
        sum_policy_loss, sum_value_loss, sum_entropy = 0.0, 0.0, 0.0

        for _ in range(config.RL_PPO_EPOCHS):
            for b_obs, b_cont, b_disc, b_old_lp, b_adv, b_ret in buf.batches(
                config.RL_PPO_MINIBATCH_SIZE,
            ):
                b_adv_norm = (b_adv - adv_mean) / adv_std

                new_lp, entropy = policy.evaluate(b_obs, b_cont, b_disc)
                ratio = (new_lp - b_old_lp).exp()

                surr1 = ratio * b_adv_norm
                surr2 = ratio.clamp(
                    1 - config.RL_PPO_CLIP_EPSILON,
                    1 + config.RL_PPO_CLIP_EPSILON,
                ) * b_adv_norm
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()

                values = value_net(b_obs)
                value_loss = nn.functional.mse_loss(values, b_ret)

                loss = (
                    policy_loss
                    + config.RL_PPO_ENTROPY_COEF * entropy_loss
                    + 0.5 * value_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(policy.parameters()) + list(value_net.parameters()),
                    config.RL_PPO_MAX_GRAD_NORM,
                )
                optimizer.step()

                n_updates += 1
                sum_policy_loss += policy_loss.item()
                sum_value_loss += value_loss.item()
                sum_entropy += entropy.mean().item()

        tb_writer.add_scalar("Charts/Policy_Loss", sum_policy_loss / n_updates, total_frames)
        tb_writer.add_scalar("Charts/Value_Loss", sum_value_loss / n_updates, total_frames)
        tb_writer.add_scalar("Charts/Entropy", sum_entropy / n_updates, total_frames)

        # Periodic checkpoint
        if total_frames >= next_ckpt:
            save_checkpoint(
                ckpt_dir / f"step_{total_frames}.pt",
                policy, value_net, optimizer, total_frames, episode_count,
                milestones,
            )
            print(f"  Checkpoint saved at frame {total_frames}")
            next_ckpt += config.RL_CHECKPOINT_INTERVAL

    # Final save
    save_checkpoint(
        ckpt_dir / "final.pt",
        policy, value_net, optimizer, total_frames, episode_count,
        milestones,
    )
    log_file.close()
    tb_writer.close()
    print(f"Training complete: {total_frames} frames, {episode_count} episodes")


def main():
    parser = argparse.ArgumentParser(description="PPO training for RL climbing baseline")
    parser.add_argument("--dataset", default=str(config.DATA_DIR / "dataset.npz"))
    parser.add_argument("--total-frames", type=int, default=config.RL_TOTAL_FRAMES)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None, help="Path to checkpoint .pt to resume from")
    parser.add_argument(
        "--ref-routes", default=str(config.DATA_DIR / "rl_reference_routes.txt"),
        help="Text file with one climb name per line for eval videos",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()