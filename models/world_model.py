"""
Pose prediction transformer with route context.

The model attends over both a context window of recent poses and the
route's hold positions, so it can learn route-specific movement patterns.

Architecture:
    Pose frames and hold features are independently projected, then
    concatenated into a single sequence for the transformer encoder.
    Role embeddings are added to hold features before projection.

The hold tokens and pose tokens are concatenated into a single sequence.
The transformer's self-attention lets pose tokens attend to hold tokens
(learning which holds matter for the current movement) and to other pose
tokens (learning motion dynamics).

Input:  poses  (batch, context_window, NUM_CLIMBING_KEYPOINTS * 2) - flattened keypoints
        holds  (batch, max_holds, 2)       - normalized (x, y) positions
        roles  (batch, max_holds)          - role IDs (12=start, 13=mid, 14=finish, 15=foot)
        hold_mask (batch, max_holds)       - True for padded positions

Output: (batch, NUM_CLIMBING_KEYPOINTS * 2), predicted delta for next frame
"""

import torch
import torch.nn as nn
import numpy as np

import config
from pipeline.routes import pad_holds, normalize_board_coords


# Number of distinct placement roles (12-15 in the Kilter DB, plus 0 for padding)
NUM_ROLES = 16
ROLE_EMBED_DIM = 8


class PoseTransformer(nn.Module):
    """
    Transformer for next-frame pose delta prediction with route context.

    Concatenates pose frame tokens and hold tokens into a single sequence.
    Self-attention allows pose tokens to attend to holds and vice versa.

    Args:
        pose_dim: Flattened pose dimension (NUM_CLIMBING_KEYPOINTS × 2).
        hold_dim: Hold feature dimension (2 for normalized x, y).
        hidden_dim: Transformer hidden / embedding dimension.
        n_layers: Number of transformer encoder layers.
        n_heads: Number of attention heads.
        context_len: Maximum context window length.
        max_holds: Maximum number of holds in a route (for positional encoding).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        pose_dim: int = config.NUM_CLIMBING_KEYPOINTS * 2,
        hold_dim: int = 2,
        hidden_dim: int = config.MODEL_HIDDEN_DIM,
        n_layers: int = config.MODEL_LAYERS,
        n_heads: int = config.MODEL_HEADS,
        context_len: int = config.CONTEXT_WINDOW,
        max_holds: int = config.MAX_ROUTE_HOLDS,
        dropout: float = config.MODEL_DROPOUT,
    ):
        super().__init__()
        self.pose_dim = pose_dim
        self.hidden_dim = hidden_dim
        self.context_len = context_len
        self.max_holds = max_holds

        # Pose pathway
        self.pose_proj = nn.Linear(pose_dim, hidden_dim)
        self.pose_pos = nn.Parameter(
            torch.randn(1, context_len, hidden_dim) * 0.02
        )

        # Hold pathway
        self.hold_proj = nn.Linear(hold_dim + ROLE_EMBED_DIM, hidden_dim)
        self.role_embed = nn.Embedding(NUM_ROLES, ROLE_EMBED_DIM)

        # Token type embeddings to distinguish pose vs hold tokens
        self.type_embed = nn.Embedding(2, hidden_dim)  # 0 = pose, 1 = hold

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        self.output_proj = nn.Linear(hidden_dim, pose_dim)

    def forward(
        self,
        poses: torch.Tensor,
        holds: torch.Tensor,
        roles: torch.Tensor,
        hold_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Predict the pose delta for the next frame.

        Args:
            poses: (batch, seq_len, NUM_CLIMBING_KEYPOINTS * 2) context window of flattened poses.
            holds: (batch, n_holds, 2) normalized hold positions.
            roles: (batch, n_holds) role IDs.
            hold_mask: (batch, n_holds) True where holds are padding.

        Returns:
            (batch, NUM_CLIMBING_KEYPOINTS * 2) predicted delta from the last frame in the window.
        """
        B, S, _ = poses.shape
        _, H, _ = holds.shape

        # Embed poses
        pose_tokens = self.pose_proj(poses)
        pose_tokens = pose_tokens + self.pose_pos[:, :S, :]
        pose_tokens = pose_tokens + self.type_embed(
            torch.zeros(B, S, dtype=torch.long, device=poses.device)
        )

        # Embed holds: concat position features with role embedding
        role_features = self.role_embed(roles)
        hold_features = torch.cat([holds, role_features], dim=-1)
        hold_tokens = self.hold_proj(hold_features)
        hold_tokens = hold_tokens + self.type_embed(
            torch.ones(B, H, dtype=torch.long, device=holds.device)
        )

        # Concatenate: [hold_tokens, pose_tokens]
        # Holds first so pose tokens can attend to them at every layer
        tokens = torch.cat([hold_tokens, pose_tokens], dim=1)

        # Build attention mask: padded hold positions should not be attended to
        if hold_mask is not None:
            # mask shape: (batch, n_holds + seq_len)
            # False = attend, True = ignore
            pose_mask = torch.zeros(B, S, dtype=torch.bool, device=poses.device)
            src_key_padding_mask = torch.cat([hold_mask, pose_mask], dim=1)
        else:
            src_key_padding_mask = None

        tokens = self.encoder(tokens, src_key_padding_mask=src_key_padding_mask)

        # Take the last pose token's output (it's at position H + S - 1)
        last_pose = tokens[:, H + S - 1, :]
        return self.output_proj(last_pose)

    def predict_absolute(
        self,
        poses: torch.Tensor,
        holds: torch.Tensor,
        roles: torch.Tensor,
        hold_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Predict the next frame's absolute pose (current + delta).

        Returns:
            (batch, NUM_CLIMBING_KEYPOINTS * 2) predicted absolute pose for the next frame.
        """
        delta = self.forward(poses, holds, roles, hold_mask)
        return poses[:, -1, :] + delta
    
    
class StructuredPoseTransformer(nn.Module):
    """
    Structured variant: receives target hold position as explicit input.

    Same architecture as PoseTransformer but with an additional conditioning
    token that encodes the current target hold (x, y). This collapses
    the move-planning ambiguity, letting the model focus on motion execution.

    The conditioning token is prepended to the pose tokens so it's always
    in the attention window.

    Args:
        pose_dim: Flattened pose dimension (NUM_CLIMBING_KEYPOINTS * 2).
        hold_dim: Hold feature dimension (2).
        hidden_dim: Transformer hidden dimension.
        n_layers: Number of transformer encoder layers.
        n_heads: Number of attention heads.
        context_len: Maximum context window length.
        max_holds: Maximum holds per route.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        pose_dim: int = config.NUM_CLIMBING_KEYPOINTS * 2,
        hold_dim: int = 2,
        hidden_dim: int = config.MODEL_HIDDEN_DIM,
        n_layers: int = config.MODEL_LAYERS,
        n_heads: int = config.MODEL_HEADS,
        context_len: int = config.CONTEXT_WINDOW,
        max_holds: int = config.MAX_ROUTE_HOLDS,
        dropout: float = config.MODEL_DROPOUT,
    ):
        super().__init__()
        self.pose_dim = pose_dim
        self.hidden_dim = hidden_dim
        self.context_len = context_len
        self.max_holds = max_holds

        # Pose pathway (same as direct model)
        self.pose_proj = nn.Linear(pose_dim, hidden_dim)
        self.pose_pos = nn.Parameter(
            torch.randn(1, context_len, hidden_dim) * 0.02
        )

        # Hold pathway (same as direct model)
        self.hold_proj = nn.Linear(hold_dim + ROLE_EMBED_DIM, hidden_dim)
        self.role_embed = nn.Embedding(NUM_ROLES, ROLE_EMBED_DIM)

        # Structured conditioning: target hold position only
        self.condition_proj = nn.Linear(hold_dim, hidden_dim)

        # Token type embeddings: 0=pose, 1=hold, 2=condition
        self.type_embed = nn.Embedding(3, hidden_dim)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        self.output_proj = nn.Linear(hidden_dim, pose_dim)

    def forward(
        self,
        poses: torch.Tensor,
        holds: torch.Tensor,
        roles: torch.Tensor,
        target_pos: torch.Tensor,
        hold_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Predict the pose delta for the next frame.

        Args:
            poses: (batch, seq_len, NUM_CLIMBING_KEYPOINTS * 2) context window of flattened poses.
            holds: (batch, n_holds, 2) normalized hold positions.
            roles: (batch, n_holds) role IDs.
            target_pos: (batch, 2) normalized target hold (x, y).
            hold_mask: (batch, n_holds) True where holds are padding.

        Returns:
            (batch, NUM_CLIMBING_KEYPOINTS * 2) predicted delta from the last frame in the window.
        """
        B, S, _ = poses.shape
        _, H, _ = holds.shape

        # Embed poses
        pose_tokens = self.pose_proj(poses)
        pose_tokens = pose_tokens + self.pose_pos[:, :S, :]
        pose_tokens = pose_tokens + self.type_embed(
            torch.zeros(B, S, dtype=torch.long, device=poses.device)
        )

        # Embed holds
        role_features = self.role_embed(roles)
        hold_features = torch.cat([holds, role_features], dim=-1)
        hold_tokens = self.hold_proj(hold_features)
        hold_tokens = hold_tokens + self.type_embed(
            torch.ones(B, H, dtype=torch.long, device=holds.device)
        )

        # Embed condition: target hold position
        cond_token = self.condition_proj(target_pos).unsqueeze(1)  # (B, 1, hidden)
        cond_token = cond_token + self.type_embed(
            torch.full((B, 1), 2, dtype=torch.long, device=poses.device)
        )

        # Concatenate: [condition, hold_tokens, pose_tokens]
        tokens = torch.cat([cond_token, hold_tokens, pose_tokens], dim=1)

        # Attention mask
        if hold_mask is not None:
            cond_mask = torch.zeros(B, 1, dtype=torch.bool, device=poses.device)
            pose_mask = torch.zeros(B, S, dtype=torch.bool, device=poses.device)
            src_key_padding_mask = torch.cat([cond_mask, hold_mask, pose_mask], dim=1)
        else:
            src_key_padding_mask = None

        tokens = self.encoder(tokens, src_key_padding_mask=src_key_padding_mask)

        # Last pose token is at position 1 + H + S - 1
        last_pose = tokens[:, 1 + H + S - 1, :]
        return self.output_proj(last_pose)

    def predict_absolute(
        self,
        poses: torch.Tensor,
        holds: torch.Tensor,
        roles: torch.Tensor,
        target_pos: torch.Tensor,
        hold_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Predict the next frame's absolute pose (current + delta).

        Returns:
            (batch, NUM_CLIMBING_KEYPOINTS * 2) predicted absolute pose for the next frame.
        """
        delta = self.forward(poses, holds, roles, target_pos, hold_mask)
        return poses[:, -1, :] + delta
    

class PoseDataset(torch.utils.data.Dataset):
    """
    Sliding window dataset with route context.

    Each sample is a (context_window, target_delta, hold_positions, hold_roles,
    displacement) tuple.

    Args:
        sequences: List of (T_i, 17, 2) arrays in board space
            (filtered to NUM_CLIMBING_KEYPOINTS climbing keypoints internally).
        scores: List of (T_i, 17) confidence arrays.
        hold_positions: List of (N_i, 2) arrays of normalized hold positions.
        hold_roles: List of (N_i,) arrays of role IDs.
        context_len: Number of frames in the input window.
        max_holds: Maximum holds per route (for padding).
        strides: List of temporal strides for sample construction. Stride 1
            uses consecutive frames; stride N subsamples every Nth frame,
            making transitions more visible to the model. Default [1].
    """

    def __init__(
        self,
        sequences: list,
        scores: list,
        hold_positions: list,
        hold_roles: list,
        context_len: int = config.CONTEXT_WINDOW,
        max_holds: int = config.MAX_ROUTE_HOLDS,
        strides: list[int] | None = None,
    ):
        self.context_len = context_len
        self.max_holds = max_holds
        self.samples = []

        if strides is None:
            strides = [1]

        for seq, sc, h_pos, h_roles in zip(
            sequences, scores, hold_positions, hold_roles
        ):
            T = seq.shape[0]
            seq_filtered = seq[:, config.CLIMBING_KEYPOINT_INDICES, :]
            flat = seq_filtered.reshape(T, -1).astype("float32")

            padded_pos, padded_roles, mask = pad_holds(h_pos, h_roles, max_holds)

            for stride in strides:
                # Need context_len * stride frames before the target,
                # plus the target frame itself
                min_frames = context_len * stride + stride
                if T < min_frames:
                    continue

                for t in range(context_len * stride, T, stride):
                    context_indices = range(t - context_len * stride, t, stride)
                    context = flat[list(context_indices)]
                    target_abs = flat[t]
                    delta = flat[t] - flat[t - stride]
                    displacement = min(
                        float((delta.reshape(-1, 2) ** 2).sum(axis=1).mean() ** 0.5),
                        10.0,
                    )

                    self.samples.append((
                        context, target_abs, padded_pos, padded_roles, mask,
                        np.float32(displacement),
                    ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        context, target_abs, h_pos, h_roles, mask, disp = self.samples[idx]
        return (
            torch.from_numpy(context),
            torch.from_numpy(target_abs),
            torch.from_numpy(h_pos),
            torch.from_numpy(h_roles.copy()),
            torch.from_numpy(mask.copy()),
            disp,
        )
        
        
class StructuredPoseDataset(torch.utils.data.Dataset):
    """
    Sliding window dataset with route context and per-frame target hold.

    Like PoseDataset but each sample additionally includes the normalized
    target hold position for the target frame. The hold sequence is derived
    automatically from ground truth poses.

    Args:
        sequences: List of (T_i, 17, 2) arrays in board space
            (filtered to NUM_CLIMBING_KEYPOINTS climbing keypoints internally).
        scores: List of (T_i, 17) confidence arrays.
        hold_positions: List of (N_i, 2) arrays of normalized hold positions.
        hold_roles: List of (N_i,) arrays of role IDs.
        route_holds: List of unordered hold dict lists (with 'x', 'y' in
            board coordinates) for deriving hold sequences.
        context_len: Number of frames in the input window.
        max_holds: Maximum holds per route (for padding).
        strides: List of temporal strides for sample construction.
    """

    def __init__(
        self,
        sequences: list,
        scores: list,
        hold_positions: list,
        hold_roles: list,
        route_holds: list[list[dict]],
        context_len: int = config.CONTEXT_WINDOW,
        max_holds: int = config.MAX_ROUTE_HOLDS,
        strides: list[int] | None = None,
    ):
        self.context_len = context_len
        self.max_holds = max_holds
        self.samples = []

        if strides is None:
            strides = [1]

        for seq, sc, h_pos, h_roles, r_holds in zip(
            sequences, scores, hold_positions, hold_roles, route_holds
        ):
            T = seq.shape[0]
            seq_filtered = seq[:, config.CLIMBING_KEYPOINT_INDICES, :]
            flat = seq_filtered.reshape(T, -1).astype("float32")

            # Derive hold sequence and per-frame targets (board space)
            hold_seq = derive_hold_sequence(seq_filtered, r_holds)
            if len(hold_seq) == 0:
                continue
            targets_board = extract_move_targets(seq_filtered, hold_seq)

            targets_norm = normalize_board_coords(targets_board)

            padded_pos, padded_roles, mask = pad_holds(h_pos, h_roles, max_holds)

            for stride in strides:
                min_frames = context_len * stride + stride
                if T < min_frames:
                    continue

                for t in range(context_len * stride, T, stride):
                    context_indices = range(t - context_len * stride, t, stride)
                    context = flat[list(context_indices)]
                    target_abs = flat[t]
                    delta = flat[t] - flat[t - stride]
                    displacement = min(
                        float((delta.reshape(-1, 2) ** 2).sum(axis=1).mean() ** 0.5),
                        10.0,
                    )

                    self.samples.append((
                        context, target_abs, padded_pos, padded_roles, mask,
                        np.float32(displacement),
                        targets_norm[t].copy(),
                    ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        (context, target_abs, h_pos, h_roles, mask, disp,
         tgt_pos) = self.samples[idx]
        return (
            torch.from_numpy(context),
            torch.from_numpy(target_abs),
            torch.from_numpy(h_pos),
            torch.from_numpy(h_roles.copy()),
            torch.from_numpy(mask.copy()),
            disp,
            torch.from_numpy(tgt_pos),
        )


# Bone pairs for length constraints: (parent_idx, child_idx)
# Rigid body segments only (excludes noisy head keypoints).
BONE_PAIRS = [
    (0, 2), (2, 4),     # left upper arm, left forearm
    (1, 3), (3, 5),     # right upper arm, right forearm
    (0, 6), (1, 7),     # left torso, right torso
    (6, 7),              # hips
    (0, 1),              # shoulders
    (6, 8), (8, 10),    # left thigh, left shin
    (7, 9), (9, 11),    # right thigh, right shin
]


def weighted_mse_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    displacement: torch.Tensor,
) -> torch.Tensor:
    """
    MSE loss with per-sample weighting by displacement magnitude.

    Frames with larger keypoint displacement get higher weight so the
    model pays attention to rare but important moments of movement
    rather than learning to predict "stay still."

    Args:
        predicted: (batch, NUM_CLIMBING_KEYPOINTS * 2) predicted absolute poses.
        target: (batch, NUM_CLIMBING_KEYPOINTS * 2) ground truth absolute poses.
        displacement: (batch,) mean keypoint displacement for each sample.

    Returns:
        Scalar weighted MSE loss.
    """
    weights = 1.0 + (2.0 * displacement)
    per_sample = ((predicted - target) ** 2).mean(dim=1)
    return (per_sample * weights).mean()


def hold_proximity_loss(
    predicted: torch.Tensor,
    target_hold: torch.Tensor,
) -> torch.Tensor:
    """
    Penalize distance between limb endpoints and the target hold.

    Encourages the model to produce poses where at least one
    limb is reaching toward or contacting the target hold.
    Uses the minimum distance across all four limb endpoints
    so the model is rewarded for any limb reaching the hold.

    Args:
        predicted: (batch, NUM_CLIMBING_KEYPOINTS * 2) predicted absolute poses.
        target_hold: (batch, 2) normalized target hold position.

    Returns:
        Scalar mean minimum-limb-to-hold distance.
    """
    poses = predicted.reshape(-1, config.NUM_CLIMBING_KEYPOINTS, 2)
    limb_positions = poses[:, [4, 5, 10, 11], :]  # wrists and ankles in climbing indices
    target = target_hold.unsqueeze(1)  # (batch, 1, 2)
    distances = (limb_positions - target).norm(dim=2)  # (batch, 4)
    return distances.min(dim=1).values.mean()


def compute_bone_lengths(poses_flat: torch.Tensor) -> torch.Tensor:
    """
    Compute bone lengths for all rigid segments.

    Args:
        poses_flat: (batch, NUM_CLIMBING_KEYPOINTS * 2) flattened keypoints.

    Returns:
        (batch, n_bones) bone lengths.
    """
    poses = poses_flat.reshape(-1, config.NUM_CLIMBING_KEYPOINTS, 2)
    lengths = []
    for i, j in BONE_PAIRS:
        diff = poses[:, i, :] - poses[:, j, :]
        lengths.append(diff.norm(dim=1))
    return torch.stack(lengths, dim=1)

def project_bone_lengths(
    poses_flat: torch.Tensor,
    max_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Differentiable bone-length projection: clamp bones exceeding max length.

    Iterates parent -> child through BONE_PAIRS, pinning the parent and
    pulling the child inward when the bone exceeds its max. Only shortens,
    never lengthens (preserves foreshortening).

    Because this uses in-place-safe tensor operations and no conditionals
    on values, gradients flow through the projection, teaching the model
    to produce valid skeletons directly.

    Args:
        poses_flat: (batch, NUM_CLIMBING_KEYPOINTS * 2) predicted absolute poses.
        max_lengths: (n_bones,) maximum valid length per bone.

    Returns:
        (batch, NUM_CLIMBING_KEYPOINTS * 2) projected poses with no bone exceeding its max.
    """
    poses = poses_flat.reshape(-1, config.NUM_CLIMBING_KEYPOINTS, 2).clone()

    for bone_idx, (parent, child) in enumerate(BONE_PAIRS):
        diff = poses[:, child, :] - poses[:, parent, :]
        current_len = diff.norm(dim=1, keepdim=True).clamp(min=1e-6)
        scale = (max_lengths[bone_idx] / current_len).clamp(max=1.0)
        poses[:, child, :] = poses[:, parent, :] + diff * scale

    return poses.reshape(-1, config.NUM_CLIMBING_KEYPOINTS * 2)


def bone_length_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Penalize bone-length deviations between predicted and target poses.

    Computes L1 loss on bone lengths, which directly penalizes limb
    stretching/shrinking without constraining joint positions.

    Args:
        predicted: (batch, NUM_CLIMBING_KEYPOINTS * 2) predicted absolute poses.
        target: (batch, NUM_CLIMBING_KEYPOINTS * 2) ground truth absolute poses.

    Returns:
        Scalar mean bone-length deviation.
    """
    pred_lengths = compute_bone_lengths(predicted)
    target_lengths = compute_bone_lengths(target)
    return (pred_lengths - target_lengths).abs().mean()


def enforce_bone_lengths(
    pose: np.ndarray,
    max_lengths: np.ndarray,
) -> np.ndarray:
    """
    Clamp bones that exceed their maximum observed length.

    Only shortens bones, never lengthens. This respects foreshortening
    (bones appear shorter in 2D when angled toward/away from camera)
    while preventing skeleton explosion during autoregressive rollout.

    Iterates parent -> child through BONE_PAIRS, pinning the parent and
    pulling the child inward when the bone exceeds its max length.

    Args:
        pose: (N_kp, 2) predicted keypoints.
        max_lengths: (n_bones,) maximum valid length for each bone in BONE_PAIRS.

    Returns:
        (N_kp, 2) corrected pose with no bone exceeding its max length.
    """
    out = pose.copy()

    for bone_idx, (parent, child) in enumerate(BONE_PAIRS):
        diff = out[child] - out[parent]
        current_len = np.linalg.norm(diff)
        if current_len < 1e-6:
            continue
        if current_len > max_lengths[bone_idx]:
            out[child] = out[parent] + diff * (max_lengths[bone_idx] / current_len)

    return out


def compute_reference_bone_lengths(
    sequences: list[np.ndarray],
    percentile: float = 95.0,
) -> np.ndarray:
    """
    Compute maximum plausible bone lengths from training sequences.

    Uses a high percentile rather than true max to be robust to
    occasional ViTPose tracking errors that produce spuriously long bones.

    Args:
        sequences: List of (T, 17, 2) arrays from the training set.
        percentile: Percentile to use as the max (default 95th).

    Returns:
        (n_bones,) array of max plausible bone lengths.
    """
    all_lengths = []
    for seq in sequences:
        seq_f = seq[:, config.CLIMBING_KEYPOINT_INDICES, :] if seq.shape[1] > config.NUM_CLIMBING_KEYPOINTS else seq
        for t in range(seq_f.shape[0]):
            lengths = []
            for i, j in BONE_PAIRS:
                diff = seq_f[t, i] - seq_f[t, j]
                lengths.append(np.linalg.norm(diff))
            all_lengths.append(lengths)
    return np.percentile(all_lengths, percentile, axis=0).astype(np.float32)

def _limb_arrival_threshold(limb_id: int) -> float:
    """Return the arrival threshold for a given limb ID."""
    if limb_id in config.HAND_LIMBS:
        return config.HAND_ARRIVAL_THRESHOLD
    return config.FOOT_ARRIVAL_THRESHOLD


def check_hold_arrival(
    pose: np.ndarray,
    hold_xy: np.ndarray,
) -> bool:
    """
    Check if any limb endpoint is within its arrival threshold of a hold.

    Args:
        pose: (NUM_CLIMBING_KEYPOINTS, 2) predicted pose in board space.
        hold_xy: (2,) board-space hold position.

    Returns:
        True if any limb is within threshold.
    """
    for limb_id in range(config.NUM_LIMBS):
        kp_idx = config.LIMB_KEYPOINTS[limb_id]
        threshold = _limb_arrival_threshold(limb_id)
        if np.linalg.norm(pose[kp_idx] - hold_xy) < threshold:
            return True
    return False


def derive_hold_sequence(
    sequence_poses: np.ndarray,
    route_holds: list[dict],
    arrival_frames: int = config.HOLD_ARRIVAL_FRAMES,
) -> list[dict]:
    """
    Derive the hold visit order from ground truth poses.

    Watches limb endpoints (wrists, ankles) over time and records
    the order in which route holds are reached. Holds can appear
    multiple times (matching, foot bumps, returning to a hold).
    Uses separate thresholds for hands and feet since ankle keypoints
    are offset from footholds.

    A hold is locked out after triggering to prevent re-triggering
    during sustained contact. The lockout clears when all limbs leave
    the hold's threshold, allowing re-detection on a later visit.

    Args:
        sequence_poses: (T, 17, 2) board-space keypoints.
        route_holds: Unordered list of hold dicts with 'x', 'y'.
        arrival_frames: Consecutive frames required to confirm arrival.

    Returns:
        Ordered list of hold dicts in the order they were visited.
        May contain duplicates if a hold is revisited.
    """
    T = sequence_poses.shape[0]
    ordered = []
    near_counts = [0] * len(route_holds)
    locked = set()  # holds locked out during sustained contact

    for t in range(T):
        for hold_idx, hold in enumerate(route_holds):
            hold_xy = np.array([hold["x"], hold["y"]])
            any_near = check_hold_arrival(sequence_poses[t], hold_xy)

            if any_near:
                if hold_idx in locked:
                    continue  # still in contact, don't re-count
                near_counts[hold_idx] += 1
                if near_counts[hold_idx] >= arrival_frames:
                    ordered.append(hold)
                    locked.add(hold_idx)
                    near_counts[hold_idx] = 0
            else:
                near_counts[hold_idx] = 0
                locked.discard(hold_idx)  # left the hold, can detect again

    return ordered


def extract_move_targets(
    sequence_poses: np.ndarray,
    hold_sequence: list[dict],
    arrival_frames: int = config.HOLD_ARRIVAL_FRAMES,
) -> np.ndarray:
    """
    Derive per-frame target hold from an ordered hold sequence.

    Walks through hold_sequence in order. The current target is held
    constant until any limb endpoint stays within its arrival threshold
    for arrival_frames consecutive frames, then advances. If the climber
    is already at the next hold when advancing, skips ahead to avoid
    getting stuck on holds being departed.

    Args:
        sequence_poses: (T, 17, 2) board-space keypoints.
        hold_sequence: Ordered list of hold dicts with 'x', 'y' in
            board coordinates.
        arrival_frames: Consecutive frames a limb must be within
            threshold before advancing.

    Returns:
        (T, 2) board-space (x, y) of the target hold per frame.
    """
    T = sequence_poses.shape[0]
    targets = np.zeros((T, 2), dtype=np.float32)

    seq_idx = 0
    n_holds = len(hold_sequence)
    consecutive_near = 0

    for t in range(T):
        target = hold_sequence[min(seq_idx, n_holds - 1)]
        targets[t] = [target["x"], target["y"]]

        if seq_idx < n_holds:
            if check_hold_arrival(sequence_poses[t], np.array([target["x"], target["y"]])):
                consecutive_near += 1
                if consecutive_near >= arrival_frames:
                    seq_idx += 1
                    consecutive_near = 0
                    # Skip past any subsequent holds we're already at
                    while seq_idx < n_holds and check_hold_arrival(
                        sequence_poses[t],
                        np.array([hold_sequence[seq_idx]["x"], hold_sequence[seq_idx]["y"]]),
                    ):
                        seq_idx += 1
                    # Update target for this frame
                    target = hold_sequence[min(seq_idx, n_holds - 1)]
                    targets[t] = [target["x"], target["y"]]
            else:
                consecutive_near = 0

    return targets