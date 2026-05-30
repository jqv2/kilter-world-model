"""
Pymunk-based 2D ragdoll for the RL climbing baseline.

9 rigid body segments connected by 8 pivot joints with rotary limits
and motors.  Gravity acts along the wall plane (projected by wall angle).
All segments share a collision group so they pass through each other,
which is correct for a 2D projection of 3D climbing.

The ragdoll produces (12, 2) keypoint arrays in board units, identical
in format to the world model output, and plugs directly into the
existing evaluation and visualization pipelines.

Public API
----------
create_space            Create a pre-configured Pymunk Space.
compute_rl_bone_lengths Derive segment lengths from training pose data.
create_ragdoll          Build the full ragdoll in a Space.
extract_keypoints       Read (12, 2) board-unit keypoints from body state.
create_hold_joint       Attach a limb to a hold.
destroy_hold_joint      Release a limb from a hold.
set_motor_rate          Set a joint motor's target angular velocity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pymunk

import config

# ──────────────────────────────────────────────────────────────────────
# De Leva (1996) mass fractions — combined segments for the ragdoll
# ──────────────────────────────────────────────────────────────────────
# torso  = head (0.0694) + trunk (0.4346)  = 0.5040
# thigh  = thigh                            = 0.1416  (×2)
# shin   = shank (0.0433) + foot (0.0137)  = 0.0570  (×2)
# upper_arm = upper arm                    = 0.0271  (×2)
# forearm = forearm (0.0162) + hand (0.0061) = 0.0223  (×2)
# Total: 0.5040 + 2*(0.1416+0.0570+0.0271+0.0223) = 1.0000

_MASS_FRACTIONS = {
    "torso": 0.5040,
    "upper_arm": 0.0271,
    "forearm": 0.0223,
    "thigh": 0.1416,
    "shin": 0.0570,
}

# Mass sub-components needed for torso CoM and combined-segment CoM
_TRUNK_MASS_FRAC = 0.4346
_HEAD_MASS_FRAC = 0.0694

# CoM as fraction of segment length from proximal joint (De Leva 1996)
_COM_PROXIMAL = {
    "trunk": 0.5138,
    "head": 0.5002,
    "upper_arm": 0.5772,
    "forearm": 0.4574,
    "hand": 0.7900,
    "thigh": 0.4095,
    "shank": 0.4395,
    "foot": 0.4415,
}

# Approximate appendage length as fraction of parent segment length
_HAND_FOREARM_RATIO = 0.4
_FOOT_SHANK_RATIO = 0.4


def _combined_com_fraction(
    main_mass: float,
    main_com_frac: float,
    appendage_mass: float,
    appendage_com_frac: float,
    appendage_length_ratio: float,
) -> float:
    """CoM fraction from proximal for a merged segment (e.g. forearm+hand).

    The main segment runs from the proximal joint to the distal joint
    (length = 1.0 in normalised units).  The appendage hangs off the
    distal end with length = appendage_length_ratio × main length.

    Returns the combined CoM as a fraction of the *main* segment length
    measured from the proximal joint.
    """
    main_com = main_com_frac
    appendage_com = 1.0 + appendage_com_frac * appendage_length_ratio
    return (main_mass * main_com + appendage_mass * appendage_com) / (
        main_mass + appendage_mass
    )


# Precomputed CoM fractions for each segment type
_SEGMENT_COM_FRACS: dict[str, float] = {
    "upper_arm": _COM_PROXIMAL["upper_arm"],
    "forearm": _combined_com_fraction(
        0.0162, _COM_PROXIMAL["forearm"],
        0.0061, _COM_PROXIMAL["hand"],
        _HAND_FOREARM_RATIO,
    ),
    "thigh": _COM_PROXIMAL["thigh"],
    "shin": _combined_com_fraction(
        0.0433, _COM_PROXIMAL["shank"],
        0.0137, _COM_PROXIMAL["foot"],
        _FOOT_SHANK_RATIO,
    ),
}

# ──────────────────────────────────────────────────────────────────────
# Joint limits and motor torques
# ──────────────────────────────────────────────────────────────────────

def _get_joint_limits(joint_name: str) -> tuple[float, float] | None:
    """Angle limits for a joint, or None for free rotation.

    Elbows are side-dependent: bending is clockwise on the left
    (negative relative angle) and counter-clockwise on the right.
    Knees get the full 360° range but cannot do multiple revolutions.
    Hips and shoulders rotate freely (no limit joint created).
    """
    if "shoulder" in joint_name or "hip" in joint_name:
        return None
    if "knee" in joint_name:
        return (-math.pi, math.pi)
    if "elbow" in joint_name:
        if "left" in joint_name:
            return (-math.pi, 0.0)
        return (0.0, math.pi)
    return None

_MOTOR_TORQUES: dict[str, float] = {
    "shoulder": config.RL_MOTOR_TORQUE_SHOULDER,
    "elbow": config.RL_MOTOR_TORQUE_ELBOW,
    "hip": config.RL_MOTOR_TORQUE_HIP,
    "knee": config.RL_MOTOR_TORQUE_KNEE,
}

# ──────────────────────────────────────────────────────────────────────
# Structural constants
# ──────────────────────────────────────────────────────────────────────

_SEGMENT_RADIUS = 0.015  # meters, cosmetic half-width for Segment shapes
_COLLISION_GROUP = 1      # all ragdoll segments share this group

# Limb-name → (end-effector body key, segment type for that body)
_LIMB_BODIES: dict[str, tuple[str, str]] = {
    "left_hand": ("left_forearm", "forearm"),
    "right_hand": ("right_forearm", "forearm"),
    "left_foot": ("left_shin", "shin"),
    "right_foot": ("right_shin", "shin"),
}

# ──────────────────────────────────────────────────────────────────────
# Bone-length computation from training data
# ──────────────────────────────────────────────────────────────────────

# COCO 17-keypoint index pairs: [(left_pair), (right_pair)]
_BONE_PAIRS_COCO: dict[str, list[tuple[int, int]]] = {
    "upper_arm": [(5, 7), (6, 8)],
    "forearm": [(7, 9), (8, 10)],
    "thigh": [(11, 13), (12, 14)],
    "shin": [(13, 15), (14, 16)],
    "torso": [(5, 11), (6, 12)],
}
_WIDTH_PAIRS_COCO: dict[str, tuple[int, int]] = {
    "half_shoulder_width": (5, 6),
    "half_hip_width": (11, 12),
}


def compute_rl_bone_lengths(
    sequences: list[np.ndarray],
    percentile: float = config.RL_BONE_LENGTH_PERCENTILE,
) -> dict[str, float]:
    """Compute representative bone lengths from training pose sequences.

    For each bone, computes the Euclidean distance per frame across all
    sequences, averages left and right sides, and takes the given
    percentile.  Returns 5 bone lengths and 2 half-widths, all in
    board units.

    Args:
        sequences: List of (T_i, 17, 2) arrays in board space
            (full COCO keypoints).
        percentile: Percentile to use (e.g. 95 for 95th-percentile).

    Returns:
        Dict with keys ``upper_arm``, ``forearm``, ``thigh``, ``shin``,
        ``torso``, ``half_shoulder_width``, ``half_hip_width``.
    """
    result: dict[str, float] = {}

    for name, pairs in _BONE_PAIRS_COCO.items():
        all_lengths: list[np.ndarray] = []
        for seq in sequences:
            for i, j in pairs:
                all_lengths.append(np.linalg.norm(seq[:, i] - seq[:, j], axis=1))
        result[name] = float(np.percentile(np.concatenate(all_lengths), percentile))

    for name, (i, j) in _WIDTH_PAIRS_COCO.items():
        all_widths = [np.linalg.norm(seq[:, i] - seq[:, j], axis=1) for seq in sequences]
        result[name] = float(np.percentile(np.concatenate(all_widths), percentile)) / 2.0

    return result


# ──────────────────────────────────────────────────────────────────────
# Ragdoll dataclass
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Ragdoll:
    """All Pymunk references for the 9-segment ragdoll.

    Attributes:
        bodies: Segment name → ``pymunk.Body``.
        shapes: Segment name → list of ``pymunk.Shape``.
        joints: Joint name → ``(PivotJoint, RotaryLimitJoint, SimpleMotor)``.
        hold_joints: Limb name → active hold ``PivotJoint`` (mutable).
        bone_lengths_m: Bone name → length in metres.
        torso_com_offset: Distance from hip-center to torso CoM along
            the torso's local x-axis (metres).  Needed for keypoint
            extraction and hold-joint anchoring.
    """

    bodies: dict[str, pymunk.Body]
    shapes: dict[str, list[pymunk.Shape]]
    joints: dict[str, tuple[pymunk.PivotJoint, pymunk.RotaryLimitJoint | None, pymunk.SimpleMotor]]
    hold_joints: dict[str, pymunk.PivotJoint] = field(default_factory=dict)
    bone_lengths_m: dict[str, float] = field(default_factory=dict)
    torso_com_offset: float = 0.0


# ──────────────────────────────────────────────────────────────────────
# Space and ragdoll creation
# ──────────────────────────────────────────────────────────────────────

def create_space() -> pymunk.Space:
    """Create a Pymunk Space with climbing-wall gravity and damping.

    Gravity is projected onto the wall plane:
    ``(0, -g * cos(wall_angle))``.
    """
    space = pymunk.Space()
    angle_rad = math.radians(config.RL_WALL_ANGLE_DEG)
    space.gravity = (0, -config.RL_GRAVITY * math.cos(angle_rad))
    space.damping = config.RL_SPACE_DAMPING
    return space


def _tip_local(seg_type: str, bone_lengths_m: dict[str, float]) -> tuple[float, float]:
    """Local coordinates of the distal tip of a limb segment."""
    L = bone_lengths_m[seg_type]
    return (L * (1.0 - _SEGMENT_COM_FRACS[seg_type]), 0.0)


def _proximal_local(seg_type: str, bone_lengths_m: dict[str, float]) -> tuple[float, float]:
    """Local coordinates of the proximal end of a limb segment."""
    L = bone_lengths_m[seg_type]
    return (-L * _SEGMENT_COM_FRACS[seg_type], 0.0)


def _distal_local(seg_type: str, bone_lengths_m: dict[str, float]) -> tuple[float, float]:
    """Alias for _tip_local (distal end = tip)."""
    return _tip_local(seg_type, bone_lengths_m)


def _place_body(body: pymunk.Body, anchor_local: tuple[float, float],
                target_world: tuple[float, float], angle: float) -> None:
    """Position *body* so that *anchor_local* lands at *target_world*."""
    body.angle = angle
    ca, sa = math.cos(angle), math.sin(angle)
    ax, ay = anchor_local
    body.position = (
        target_world[0] - (ax * ca - ay * sa),
        target_world[1] - (ax * sa + ay * ca),
    )


def create_ragdoll(
    space: pymunk.Space,
    bone_lengths: dict[str, float],
    position: tuple[float, float] = (0.0, 0.0),
) -> Ragdoll:
    """Create a 9-segment ragdoll in the given Pymunk Space.

    The ragdoll is placed at *position* (metres) with the torso upright
    and limbs hanging in a neutral pose.  All segments share a collision
    group so they pass through each other.

    Args:
        space: Pymunk Space to populate.
        bone_lengths: Dict from :func:`compute_rl_bone_lengths`
            (board units).  Keys: ``upper_arm``, ``forearm``, ``thigh``,
            ``shin``, ``torso``, ``half_shoulder_width``,
            ``half_hip_width``.
        position: Initial torso centre-of-mass in metres ``(x, y)``.

    Returns:
        :class:`Ragdoll` with all body, shape, and joint references.
    """
    bu2m = config.RL_BOARD_UNIT_TO_METERS
    M = config.RL_BODY_MASS_KG
    R_head = config.RL_HEAD_RADIUS
    sf = pymunk.ShapeFilter(group=_COLLISION_GROUP)

    # Convert bone lengths to metres
    bl = {k: v * bu2m for k, v in bone_lengths.items()}

    bodies: dict[str, pymunk.Body] = {}
    shapes: dict[str, list[pymunk.Shape]] = {}

    # ── Torso (trunk + head on one body) ───────────────────────────
    L_torso = bl["torso"]
    m_trunk = _TRUNK_MASS_FRAC * M
    m_head = _HEAD_MASS_FRAC * M
    m_torso = _MASS_FRACTIONS["torso"] * M

    trunk_com_from_hip = _COM_PROXIMAL["trunk"] * L_torso
    head_com_from_hip = L_torso + _COM_PROXIMAL["head"] * (2 * R_head)
    torso_com = (m_trunk * trunk_com_from_hip + m_head * head_com_from_hip) / m_torso

    hip_local = (-torso_com, 0.0)
    shoulder_local = (L_torso - torso_com, 0.0)
    head_center_local = (L_torso + R_head - torso_com, 0.0)

    I_torso = (
        pymunk.moment_for_segment(m_trunk, hip_local, shoulder_local, _SEGMENT_RADIUS)
        + pymunk.moment_for_circle(m_head, 0, R_head, head_center_local)
    )

    torso_body = pymunk.Body(m_torso, I_torso)
    torso_body.position = position
    torso_body.angle = math.pi / 2  # upright

    trunk_shape = pymunk.Segment(torso_body, hip_local, shoulder_local, _SEGMENT_RADIUS)
    trunk_shape.filter = sf
    head_shape = pymunk.Circle(torso_body, R_head, head_center_local)
    head_shape.filter = sf

    space.add(torso_body, trunk_shape, head_shape)
    bodies["torso"] = torso_body
    shapes["torso"] = [trunk_shape, head_shape]

    # ── Limb segments ──────────────────────────────────────────────
    seg_defs = [
        ("left_upper_arm", "upper_arm"),
        ("right_upper_arm", "upper_arm"),
        ("left_forearm", "forearm"),
        ("right_forearm", "forearm"),
        ("left_thigh", "thigh"),
        ("right_thigh", "thigh"),
        ("left_shin", "shin"),
        ("right_shin", "shin"),
    ]

    for seg_name, seg_type in seg_defs:
        length = bl[seg_type]
        mass = _MASS_FRACTIONS[seg_type] * M
        com_frac = _SEGMENT_COM_FRACS[seg_type]
        prox = (-com_frac * length, 0.0)
        dist = ((1.0 - com_frac) * length, 0.0)

        moment = pymunk.moment_for_segment(mass, prox, dist, _SEGMENT_RADIUS)
        body = pymunk.Body(mass, moment)
        # Temporary position; corrected below when initial pose is set
        body.position = position

        shape = pymunk.Segment(body, prox, dist, _SEGMENT_RADIUS)
        shape.filter = sf

        space.add(body, shape)
        bodies[seg_name] = body
        shapes[seg_name] = [shape]

    # ── Joint anchor helpers ───────────────────────────────────────
    hsw = bl["half_shoulder_width"]
    hhw = bl["half_hip_width"]
    shoulder_x = L_torso - torso_com
    hip_x = -torso_com

    # Torso-local anchors (x = along spine, y = perpendicular)
    # When torso angle = π/2: local +y → world −x, so positive y = left.
    torso_anchors = {
        "left_shoulder": (shoulder_x, hsw),
        "right_shoulder": (shoulder_x, -hsw),
        "left_hip": (hip_x, hhw),
        "right_hip": (hip_x, -hhw),
    }

    # ── Joints ─────────────────────────────────────────────────────
    joint_defs = [
        # (name, parent_key, child_key, parent_anchor, child_seg_type, joint_type)
        ("left_shoulder", "torso", "left_upper_arm",
         torso_anchors["left_shoulder"], "upper_arm", "shoulder"),
        ("right_shoulder", "torso", "right_upper_arm",
         torso_anchors["right_shoulder"], "upper_arm", "shoulder"),
        ("left_elbow", "left_upper_arm", "left_forearm",
         _distal_local("upper_arm", bl), "forearm", "elbow"),
        ("right_elbow", "right_upper_arm", "right_forearm",
         _distal_local("upper_arm", bl), "forearm", "elbow"),
        ("left_hip", "torso", "left_thigh",
         torso_anchors["left_hip"], "thigh", "hip"),
        ("right_hip", "torso", "right_thigh",
         torso_anchors["right_hip"], "thigh", "hip"),
        ("left_knee", "left_thigh", "left_shin",
         _distal_local("thigh", bl), "shin", "knee"),
        ("right_knee", "right_thigh", "right_shin",
         _distal_local("thigh", bl), "shin", "knee"),
    ]

    joints: dict[str, tuple[pymunk.PivotJoint, pymunk.RotaryLimitJoint, pymunk.SimpleMotor]] = {}

    for name, parent_key, child_key, parent_anchor, child_seg_type, joint_type in joint_defs:
        parent_body = bodies[parent_key]
        child_body = bodies[child_key]
        child_anchor = _proximal_local(child_seg_type, bl)

        pivot = pymunk.PivotJoint(parent_body, child_body, parent_anchor, child_anchor)
        pivot.collide_bodies = False

        limits = _get_joint_limits(name)
        if limits is not None:
            limit = pymunk.RotaryLimitJoint(parent_body, child_body, *limits)
            space.add(limit)
        else:
            limit = None

        motor = pymunk.SimpleMotor(parent_body, child_body, 0.0)
        motor.max_force = 0.0

        space.add(pivot, motor)
        joints[name] = (pivot, limit, motor)

    # ── Set initial pose (upright, limbs hanging) ──────────────────
    _set_initial_pose(bodies, joints, torso_anchors, bl, torso_com, position)

    return Ragdoll(
        bodies=bodies,
        shapes=shapes,
        joints=joints,
        bone_lengths_m=bl,
        torso_com_offset=torso_com,
    )


def _set_initial_pose(
    bodies: dict[str, pymunk.Body],
    joints: dict,
    torso_anchors: dict[str, tuple[float, float]],
    bl: dict[str, float],
    torso_com: float,
    position: tuple[float, float],
) -> None:
    """Place every body in an upright, arms-and-legs-hanging pose.

    Called once at creation time so the ragdoll starts in a
    physically sensible configuration rather than overlapping at the
    origin.
    """
    torso = bodies["torso"]
    torso.position = position
    torso.angle = math.pi / 2

    arm_angle = -math.pi / 2  # hanging straight down
    leg_angle = -math.pi / 2

    for side in ("left", "right"):
        # ── Arm chain ──
        shoulder_world = torso.local_to_world(torso_anchors[f"{side}_shoulder"])
        ua = bodies[f"{side}_upper_arm"]
        _place_body(ua, _proximal_local("upper_arm", bl), shoulder_world, arm_angle)

        elbow_world = ua.local_to_world(_distal_local("upper_arm", bl))
        fa = bodies[f"{side}_forearm"]
        _place_body(fa, _proximal_local("forearm", bl), elbow_world, arm_angle)

        # ── Leg chain ──
        hip_world = torso.local_to_world(torso_anchors[f"{side}_hip"])
        th = bodies[f"{side}_thigh"]
        _place_body(th, _proximal_local("thigh", bl), hip_world, leg_angle)

        knee_world = th.local_to_world(_distal_local("thigh", bl))
        sh = bodies[f"{side}_shin"]
        _place_body(sh, _proximal_local("shin", bl), knee_world, leg_angle)


# ──────────────────────────────────────────────────────────────────────
# Keypoint extraction
# ──────────────────────────────────────────────────────────────────────

def extract_keypoints(ragdoll: Ragdoll) -> np.ndarray:
    """Read 12 climbing keypoints from current Pymunk body positions.

    Returns:
        ``(12, 2)`` float32 array in board units, ordered:
        left_shoulder, right_shoulder, left_elbow, right_elbow,
        left_wrist, right_wrist, left_hip, right_hip, left_knee,
        right_knee, left_ankle, right_ankle.
    """
    m2bu = 1.0 / config.RL_BOARD_UNIT_TO_METERS
    bl = ragdoll.bone_lengths_m

    def _joint_world(name: str) -> tuple[float, float]:
        pivot = ragdoll.joints[name][0]
        return pivot.a.local_to_world(pivot.anchor_a)

    def _tip_world(body_name: str, seg_type: str) -> tuple[float, float]:
        body = ragdoll.bodies[body_name]
        return body.local_to_world(_tip_local(seg_type, bl))

    pts = [
        _joint_world("left_shoulder"),
        _joint_world("right_shoulder"),
        _joint_world("left_elbow"),
        _joint_world("right_elbow"),
        _tip_world("left_forearm", "forearm"),
        _tip_world("right_forearm", "forearm"),
        _joint_world("left_hip"),
        _joint_world("right_hip"),
        _joint_world("left_knee"),
        _joint_world("right_knee"),
        _tip_world("left_shin", "shin"),
        _tip_world("right_shin", "shin"),
    ]

    return (np.array(pts, dtype=np.float64) * m2bu).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# Hold attachment / release
# ──────────────────────────────────────────────────────────────────────

def create_hold_joint(
    ragdoll: Ragdoll,
    space: pymunk.Space,
    limb: str,
    hold_pos_board: tuple[float, float],
) -> bool:
    """Attach a limb's end effector to a hold via a pivot joint.

    The joint anchors the limb at its current tip position (no
    teleporting).  Rejects the grab if the limb is already anchored
    or the tip is beyond ``RL_GRAB_THRESHOLD`` of the hold.

    Args:
        ragdoll: Ragdoll to modify.
        space: Pymunk Space containing the ragdoll.
        limb: ``"left_hand"``, ``"right_hand"``, ``"left_foot"``,
            or ``"right_foot"``.
        hold_pos_board: Hold position in board units ``(x, y)``.

    Returns:
        True if the joint was created.
        
    For initialization (episode reset, testing), use :func:`reset_pose`
    instead, which positions the ragdoll via IK and creates joints
    without a distance check.
    """
    if limb in ragdoll.hold_joints:
        return False

    body_name, seg_type = _LIMB_BODIES[limb]
    body = ragdoll.bodies[body_name]
    tip_l = _tip_local(seg_type, ragdoll.bone_lengths_m)
    tip_w = body.local_to_world(tip_l)

    bu2m = config.RL_BOARD_UNIT_TO_METERS
    hold_m = (hold_pos_board[0] * bu2m, hold_pos_board[1] * bu2m)

    dist_bu = math.hypot(tip_w[0] - hold_m[0], tip_w[1] - hold_m[1]) / bu2m
    if dist_bu > config.RL_GRAB_THRESHOLD:
        return False

    joint = pymunk.PivotJoint(body, space.static_body, tip_l, hold_m)
    joint.max_force = config.RL_GRIP_MAX_FORCE
    joint.collide_bodies = False
    space.add(joint)
    ragdoll.hold_joints[limb] = joint
    return True


def destroy_hold_joint(
    ragdoll: Ragdoll,
    space: pymunk.Space,
    limb: str,
) -> bool:
    """Release a limb from its currently held position.

    Args:
        ragdoll: Ragdoll to modify.
        space: Pymunk Space containing the ragdoll.
        limb: ``"left_hand"``, ``"right_hand"``, ``"left_foot"``,
            or ``"right_foot"``.

    Returns:
        True if a joint was destroyed, False if the limb was free.
    """
    joint = ragdoll.hold_joints.pop(limb, None)
    if joint is None:
        return False
    space.remove(joint)
    return True


# ──────────────────────────────────────────────────────────────────────
# Motor control
# ──────────────────────────────────────────────────────────────────────

def set_motor_rate(ragdoll: Ragdoll, joint_name: str, rate: float) -> None:
    """Set the angular velocity target for a joint's motor.

    Enables the motor at its configured torque limit.  A rate of 0
    brakes the joint (resists rotation up to max torque).  To fully
    disable a motor, set ``ragdoll.joints[name][2].max_force = 0``.

    Args:
        ragdoll: Ragdoll containing the joint.
        joint_name: One of ``left_shoulder``, ``right_shoulder``,
            ``left_elbow``, ``right_elbow``, ``left_hip``,
            ``right_hip``, ``left_knee``, ``right_knee``.
        rate: Target angular velocity in rad/s.  Clamped to
            ±``RL_MAX_MOTOR_SPEED``.
    """
    _, _, motor = ragdoll.joints[joint_name]
    motor.rate = max(-config.RL_MAX_MOTOR_SPEED, min(config.RL_MAX_MOTOR_SPEED, rate))
    # Infer joint type from name and set the configured torque limit
    joint_type = joint_name.split("_", 1)[1]  # "left_shoulder" → "shoulder"
    motor.max_force = _MOTOR_TORQUES[joint_type]
    
    
# ──────────────────────────────────────────────────────────────────────
# Pose initialization (IK-based)
# ──────────────────────────────────────────────────────────────────────

def _solve_ik_2bone(
    root: tuple[float, float],
    target: tuple[float, float],
    len_a: float,
    len_b: float,
    bend_sign: float = -1.0,
) -> tuple[float, float]:
    """2D two-bone IK: find the mid-joint (elbow/knee) position.

    Args:
        root: Proximal joint position (shoulder or hip) in metres.
        target: End-effector target (wrist or ankle) in metres.
        len_a: Upper bone length (upper arm or thigh).
        len_b: Lower bone length (forearm or shin).
        bend_sign: -1 bends the joint clockwise from the root→target
            line (elbows down), +1 bends counter-clockwise (knees
            forward).

    Returns:
        Mid-joint world position in metres.
    """
    dx = target[0] - root[0]
    dy = target[1] - root[1]
    dist = math.hypot(dx, dy)

    if dist < 1e-6:
        return (root[0], root[1] - len_a)

    max_reach = len_a + len_b
    if dist >= max_reach:
        s = len_a / max_reach
        return (root[0] + dx * s, root[1] + dy * s)

    cos_a = (len_a ** 2 + dist ** 2 - len_b ** 2) / (2 * len_a * dist)
    cos_a = max(-1.0, min(1.0, cos_a))
    angle = math.acos(cos_a)
    base = math.atan2(dy, dx)
    mid_angle = base + bend_sign * angle

    return (root[0] + len_a * math.cos(mid_angle),
            root[1] + len_a * math.sin(mid_angle))


def reset_pose(
    ragdoll: Ragdoll,
    space: pymunk.Space,
    hand_holds: dict[str, tuple[float, float]],
    foot_positions: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Position the ragdoll with hands at holds via IK and create joints.

    Solves 2-bone IK for each arm so wrists land exactly at hold
    positions, positions the torso below the holds at a natural
    hanging depth, and creates hold joints for all specified contacts.
    Existing hold joints are cleared first.

    Used for Phase 1 testing and Phase 2 episode resets.

    Args:
        ragdoll: Ragdoll to reposition.
        space: Pymunk Space containing the ragdoll.
        hand_holds: Limb name → ``(x, y)`` in board units.
            e.g. ``{"left_hand": (60, 120), "right_hand": (84, 120)}``.
        foot_positions: Optional limb name → ``(x, y)`` in board units.
            When omitted, legs hang straight down from the hips.
    """
    bu2m = config.RL_BOARD_UNIT_TO_METERS
    bl = ragdoll.bone_lengths_m

    # Clear existing hold joints
    for limb in list(ragdoll.hold_joints):
        destroy_hold_joint(ragdoll, space, limb)

    # Convert to metres
    hands_m = {k: (v[0] * bu2m, v[1] * bu2m) for k, v in hand_holds.items()}
    feet_m = {k: (v[0] * bu2m, v[1] * bu2m) for k, v in (foot_positions or {}).items()}

    # ── Torso: centred between holds, shoulders below hold level ───
    all_hand = list(hands_m.values())
    mid_x = sum(p[0] for p in all_hand) / len(all_hand)
    mid_y = sum(p[1] for p in all_hand) / len(all_hand)

    shoulder_offset = bl["torso"] - ragdoll.torso_com_offset
    arm_reach = bl["upper_arm"] + bl["forearm"]

    torso_x = mid_x
    torso_y = mid_y - shoulder_offset - 0.7 * arm_reach

    torso = ragdoll.bodies["torso"]
    torso.position = (torso_x, torso_y)
    torso.angle = math.pi / 2

    # ── World positions of shoulder and hip joints ─────────────────
    hsw = bl["half_shoulder_width"]
    hhw = bl["half_hip_width"]
    shoulder_y = torso_y + shoulder_offset
    hip_y = torso_y - ragdoll.torso_com_offset

    shoulders = {
        "left": (torso_x - hsw, shoulder_y),
        "right": (torso_x + hsw, shoulder_y),
    }
    hips = {
        "left": (torso_x - hhw, hip_y),
        "right": (torso_x + hhw, hip_y),
    }

    # ── Arm IK ─────────────────────────────────────────────────────
    for limb, hold_m in hands_m.items():
        side = "left" if "left" in limb else "right"
        shoulder = shoulders[side]
        elbow = _solve_ik_2bone(shoulder, hold_m, bl["upper_arm"], bl["forearm"],
                                bend_sign=-1.0)

        ua_angle = math.atan2(elbow[1] - shoulder[1], elbow[0] - shoulder[0])
        _place_body(ragdoll.bodies[f"{side}_upper_arm"],
                    _proximal_local("upper_arm", bl), shoulder, ua_angle)

        fa_angle = math.atan2(hold_m[1] - elbow[1], hold_m[0] - elbow[0])
        _place_body(ragdoll.bodies[f"{side}_forearm"],
                    _proximal_local("forearm", bl), elbow, fa_angle)

    # ── Leg IK or straight hang ────────────────────────────────────
    for side in ("left", "right"):
        limb = f"{side}_foot"
        hip = hips[side]

        if limb in feet_m:
            foot_m = feet_m[limb]
            knee = _solve_ik_2bone(hip, foot_m, bl["thigh"], bl["shin"],
                                   bend_sign=1.0)
            th_angle = math.atan2(knee[1] - hip[1], knee[0] - hip[0])
            sh_angle = math.atan2(foot_m[1] - knee[1], foot_m[0] - knee[0])
        else:
            th_angle = -math.pi / 2
            sh_angle = -math.pi / 2

        th = ragdoll.bodies[f"{side}_thigh"]
        _place_body(th, _proximal_local("thigh", bl), hip, th_angle)

        knee_world = th.local_to_world(_distal_local("thigh", bl))
        sh = ragdoll.bodies[f"{side}_shin"]
        _place_body(sh, _proximal_local("shin", bl), knee_world, sh_angle)

    # ── Zero all velocities ────────────────────────────────────────
    for body in ragdoll.bodies.values():
        body.velocity = (0, 0)
        body.angular_velocity = 0

    # ── Create hold joints (no distance check) ────────────────────
    all_contacts = {**hand_holds, **(foot_positions or {})}
    for limb, pos_board in all_contacts.items():
        body_name, seg_type = _LIMB_BODIES[limb]
        body = ragdoll.bodies[body_name]
        tip_l = _tip_local(seg_type, bl)
        pos_m = (pos_board[0] * bu2m, pos_board[1] * bu2m)

        joint = pymunk.PivotJoint(body, space.static_body, tip_l, pos_m)
        joint.max_force = config.RL_GRIP_MAX_FORCE
        joint.collide_bodies = False
        space.add(joint)
        ragdoll.hold_joints[limb] = joint
    
    
def extract_head_position(ragdoll: Ragdoll) -> np.ndarray:
    """Head centre in board units, for visualization only.

    Not part of the 12 climbing keypoints — the head is drawn as an
    extra element in RL-specific renders.

    Returns:
        (2,) float32 array in board units.
    """
    torso = ragdoll.bodies["torso"]
    L_torso = ragdoll.bone_lengths_m["torso"]
    R_head = config.RL_HEAD_RADIUS
    head_local = (L_torso + R_head - ragdoll.torso_com_offset, 0.0)
    world = torso.local_to_world(head_local)
    m2bu = 1.0 / config.RL_BOARD_UNIT_TO_METERS
    return np.array([world[0] * m2bu, world[1] * m2bu], dtype=np.float32)