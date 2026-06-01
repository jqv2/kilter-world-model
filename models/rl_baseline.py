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
prepare_routes_for_rl   Package dataset routes for the Gym environment.
ClimbingEnv             Gymnasium environment for RL climbing.
rollout_episode         Run one episode and collect rendering/metric data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pymunk
import gymnasium
from gymnasium import spaces
from models.world_model import resolve_hold_sequence_and_targets
from pipeline.routes import load_hold_order_edit, BOARD_X_MIN, BOARD_X_MAX, BOARD_Y_MIN, BOARD_Y_MAX

import config

################################################
# De Leva (1996) mass fractions. Combined segments for the ragdoll
################################################
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

################################################
# Joint limits and motor torques
################################################

def _get_joint_limits(joint_name: str) -> tuple[float, float] | None:
    """Angle limits for a joint, or None for free rotation.

    Elbows are side-dependent: bending is clockwise on the left
    (negative relative angle) and counter-clockwise on the right.
    Knees allow full (-π, π) range for drop-knees and heel hooks.
    Hips and shoulders rotate freely (no limit joint created).
    """
    if "shoulder" in joint_name or "hip" in joint_name:
        return None
    if "knee" in joint_name:
        if "right" in joint_name:
            return (-5 * math.pi / 6, 0.0)
        return (0.0, 5 * math.pi / 6)
    if "elbow" in joint_name:
        if "left" in joint_name:
            return (-5 * math.pi / 6, 0.0)
        return (0.0, 5 * math.pi / 6)
    return None

_MOTOR_TORQUES: dict[str, float] = {
    "shoulder": config.RL_MOTOR_TORQUE_SHOULDER,
    "elbow": config.RL_MOTOR_TORQUE_ELBOW,
    "hip": config.RL_MOTOR_TORQUE_HIP,
    "knee": config.RL_MOTOR_TORQUE_KNEE,
}

################################################
# Structural constants
################################################

_SEGMENT_RADIUS = 0.015  # meters, cosmetic half-width for Segment shapes
_COLLISION_GROUP = 1      # all ragdoll segments share this group

# Limb-name -> (end-effector body key, segment type for that body)
_LIMB_BODIES: dict[str, tuple[str, str]] = {
    "left_hand": ("left_forearm", "forearm"),
    "right_hand": ("right_forearm", "forearm"),
    "left_foot": ("left_shin", "shin"),
    "right_foot": ("right_shin", "shin"),
}

# Ordered names for vectorized observation/action
_JOINT_NAMES = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
]
_LIMB_NAMES = ["left_hand", "right_hand", "left_foot", "right_foot"]

# Joint name -> (parent body name, child body name)
_JOINT_BODIES = {
    "left_shoulder": ("torso", "left_upper_arm"),
    "right_shoulder": ("torso", "right_upper_arm"),
    "left_elbow": ("left_upper_arm", "left_forearm"),
    "right_elbow": ("right_upper_arm", "right_forearm"),
    "left_hip": ("torso", "left_thigh"),
    "right_hip": ("torso", "right_thigh"),
    "left_knee": ("left_thigh", "left_shin"),
    "right_knee": ("right_thigh", "right_shin"),
}

# Canonical role mapping (same as routes.py holds_to_array)
_ROLE_MAP = {
    12: 12, 13: 13, 14: 14, 15: 15,
    20: 12, 21: 13, 22: 14, 23: 15,
    24: 12, 25: 13, 26: 14, 27: 15,
    28: 12, 29: 13, 30: 14, 31: 15,
    32: 12, 33: 13, 34: 14, 35: 15,
    42: 12, 43: 13, 44: 14, 45: 15,
}

_ROLE_START = 12
_ROLE_FINISH = 14
_ROLE_FOOT_ONLY = 15

################################################
# Bone-length computation from training data
################################################

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
        Dict with keys upper_arm, forearm, thigh, shin,
        torso, half_shoulder_width, half_hip_width.
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

################################################
# Ragdoll dataclass
################################################

@dataclass
class Ragdoll:
    """All Pymunk references for the 9-segment ragdoll.

    Attributes:
        bodies: Segment name -> pymunk.Body.
        shapes: Segment name -> list of pymunk.Shape.
        joints: Joint name -> (PivotJoint, RotaryLimitJoint, SimpleMotor).
        hold_joints: Limb name -> active hold PivotJoint (mutable).
        bone_lengths_m: Bone name -> length in metres.
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

################################################
# Space and ragdoll creation
################################################

def create_space() -> pymunk.Space:
    """Create a Pymunk Space with climbing-wall gravity and damping.

    Gravity is projected onto the wall plane:
    (0, -g * cos(wall_angle)).
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
        bone_lengths: Dict from :func:compute_rl_bone_lengths
            (board units).  Keys: upper_arm, forearm, thigh,
            shin, torso, half_shoulder_width,
            half_hip_width.
        position: Initial torso centre-of-mass in metres (x, y).

    Returns:
        :class:Ragdoll with all body, shape, and joint references.
    """
    bu2m = config.RL_BOARD_UNIT_TO_METERS
    M = config.RL_BODY_MASS_KG
    R_head = config.RL_HEAD_RADIUS
    sf = pymunk.ShapeFilter(group=_COLLISION_GROUP)

    # Convert bone lengths to metres
    bl = {k: v * bu2m for k, v in bone_lengths.items()}

    bodies: dict[str, pymunk.Body] = {}
    shapes: dict[str, list[pymunk.Shape]] = {}

    # Torso (trunk + head on one body)
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

    # Limb segments
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

    # Joint anchor helpers
    hsw = bl["half_shoulder_width"]
    hhw = bl["half_hip_width"]
    shoulder_x = L_torso - torso_com
    hip_x = -torso_com

    # Torso-local anchors (x = along spine, y = perpendicular)
    # When torso angle = π/2: local +y -> world −x, so positive y = left.
    torso_anchors = {
        "left_shoulder": (shoulder_x, hsw),
        "right_shoulder": (shoulder_x, -hsw),
        "left_hip": (hip_x, hhw),
        "right_hip": (hip_x, -hhw),
    }

    # Joints
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

    # Set initial pose (upright, limbs hanging)
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
        # Arm chain
        shoulder_world = torso.local_to_world(torso_anchors[f"{side}_shoulder"])
        ua = bodies[f"{side}_upper_arm"]
        _place_body(ua, _proximal_local("upper_arm", bl), shoulder_world, arm_angle)

        elbow_world = ua.local_to_world(_distal_local("upper_arm", bl))
        fa = bodies[f"{side}_forearm"]
        _place_body(fa, _proximal_local("forearm", bl), elbow_world, arm_angle)

        # Leg chain
        hip_world = torso.local_to_world(torso_anchors[f"{side}_hip"])
        th = bodies[f"{side}_thigh"]
        _place_body(th, _proximal_local("thigh", bl), hip_world, leg_angle)

        knee_world = th.local_to_world(_distal_local("thigh", bl))
        sh = bodies[f"{side}_shin"]
        _place_body(sh, _proximal_local("shin", bl), knee_world, leg_angle)

################################################
# Keypoint extraction
################################################

def extract_keypoints(ragdoll: Ragdoll) -> np.ndarray:
    """Read 12 climbing keypoints from current Pymunk body positions.

    Returns:
        (12, 2) float32 array in board units, ordered:
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

################################################
# Hold attachment / release
################################################

def create_hold_joint(
    ragdoll: Ragdoll,
    space: pymunk.Space,
    limb: str,
    hold_pos_board: tuple[float, float],
) -> bool:
    """Attach a limb's end effector to a hold via a pivot joint.

    The joint anchors the limb at its current tip position (no
    teleporting).  Rejects the grab if the limb is already anchored
    or the tip is beyond RL_GRAB_THRESHOLD of the hold.

    Args:
        ragdoll: Ragdoll to modify.
        space: Pymunk Space containing the ragdoll.
        limb: "left_hand", "right_hand", "left_foot",
            or "right_foot".
        hold_pos_board: Hold position in board units (x, y).

    Returns:
        True if the joint was created.

    For initialization (episode reset, testing), use :func:reset_pose
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
        limb: "left_hand", "right_hand", "left_foot",
            or "right_foot".

    Returns:
        True if a joint was destroyed, False if the limb was free.
    """
    joint = ragdoll.hold_joints.pop(limb, None)
    if joint is None:
        return False
    space.remove(joint)
    return True

################################################
# Motor control
################################################

def set_motor_rate(ragdoll: Ragdoll, joint_name: str, rate: float) -> None:
    """Set the angular velocity target for a joint's motor.

    Enables the motor at its configured torque limit.  A rate of 0
    brakes the joint (resists rotation up to max torque).  To fully
    disable a motor, set ragdoll.joints[name][2].max_force = 0.

    Args:
        ragdoll: Ragdoll containing the joint.
        joint_name: One of left_shoulder, right_shoulder,
            left_elbow, right_elbow, left_hip,
            right_hip, left_knee, right_knee.
        rate: Target angular velocity in rad/s.  Clamped to
            ±RL_MAX_MOTOR_SPEED.
    """
    _, _, motor = ragdoll.joints[joint_name]
    motor.rate = max(-config.RL_MAX_MOTOR_SPEED, min(config.RL_MAX_MOTOR_SPEED, rate))
    # Infer joint type from name and set the configured torque limit
    joint_type = joint_name.split("_", 1)[1]  # "left_shoulder" -> "shoulder"
    motor.max_force = _MOTOR_TORQUES[joint_type]

################################################
# Pose initialization (IK-based)
################################################

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
        bend_sign: -1 bends the joint clockwise from the root->target
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
    """Position the ragdoll at start holds via IK and create joints.

    Places the torso at the centroid of all contacts (hands + feet)
    so the body starts centered rather than hanging at full extension.
    Arms are allowed to start bent.  Adjusts downward only if a
    shoulder can't reach its hand hold.
    
    For arms, elbows always point outward (right elbow rightward,
    left elbow leftward) regardless of shoulder-hand geometry.

    For legs, both IK solutions are evaluated and the knee is chosen
    based on whether the hip is above or below the foot:
      - Right leg: hip below → smaller knee x, hip above → larger knee x
      - Left leg:  hip below → larger knee x, hip above → smaller knee x
    This avoids drop-knee artifacts regardless of the hip-foot geometry.

    Existing hold joints are cleared first.

    Args:
        ragdoll: Ragdoll to reposition.
        space: Pymunk Space containing the ragdoll.
        hand_holds: Limb name -> (x, y) in board units.
            e.g. {"left_hand": (60, 120), "right_hand": (84, 120)}.
        foot_positions: Optional limb name -> (x, y) in board units.
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

    shoulder_offset = bl["torso"] - ragdoll.torso_com_offset
    arm_reach = bl["upper_arm"] + bl["forearm"]

    # ── Torso placement: centroid of all contacts ──
    all_contacts = list(hands_m.values()) + list(feet_m.values())
    torso_x = sum(p[0] for p in all_contacts) / len(all_contacts)

    if feet_m:
        # Center vertically among contacts; clamp so shoulders
        # don't end up above the highest hand
        torso_y = sum(p[1] for p in all_contacts) / len(all_contacts)
        max_hand_y = max(p[1] for p in hands_m.values())
        torso_y = min(torso_y, max_hand_y - shoulder_offset)
    else:
        # No feet: hang below hands
        mid_y = sum(p[1] for p in hands_m.values()) / len(hands_m)
        torso_y = mid_y - shoulder_offset - 0.7 * arm_reach

    torso = ragdoll.bodies["torso"]
    torso.position = (torso_x, torso_y)
    torso.angle = math.pi / 2

    # World positions of shoulder and hip joints
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

    # Lower torso if either arm can't reach its hold
    for _ in range(10):
        all_ok = True
        for limb, hold_m in hands_m.items():
            side = "left" if "left" in limb else "right"
            dist = math.hypot(hold_m[0] - shoulders[side][0],
                              hold_m[1] - shoulders[side][1])
            if dist >= arm_reach * 0.98:
                all_ok = False
                break
        if all_ok:
            break
        torso_y -= 0.05 * arm_reach
        torso.position = (torso_x, torso_y)
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

    # ── Arm IK: elbows always point outward ──
    for limb, hold_m in hands_m.items():
        side = "left" if "left" in limb else "right"
        shoulder = shoulders[side]
        elbow_a = _solve_ik_2bone(shoulder, hold_m, bl["upper_arm"], bl["forearm"],
                                   bend_sign=-1.0)
        elbow_b = _solve_ik_2bone(shoulder, hold_m, bl["upper_arm"], bl["forearm"],
                                   bend_sign=1.0)
        # Right elbow: prefer larger x (outward right)
        # Left elbow: prefer smaller x (outward left)
        if side == "left":
            elbow = elbow_a if elbow_a[0] <= elbow_b[0] else elbow_b
        else:
            elbow = elbow_a if elbow_a[0] >= elbow_b[0] else elbow_b

        ua_angle = math.atan2(elbow[1] - shoulder[1], elbow[0] - shoulder[0])
        _place_body(ragdoll.bodies[f"{side}_upper_arm"],
                    _proximal_local("upper_arm", bl), shoulder, ua_angle)

        fa_angle = math.atan2(hold_m[1] - elbow[1], hold_m[0] - elbow[0])
        _place_body(ragdoll.bodies[f"{side}_forearm"],
                    _proximal_local("forearm", bl), elbow, fa_angle)

    # ── Leg IK: pick knee based on hip-above/below-foot ──
    for side in ("left", "right"):
        limb = f"{side}_foot"
        hip = hips[side]

        if limb in feet_m:
            foot_m = feet_m[limb]
            knee_a = _solve_ik_2bone(hip, foot_m, bl["thigh"], bl["shin"],
                                     bend_sign=-1.0)
            knee_b = _solve_ik_2bone(hip, foot_m, bl["thigh"], bl["shin"],
                                     bend_sign=1.0)
            hip_below = hip[1] < foot_m[1]
            if side == "right":
                # hip below foot → smaller knee x
                # hip above foot → larger knee x
                prefer_smaller_x = hip_below
            else:
                # hip below foot → larger knee x
                # hip above foot → smaller knee x
                prefer_smaller_x = not hip_below
            if prefer_smaller_x:
                knee = knee_a if knee_a[0] <= knee_b[0] else knee_b
            else:
                knee = knee_a if knee_a[0] >= knee_b[0] else knee_b
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

    # Zero all velocities
    for body in ragdoll.bodies.values():
        body.velocity = (0, 0)
        body.angular_velocity = 0

    # Create hold joints (no distance check)
    all_contacts_bu = {**hand_holds, **(foot_positions or {})}
    for limb, pos_board in all_contacts_bu.items():
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

    Not part of the 12 climbing keypoints. The head is drawn as an
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

################################################
# Route preparation for RL
################################################

@dataclass
class RouteConfig:
    """Pre-processed route for the RL environment.

    Attributes:
        holds: Route hold dicts with x, y, name, role_id (board units).
        hold_sequence: Ordered hand-only visit sequence, each dict has
            x, y, hand ('L'/'R'), and name.
        start_hands: Maps 'L'/'R' to the hold name each hand starts on.
            When None, falls back to x-sorted start-role holds.
        start_feet: Maps 'L'/'R' to the hold name each foot starts on.
            When None, route is skipped (starting feet are required).
        stem: Video filename stem this route was derived from.
    """
    holds: list[dict]
    hold_sequence: list[dict]
    start_hands: dict[str, str] | None = None
    start_feet: dict[str, str] | None = None
    stem: str = ""

def prepare_routes_for_rl(
    dataset: dict,
) -> tuple[list[RouteConfig], dict[str, float]]:
    """Prepare route configs and bone lengths from a loaded dataset.

    Loads routes from both train and test splits so that any route
    can be used for evaluation or reference videos.  Use --train-routes
    to restrict which routes are sampled during training.

    For each video, resolves the hand-only hold visit sequence
    from GT poses and packages it with the route holds.  When a manual
    hold order override exists, extracts start-hand assignments from it.
    Skips routes with no detectable hold sequence.

    Args:
        dataset: Dict from load_dataset with train_sequences,
            train_route_holds, train_stems.

    Returns:
        (routes, bone_lengths) where *routes* is a list of
        :class:RouteConfig and *bone_lengths* is a dict from
        :func:compute_rl_bone_lengths.
    """

    bone_lengths = compute_rl_bone_lengths(dataset["train_sequences"])

    routes: list[RouteConfig] = []
    all_sequences = list(dataset["train_sequences"])
    all_route_holds = list(dataset["train_route_holds"])
    all_stems = list(dataset["train_stems"])

    if "test_sequences" in dataset:
        all_sequences += list(dataset["test_sequences"])
        all_route_holds += list(dataset["test_route_holds"])
        all_stems += list(dataset["test_stems"])

    for seq, route_holds, stem in zip(all_sequences, all_route_holds, all_stems):
        climbing_seq = seq[:, config.CLIMBING_KEYPOINT_INDICES, :]
        hold_seq, _ = resolve_hold_sequence_and_targets(
            climbing_seq, route_holds, video_stem=stem,
        )
        if not hold_seq:
            continue

        # Extract start-hand and start-feet assignments from hold order override
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

    return routes, bone_lengths

################################################
# Geometry helpers (reward computation)
################################################

def _compute_cog_meters(ragdoll: Ragdoll) -> np.ndarray:
    """Weighted centre of mass from all Pymunk bodies, in metres."""
    total_mass = 0.0
    cx, cy = 0.0, 0.0
    for body in ragdoll.bodies.values():
        m = body.mass
        px, py = body.position
        cx += m * px
        cy += m * py
        total_mass += m
    return np.array([cx / total_mass, cy / total_mass])

def _point_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance from point *p* to line segment *ab*."""
    ab = b - a
    ab_sq = float(np.dot(ab, ab))
    if ab_sq < 1e-12:
        return float(np.linalg.norm(p - a))
    t = np.clip(np.dot(p - a, ab) / ab_sq, 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))

def _cog_support_distance(cog: np.ndarray, support: np.ndarray) -> float:
    """Distance from CoG to the support polygon (convex hull of contacts).

    Returns 0.0 if the CoG is inside the polygon, otherwise the
    Euclidean distance to the nearest edge.  For 2 contacts the
    polygon degenerates to a line segment.

    Args:
        cog: (2,) centre of gravity position.
        support: (N, 2) anchored end-effector positions, N >= 2.
    """
    n = len(support)
    if n == 2:
        return _point_segment_dist(cog, support[0], support[1])

    # Order vertices CCW by angle from centroid
    centroid = support.mean(axis=0)
    angles = np.arctan2(support[:, 1] - centroid[1],
                        support[:, 0] - centroid[0])
    pts = support[np.argsort(angles)]

    # Inside test via cross products (CCW winding -> all crosses ≥ 0)
    inside = True
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        cross = (b[0] - a[0]) * (cog[1] - a[1]) - (b[1] - a[1]) * (cog[0] - a[0])
        if cross < 0:
            inside = False
            break
    if inside:
        return 0.0

    return min(
        _point_segment_dist(cog, pts[i], pts[(i + 1) % n])
        for i in range(n)
    )
    
def _segments_intersect(a, b, c, d):
    """Check if segment AB intersects segment CD using cross products."""
    def _cross2d(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])
    d1 = _cross2d(c, d, a)
    d2 = _cross2d(c, d, b)
    d3 = _cross2d(a, b, c)
    d4 = _cross2d(a, b, d)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def _is_valid_support_quad(anchored_positions: dict[str, np.ndarray]) -> bool:
    """Check if 4-contact support polygon is non-self-intersecting.

    Uses fixed anatomical winding: LH → RH → RF → LF.  If opposite
    edges of this quadrilateral cross, the body configuration is
    nonsensical (e.g. foot above both hands) and the polygon is invalid.

    Args:
        anchored_positions: Maps limb name to (2,) board-unit position.
            Must contain all four limbs.

    Returns:
        True if the quadrilateral is simple (valid support base).
    """
    lh = anchored_positions["left_hand"]
    rh = anchored_positions["right_hand"]
    rf = anchored_positions["right_foot"]
    lf = anchored_positions["left_foot"]
    # Opposite edge pairs: (LH→RH vs RF→LF) and (RH→RF vs LF→LH)
    if _segments_intersect(lh, rh, rf, lf):
        return False
    if _segments_intersect(rh, rf, lf, lh):
        return False
    return True

################################################
# Observation helpers
################################################

def _get_joint_angle(ragdoll: Ragdoll, joint_name: str) -> float:
    """Relative angle between parent and child body (radians).

    Normalized to (-π, π) and clamped to joint limits to prevent
    Pymunk's unwrapped body angles from accumulating past
    RotaryLimitJoint boundaries or producing unbounded observations.
    """
    parent_name, child_name = _JOINT_BODIES[joint_name]
    raw = ragdoll.bodies[child_name].angle - ragdoll.bodies[parent_name].angle
    angle = (raw + math.pi) % (2 * math.pi) - math.pi
    limits = _get_joint_limits(joint_name)
    if limits is not None:
        angle = max(limits[0], min(limits[1], angle))
    return angle

def _get_joint_angular_velocity(ragdoll: Ragdoll, joint_name: str) -> float:
    """Relative angular velocity between parent and child (rad/s)."""
    parent_name, child_name = _JOINT_BODIES[joint_name]
    return (ragdoll.bodies[child_name].angular_velocity
            - ragdoll.bodies[parent_name].angular_velocity)

def _get_end_effector_positions_bu(ragdoll: Ragdoll) -> np.ndarray:
    """End-effector tip positions in board units, shape (4, 2).

    Order: left_hand, right_hand, left_foot, right_foot.
    """
    m2bu = 1.0 / config.RL_BOARD_UNIT_TO_METERS
    bl = ragdoll.bone_lengths_m
    pts = []
    for limb in _LIMB_NAMES:
        body_name, seg_type = _LIMB_BODIES[limb]
        body = ragdoll.bodies[body_name]
        tip_w = body.local_to_world(_tip_local(seg_type, bl))
        pts.append([tip_w[0] * m2bu, tip_w[1] * m2bu])
    return np.array(pts, dtype=np.float32)

################################################
# Gymnasium environment
################################################

def _normalize_pos(xy: np.ndarray) -> np.ndarray:
    """Normalize board-unit (x, y) positions to ~[0, 1]."""
    out = xy.copy()
    out[..., 0] = (out[..., 0] - BOARD_X_MIN) / (BOARD_X_MAX - BOARD_X_MIN)
    out[..., 1] = (out[..., 1] - BOARD_Y_MIN) / (BOARD_Y_MAX - BOARD_Y_MIN)
    return out

class ClimbingEnv(gymnasium.Env):
    """Gymnasium environment for the RL climbing baseline.

    The agent controls 8 joint-angle deltas and 4 grip state signals
    (1=engage, 0=disengage)
    to climb a Kilter Board route.  Each episode samples a random
    training route (with optional hold-position jitter), positions the
    ragdoll at the start holds, and runs until the route is completed,
    the climber falls, or the step limit is reached.

    The observation is a flat float32 vector containing torso state,
    joint angles/velocities, end-effector positions, anchor status,
    padded route holds, and the current target.  See _build_obs
    for the exact layout.

    Args:
        routes: List of :class:RouteConfig from :func:prepare_routes_for_rl.
        bone_lengths: Dict from :func:compute_rl_bone_lengths (board units).
        seed: RNG seed for route sampling and jitter.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        routes: list[RouteConfig],
        bone_lengths: dict[str, float],
        seed: int | None = None,
    ):
        super().__init__()
        assert routes, "At least one route is required"
        self._routes = routes
        self._bone_lengths = bone_lengths

        obs_dim = (
            2 + 1 + 2 + 1           # torso pos, angle, vel, ang_vel
            + 8 + 8                  # joint angles, angular velocities
            + 8                      # end-effector positions (4 × 2)
            + 4                      # anchor status
            + config.MAX_ROUTE_HOLDS * 3  # hold positions (×2) + roles
            + 2 + 1                  # target position + target hand
        )
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Dict({
            "joint_deltas": spaces.Box(
                -np.pi, np.pi, shape=(8,), dtype=np.float32,
            ),
            "grab_release": spaces.MultiBinary(4),
        })

        self._rng = np.random.default_rng(seed)

        # Per-episode state (initialised in reset)
        self._space: pymunk.Space | None = None
        self._ragdoll: Ragdoll | None = None
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._footholds_established = 0
        self._prev_target_dist: float | None = None
        self._last_reward_breakdown: dict[str, float] = {}

        # Route state
        self._hold_positions_bu: np.ndarray | None = None  # (N, 2)
        self._hold_roles: np.ndarray | None = None          # (N,)
        self._obs_holds: np.ndarray | None = None            # (MAX, 2) normalised
        self._obs_roles: np.ndarray | None = None            # (MAX,) normalised
        self._hold_sequence: list[dict] = []
        self._seq_idx = 0
        self._anchor_hold_idx: dict[str, int] = {}  # limb -> hold idx

    # Observation

    def _build_obs(self) -> np.ndarray:
        """Flat observation vector.

        Layout (all float32):
            [0:2]   torso (x, y) normalised
            [2]     torso angle (rad)
            [3:5]   torso linear velocity
            [5]     torso angular velocity
            [6:14]  joint angles (8)
            [14:22] joint angular velocities (8)
            [22:30] end-effector positions (4×2, normalised)
            [30:34] anchor status per limb
            [34 : 34+MAX*2]  hold positions (normalised, padded)
            [+MAX*2 : +MAX*3] hold roles (normalised, padded)
            [-3:-1] target position (normalised)
            [-1]    target hand (0=left, 1=right)
        """
        rg = self._ragdoll
        torso = rg.bodies["torso"]
        m2bu = 1.0 / config.RL_BOARD_UNIT_TO_METERS

        torso_pos_bu = np.array(torso.position, dtype=np.float32) * m2bu
        torso_pos_norm = _normalize_pos(torso_pos_bu.reshape(1, 2)).ravel()
        torso_vel = np.array(torso.velocity, dtype=np.float32) * m2bu

        joint_angles = np.array(
            [_get_joint_angle(rg, j) for j in _JOINT_NAMES], dtype=np.float32,
        )
        joint_ang_vels = np.array(
            [_get_joint_angular_velocity(rg, j) for j in _JOINT_NAMES],
            dtype=np.float32,
        )

        ee_pos = _normalize_pos(_get_end_effector_positions_bu(rg)).ravel()

        anchor = np.full(4, -1.0, dtype=np.float32)
        for i, limb in enumerate(_LIMB_NAMES):
            if limb in self._anchor_hold_idx:
                anchor[i] = float(self._anchor_hold_idx[limb])

        target_entry = self._current_target()
        target_pos = _normalize_pos(
            np.array([[target_entry["x"], target_entry["y"]]], dtype=np.float32),
        ).ravel()
        target_hand = np.float32(0.0 if target_entry["hand"] == "L" else 1.0)

        return np.concatenate([
            torso_pos_norm,
            [torso.angle],
            torso_vel,
            [torso.angular_velocity],
            joint_angles,
            joint_ang_vels,
            ee_pos,
            anchor,
            self._obs_holds.ravel(),
            self._obs_roles,
            target_pos,
            [target_hand],
        ]).astype(np.float32)

    # Target tracking

    def _current_target(self) -> dict:
        """Current target hold dict (x, y, hand, name)."""
        idx = min(self._seq_idx, len(self._hold_sequence) - 1)
        return self._hold_sequence[idx]

    def _sequence_complete(self) -> bool:
        return self._seq_idx >= len(self._hold_sequence)

    # Step

    def step(
        self, action: dict,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Run one environment step.

        Args:
            action: Dict with joint_deltas (8,) and
                grab_release (4,) arrays.  Grip values are
                absolute: 1=engage (grab if near hold), 0=disengage
                (release if anchored).

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        # Clip joint deltas to prevent extreme output deltas
        joint_deltas = np.clip(
            np.asarray(action["joint_deltas"], dtype=np.float32),
            -np.pi, np.pi,
        )
        grab_release = np.asarray(action["grab_release"], dtype=np.int32)

        rg = self._ragdoll
        space = self._space

        reward_from_grab = 0.0

        # 1. Apply motor targets
        for i, jname in enumerate(_JOINT_NAMES):
            current = _get_joint_angle(rg, jname)
            target = current + float(joint_deltas[i])
            limits = _get_joint_limits(jname)
            if limits is not None:
                target = max(limits[0], min(limits[1], target))
            error = target - current
            rate = np.clip(
                config.RL_MOTOR_GAIN * error,
                -config.RL_MAX_MOTOR_SPEED,
                config.RL_MAX_MOTOR_SPEED,
            )
            set_motor_rate(rg, jname, rate)

        # 2. Resolve grab/release
        for i, limb in enumerate(_LIMB_NAMES):
            wants_grip = grab_release[i] == 1
            is_gripping = limb in rg.hold_joints

            if wants_grip and not is_gripping:
                # Try to grab a route hold
                hold_idx = self._find_nearest_hold(limb)
                if hold_idx is not None:
                    pos = self._hold_positions_bu[hold_idx]
                    if create_hold_joint(rg, space, limb, tuple(pos)):
                        self._anchor_hold_idx[limb] = hold_idx
                        reward_from_grab += self._on_grab(limb, hold_idx)
            elif not wants_grip and is_gripping:
                # Release
                destroy_hold_joint(rg, space, limb)
                self._anchor_hold_idx.pop(limb, None)

        # 3. Step physics
        dt = 1.0 / config.RL_PHYSICS_HZ
        substeps = config.RL_PHYSICS_HZ // config.RL_CONTROL_HZ
        for _ in range(substeps):
            space.step(dt)

        # 4. Detect broken grips (force limit exceeded -> drift)
        bu2m = config.RL_BOARD_UNIT_TO_METERS
        for limb in list(rg.hold_joints):
            body_name, seg_type = _LIMB_BODIES[limb]
            body = rg.bodies[body_name]
            tip_w = body.local_to_world(_tip_local(seg_type, rg.bone_lengths_m))
            joint = rg.hold_joints[limb]
            anchor_w = joint.b.local_to_world(joint.anchor_b)
            drift = math.hypot(tip_w[0] - anchor_w[0], tip_w[1] - anchor_w[1])
            if drift / bu2m > config.RL_GRAB_THRESHOLD * 2:
                destroy_hold_joint(rg, space, limb)
                self._anchor_hold_idx.pop(limb, None)

        self._step_count += 1
        self._steps_since_arrival += 1

        # 5. Compute reward
        reward = self._compute_reward() + reward_from_grab
        self._last_reward_breakdown["grab_bonus"] = reward_from_grab
        self._cumulative_reward += reward

        # 6. Check termination
        terminated, truncated, outcome = self._check_termination()
        if terminated and outcome == "fall":
            reward += config.RL_CONTACT_PENALTY_TERMINAL
            self._cumulative_reward += config.RL_CONTACT_PENALTY_TERMINAL
            self._last_reward_breakdown["fall_penalty"] = config.RL_CONTACT_PENALTY_TERMINAL
        elif truncated and outcome == "timeout":
            reward += config.RL_CONTACT_PENALTY_TIMEOUT
            self._cumulative_reward += config.RL_CONTACT_PENALTY_TIMEOUT
            self._last_reward_breakdown["timeout_penalty"] = config.RL_CONTACT_PENALTY_TIMEOUT

        self._last_reward_breakdown["cumulative"] = self._cumulative_reward

        obs = self._build_obs()
        info = self._build_info(outcome)

        return obs, reward, terminated, truncated, info

    # Reset

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment with a random training route.

        Args:
            seed: Optional RNG seed.
            options: Optional dict.
                route_index selects a specific
                route (for deterministic testing).
                hold_jitter overrides the jitter magnitude
                (default RL_HOLD_JITTER; pass 0.0 for eval).

        Returns:
            (obs, info)
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        options = options or {}
        route_idx = options.get(
            "route_index", self._rng.integers(len(self._routes)),
        )
        route = self._routes[route_idx]
        self._route_idx = route_idx

        # Apply hold jitter
        jitter_amount = options.get("hold_jitter", config.RL_HOLD_JITTER)
        jitter = self._rng.uniform(
            -jitter_amount, jitter_amount,
            size=(len(route.holds), 2),
        )
        jittered_positions = np.array(
            [[h["x"], h["y"]] for h in route.holds], dtype=np.float32,
        ) + jitter

        name_to_jittered = {
            h["name"]: jittered_positions[i]
            for i, h in enumerate(route.holds)
        }

        # Store route state
        self._hold_positions_bu = jittered_positions
        self._hold_roles = np.array(
            [_ROLE_MAP.get(h["role_id"], 13) for h in route.holds],
            dtype=np.int64,
        )
        self._hold_sequence = []
        for entry in route.hold_sequence:
            pos = name_to_jittered.get(entry.get("name"))
            if pos is not None:
                self._hold_sequence.append(
                    {**entry, "x": float(pos[0]), "y": float(pos[1])},
                )
            else:
                self._hold_sequence.append(entry)

        # Observation hold arrays (padded + normalised)
        n_holds = len(self._hold_positions_bu)
        self._obs_holds = np.zeros(
            (config.MAX_ROUTE_HOLDS, 2), dtype=np.float32,
        )
        self._obs_holds[:n_holds] = _normalize_pos(self._hold_positions_bu)
        self._obs_roles = np.zeros(config.MAX_ROUTE_HOLDS, dtype=np.float32)
        self._obs_roles[:n_holds] = self._hold_roles / 15.0

        if route.start_hands is not None:
            hand_holds = {
                "left_hand": tuple(name_to_jittered[route.start_hands["L"]]),
                "right_hand": tuple(name_to_jittered[route.start_hands["R"]]),
            }
        else:
            # Fallback: start-role holds sorted by x (lower x = left)
            start_indices = [
                i for i, r in enumerate(self._hold_roles) if r == _ROLE_START
            ]
            if len(start_indices) < 2:
                start_indices = list(range(min(2, n_holds)))
            start_positions = self._hold_positions_bu[start_indices[:2]]
            sorted_starts = start_positions[np.argsort(start_positions[:, 0])]
            hand_holds = {
                "left_hand": tuple(sorted_starts[0]),
                "right_hand": tuple(sorted_starts[-1]),
            }

        # Foot holds from annotation
        foot_holds = {}
        if route.start_feet is not None:
            for side_key, hold_name in route.start_feet.items():
                limb = "left_foot" if side_key == "L" else "right_foot"
                if hold_name in name_to_jittered:
                    foot_holds[limb] = tuple(name_to_jittered[hold_name])

        # Create physics world + ragdoll
        self._space = create_space()
        self._ragdoll = create_ragdoll(
            self._space, self._bone_lengths,
        )
        reset_pose(
            self._ragdoll, self._space, hand_holds,
            foot_holds if foot_holds else None,
        )

        # Settle physics with motors braking to preserve IK pose
        for jname in _JOINT_NAMES:
            set_motor_rate(self._ragdoll, jname, 0.0)
        dt = 1.0 / config.RL_PHYSICS_HZ
        for _ in range(config.RL_SETTLE_STEPS):
            self._space.step(dt)
        # Release brakes so the agent starts with passive joints
        for jname in _JOINT_NAMES:
            self._ragdoll.joints[jname][2].max_force = 0.0

        # Track anchor state
        self._anchor_hold_idx = {}
        for limb, hold_pos_bu in hand_holds.items():
            idx = self._nearest_hold_index(np.array(hold_pos_bu))
            if idx is not None:
                self._anchor_hold_idx[limb] = idx
        for limb, hold_pos_bu in foot_holds.items():
            idx = self._nearest_hold_index(np.array(hold_pos_bu))
            if idx is not None:
                self._anchor_hold_idx[limb] = idx

        # Episode bookkeeping
        self._seq_idx = 0
        self._step_count = 0
        self._steps_since_arrival = 0
        self._cumulative_reward = 0.0
        self._footholds_established = 0
        self._prev_target_dist = self._target_distance()

        return self._build_obs(), self._build_info("running")

    # Grab helpers

    def _find_nearest_hold(self, limb: str) -> int | None:
        """Find nearest grabbable hold for *limb*, or None."""
        ee_pos = _get_end_effector_positions_bu(self._ragdoll)
        limb_idx = _LIMB_NAMES.index(limb)
        pos = ee_pos[limb_idx]
        is_hand = "hand" in limb

        dists = np.linalg.norm(self._hold_positions_bu - pos, axis=1)
        for idx in np.argsort(dists):
            if dists[idx] > config.RL_GRAB_THRESHOLD:
                return None
            if is_hand and self._hold_roles[idx] == _ROLE_FOOT_ONLY:
                continue
            return int(idx)
        return None

    def _nearest_hold_index(self, pos_bu: np.ndarray) -> int | None:
        """Index of nearest hold to a board-unit position (no threshold)."""
        if len(self._hold_positions_bu) == 0:
            return None
        dists = np.linalg.norm(self._hold_positions_bu - pos_bu, axis=1)
        return int(np.argmin(dists))

    def _on_grab(self, limb: str, hold_idx: int) -> float:
        """Bookkeeping after a successful grab.

        Returns:
            Bonus reward earned (arrival or finish), 0.0 otherwise.
        """
        bonus = 0.0
        if "foot" in limb and hold_idx >= 0:
            self._footholds_established += 1

        if self._seq_idx < len(self._hold_sequence):
            target = self._current_target()
            expected_limb = "left_hand" if target["hand"] == "L" else "right_hand"
            if limb == expected_limb:
                target_pos = np.array([target["x"], target["y"]])
                hold_pos = self._hold_positions_bu[hold_idx]
                if np.linalg.norm(hold_pos - target_pos) < config.RL_GRAB_THRESHOLD:
                    self._seq_idx += 1
                    self._steps_since_arrival = 0
                    self._prev_target_dist = self._target_distance()
                    bonus = config.RL_REWARD_ARRIVAL_BONUS
                    if self._sequence_complete():
                        bonus += config.RL_REWARD_FINISH_BONUS
        return bonus

    # Reward

    def _target_distance(self) -> float:
        """Distance from target wrist to target hold (board units)."""
        if self._sequence_complete():
            return 0.0
        target = self._current_target()
        target_limb_idx = 0 if target["hand"] == "L" else 1  # index into ee
        ee = _get_end_effector_positions_bu(self._ragdoll)
        hold = np.array([target["x"], target["y"]])
        return float(np.linalg.norm(ee[target_limb_idx] - hold))

    def _compute_reward(self) -> float:
        """Compute shaped reward for the current step.

        Stores component breakdown in _last_reward_breakdown for
        debug visualization.
        """
        bd: dict[str, float] = {}
        reward = config.RL_REWARD_STEP_PENALTY
        bd["step"] = config.RL_REWARD_STEP_PENALTY

        # Hand proximity: distance reduction toward target
        hand_prox = 0.0
        if not self._sequence_complete():
            curr_dist = self._target_distance()
            if self._prev_target_dist is not None:
                hand_prox = (self._prev_target_dist - curr_dist) * config.RL_HAND_PROXIMITY_SCALE
            self._prev_target_dist = curr_dist
        reward += hand_prox
        bd["hand_prox"] = hand_prox

        # Contact / stability
        contact = self._contact_stability_reward()
        reward += contact
        bd["contact"] = contact

        self._last_reward_breakdown = bd
        bd["steps_remaining"] = config.RL_STEPS_PER_TARGET - self._steps_since_arrival
        return reward

    def _contact_stability_reward(self) -> float:
        """Tiered contact reward based on anchor configuration.

        For 4-contact configurations, validates that the support
        quadrilateral (LH → RH → RF → LF) is non-self-intersecting.
        Self-intersecting polygons (e.g. foot above both hands) are
        treated as having no useful foot contacts.
        """
        hand_contacts = sum(
            1 for limb in ("left_hand", "right_hand")
            if limb in self._anchor_hold_idx
        )
        foot_contacts = sum(
            1 for limb in ("left_foot", "right_foot")
            if limb in self._anchor_hold_idx
        )

        if hand_contacts == 0:
            return 0.0  # will be terminated

        if foot_contacts == 0:
            if hand_contacts == 2:
                return config.RL_CONTACT_PENALTY_2HANDS
            return config.RL_CONTACT_PENALTY_1HAND

        # At least 1 hand + 1 foot on wall: CoG-based signal
        m2bu = 1.0 / config.RL_BOARD_UNIT_TO_METERS
        cog = _compute_cog_meters(self._ragdoll) * m2bu

        anchored_pos = {}
        for limb in _LIMB_NAMES:
            idx = self._anchor_hold_idx.get(limb)
            if idx is not None and idx >= 0:
                anchored_pos[limb] = self._hold_positions_bu[idx]

        if len(anchored_pos) < 2:
            return 0.0

        # Reject nonsensical 4-contact configurations (e.g. foot above hands)
        if len(anchored_pos) == 4 and not _is_valid_support_quad(anchored_pos):
            return config.RL_CONTACT_PENALTY_2HANDS

        anchored = np.array(list(anchored_pos.values()))
        dist = _cog_support_distance(cog, anchored)
        if dist <= 0.0:
            return config.RL_CONTACT_REWARD_COG_INSIDE
        return config.RL_CONTACT_COG_DISTANCE_SCALE * dist

    # Termination

    def _check_termination(self) -> tuple[bool, bool, str]:
        """Check episode end conditions.

        Uses a per-target step budget: the agent gets
        ``RL_STEPS_PER_TARGET`` steps to reach each successive
        target hand hold.  The counter resets on every arrival.

        Returns:
            (terminated, truncated, outcome) where outcome is one of
            "running", "success", "fall", "timeout".
        """
        hand_contacts = sum(
            1 for limb in ("left_hand", "right_hand")
            if limb in self._ragdoll.hold_joints
        )

        if hand_contacts == 0:
            return True, False, "fall"

        # Success: both hands on finish holds
        if self._sequence_complete():
            hands_on_finish = all(
                self._anchor_hold_idx.get(limb) is not None
                and self._anchor_hold_idx[limb] >= 0
                and self._hold_roles[self._anchor_hold_idx[limb]] == _ROLE_FINISH
                for limb in ("left_hand", "right_hand")
            )
            if hands_on_finish:
                return True, False, "success"

        if self._steps_since_arrival >= config.RL_STEPS_PER_TARGET:
            return False, True, "timeout"

        return False, False, "running"

    # Info

    def _build_info(self, outcome: str) -> dict:
        return {
            "holds_visited": self._seq_idx,
            "total_holds": len(self._hold_sequence),
            "hold_visit_rate": (
                self._seq_idx / max(len(self._hold_sequence), 1)
            ),
            "footholds_established": self._footholds_established,
            "outcome": outcome,
            "cumulative_reward": self._cumulative_reward,
            "step_count": self._step_count,
            "route_stem": self._routes[self._route_idx].stem,
        }

    # Keypoint extraction (for evaluation)

    def extract_episode_keypoints(self) -> np.ndarray:
        """Current-frame keypoints for recording, shape (12, 2)."""
        return extract_keypoints(self._ragdoll)

def rollout_episode(
    env: ClimbingEnv,
    action_fn: callable,
    route_index: int = 0,
    max_steps: int | None = None,
) -> dict:
    """Run one episode and collect per-step data for rendering/metrics.

    Args:
        env: ClimbingEnv instance.
        action_fn: (env, step) -> action dict. Called each step.
        route_index: Which route to reset with.

    Returns:
        Dict with keys poses, head_positions, targets,
        cog_positions, support_polygons, rewards,
        outcome, info.  Each list has one entry per step
        (including the initial reset frame).
    """
    obs, info = env.reset(options={"route_index": route_index, "hold_jitter": 0.0})
    m2bu = 1.0 / config.RL_BOARD_UNIT_TO_METERS

    poses, head_positions, targets = [], [], []
    cog_positions, support_polygons = [], []
    target_hands = []
    rewards = []
    reward_breakdowns = []

    def snapshot():
        poses.append(extract_keypoints(env._ragdoll))
        head_positions.append(extract_head_position(env._ragdoll))
        t = env._current_target()
        targets.append((t["x"], t["y"]))
        target_hands.append(t["hand"])  # "L" or "R"
        cog_positions.append(_compute_cog_meters(env._ragdoll) * m2bu)

        anchored_map = {}
        for limb in _LIMB_NAMES:
            idx = env._anchor_hold_idx.get(limb)
            if idx is not None and idx >= 0:
                anchored_map[limb] = env._hold_positions_bu[idx]
        if len(anchored_map) == 4 and not _is_valid_support_quad(anchored_map):
            support_polygons.append(np.empty((0, 2)))
        else:
            anchored = list(anchored_map.values())
            support_polygons.append(
                np.array(anchored) if anchored else np.empty((0, 2)),
            )

    snapshot()
    outcome = "running"
    step = 0

    while max_steps is None or step < max_steps:
        action = action_fn(env, step)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        reward_breakdowns.append(dict(env._last_reward_breakdown))
        snapshot()
        step += 1
        if terminated or truncated:
            outcome = info["outcome"]
            break

    return {
        "poses": poses,
        "head_positions": head_positions,
        "targets": targets,
        "target_hands": target_hands,
        "cog_positions": cog_positions,
        "support_polygons": support_polygons,
        "rewards": rewards,
        "reward_breakdowns": reward_breakdowns,
        "outcome": outcome,
        "info": info,
        "route_holds": env._routes[route_index].holds,
    }