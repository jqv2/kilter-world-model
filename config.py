from pathlib import Path

import torch

########################
# Paths
########################
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_VIDEO_DIR = DATA_DIR / "raw"
POSES_DIR = DATA_DIR / "poses"


########################
# Board Information
########################
PRODUCT_SIZE_ID = 10
MAX_ROUTE_HOLDS = 40  # max holds per route (for padding); most Kilter routes have 6-15

########################
# ViTPose
########################

# Variants:
# "usyd-community/vitpose-base-simple"
# "usyd-community/vitpose-plus-base"
# "usyd-community/vitpose-plus-large"
VITPOSE_MODEL_ID = "usyd-community/vitpose-base-simple"

# Person detector to crop bounding box before ViTPose
PERSON_DETECTOR_ID = "PekingU/rtdetr_r50vd_coco_o365"

# Minimum confidence for person detector to accept bounding box
PERSON_DETECTION_THRESHOLD = 0.5

# Minimum confidence for keypoint to be considered valid
KEYPOINT_CONFIDENCE_THRESHOLD = 0.3


########################
# COCO Keypoint Schema
########################

# ViTPose (COCO) outputs these keypoints in this order
COCO_KEYPOINT_NAMES = [
    "nose",             # 0
    "left_eye",         # 1
    "right_eye",        # 2
    "left_ear",         # 3
    "right_ear",        # 4
    "left_shoulder",    # 5
    "right_shoulder",   # 6
    "left_elbow",       # 7
    "right_elbow",      # 8
    "left_wrist",       # 9
    "right_wrist",      # 10
    "left_hip",         # 11
    "right_hip",        # 12
    "left_knee",        # 13
    "right_knee",       # 14
    "left_ankle",       # 15
    "right_ankle",      # 16
]

# Keypoints used by the model (drop head: nose, eyes, ears)
CLIMBING_KEYPOINT_INDICES = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
NUM_CLIMBING_KEYPOINTS = len(CLIMBING_KEYPOINT_INDICES)

# Skeletal connections between keypoints for visualization
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),         # head
    (5, 3), (6, 4),                         # shoulders to ears
    (5, 6),                                 # shoulders
    (5, 7), (7, 9),                         # left arm
    (6, 8), (8, 10),                        # right arm
    (5, 11), (6, 12),                       # torso
    (11, 12),                               # hips
    (11, 13), (13, 15),                     # left leg
    (12, 14), (14, 16),                     # right leg
]

# Skeleton connections for climbing keypoints only (indices into CLIMBING_KEYPOINT_INDICES)
CLIMBING_SKELETON = [
    (0, 1),              # shoulders
    (0, 2), (2, 4),     # left arm
    (1, 3), (3, 5),     # right arm
    (0, 6), (1, 7),     # torso
    (6, 7),              # hips
    (6, 8), (8, 10),    # left leg
    (7, 9), (9, 11),    # right leg
]

########################
# Video
########################

VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv"}

########################
# Pose Cleaning
########################

BOARD_SPACE_MAX_DISPLACEMENT = 5.0  # max mean keypoint displacement per frame in board units

########################
# Evaluation
########################

# Metrics are computed only on keypoints above this confidence
EVAL_CONFIDENCE_THRESHOLD = KEYPOINT_CONFIDENCE_THRESHOLD

########################
# Calibration
########################

CALIBRATIONS_DIR = DATA_DIR / "calibrations"
CALIBRATION_FRAMES_DIR = DATA_DIR / "calibration_frames"
DETECTED_FEATURES_DIR = DATA_DIR / "detected_features"
WARPED_DIR = DATA_DIR / "warped"
ROUTE_EDITS_DIR = DATA_DIR / "route_edits"
HOLD_ORDERS_DIR = DATA_DIR / "hold_orders"

WARP_SCALE = 4        # pixels per board unit for warped editing videos
WARP_FPS = 10         # output frame rate for warped editing videos

########################
# Model
########################

NUM_KEYPOINTS = 17
CONTEXT_WINDOW = 30      # ~1 second at 30fps
MODEL_HIDDEN_DIM = 128
MODEL_LAYERS = 2
MODEL_HEADS = 4
MODEL_DROPOUT = 0.1

########################
# Training
########################

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
BONE_LOSS_WEIGHT = 0.5          # weight of bone-length constraint relative to MSE
NOISE = 0.5
SCHEDULED_SAMPLING_MAX = 0.5    # max probability of replacing last context frame with model's own prediction
TRAINING_STRIDES = [1,3,6,10]   # temporal strides for dataset construction
ROLLOUT_STRIDE = 1              # temporal stride during autoregressive rollout (match single-stride training)

########################
# Structured Model
########################

# Limb endpoint keypoint indices (wrists and ankles)
LIMB_KEYPOINTS = {
    0: 4,   # left hand  -> left_wrist (COCO idx 9, climbing idx 4)
    1: 5,   # right hand -> right_wrist (COCO idx 10, climbing idx 5)
    2: 10,  # left foot  -> left_ankle (COCO idx 15, climbing idx 10)
    3: 11,  # right foot -> right_ankle (COCO idx 16, climbing idx 11)
}
LIMB_KEYPOINTS_COCO = {
    0: 9,   # left hand  -> left_wrist
    1: 10,  # right hand -> right_wrist
    2: 15,  # left foot  -> left_ankle
    3: 16,  # right foot -> right_ankle
}
NUM_LIMBS = 4
HAND_LIMBS = {0, 1}
FOOT_LIMBS = {2, 3}

HAND_ARRIVAL_THRESHOLD = 8.0    # board units
FOOT_ARRIVAL_THRESHOLD = 8.0    # board units
HOLD_ARRIVAL_FRAMES = 15        # consecutive frames to confirm arrival
ROLLOUT_ARRIVAL_THRESHOLD_HAND = 8.0   # relaxed threshold for autoregressive rollout
ROLLOUT_ARRIVAL_THRESHOLD_FOOT = 8.0   # relaxed threshold for autoregressive rollout
ROLLOUT_HOLD_TIMEOUT = 120               # advance to next hold after this many rollout steps with no arrival

########################
# RL Baseline — Physics
########################

RL_BOARD_UNIT_TO_METERS = 0.025       # 4 board units = 10 cm
RL_WALL_ANGLE_DEG = 30.0              # overhang angle (degrees)
RL_GRAVITY = 9.81                     # m/s^2
RL_PHYSICS_HZ = 60                    # physics simulation rate (timestep = 1/60 s)
RL_CONTROL_HZ = 10                    # agent decisions per second of sim time
RL_BODY_MASS_KG = 77.0                # total ragdoll mass
RL_SPACE_DAMPING = 0.93              # global velocity damping (1.0 = none)
RL_BONE_LENGTH_PERCENTILE = 97.0      # percentile for bone length estimation
RL_HEAD_RADIUS = 0.10                 # meters

RL_MOTOR_TORQUE_HIP = 200.0 * 2     # Nm (2x human, compensates for 2D)
RL_MOTOR_TORQUE_KNEE = 175.0 * 2
RL_MOTOR_TORQUE_SHOULDER = 100.0 * 2
RL_MOTOR_TORQUE_ELBOW = 80.0 * 2
RL_MAX_MOTOR_SPEED = 4.0              # rad/s, joint motor speed cap

RL_GRIP_MAX_FORCE = 50000.0            # Newtons, per hold joint
RL_GRAB_THRESHOLD = 6.0               # board units, max distance to grab a hold

# RL Baseline — Environment
RL_STEP_LIMIT = 350
RL_STEPS_PER_TARGET = 60             # steps allowed per target hand hold
RL_HOLD_JITTER = 0.5               # board units, ±random offset per hold per reset
RL_SETTLE_STEPS = 30               # Pymunk substeps during reset to find equilibrium
RL_MOTOR_GAIN = 10.0               # proportional gain for motor target chasing

# RL Baseline — Rewards
RL_REWARD_STEP_PENALTY = -0.01
RL_REWARD_ARRIVAL_BONUS = 50.0
RL_REWARD_FINISH_BONUS = 100.0
RL_HAND_PROXIMITY_SCALE = 2.0
RL_CONTACT_REWARD_COG_INSIDE = 0.1
RL_CONTACT_COG_DISTANCE_SCALE = -0.05
RL_CONTACT_PENALTY_2HANDS = -0.5
RL_CONTACT_PENALTY_1HAND = -1.0
RL_CONTACT_PENALTY_TERMINAL = -20
RL_CONTACT_PENALTY_TIMEOUT = 0
RL_BEST_RETURN_THRESHOLD = 2.0        # minimum improvement to trigger best_return milestone

# RL Baseline — PPO Training
RL_PPO_LR = 1e-4
RL_PPO_BATCH_SIZE = 4096           # frames collected per PPO update
RL_PPO_MINIBATCH_SIZE = 128
RL_PPO_EPOCHS = 5                 # SGD passes per PPO update
RL_PPO_CLIP_EPSILON = 0.2
RL_PPO_GAMMA = 0.993
RL_PPO_GAE_LAMBDA = 0.95
RL_PPO_ENTROPY_COEF = 0.0075
RL_PPO_MAX_GRAD_NORM = 0.5
RL_HIDDEN_DIM = 128
RL_HIDDEN_LAYERS = 2
RL_TOTAL_FRAMES = 1_000_000
RL_CHECKPOINT_INTERVAL = 50_000
RL_CHECKPOINT_DIR = DATA_DIR / "rl_checkpoints"
RL_LOG_DIR = DATA_DIR / "rl_logs"
RL_EVAL_VIDEO_INTERVAL = 500        # episodes between eval video renders
RL_VIZ_DIR = DATA_DIR / "rl_viz"

def get_device(override: str | None = None) -> "torch.device":
    """Auto-detect best available device, or use override if given."""
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")