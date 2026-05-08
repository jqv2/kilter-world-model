from pathlib import Path


########################
# Paths
########################
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_VIDEO_DIR = DATA_DIR / "raw"
POSES_DIR = DATA_DIR / "poses"


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


########################
# Video
########################

VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv"}