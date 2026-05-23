"""
ViTPose Wrapper using HuggingFace Transformers

Pipeline:
1. RT-DETR to detect person bounding boxes in each frame
2. ViTPose to estimate keypoints within each bounding box
"""

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)

import config

def load_models(device: str | None = None) -> dict:
    """
    Load the person detector and ViTPose model.

    Args:
        device: 'cuda', 'mps', 'cpu', or None to auto-detect.

    Returns:
        Dict with keys 'person_processor', 'person_model',
        'pose_processor', 'pose_model', 'device'.
    """
    
    if device is None:
        device = str(config.get_device())

    person_processor = AutoProcessor.from_pretrained(config.PERSON_DETECTOR_ID)
    person_model = RTDetrForObjectDetection.from_pretrained(
        config.PERSON_DETECTOR_ID
    ).to(device)

    pose_processor = AutoProcessor.from_pretrained(config.VITPOSE_MODEL_ID)
    pose_model = VitPoseForPoseEstimation.from_pretrained(
        config.VITPOSE_MODEL_ID
    ).to(device)

    return {
        "person_processor": person_processor,
        "person_model": person_model,
        "pose_processor": pose_processor,
        "pose_model": pose_model,
        "device": device,
    }
    
    
def detect_person(image: Image.Image, models: dict) -> np.ndarray | None:
    """
    Detect the largest person bounding box in the image.

    Args:
        image: PIL Image (RGB).
        models: Dict returned by load_models().

    Returns:
        Bounding box as np.ndarray [x1, y1, x2, y2] or None if no person found.
    """
    
    processor = models["person_processor"]
    model = models["person_model"]
    device = models["device"]

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([(image.height, image.width)]),
        threshold=config.PERSON_DETECTION_THRESHOLD,
    )[0]

    # COCO label 0 is a person
    person_mask = results["labels"] == 0
    if not person_mask.any():
        return None

    person_boxes = results["boxes"][person_mask]
    areas = (person_boxes[:, 2] - person_boxes[:, 0]) * (person_boxes[:, 3] - person_boxes[:, 1])
    best_idx = areas.argmax()
    return person_boxes[best_idx].cpu().numpy()


def estimate_keypoints(image: Image.Image, box: np.ndarray, models: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate keypoints for a single person bounding box.

    Args:
        image: PIL Image (RGB).
        box: Bounding box as np.ndarray [x1, y1, x2, y2].
        models: Dict returned by load_models().

    Returns:
        Tuple of (keypoints, scores):
            keypoints: np.ndarray shape (17, 2); (x, y) pixel coordinates.
            scores: np.ndarray shape (17,); confidence per keypoint.
    """
    
    processor = models["pose_processor"]
    model = models["pose_model"]
    device = models["device"]

    # RT-DETR returns [x1, y1, x2, y2], ViTPose expects COCO format [x, y, width, height]
    x1, y1, x2, y2 = box.tolist()
    coco_box = [x1, y1, x2 - x1, y2 - y1]
    boxes_input = [[coco_box]]
    inputs = processor(image, boxes=boxes_input, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process: convert probability heatmaps to keypoint coordinates
    results = processor.post_process_pose_estimation(
        outputs,
        boxes=boxes_input,
    )[0]

    # results is a list of dicts, one per detected box
    if not results:
        return np.zeros((17, 2)), np.zeros(17)

    result = results[0]
    keypoints = result["keypoints"].cpu().numpy()
    scores = result["scores"].cpu().numpy()
    
    return keypoints, scores


def extract_frame_pose(image: Image.Image, models: dict) -> dict | None:
    """
    Full single-frame pipeline: detect person -> estimate keypoints.

    Args:
        image: PIL Image (RGB).
        models: Dict returned by load_models().

    Returns:
        Dict with 'box', 'keypoints', 'scores', or None if no person detected.
    """
    box = detect_person(image, models)
    if box is None:
        return None

    keypoints, scores = estimate_keypoints(image, box, models)
    return {
        "box": box,
        "keypoints": keypoints,
        "scores": scores,
    }