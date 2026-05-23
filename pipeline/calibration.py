"""
Calibration utilities: load saved calibrations, compute homographies,
and transform pixel-space keypoints to board-space coordinates.

Calibration JSONs are produced by the calibration UI (calibration_server.py)
and stored in data/calibrations/ with the structure:
    {
        "video_id": "session1/climb_001",
        "main_points": [{"pixel": [px, py], "board": [bx, by], "label": "..."}],
        "kick_points": [{"pixel": [px, py], "board": [bx, by], "label": "..."}]
    }

Homographies are computed from point pairs on demand, not stored.
Two separate homographies handle the main board (angled) and kickboard (vertical).
"""

import copy
import json
from pathlib import Path

import cv2
import numpy as np

import config

# Minimum calibration points required per plane
MIN_CALIBRATION_POINTS = 4


def load_calibration(video_stem: str, calibrations_dir: Path | None = None) -> dict:
    """
    Load a calibration JSON for a video.

    Searches recursively in calibrations_dir for a file matching
    '{video_stem}_calibration.json'.

    Args:
        video_stem: Video filename without extension (e.g. 'climb_001').
        calibrations_dir: Directory to search. Defaults to config.CALIBRATIONS_DIR.

    Returns:
        Parsed calibration dict with 'main_points' and 'kick_points'.

    Raises:
        FileNotFoundError: If no calibration file exists for this video.
    """
    cal_dir = calibrations_dir or config.CALIBRATIONS_DIR
    filename = f"{video_stem}_calibration.json"

    matches = list(cal_dir.rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"No calibration found for '{video_stem}' in {cal_dir}"
        )

    with open(matches[0]) as f:
        return json.load(f)


def compute_homographies(
    calibration: dict,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Compute homography matrices from calibration point pairs.

    Args:
        calibration: Dict with 'main_points' and 'kick_points', each a list
            of {'pixel': [px, py], 'board': [bx, by]}.

    Returns:
        (H_main, H_kick) where each is a 3x3 homography matrix.
        H_kick is None if fewer than MIN_CALIBRATION_POINTS kick points exist.

    Raises:
        ValueError: If main board has fewer than MIN_CALIBRATION_POINTS points.
        RuntimeError: If homography computation fails.
    """
    main_pts = calibration.get("main_points", [])
    kick_pts = calibration.get("kick_points", [])

    if len(main_pts) < MIN_CALIBRATION_POINTS:
        raise ValueError(
            f"Main board has {len(main_pts)} points, "
            f"need at least {MIN_CALIBRATION_POINTS}"
        )

    H_main = _compute_homography(main_pts, "main board")

    H_kick = None
    if len(kick_pts) >= MIN_CALIBRATION_POINTS:
        H_kick = _compute_homography(kick_pts, "kickboard")

    return H_main, H_kick


def _compute_homography(points: list[dict], label: str) -> np.ndarray:
    """Compute a single homography from a list of point pair dicts."""
    pixel = np.array([p["pixel"] for p in points], dtype=np.float64)
    board = np.array([p["board"] for p in points], dtype=np.float64)

    H, status = cv2.findHomography(pixel, board, cv2.RANSAC, 3.0)

    if H is None:
        raise RuntimeError(f"Homography computation failed for {label}")

    return H


def kick_threshold_px(calibration: dict) -> float | None:
    """
    Estimate the pixel y-coordinate separating main board from kickboard.

    Uses the midpoint between the lowest main-board calibration point
    and the highest kickboard calibration point in pixel space.

    Returns:
        Pixel y threshold, or None if kickboard has no calibration points.
    """
    main_pts = calibration.get("main_points", [])
    kick_pts = calibration.get("kick_points", [])

    if not kick_pts or not main_pts:
        return None

    # In pixel space, y increases downward. Kickboard points have larger y
    main_max_y = max(p["pixel"][1] for p in main_pts)
    kick_min_y = min(p["pixel"][1] for p in kick_pts)

    return (main_max_y + kick_min_y) / 2.0


def transform_keypoints(
    keypoints: np.ndarray,
    H_main: np.ndarray,
    H_kick: np.ndarray | None = None,
    kick_y_threshold: float | None = None,
) -> np.ndarray:
    """
    Transform pixel-space keypoints to board coordinates.

    Each keypoint is transformed by either the main board or kickboard
    homography depending on its pixel y-coordinate relative to the threshold.
    If no kickboard homography is provided, all keypoints use the main board.

    Args:
        keypoints: (17, 2) array of pixel coordinates.
        H_main: 3x3 main board homography.
        H_kick: 3x3 kickboard homography, or None.
        kick_y_threshold: Pixel y value separating main board (above/smaller y)
            from kickboard (below/larger y). Required if H_kick is provided.

    Returns:
        (17, 2) array of board-space coordinates.
    """
    if H_kick is not None and kick_y_threshold is not None:
        main_mask = keypoints[:, 1] < kick_y_threshold
        kick_mask = ~main_mask

        result = np.empty_like(keypoints)

        if main_mask.any():
            result[main_mask] = _apply_homography(
                keypoints[main_mask], H_main
            )
        if kick_mask.any():
            result[kick_mask] = _apply_homography(
                keypoints[kick_mask], H_kick
            )
        return result

    return _apply_homography(keypoints, H_main)


def _apply_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply a homography to an (N, 2) array of points."""
    if len(points) == 0:
        return points
    reshaped = points.reshape(-1, 1, 2).astype(np.float64)
    transformed = cv2.perspectiveTransform(reshaped, H)
    return transformed.reshape(-1, 2)


def transform_pose_sequence(
    frames: list[dict],
    calibration: dict,
) -> list[dict]:
    """
    Transform all keypoints in a pose sequence from pixel to board space.

    Frames with no keypoints are passed through unchanged.

    Args:
        frames: List of frame dicts, each with 'keypoints' (list or None).
        calibration: Calibration dict for this video.

    Returns:
        New list of frame dicts with keypoints in board-space coordinates.
        Original frames are not modified.
    """
    H_main, H_kick = compute_homographies(calibration)
    threshold = kick_threshold_px(calibration)

    out = []
    for frame in frames:
        f = copy.deepcopy(frame)
        if f["keypoints"] is not None:
            kp = np.array(f["keypoints"])
            kp_board = transform_keypoints(kp, H_main, H_kick, threshold)
            f["keypoints"] = kp_board.tolist()
        out.append(f)
    return out